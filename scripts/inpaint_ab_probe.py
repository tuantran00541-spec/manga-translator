"""A/B probe for inpaint colour problems on selected slices of a real chapter.

Requires the chapter downloaded by scripts/inpaint_audit_run.py (same CHAPTER_ID).
For each selected slice it records which route every inpaint region took
(flat "smart fill" or LaMa, with the fill colour) and cleans the slice twice:

  A  current code
  B  masked pixels zeroed before they reach LaMa (what LaMa's own
     img * (1 - mask) does; a no-op if the ONNX graph already masks)

and writes original | A | B crops of every changed region, plus paths.json.

Usage:
    python scripts/inpaint_ab_probe.py --pages 16,27,31 --out audit-results/ab
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import PROCESSED_DIR, RAW_DIR, LAMA_DYNAMIC_MODEL  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.inpaint.lama_inpainter import Inpainter  # noqa: E402
from app.manifest_utils import load_manifest_raw  # noqa: E402
from app.processing_pipeline_factory import build_processing_pipeline  # noqa: E402
from app.security import validate_managed_path  # noqa: E402
import lama_color_audit  # noqa: E402
from inpaint_audit_run import CHAPTER_ID  # noqa: E402

_state = {"page": None}
ROUTES: list[dict] = []
ZERO_MASK = {"on": False}

_original_smart_paint = Inpainter._smart_paint_region
_original_run_lama = Inpainter._run_lama


def _logged_smart_paint(self, image, local_mask, crop_box, feather=False, force_lama=False):
    cx1, cy1, cx2, cy2 = crop_box
    crop = image[cy1:cy2, cx1:cx2]
    fill = None if force_lama else self._smart_fill_color(crop, local_mask)
    ys, xs = np.nonzero(local_mask > 127)
    if len(xs):
        ROUTES.append({
            "page": _state["page"],
            "variant": "B" if ZERO_MASK["on"] else "A",
            "mask_box": [int(cx1 + xs.min()), int(cy1 + ys.min()), int(cx1 + xs.max()), int(cy1 + ys.max())],
            "route": "flat-fill" if fill is not None else "lama",
            "fill_bgr": None if fill is None else [int(v) for v in fill],
            "feather": bool(feather),
        })
    return _original_smart_paint(self, image, local_mask, crop_box, feather=feather, force_lama=force_lama)


def _zeroing_run_lama(self, canvas, mask_canvas):
    if ZERO_MASK["on"]:
        canvas = canvas.copy()
        canvas[mask_canvas > 127] = 0
    return _original_run_lama(self, canvas, mask_canvas)


def describe_model_masking(path: Path) -> dict:
    """Report whether the ONNX graph multiplies the image input by the mask."""
    try:
        import onnx
    except ImportError:
        return {"error": "onnx not installed"}
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    inputs = [i.name for i in graph.input]
    consumers = {name: [] for name in inputs}
    for node in graph.node:
        for name in node.input:
            if name in consumers:
                consumers[name].append({"op": node.op_type, "name": node.name, "inputs": list(node.input)})
    first_ops = [{"op": n.op_type, "inputs": list(n.input)[:3]} for n in graph.node[:12]]
    return {"inputs": inputs, "input_consumers": consumers, "first_nodes": first_ops}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = [int(p) for p in args.pages.split(",") if p.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    if LAMA_DYNAMIC_MODEL.is_file():
        (args.out / "model_masking.json").write_text(
            json.dumps(describe_model_masking(LAMA_DYNAMIC_MODEL), indent=1), encoding="utf-8")

    Inpainter._smart_paint_region = _logged_smart_paint
    Inpainter._run_lama = _zeroing_run_lama
    pipeline = build_processing_pipeline()
    manifest = load_manifest_raw(CHAPTER_ID)
    work = PROCESSED_DIR / CHAPTER_ID / "ab"
    work.mkdir(exist_ok=True)

    results: dict[str, dict[int, Path]] = {"A": {}, "B": {}}
    for variant in ("A", "B"):
        ZERO_MASK["on"] = variant == "B"
        for index in pages:
            if index >= len(manifest["pages"]):
                continue
            _state["page"] = index
            pipeline.process_pages(CHAPTER_ID, [index], workers=1)
            page = load_manifest_raw(CHAPTER_ID)["pages"][index]
            clean_path = validate_managed_path(page["clean"], PROCESSED_DIR / CHAPTER_ID)
            target = work / f"{variant}_{index:03d}.png"
            shutil.copy2(clean_path, target)
            results[variant][index] = target
    (args.out / "paths.json").write_text(json.dumps(ROUTES, indent=1), encoding="utf-8")

    summary = []
    for index in pages:
        if index not in results["A"]:
            continue
        original = read_image(validate_managed_path(manifest["pages"][index]["original"], RAW_DIR / CHAPTER_ID))
        a, b = read_image(results["A"][index]), read_image(results["B"][index])
        records_a = {tuple(r["box"]): r for r in lama_color_audit.audit_pair(original, a)}
        records_b = {tuple(r["box"]): r for r in lama_color_audit.audit_pair(original, b)}

        def overlaps(p, q):
            ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
            iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
            return ix * iy > 0.3 * min((p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1]))

        regions = dict(records_a)
        for box_b, rec_b in records_b.items():
            if not any(overlaps(box_b, box_a) for box_a in records_a):
                regions[box_b] = rec_b
        for rank, (box, rec) in enumerate(sorted(regions.items(), key=lambda kv: -kv[1]["pixels"])[:8]):
            x1, y1, x2, y2 = box
            m = 40
            x1, y1 = max(0, x1 - m), max(0, y1 - m)
            x2, y2 = min(original.shape[1], x2 + m), min(original.shape[0], y2 + m)
            gap = np.full((y2 - y1, 10, 3), (0, 0, 255), np.uint8)
            strip = np.hstack([original[y1:y2, x1:x2], gap, a[y1:y2, x1:x2], gap, b[y1:y2, x1:x2]])
            if strip.shape[1] > 1500:
                strip = cv2.resize(strip, (1500, round(strip.shape[0] * 1500 / strip.shape[1])), interpolation=cv2.INTER_AREA)
            elif strip.shape[1] < 900:
                f = min(3.0, 900 / strip.shape[1])
                strip = cv2.resize(strip, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            name = f"slice{index:03d}_{rank:02d}.jpg"
            cv2.imwrite(str(args.out / name), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rb = records_b.get(box) or next((r for q, r in records_b.items() if overlaps(q, box)), None)
            ra = records_a.get(box) or next((r for q, r in records_a.items() if overlaps(q, box)), None)
            diff = float(np.abs(a[box[1]:box[3], box[0]:box[2]].astype(int) - b[box[1]:box[3], box[0]:box[2]].astype(int)).mean())
            summary.append({
                "image": name, "box": list(box), "seam_A": ra["seam_delta_e"] if ra else None,
                "seam_B": rb["seam_delta_e"] if rb else None, "mean_abs_A_vs_B": round(diff, 2),
            })
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    routes_a = [r for r in ROUTES if r["variant"] == "A"]
    flat = sum(1 for r in routes_a if r["route"] == "flat-fill")
    print(f"routes (variant A): {len(routes_a)} regions, {flat} flat-fill, {len(routes_a) - flat} LaMa")
    for s in summary:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
