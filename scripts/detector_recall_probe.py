"""Compare text-detector recall and cost: one-shot vs windowed passes.

Runs on the chapter processed by scripts/inpaint_audit_run.py (same
CHAPTER_ID). For every slice it times each detector variant; for the slices
given with --show it draws the boxes each variant found on the slice's stitch
core so missed text can be checked by eye.

Variants (all use the production text-segmenter model):
  oneshot   whole slice letterboxed into one 1024 input (production today)
  oneshot12 same forward decoded at the 0.12 rescue threshold (no extra cost)
  win1024   1024-tall windows, 200px overlap, native scale (+ global pass)
  win1536   1536-tall windows, 256px overlap, scaled to 1024 (~0.67x)

Usage:
    python scripts/detector_recall_probe.py --show 16,23,33,52,67,82,83 --out audit-results/detector
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import RAW_DIR  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.manifest_utils import load_manifest_raw  # noqa: E402
from app.one_shot_cleanup import OneShotTextMaskDetector  # noqa: E402
from app.security import validate_managed_path  # noqa: E402
from inpaint_audit_run import CHAPTER_ID  # noqa: E402

COLOURS = {"oneshot": (0, 0, 255), "oneshot12": (200, 0, 200), "win1024": (0, 170, 0), "win1536": (255, 120, 0)}


def windows(height: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if height <= window:
        return [(0, height)]
    step = window - overlap
    spans, y = [], 0
    while True:
        y2 = min(height, y + window)
        spans.append((max(0, y2 - window), y2))
        if y2 >= height:
            return spans
        y += step


def run_windows(core: OneShotTextMaskDetector, image: np.ndarray, window: int, overlap: int, *, global_pass: bool):
    detector = core.detector
    boxes, forwards = [], 0
    for y1, y2 in windows(image.shape[0], window, overlap):
        crop = image[y1:y2]
        blob, transform = detector._preprocess(crop, offset_x=0, offset_y=y1)
        outputs = core._run_session(blob)
        forwards += 1
        boxes.extend(core._accept_many(detector._postprocess(outputs, transform)))
    if global_pass:
        found, _ = core.detect(image)
        boxes.extend(found)
        forwards += 1
    return detector._nms_boxes(boxes), forwards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    show = {int(v) for v in args.show.split(",") if v.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    core = OneShotTextMaskDetector()
    manifest = load_manifest_raw(CHAPTER_ID)
    pages = manifest["pages"]
    # Warm the session so the first timing is not the model load.
    first = read_image(validate_managed_path(pages[0]["original"], RAW_DIR / CHAPTER_ID))
    core.detect(first)

    totals = {name: {"seconds": 0.0, "forwards": 0, "boxes": 0} for name in COLOURS}
    # Preprocess + ONNX forward alone (no mask decode / NMS), from oneshot12.
    forward_s = 0.0
    rows = []
    for index, page in enumerate(pages):
        full = read_image(validate_managed_path(page["original"], RAW_DIR / CHAPTER_ID))
        # Production (page_processing._process_page) detects on the stitch
        # core only, so every variant sees exactly that crop.
        core_meta = page.get("stitch_core") or {}
        y1 = max(0, min(int(core_meta.get("core_y1", 0)), full.shape[0]))
        y2 = max(y1 + 1, min(int(core_meta.get("core_y2", full.shape[0])), full.shape[0]))
        image = full[y1:y2]
        found = {}
        for name in COLOURS:
            started = time.perf_counter()
            if name == "oneshot":
                boxes, _ = core.detect(image)
                forwards = 1
            elif name == "oneshot12":
                outputs, transform = core._single_forward_outputs(image)
                forward_s += time.perf_counter() - started
                raw = core._postprocess_at_threshold(outputs, transform, core.RESCUE_CONF_THRESHOLD)
                boxes, forwards = core._accept_many(raw), 1
            elif name == "win1024":
                boxes, forwards = run_windows(core, image, 1024, 200, global_pass=True)
            else:
                boxes, forwards = run_windows(core, image, 1536, 256, global_pass=False)
            elapsed = time.perf_counter() - started
            totals[name]["seconds"] += elapsed
            totals[name]["forwards"] += forwards
            totals[name]["boxes"] += len(boxes)
            found[name] = boxes
        rows.append({"slice": index, "core_h": int(image.shape[0]),
                     **{f"{n}_boxes": len(b) for n, b in found.items()}})

        if index in show:
            panels = []
            for name, colour in COLOURS.items():
                canvas = image.copy()
                for b in found[name]:
                    cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 4)
                panel = canvas
                label = np.full((40, panel.shape[1], 3), 255, np.uint8)
                cv2.putText(label, f"{name}: {len(found[name])}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                panels.append(np.vstack([label, panel]))
                panels.append(np.full((panels[-1].shape[0], 16, 3), 255, np.uint8))
            strip = np.hstack(panels[:-1])
            scale = min(1.0, 1800 / strip.shape[1])
            strip = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.out / f"slice{index:03d}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])

    summary = {name: {**t, "seconds": round(t["seconds"], 1),
                      "per_slice_s": round(t["seconds"] / max(1, len(pages)), 3)} for name, t in totals.items()}
    summary["forward_only_s_per_slice"] = round(forward_s / max(1, len(pages)), 3)
    (args.out / "summary.json").write_text(json.dumps({"slices": len(pages), "variants": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
