"""One chapter cleaned by the current pipeline and by the flow strip, compared block by block.

Usage: run.py <chapter-url> <out-dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import flow_clean  # noqa: E402
from app.config import KIUYHA_TEXT_MODEL, RAW_DIR  # noqa: E402
from app.detector.kiuyha_detector import KiuyhaTextDetector  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.inpaint.lama_inpainter import Inpainter  # noqa: E402
from app.manifest_utils import load_manifest_raw  # noqa: E402
from app.processing_pipeline_factory import build_processing_pipeline  # noqa: E402

CHAPTER = "f10ec0de"
LAMA_CAP = 512


def old_pages(width: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Original and cleaned pages rebuilt from the slice cores the current pipeline wrote."""
    by_source: dict[int, list] = {}
    for page in load_manifest_raw(CHAPTER)["pages"]:
        by_source.setdefault(int(page.get("source_page", 0)), []).append(page)
    originals, cleans = [], []
    for source in sorted(by_source):
        o_parts, c_parts = [], []
        for page in sorted(by_source[source], key=lambda p: int(p.get("slice_index", 0))):
            core = page.get("stitch_core") or {}
            o = read_image(Path(page["original"]))
            c = read_image(Path(page["clean"])) if page.get("clean") else o
            y0, y1 = int(core.get("core_y1", 0)), int(core.get("core_y2", o.shape[0]))
            o_parts.append(o[y0:y1])
            c_parts.append(c[y0:y1])
        originals.append(np.vstack(o_parts))
        cleans.append(np.vstack(c_parts))
    return originals, cleans


def text_mask(det, strip: np.ndarray) -> np.ndarray:
    """What the current rule masks call text, window by window, as the neutral measuring stick."""
    mask = np.zeros(strip.shape[:2], bool)
    H, y = strip.shape[0], 0
    while y < H:
        end = min(H, y + 2400)
        for b in det.text_boxes(np.ascontiguousarray(strip[y:end])):
            if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
                mask[y + b.y1:y + b.y2, b.x1:b.x2] |= b.mask > 127
        y = end if end == H else end - 256
    return mask


def main(url: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pipeline = build_processing_pipeline()
    started = time.perf_counter()
    pages = pipeline.download_chapter(url, CHAPTER, workers=2)["pages"]
    print(f"downloaded {len(pages)} slices in {time.perf_counter() - started:.0f} s", flush=True)

    started = time.perf_counter()
    pipeline.process_pages(CHAPTER, list(range(len(pages))))
    old_s = time.perf_counter() - started
    print(f"current pipeline: {old_s:.1f} s", flush=True)

    originals, cleans_old = old_pages(0)
    width = int(np.median([im.shape[1] for im in originals]))
    strip = flow_clean.build_strip(originals, width)
    old = flow_clean.build_strip(cleans_old, width)

    import app.inpaint.lama_inpainter as li
    import ctd_mask
    li.DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM = LAMA_CAP  # the fill is as good at a 640 px long side and ~4x cheaper
    li.INPAINT_NATIVE_TILE_ENABLED = False
    li.FIXED_LAMA_TILE_ASPECT = 1e9
    det = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    inpainter = Inpainter()
    stages: dict[str, float] = {}

    def timed(obj, name, label):
        f = getattr(obj, name)

        def g(*a, **k):
            t = time.perf_counter()
            r = f(*a, **k)
            stages[label] = stages.get(label, 0.0) + time.perf_counter() - t
            return r
        setattr(obj, name, g)

    timed(det, "detect_slice", "kiuyha")
    timed(ctd_mask, "_prob", "ctd")
    timed(ctd_mask, "grow", "grow")
    timed(inpainter, "inpaint", "inpaint")
    timed(inpainter, "_run_lama", "lama model")
    calls = {"lama_calls": 0}
    run_lama = inpainter._run_lama

    def counted(*a, **k):
        calls["lama_calls"] += 1
        return run_lama(*a, **k)
    inpainter._run_lama = counted
    started = time.perf_counter()
    new, boxes = flow_clean.clean(strip, det, inpainter, calls)
    new_s = time.perf_counter() - started
    stages = {**{k: round(v, 1) for k, v in stages.items()}, **calls}  # frozen before scoring runs Kiuyha again
    print(f"flow strip: {new_s:.1f} s, {len(boxes)} blocks, stages {stages}", flush=True)

    text, rest_old, rest_new = text_mask(det, strip), text_mask(det, old), text_mask(det, new)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(text.astype(np.uint8), np.ones((15, 15), np.uint8)))
    seams = np.cumsum([0] + [flow_clean.page_height(im.shape, width) for im in originals])[1:-1]
    rows, left_old, left_new, blocks = [], 0, 0, 0
    (out / "blocks").mkdir(exist_ok=True)
    for k in range(1, count):
        block = (labels == k) & text
        if block.sum() < 150:
            continue
        blocks += 1
        lo = bool((rest_old & block).sum() >= 0.3 * block.sum())
        ln = bool((rest_new & block).sum() >= 0.3 * block.sum())
        left_old += lo
        left_new += ln
        x, y, w, h = (int(v) for v in stats[k, :4])
        on_seam = bool(any(y <= s <= y + h for s in seams))
        m = 48
        a, b, c, d = max(0, y - m), min(strip.shape[0], y + h + m), max(0, x - m), min(width, x + w + m)
        tiles = []
        for label, im in (("original", strip), ("current" + (" LEFT" if lo else ""), old), ("flow" + (" LEFT" if ln else ""), new)):
            t = im[a:b, c:d]
            s = min(1.0, 460 / max(1, t.shape[0]), 560 / max(1, t.shape[1]))
            t = cv2.resize(t, (max(1, int(t.shape[1] * s)), max(1, int(t.shape[0] * s))))
            t = cv2.copyMakeBorder(t, 24, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200) if "LEFT" in label else (0, 0, 0), 1)
            tiles.append(t)
        hh = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        name = f"y{y:06d}{'-seam' if on_seam else ''}.jpg"
        cv2.imwrite(str(out / "blocks" / name), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 86])
        rows.append({"block": name, "left_current": lo, "left_flow": ln, "on_page_seam": on_seam})

    (out / "flow-slices").mkdir(exist_ok=True)
    cuts = flow_clean.cut_rows(new, boxes)
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        part = new[a:b]
        s = min(1.0, 900 / width)
        cv2.imwrite(str(out / "flow-slices" / f"{i:03d}.jpg"), cv2.resize(part, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])

    summary = {
        "url": url, "pages": len(originals), "strip": [int(strip.shape[0]), width],
        "seconds": {"current": round(old_s, 1), "flow": round(new_s, 1),
                    "flow_stages": stages, "lama_cap": LAMA_CAP},
        "blocks": blocks, "left": {"current": left_old, "flow": left_new},
        "on_page_seams": {"blocks": sum(r["on_page_seam"] for r in rows),
                          "left_current": sum(r["on_page_seam"] and r["left_current"] for r in rows),
                          "left_flow": sum(r["on_page_seam"] and r["left_flow"] for r in rows)},
        "flow_blocks": len(boxes), "flow_slices": len(cuts) - 1, "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
