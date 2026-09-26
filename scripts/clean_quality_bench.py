"""Clean the same real slices with each text detector and compare.

Downloads and slices a chapter the way the app does, then cleans every slice
with LaMa after each detector:

  segmenter   text_segmenter.onnx, two halves side by side in one pass
              (the app's detector when the Kiuyha model is absent)
  kiuyha      kiuyha_text_1280.onnx boxes with Otsu letter masks (the app's
              detector when it is present)

and reports, per detector: detect_s / inpaint_s, blocks_left (text blocks found
on the original by either detector that either detector still finds after
cleaning, the miss rate that matters) and over_erase (mask farther than 10 px
from any text either detector found). Crops of the largest fills and of every
block left behind are saved for checking by eye: the counts cannot see a
letter outline left as a ghost.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.detector.mask_builder import build_mask
from app.image_io import read_image
from app.inpaint.lama_inpainter import Inpainter
from app.one_shot_cleanup import OneShotTextMaskDetector
from app.processing_pipeline_factory import build_processing_pipeline


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for b in boxes:
        if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
            mask[b.y1:b.y2, b.x1:b.x2] |= b.mask > 127
    return mask


def crop_sheet(images: list[tuple[str, np.ndarray]], box: tuple[int, int, int, int], path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = []
    for label, image in images:
        tile = image[y1:y2, x1:x2]
        scale = min(1.0, 360 / max(1, tile.shape[0]))
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--skip", type=int, default=1, help="leading slices to skip (credits)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = build_processing_pipeline()
    pages = pipeline.download_chapter(args.url, "c1ea0001", workers=2)["pages"][args.skip:args.skip + args.slices]
    segmenter = OneShotTextMaskDetector()
    segmenter.collage = True
    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    detectors = {
        "segmenter": lambda image: segmenter.detect(image)[0],
        "kiuyha": lambda image: kiuyha.text_boxes(image),
    }
    inpainter = Inpainter()

    def found(image: np.ndarray) -> np.ndarray:
        return text_mask(image.shape[:2], [b for detect in detectors.values() for b in detect(image)])

    totals = {name: {"detect_s": 0.0, "inpaint_s": 0.0, "blocks": 0, "blocks_left": 0, "mask_px": 0, "over_px": 0}
              for name in detectors}
    fills, lefts = [], []
    for number, page in enumerate(pages):
        full = read_image(Path(page["original"]))
        core = page.get("stitch_core") or {}
        y0, y1 = int(core.get("core_y1", 0)), int(core.get("core_y2", full.shape[0]))
        image = np.ascontiguousarray(full[y0:y1])
        text = found(image)
        count, labels = cv2.connectedComponents(cv2.dilate(text.astype(np.uint8), np.ones((15, 15), np.uint8)))
        blocks = [b for b in ((labels == i) & text for i in range(1, count)) if b.sum() >= 150]
        near = cv2.dilate(text.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0
        results = {}
        for name, detect in detectors.items():
            started = time.perf_counter()
            boxes = detect(image)
            detect_s = time.perf_counter() - started
            started = time.perf_counter()
            result = inpainter.inpaint(image, boxes)
            inpaint_s = time.perf_counter() - started
            mask = build_mask(image.shape[:2], boxes, image) > 0
            left = found(result)
            t = totals[name]
            t["detect_s"] += detect_s
            t["inpaint_s"] += inpaint_s
            t["mask_px"] += int(mask.sum())
            t["over_px"] += int((mask & ~near).sum())
            for block in blocks:
                t["blocks"] += 1
                if (left & block).sum() >= 0.3 * block.sum():
                    t["blocks_left"] += 1
                    ys, xs = np.nonzero(block)
                    lefts.append((int(block.sum()), number, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
            results[name] = result
            if name == "kiuyha":
                c, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                fills += [(int(a), number, (int(x), int(y), int(x + w), int(y + h))) for x, y, w, h, a in stats[1:c]]
        page["_image"], page["_results"] = image, results
        print(f"slice {number + 1}/{len(pages)} done", flush=True)

    def sheets(kind: str, regions: list, limit: int) -> None:
        seen = set()
        for _, number, (x1, y1, x2, y2) in sorted(regions, reverse=True):
            if (number, x1, y1) in seen or len(seen) >= limit:
                continue
            seen.add((number, x1, y1))
            image = pages[number]["_image"]
            box = (max(0, x1 - 40), max(0, y1 - 40), min(image.shape[1], x2 + 40), min(image.shape[0], y2 + 40))
            crop_sheet([("original", image)] + list(pages[number]["_results"].items()), box,
                       args.out / f"{kind}-{len(seen):02d}.jpg")

    sheets("fill", fills, 8)
    sheets("left", lefts, 8)
    report = {"url": args.url, "slices": len(pages), "detectors": {}}
    for name, t in totals.items():
        report["detectors"][name] = {
            "detect_s": round(t["detect_s"], 1), "inpaint_s": round(t["inpaint_s"], 1),
            "blocks": t["blocks"], "blocks_left": t["blocks_left"],
            "over_erase": round(t["over_px"] / max(1, t["mask_px"]), 4),
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
