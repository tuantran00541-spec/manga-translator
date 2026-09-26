"""Process a whole chapter through the app pipeline and report speed and text left behind."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.manifest_utils import load_manifest_raw
from app.one_shot_cleanup import OneShotTextMaskDetector
from app.processing_pipeline_factory import build_processing_pipeline


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for b in boxes:
        if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
            mask[b.y1:b.y2, b.x1:b.x2] |= b.mask > 127
    return mask


def pair(original: np.ndarray, clean: np.ndarray, box, path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = []
    for label, image in (("original", original), ("clean", clean)):
        tile = image[y1:y2, x1:x2]
        scale = min(1.0, 360 / max(1, tile.shape[0]))
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chapter-id", default="c1a90001")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = build_processing_pipeline()
    started = time.perf_counter()
    pages = pipeline.download_chapter(args.url, args.chapter_id, workers=2)["pages"]
    download_s = time.perf_counter() - started
    started = time.perf_counter()
    pipeline.process_pages(args.chapter_id, list(range(len(pages))))
    process_s = time.perf_counter() - started
    detector = "kiuyha" if getattr(pipeline.detector, "kiuyha", None) is not None else "segmenter"
    print(f"{len(pages)} slices processed in {process_s:.1f} s with {detector}", flush=True)

    segmenter = OneShotTextMaskDetector()
    segmenter.collage = True
    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)

    def found(image: np.ndarray) -> np.ndarray:
        return text_mask(image.shape[:2], segmenter.detect(image)[0] + kiuyha.text_boxes(image))

    blocks = left = 0
    lefts = []
    for number, page in enumerate(load_manifest_raw(args.chapter_id)["pages"]):
        if not page.get("clean"):
            continue
        core = page.get("stitch_core") or {}
        original, clean = read_image(Path(page["original"])), read_image(Path(page["clean"]))
        y0, y1 = int(core.get("core_y1", 0)), int(core.get("core_y2", original.shape[0]))
        before, after = np.ascontiguousarray(original[y0:y1]), np.ascontiguousarray(clean[y0:y1])
        text, rest = found(before), found(after)
        count, labels = cv2.connectedComponents(cv2.dilate(text.astype(np.uint8), np.ones((15, 15), np.uint8)))
        for i in range(1, count):
            block = (labels == i) & text
            if block.sum() < 150:
                continue
            blocks += 1
            if (rest & block).sum() >= 0.3 * block.sum():
                left += 1
                ys, xs = np.nonzero(block)
                box = (max(0, int(xs.min()) - 40), max(0, int(ys.min()) - 40),
                       min(before.shape[1], int(xs.max()) + 40), min(before.shape[0], int(ys.max()) + 40))
                lefts.append({"slice": number, "box": box, "px": int(block.sum())})
                pair(before, after, box, args.out / f"left-{len(lefts):02d}.jpg")

    report = {
        "url": args.url, "detector": detector, "slices": len(pages),
        "download_s": round(download_s, 1), "process_s": round(process_s, 1),
        "per_slice_s": round(process_s / max(1, len(pages)), 2),
        "blocks": blocks, "blocks_left": left, "left": lefts,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "left"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
