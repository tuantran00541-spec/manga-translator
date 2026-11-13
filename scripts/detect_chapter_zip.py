from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import BOX_PAD, KiuyhaTextDetector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def raw_slice_boxes(detector, image):
    height, width = image.shape[:2]
    found = detector._raw_slice_boxes(image)
    return [(max(0, x1 - BOX_PAD), max(0, y1 - BOX_PAD), min(width, x2 + BOX_PAD), min(height, y2 + BOX_PAD), score)
            for x1, y1, x2, y2, score in found]


def log(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--chapter", type=int, required=True)
    parser.add_argument("--chapter-id", default="")
    parser.add_argument("--model", default=str(KIUYHA_TEXT_MODEL))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    detector = KiuyhaTextDetector(args.model)
    log("detector model: %s" % detector.source_model)
    images = []
    total_boxes = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if Path(name).suffix.lower() in IMAGE_EXTS)
        for name in names:
            data = np.frombuffer(archive.read(name), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit("could not decode image %s in %s" % (name, zip_path))
            height, width = image.shape[:2]
            boxes = raw_slice_boxes(detector, image)
            total_boxes += len(boxes)
            images.append({
                "file": name,
                "width": width,
                "height": height,
                "boxes": [[int(x1), int(y1), int(x2), int(y2), float(score)] for x1, y1, x2, y2, score in boxes],
            })
            log("detected %s: %d boxes" % (name, len(boxes)))
    result = {
        "chapter": args.chapter,
        "chapter_id": args.chapter_id,
        "model": detector.source_model,
        "ctd": False,
        "image_count": len(images),
        "box_count": total_boxes,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "images": images,
    }
    (out_dir / "detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chapter": args.chapter, "images": len(images), "boxes": total_boxes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
