"""Dump every detect/mask/inpaint step for a few slices of a chapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.detector.kiuyha_detector import stroke_mask
from app.image_io import read_image
from app.processing_pipeline_factory import build_processing_pipeline


def overlay(image, boxes, colour):
    out = image.copy()
    for b in boxes:
        if b.mask is not None:
            region = out[b.y1:b.y2, b.x1:b.x2]
            tint = region.copy()
            tint[b.mask > 0] = colour
            out[b.y1:b.y2, b.x1:b.x2] = cv2.addWeighted(region, 0.5, tint, 0.5, 0)
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), colour, 2)
    return out


def otsu_steps(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                           lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    diff = np.linalg.norm(lab - np.median(ring, axis=0), axis=2)
    diff = np.clip(diff * (255.0 / max(1.0, float(diff.max()))), 0, 255).astype(np.uint8)
    threshold, fg = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return diff, fg, float(threshold), np.median(ring, axis=0).round(1).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--slices", default="42,83")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline = build_processing_pipeline()
    pages = pipeline.download_chapter(args.url, "d1a90042", workers=2)["pages"]
    det, inpainter = pipeline.detector, pipeline.inpainter
    report = {}
    for number in (int(s) for s in args.slices.split(",")):
        page = pages[number]
        image = read_image(Path(page["original"]))
        core = page.get("stitch_core") or {}
        y0, y1 = int(core.get("core_y1") or 0), int(core.get("core_y2") or image.shape[0])
        image = np.ascontiguousarray(image[y0:y1])
        tag = args.out / f"s{number:02d}"
        raw1 = det.kiuyha.detect_slice(image)
        first = det.detect(image)
        clean1 = inpainter.inpaint(image, first)
        raw2 = det.kiuyha.detect_slice(clean1)
        second = det.leftover_boxes(clean1, first)
        clean2 = inpainter.inpaint(clean1, second) if second else clean1
        cv2.imwrite(f"{tag}-0-original.png", image)
        cv2.imwrite(f"{tag}-1-pass1-masks.png", overlay(image, first, (0, 0, 255)))
        cv2.imwrite(f"{tag}-2-clean1.png", clean1)
        cv2.imwrite(f"{tag}-3-pass2-masks.png", overlay(clean1, second, (0, 200, 0)))
        cv2.imwrite(f"{tag}-4-clean2.png", clean2)
        boxes = []
        for i, (x1, yy1, x2, yy2, score) in enumerate(raw1):
            diff, fg, threshold, border = otsu_steps(image[yy1:yy2, x1:x2])
            mask = stroke_mask(image[yy1:yy2, x1:x2])
            cv2.imwrite(f"{tag}-box{i}-diff.png", diff)
            cv2.imwrite(f"{tag}-box{i}-otsu.png", fg)
            cv2.imwrite(f"{tag}-box{i}-mask.png", mask.astype(np.uint8) * 255)
            boxes.append({"box": [x1, yy1, x2, yy2], "score": round(score, 3), "otsu": threshold,
                          "border_lab": border, "fg_ratio": round(float((fg > 0).mean()), 3),
                          "mask_ratio": round(float(mask.mean()), 3)})
        report[number] = {
            "core": [y0, y1], "shape": list(image.shape[:2]),
            "pass1_raw": [list(map(int, b[:4])) + [round(b[4], 3)] for b in raw1],
            "pass1_boxes": boxes,
            "pass2_raw_on_clean1": [list(map(int, b[:4])) + [round(b[4], 3)] for b in raw2],
            "pass2_kept": [[b.x1, b.y1, b.x2, b.y2] for b in second],
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
