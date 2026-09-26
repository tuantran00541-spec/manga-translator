"""Reproduce the brush repaint leaving white letters on a large SFX ("NOW...").

Runs the real manual repaint path (Inpainter.inpaint_mask) on the page the user
reported, at a few page widths, in three ways:

  current  - one page, the code as it is (feathered manual repaint, 512 px tiles)
  seam     - the page cut into two slices through the letters, each repainted on
             its own, the way the stitched Review view sends a stroke that
             crosses a slice boundary
  single   - one page, one LaMa pass over the whole crop (no 512 px tiles)

For each run it counts how many of the letter pixels are still near-white after
the repaint, and saves the crops side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import app.inpaint.lama_inpainter as lama_module
from app.inpaint.lama_inpainter import Inpainter

WHITE = 200


def letters_mask(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Near-white pixels of the NOW... caption (lower part of the page, inside the teal blob)."""
    h, w = image.shape[:2]
    y1, y2 = int(0.76 * h), int(0.85 * h)
    x1, x2 = int(0.15 * w), int(0.85 * w)
    region = image[y1:y2, x1:x2]
    white = (region.min(axis=2) > WHITE).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    keep = np.zeros_like(white)
    min_area = max(20, int(0.00002 * h * w))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 255
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = keep
    ys, xs = np.nonzero(mask)
    return mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def brush(mask: np.ndarray, scale: float) -> np.ndarray:
    """What a user scribbling over the letters with the round brush sends."""
    radius = max(3, round(10 * scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def residual(result: np.ndarray, letters: np.ndarray) -> float:
    inside = letters > 0
    return float(np.mean(result[inside].min(axis=1) > WHITE - 20))


def run(inpainter: Inpainter, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    result = inpainter.inpaint_mask(image.copy(), mask)
    return result, dict(inpainter.last_metrics())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--widths", default="468,800,1100")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    inpainter = Inpainter()
    report = {"model": str(inpainter.lama_model_path), "runs": []}

    for width in (int(v) for v in args.widths.split(",")):
        scale = width / source.shape[1]
        image = cv2.resize(source, (width, round(source.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
        letters, (bx1, by1, bx2, by2) = letters_mask(image)
        mask = brush(letters, scale)
        seam = by1 + int(0.6 * (by2 - by1))
        pad = round(60 * scale)
        cy1, cy2 = max(0, by1 - pad), min(image.shape[0], by2 + pad)
        cx1, cx2 = max(0, bx1 - pad), min(image.shape[1], bx2 + pad)

        outputs = {}
        outputs["current"], m_current = run(inpainter, image, mask)
        top, m_top = run(inpainter, image[:seam], mask[:seam])
        bottom, m_bottom = run(inpainter, image[seam:], mask[seam:])
        outputs["seam"] = np.concatenate([top, bottom])
        original_size = lama_module.INPAINT_SIZE
        lama_module.INPAINT_SIZE = 4096
        try:
            outputs["single"], m_single = run(inpainter, image, mask)
        finally:
            lama_module.INPAINT_SIZE = original_size

        entry = {
            "width": width, "height": image.shape[0], "letters_bbox": [bx1, by1, bx2, by2],
            "mask_bbox_size": [int(np.ptp(np.nonzero(mask)[1])) + 1, int(np.ptp(np.nonzero(mask)[0])) + 1],
            "seam_y": seam,
            "white_left": {name: round(residual(out, letters), 4) for name, out in outputs.items()},
            "metrics": {"current": m_current, "seam_top": m_top, "seam_bottom": m_bottom, "single": m_single},
        }
        report["runs"].append(entry)
        print(json.dumps(entry))

        tiles = [image[cy1:cy2, cx1:cx2]]
        overlay = image.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + (0, 0, 127)).astype(np.uint8)
        tiles.append(overlay[cy1:cy2, cx1:cx2])
        for name in ("current", "seam", "single"):
            tile = outputs[name][cy1:cy2, cx1:cx2].copy()
            if name == "seam":
                cv2.line(tile, (0, seam - cy1), (tile.shape[1], seam - cy1), (0, 255, 255), 1)
            tiles.append(tile)
        labelled = []
        for label, tile in zip(("original", "brush", "current", "seam", "single"), tiles):
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            labelled.append(cv2.copyMakeBorder(tile, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        sheet = np.concatenate(labelled, axis=1)
        if sheet.shape[1] > 2400:
            sheet = cv2.resize(sheet, (2400, round(sheet.shape[0] * 2400 / sheet.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / f"w{width}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
