"""One-off: compare LaMa fills of real Shadow Slave crops (current, narrow glow, edge-in, full resolution)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import app.detector.kiuyha_detector as kd
import app.inpaint.lama_inpainter as li
from app.config import MODELS_DIR

EVIDENCE = "bc2de309"  # chapter-run evidence that still holds the full-size crops
CROPS = ("s002-b0", "s001-b0", "s014-b0", "s047-b0", "s016-b0", "s050-b0")
RING = 40  # px of untouched background round the mask the fill is compared with
BAND = 32  # px committed per edge-in pass


def _crop(name: str) -> np.ndarray:
    data = subprocess.run(["git", "show", f"{EVIDENCE}:audit-results/chapter-run/raw/{name}.jpg"],
                          check=True, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _mask(image, boxes) -> np.ndarray:
    mask = np.zeros(image.shape[:2], bool)
    for b in boxes:
        mask[b.y1:b.y2, b.x1:b.x2] |= b.mask > 0
    return mask


def _detail(image: np.ndarray, where: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F))[where].mean()) if where.any() else 0.0


def _edge_in(inpainter, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill from the edge inwards, committing one band per pass, so each pass has more real context."""
    out, left = image.copy(), mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1,) * 2)
    while left.any():
        painted = inpainter._lama_fill_single(out, left.astype(np.uint8) * 255)
        inner = cv2.erode(left.astype(np.uint8), kernel) > 0
        ring = left & ~inner if inner.any() else left
        out[ring] = painted[ring]
        left = inner if inner.any() else np.zeros_like(left)
    return out


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    detector = kd.KiuyhaTextDetector(str(MODELS_DIR / "kiuyha_text_1280.onnx"))
    inpainter = li.Inpainter()
    inpainter._ensure_session()
    results = {}
    for name in CROPS:
        image = _crop(name)
        boxes = detector.text_boxes(image)
        mask = _mask(image, boxes)
        ring = (cv2.dilate(mask.astype(np.uint8), np.ones((2 * RING + 1,) * 2, np.uint8)) > 0) & ~mask
        tiles = {"A current": inpainter.inpaint(image, boxes)}
        halo, cap = kd.HALO_REACH_MAX, kd.GROW_MAX
        kd.HALO_REACH_MAX, kd.GROW_MAX = 16, 15
        narrow = detector.text_boxes(image)
        kd.HALO_REACH_MAX, kd.GROW_MAX = halo, cap
        tiles["B narrow glow"] = inpainter.inpaint(image, narrow)
        single, pixels = li.DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM, li.DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS
        li.DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM, li.DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS = 4096, 4096 * 4096
        tiles["D full resolution"] = inpainter.inpaint(image, boxes)
        tiles["C edge-in"] = _edge_in(inpainter, image, mask)
        li.DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM, li.DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS = single, pixels
        results[name] = {"ring_detail": round(_detail(image, ring), 2), "mask_share": round(float(mask.mean()), 3)}
        for label, filled in tiles.items():
            where = _mask(image, narrow) if label.startswith("B") else mask
            results[name][label] = {"fill_detail": round(_detail(filled, where), 2),
                                    "fill_minus_ring_L": round(float(cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)[where].mean()
                                                                     - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[ring].mean()), 1)}
        strip = [image] + list(tiles.values())
        labelled = []
        for label, tile in zip(["original", *tiles], strip):
            tile = tile.copy()
            cv2.putText(tile, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            labelled.append(tile)
        sheet = np.concatenate(labelled, axis=0)
        cv2.imwrite(str(out / f"{name}.jpg"), cv2.resize(sheet, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "audit-results/lama-variants")))
