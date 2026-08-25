import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from app.detector.bubble_detector import YoloDetector
from app.config import TEXT_SEGMENTER_MODEL


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def inspect(detector, image):
    h, w = image.shape[:2]
    blob, scale, pad = detector._preprocess(image)
    outputs = detector.session.run(None, {detector.input_name: blob})
    out_arr = np.squeeze(outputs[0])
    if out_arr.ndim == 1:
        out_arr = out_arr[np.newaxis, :]
    if out_arr.ndim == 2 and out_arr.shape[0] < out_arr.shape[1]:
        out_arr = out_arr.T

    has_proto = len(outputs) > 1 and outputs[1].ndim == 4
    num_mask_coeffs = outputs[1].shape[1] if has_proto else 0
    num_classes = max(1, out_arr.shape[1] - 4 - num_mask_coeffs)

    rows = []
    for pred in out_arr:
        if num_classes == 1:
            conf = float(pred[4])
        else:
            class_scores = pred[4:4 + num_classes]
            conf = float(np.max(class_scores))
        cx, cy, bw, bh = pred[:4]
        x1 = (cx - bw / 2 - pad[0]) / scale
        y1 = (cy - bh / 2 - pad[1]) / scale
        x2 = (cx + bw / 2 - pad[0]) / scale
        y2 = (cy + bh / 2 - pad[1]) / scale
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            rows.append((conf, x1, y1, x2, y2))
    rows.sort(reverse=True)
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python debug_text_threshold_sweep.py IMAGE [threshold ...]")
    image = read_image(sys.argv[1])
    if image is None:
        raise SystemExit(f"Cannot read image: {sys.argv[1]}")
    thresholds = [float(x) for x in sys.argv[2:]] or [0.20, 0.15, 0.10, 0.05]

    for threshold in thresholds:
        detector = YoloDetector(TEXT_SEGMENTER_MODEL, threshold)
        rows = inspect(detector, image)
        print(f"\n=== threshold={threshold:.3f} raw candidates={len(rows)} ===")
        for i, (conf, x1, y1, x2, y2) in enumerate(rows[:30]):
            print(f"[{i:02d}] conf={conf:.3f} box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
