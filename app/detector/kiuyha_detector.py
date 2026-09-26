"""Kiuyha/Manga-Bubble-YOLO (YOLO26, boxes only) exported to ONNX.

YOLO26 has an end-to-end head: the ONNX output is already the final list of
detections, ``[1, N, 6]`` rows of ``x1, y1, x2, y2, score, class`` in input
pixels, so there is no NMS to run. The model accepts either a fixed square
input (a static export) or any size that is a multiple of 32 (a dynamic
export); a static model gets the slice letterboxed into its square.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.ort_utils import make_session

LETTERBOX_VALUE = 114
STRIDE = 32


class KiuyhaTextDetector:
    def __init__(self, model_path, conf_threshold: float = 0.25, session=None):
        self.session = session if session is not None else make_session(model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        height, width = inp.shape[2], inp.shape[3]
        self.fixed = (int(height), int(width)) if isinstance(height, int) and isinstance(width, int) else None
        self.conf_threshold = float(conf_threshold)

    def _input_size(self, height: int, width: int) -> tuple[int, int]:
        if self.fixed is not None:
            return self.fixed
        return int(math.ceil(height / STRIDE) * STRIDE), int(math.ceil(width / STRIDE) * STRIDE)

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Boxes (x1, y1, x2, y2, score) in ``image`` pixels."""
        h, w = image.shape[:2]
        in_h, in_w = self._input_size(h, w)
        scale = min(in_w / w, in_h / h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        pad_x, pad_y = (in_w - rw) // 2, (in_h - rh) // 2
        canvas = np.full((in_h, in_w, 3), LETTERBOX_VALUE, np.uint8)
        resized = image if (rw, rh) == (w, h) else cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        rows = np.asarray(self.session.run(None, {self.input_name: blob})[0]).reshape(-1, 6)
        boxes = []
        for x1, y1, x2, y2, score, _cls in rows:
            if score < self.conf_threshold:
                continue
            bx1 = int(max(0.0, (x1 - pad_x) / scale))
            by1 = int(max(0.0, (y1 - pad_y) / scale))
            bx2 = int(min(float(w), math.ceil((x2 - pad_x) / scale)))
            by2 = int(min(float(h), math.ceil((y2 - pad_y) / scale)))
            if bx2 > bx1 and by2 > by1:
                boxes.append((bx1, by1, bx2, by2, float(score)))
        return boxes
