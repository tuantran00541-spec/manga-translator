"""Kiuyha/Manga-Bubble-YOLO text boxes (ONNX) with Otsu letter masks."""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.ort_utils import make_session

LETTERBOX_VALUE = 114
STRIDE = 32
NMS_IOU = 0.7  # Ultralytics' default for this head
BOX_PAD = 16  # Kiuyha's boxes can stop short of the last letter of a wide line
HALVES_GAP = 16
HALVES_MIN_OVERLAP = 256
CLOSE_KERNEL = 15  # letters -> word blobs
GROW_KERNEL = 13  # past the letter outline (a white stroke round brown text)


def stroke_mask(crop: np.ndarray) -> np.ndarray:
    """Letters and their outline inside a box crop, found by Otsu against the border colour."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    if min(lab.shape[:2]) < 8:
        return np.zeros(crop.shape[:2], bool)
    ring = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                           lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    diff = np.linalg.norm(lab - np.median(ring, axis=0), axis=2)
    diff = np.clip(diff * (255.0 / max(1.0, float(diff.max()))), 0, 255).astype(np.uint8)
    _, fg = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros(count, bool)
    for label in range(1, count):
        x, y, w, h, _area = stats[label]
        keep[label] = x > 0 and y > 0 and x + w < fg.shape[1] and y + h < fg.shape[0]
    part = keep[labels].astype(np.uint8)
    part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL,) * 2))
    return cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROW_KERNEL,) * 2)) > 0


def _merge(boxes: list[tuple[int, int, int, int, float]]) -> list[tuple[int, int, int, int, float]]:
    """Fold boxes that are mostly inside a bigger one."""
    kept: list[list[float]] = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        area = (box[2] - box[0]) * (box[3] - box[1])
        for k in kept:
            ix = max(0, min(k[2], box[2]) - max(k[0], box[0]))
            iy = max(0, min(k[3], box[3]) - max(k[1], box[1]))
            if ix * iy >= 0.5 * area:
                k[:] = [min(k[0], box[0]), min(k[1], box[1]), max(k[2], box[2]), max(k[3], box[3]), max(k[4], box[4])]
                break
        else:
            kept.append(list(box))
    return [(int(a), int(b), int(c), int(d), float(e)) for a, b, c, d, e in kept]


def _rows(output: np.ndarray, conf_threshold: float) -> np.ndarray:
    """Final ``x1, y1, x2, y2, score`` rows from either output layout."""
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[-1] == 6:
        rows = output.reshape(-1, 6)
        return rows[rows[:, 4] >= conf_threshold][:, :5]
    preds = output.reshape(output.shape[-2], output.shape[-1]).T  # anchors x (4 + C)
    scores = preds[:, 4:].max(axis=1)
    keep = scores >= conf_threshold
    preds, scores = preds[keep], scores[keep]
    if not len(preds):
        return np.zeros((0, 5), np.float32)
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    picked = cv2.dnn.NMSBoxes(np.stack([boxes[:, 0], boxes[:, 1], w, h], axis=1).tolist(),
                              scores.tolist(), conf_threshold, NMS_IOU)
    picked = np.asarray(picked, dtype=int).reshape(-1)
    return np.concatenate([boxes[picked], scores[picked, None]], axis=1)


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
        rows = _rows(self.session.run(None, {self.input_name: blob})[0], self.conf_threshold)
        boxes = []
        for x1, y1, x2, y2, score in rows:
            bx1 = int(max(0.0, (x1 - pad_x) / scale))
            by1 = int(max(0.0, (y1 - pad_y) / scale))
            bx2 = int(min(float(w), math.ceil((x2 - pad_x) / scale)))
            by2 = int(min(float(h), math.ceil((y2 - pad_y) / scale)))
            if bx2 > bx1 and by2 > by1:
                boxes.append((bx1, by1, bx2, by2, float(score)))
        return boxes

    def halves_plan(self, height: int, width: int) -> tuple[float, list[tuple[int, int]]] | None:
        """Scale and overlapping halves for a static square input, or None if not worth it."""
        if self.fixed is None or self.fixed[0] != self.fixed[1]:
            return None
        size = self.fixed[0]
        scale = min(1.0, (size - HALVES_GAP) / (2.0 * width))
        overlap = int(2 * size / scale) - height
        if overlap < HALVES_MIN_OVERLAP:
            overlap = min(height, HALVES_MIN_OVERLAP)
            scale = min(scale, 2.0 * size / (height + overlap))
        if scale < 1.2 * min(size / width, size / height):
            return None
        half = (height + min(overlap, height) + 1) // 2
        return scale, [(0, half), (height - half, height)]

    def detect_slice(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Padded text boxes for a slice, detected as two halves when that helps."""
        h, w = image.shape[:2]
        plan = self.halves_plan(h, w)
        if plan is None:
            found = self.detect(image)
        else:
            scale, halves = plan
            size = self.fixed[0]
            canvas = np.full((size, size, 3), LETTERBOX_VALUE, np.uint8)
            columns, x = [], 0
            for y0, y1 in halves:
                rw, rh = min(size - x, int(w * scale)), min(size, int((y1 - y0) * scale))
                canvas[:rh, x:x + rw] = cv2.resize(image[y0:y1], (rw, rh), interpolation=cv2.INTER_LINEAR)
                columns.append((x, rw, rh, y0, y1))
                x += rw + HALVES_GAP
            found = []
            for cx1, cy1, cx2, cy2, score in self.detect(canvas):
                px, rw, rh, y0, y1 = columns[0] if (cx1 + cx2) / 2 < columns[1][0] else columns[1]
                sx, sy = rw / w, rh / (y1 - y0)
                bx1, bx2 = int((max(cx1, px) - px) / sx), int(math.ceil((min(cx2, px + rw) - px) / sx))
                by1, by2 = int(min(cy1, rh) / sy) + y0, int(math.ceil(min(cy2, rh) / sy)) + y0
                if bx2 - bx1 > 4 and by2 - by1 > 4:
                    found.append((bx1, by1, bx2, by2, score))
            found = _merge(found)
        return [(max(0, x1 - BOX_PAD), max(0, y1 - BOX_PAD), min(w, x2 + BOX_PAD), min(h, y2 + BOX_PAD), score)
                for x1, y1, x2, y2, score in found]

    def text_boxes(self, image: np.ndarray, source_model: str = "kiuyha_text") -> list[BubbleBox]:
        """Detected text blocks with letter masks, ready for inpainting and OCR."""
        boxes = []
        for x1, y1, x2, y2, score in self.detect_slice(image):
            mask = stroke_mask(image[y1:y2, x1:x2])
            if not mask.any():
                continue
            boxes.append(BubbleBox(
                x1, y1, x2, y2, score, mask.astype(np.uint8) * 255,
                source_model=source_model, class_name="text", semantic_type="free_text",
                mask_source="text_segmenter", safe_to_inpaint=True, ocr_eligible=True,
                source_role="text_segmenter",
            ))
        return boxes

    def leftover_boxes(self, clean: np.ndarray, targets, source_model: str = "kiuyha_text") -> list[BubbleBox]:
        """Text still visible after inpainting, only where a first-pass box already was."""
        def inside(box) -> bool:
            cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
            return any(t.x1 <= cx <= t.x2 and t.y1 <= cy <= t.y2 for t in targets)
        return [box for box in self.text_boxes(clean, source_model) if inside(box)]
