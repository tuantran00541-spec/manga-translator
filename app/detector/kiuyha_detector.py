"""The text detector: Kiuyha/Manga-Bubble-YOLO boxes (ONNX) with comic-text-detector letter masks."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.detector.boxes import BubbleBox
from app.detector.ctd_mask import letter_mask, still_reads
from app.knobs import knob
from app.ort_utils import make_session

LETTERBOX_VALUE = 114
STRIDE = 32
NMS_IOU = knob("detect.nms_iou")
BOX_PAD = knob("detect.box_pad")
COLUMN_GAP = 16  # letterbox gap between columns packed into one input
HALVES_MIN_OVERLAP = 256  # rows the two halves of the coarse pass share at least
PAGE_WIDTH = 400  # slice width at page scale, where the model scores big lettering highest
PAGE_OVERLAP = 0.15  # share of a page-scale window repeated in the next one
PAGE_CONFIDENCE = 0.45  # page-scale score a box needs; drawn sound effects score below it, lettering above
BAND_OVERLAP = 0.25  # share of a detection band repeated in the next one
EDGE_TOUCH = knob("detect.edge_touch")
UNION_SHARE = knob("detect.union_share")
SAME_LINE = knob("detect.same_line")


def _text_box(x1, y1, x2, y2, score, mask, source_model, letters=None) -> BubbleBox:
    box = BubbleBox(
        x1, y1, x2, y2, score, mask,
        source_model=source_model, class_name="text", semantic_type="free_text",
        mask_source="text_segmenter", safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
    )
    box.letters = letters  # the letters alone, without outline or glow; they show the lines
    return box


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


def _union_overlapping(boxes: list[tuple[int, int, int, int, float]]) -> list[tuple[int, int, int, int, float]]:
    """Join overlapping pieces of one text seen by different passes: the same line, or mostly the same area."""
    merged = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                same_line = iy >= SAME_LINE * min(a[3] - a[1], b[3] - b[1])
                if ix > 0 and iy > 0 and (same_line or ix * iy >= UNION_SHARE * max(1, smaller)):
                    merged[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), max(a[4], b[4])]
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return [(int(a), int(b), int(c), int(d), float(e)) for a, b, c, d, e in merged]


def _join_grown(boxes: list[BubbleBox]) -> list[BubbleBox]:
    """Boxes that grew into each other while taking cut letters are one text; their masks are joined."""
    rects = _union_overlapping([(b.x1, b.y1, b.x2, b.y2, b.confidence) for b in boxes])
    if len(rects) == len(boxes):
        return boxes
    joined = []
    for x1, y1, x2, y2, score in rects:
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        letters = np.zeros((y2 - y1, x2 - x1), bool)
        for b in boxes:
            if b.x1 >= x1 and b.y1 >= y1 and b.x2 <= x2 and b.y2 <= y2:
                view = mask[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1]
                np.maximum(view, b.mask, out=view)
                if getattr(b, "letters", None) is not None:
                    letters[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1] |= b.letters
        joined.append(_text_box(x1, y1, x2, y2, score, mask, boxes[0].source_model, letters))
    return joined


BLOCK_GAP = knob("detect.block_gap")
BLOCK_OFFSET = knob("detect.block_offset")
BLOCK_SPACING = knob("detect.block_spacing")
BLOCK_MIN_WIDTH = knob("detect.block_min_width")
BLOCK_LINE_SPREAD = knob("detect.block_line_spread")


def _separate(mask: np.ndarray, upper: list, lower: list) -> bool:
    """True when two groups of lines read as two texts rather than one spaced block."""
    def extent(group):
        cols = np.flatnonzero(mask[group[0][0]:group[-1][1]].any(axis=0))
        return float(cols[0]), float(cols[-1] + 1)

    # Only two real texts part: each of two lines or more, neither a sliver beside the other.
    if len(upper) < 2 or len(lower) < 2:
        return False
    (a1, a2), (b1, b2) = extent(upper), extent(lower)
    if min(a2 - a1, b2 - b1) < BLOCK_MIN_WIDTH * max(a2 - a1, b2 - b1):
        return False
    if abs((a1 + a2) - (b1 + b2)) / 2 > BLOCK_OFFSET * max(a2 - a1, b2 - b1):
        return True
    inner = [nxt[0] - line[1] for group in (upper, lower) for line, nxt in zip(group, group[1:])]
    return lower[0][0] - upper[-1][1] >= BLOCK_SPACING * max(inner)


def _split_blocks(box: BubbleBox) -> list[BubbleBox]:
    """A box holding two texts stacked with a wide gap (two captions, a staircase of boxes) becomes one box per text."""
    letters = getattr(box, "letters", None)
    if letters is None or box.mask is None or not letters.any():
        return [box]
    rows = letters.any(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
    lines = [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]
    if len(lines) < 2:
        return [box]
    height = float(np.median([end - start for start, end in lines]))
    # Rows far taller than a line are marks (a bubble edge, a drawn letter), not text; they never part texts.
    lines = [(start, end) for start, end in lines if end - start <= BLOCK_LINE_SPREAD * height]
    if len(lines) < 2:
        return [box]
    groups = [[lines[0]]]
    for line in lines[1:]:
        if line[0] - groups[-1][-1][1] >= BLOCK_GAP * height:
            groups.append([line])
        else:
            groups[-1].append(line)
    # A wide gap alone may be a spaced bubble; two texts also sit off each other's centre or space their own lines closer.
    merged = [groups[0]]
    for group in groups[1:]:
        if _separate(letters, merged[-1], group):
            merged.append(group)
        else:
            merged[-1] = merged[-1] + group
    if len(merged) < 2:
        return [box]
    # Each text takes the rows up to the middle of the gaps round it, and its own columns with a line of room.
    blocks = [(group[0][0], group[-1][1]) for group in merged]
    cuts = [0] + [(upper[1] + lower[0]) // 2 for upper, lower in zip(blocks, blocks[1:])] + [letters.shape[0]]
    # A cut through letters (two lines touching, read as one tall mark) would leave half a line in neither box.
    if rows[cuts[1:-1]].any():
        return [box]
    reach = int(height)
    pieces = []
    for (top, bottom), band_top, band_bottom in zip(blocks, cuts, cuts[1:]):
        cols = np.flatnonzero(letters[top:bottom].any(axis=0))
        y1, y2 = max(int(band_top), int(top) - reach), min(int(band_bottom), int(bottom) + reach)
        x1, x2 = max(0, int(cols[0]) - reach), min(letters.shape[1], int(cols[-1]) + 1 + reach)
        pieces.append(_text_box(box.x1 + x1, box.y1 + y1, box.x1 + x2, box.y1 + y2, box.confidence,
                                np.ascontiguousarray(box.mask[y1:y2, x1:x2]), box.source_model, letters[y1:y2, x1:x2]))
    return pieces


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
    def __init__(self, model_path, conf_threshold: float = knob("detect.confidence"), session=None):
        self.session = session if session is not None else make_session(model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        height, width = inp.shape[2], inp.shape[3]
        self.fixed = (int(height), int(width)) if isinstance(height, int) and isinstance(width, int) else None
        self.conf_threshold = float(conf_threshold)
        self._metrics = threading.local()
        self.source_model = Path(str(model_path)).name

    def _input_size(self, height: int, width: int) -> tuple[int, int]:
        if self.fixed is not None:
            return self.fixed
        return int(math.ceil(height / STRIDE) * STRIDE), int(math.ceil(width / STRIDE) * STRIDE)

    def raw_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Boxes (x1, y1, x2, y2, score) in ``image`` pixels from one letterboxed pass."""
        h, w = image.shape[:2]
        # M15: a degenerate image (w==0 or h==0) used to ZeroDivisionError here and surface as a
        # 500; reject it cleanly instead.
        if w <= 0 or h <= 0:
            raise ValueError(f"degenerate image shape: {image.shape}")
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
        scale = min(1.0, (size - COLUMN_GAP) / (2.0 * width))
        overlap = int(2 * size / scale) - height
        if overlap < HALVES_MIN_OVERLAP:
            overlap = min(height, HALVES_MIN_OVERLAP)
            scale = min(scale, 2.0 * size / (height + overlap))
        if scale < 1.2 * min(size / width, size / height):
            return None
        half = (height + min(overlap, height) + 1) // 2
        return scale, [(0, half), (height - half, height)]

    def band_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Boxes from overlapping full-width bands shrunk no more than the model input needs, for small text."""
        if self.fixed is None:
            return []
        h, w = image.shape[:2]
        scale = min(1.0, self.fixed[1] / w)
        band = int(self.fixed[0] / scale)
        if band >= h:
            return []
        step = int(band * (1 - BAND_OVERLAP))
        starts = list(range(0, h - band, step)) + [h - band]
        found = []
        for y0 in starts:
            for x1, y1, x2, y2, score in self.raw_boxes(image[y0:y0 + band]):
                # A box cut by an inner band edge is seen whole in the next band or the coarse pass.
                if (y1 <= EDGE_TOUCH and y0 > 0) or (y2 >= band - EDGE_TOUCH and y0 + band < h):
                    continue
                found.append((x1, y1 + y0, x2, y2 + y0, score))
        return found

    def detect_slice(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Padded text boxes for a slice: coarse, page-scale and near-native passes, pieces of one text merged."""
        h, w = image.shape[:2]
        found = _union_overlapping(self._coarse_boxes(image) + self.page_boxes(image) + self.band_boxes(image))
        return [(max(0, x1 - BOX_PAD), max(0, y1 - BOX_PAD), min(w, x2 + BOX_PAD), min(h, y2 + BOX_PAD), score)
                for x1, y1, x2, y2, score in found]

    def _coarse_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Boxes from the whole slice at once, as two halves when that helps."""
        h, w = image.shape[:2]
        plan = self.halves_plan(h, w)
        if plan is None:
            found = self.raw_boxes(image)
        else:
            scale, halves = plan
            size = self.fixed[0]
            canvas = np.full((size, size, 3), LETTERBOX_VALUE, np.uint8)
            columns, x = [], 0
            for y0, y1 in halves:
                rw, rh = min(size - x, int(w * scale)), min(size, int((y1 - y0) * scale))
                canvas[:rh, x:x + rw] = cv2.resize(image[y0:y1], (rw, rh), interpolation=cv2.INTER_LINEAR)
                columns.append((x, rw, rh, y0, y1))
                x += rw + COLUMN_GAP
            found = []
            for cx1, cy1, cx2, cy2, score in self.raw_boxes(canvas):
                px, rw, rh, y0, y1 = columns[0] if (cx1 + cx2) / 2 < columns[1][0] else columns[1]
                sx, sy = rw / w, rh / (y1 - y0)
                bx1, bx2 = int((max(cx1, px) - px) / sx), int(math.ceil((min(cx2, px + rw) - px) / sx))
                by1, by2 = int(min(cy1, rh) / sy) + y0, int(math.ceil(min(cy2, rh) / sy)) + y0
                if bx2 - bx1 > 4 and by2 - by1 > 4:
                    found.append((bx1, by1, bx2, by2, score))
            found = _merge(found)
        return found

    def page_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Boxes from the slice shrunk to page scale, where the model reads big lettering best, packed as columns."""
        h, w = image.shape[:2]
        if self.fixed is None:
            return self.raw_boxes(image)
        size_h, size_w = self.fixed
        scale = min(1.0, PAGE_WIDTH / w)
        sw = max(1, int(round(w * scale)))
        window = int(size_h / scale)
        step = int(window * (1 - PAGE_OVERLAP))
        starts = [0] if window >= h else list(range(0, h - window, step)) + [h - window]
        per_canvas = max(1, (size_w + COLUMN_GAP) // (sw + COLUMN_GAP))
        found = []
        for first in range(0, len(starts), per_canvas):
            canvas = np.full((size_h, size_w, 3), LETTERBOX_VALUE, np.uint8)
            columns = []
            for k, y0 in enumerate(starts[first:first + per_canvas]):
                part = image[y0:y0 + window]
                rh = min(size_h, int(round(part.shape[0] * scale)))
                x = k * (sw + COLUMN_GAP)
                canvas[:rh, x:x + sw] = cv2.resize(part, (sw, rh), interpolation=cv2.INTER_AREA)
                columns.append((x, rh, y0, part.shape[0]))
            for cx1, cy1, cx2, cy2, score in self.raw_boxes(canvas):
                if score < PAGE_CONFIDENCE:
                    continue
                centre = (cx1 + cx2) / 2
                column = next((c for c in columns if c[0] <= centre < c[0] + sw), None)
                if column is None:
                    continue
                x, rh, y0, part_h = column
                bx1, bx2 = int(max(0, cx1 - x) / scale), int(math.ceil(min(sw, cx2 - x) / scale))
                by1, by2 = int(max(0, cy1) / scale), int(math.ceil(min(rh, cy2) / scale))
                # A box cut by an inner window edge is seen whole in the next window.
                if (by1 <= EDGE_TOUCH / scale and y0 > 0) or (by2 >= part_h - EDGE_TOUCH / scale and y0 + part_h < h):
                    continue
                if bx2 - bx1 > 4 and by2 - by1 > 4:
                    found.append((min(w, bx1), by1 + y0, min(w, bx2), min(h, by2 + y0), score))
        return _merge(found)

    def text_boxes(self, image: np.ndarray) -> list[BubbleBox]:
        """Detected text blocks with letter masks, ready for inpainting and OCR."""
        boxes = []
        for x1, y1, x2, y2, score in self.detect_slice(image):
            (x1, y1, x2, y2), mask, letters = letter_mask(image, (x1, y1, x2, y2), with_letters=True)
            if mask.any():
                boxes.append(_text_box(x1, y1, x2, y2, score, mask.astype(np.uint8) * 255, self.source_model, letters))
        return [piece for box in _join_grown(boxes) for piece in _split_blocks(box)]

    def leftover_boxes(self, clean: np.ndarray, targets) -> list[BubbleBox]:
        """First-pass boxes the letter model still reads after inpainting, masked by what is left of their strokes."""
        boxes = []
        for t in targets:
            if still_reads(clean, (t.x1, t.y1, t.x2, t.y2)):
                (x1, y1, x2, y2), mask = letter_mask(clean, (t.x1, t.y1, t.x2, t.y2))
                if mask.any():
                    boxes.append(_text_box(x1, y1, x2, y2, t.confidence, mask.astype(np.uint8) * 255, self.source_model))
        return boxes

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        """Text boxes for the pipeline, timing kept for ``last_metrics``."""
        started = time.perf_counter()
        boxes = self.text_boxes(image)
        self._metrics.value = {"text_model_ms": (time.perf_counter() - started) * 1000.0, "result_boxes": len(boxes)}
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._metrics, "value", {}) or {})
