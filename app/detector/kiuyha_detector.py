"""The text detector: Kiuyha/Manga-Bubble-YOLO boxes (ONNX) with Otsu letter masks."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.detector.boxes import BubbleBox
from app.ort_utils import make_session

LETTERBOX_VALUE = 114
STRIDE = 32
NMS_IOU = 0.7  # Ultralytics' default for this head
BOX_PAD = 16  # Kiuyha's boxes can stop short of the last letter of a wide line
HALVES_GAP = 16
HALVES_MIN_OVERLAP = 256
CLOSE_KERNEL = 15  # letters -> word blobs
GROW_KERNEL = 13  # past the letter outline (a white stroke round brown text)
EDGE_GROW = 24  # room around a box for letters its edge cuts
LINE_REACH_MAX = 240  # how far sideways a line may run past its box
LINE_SHARE = 0.6  # height two letters share to be on one line
LETTER_GAP = 1.2  # widest gap, in letter heights, between neighbouring letters of a line
OUTLINE_SHARE = 0.35  # mask growth as a share of the letter height
GROW_MAX = 31  # outlines stop growing past this, so huge sound effects do not swallow the art round them
BAND_OVERLAP = 0.25  # share of a detection band repeated in the next one
EDGE_TOUCH = 3  # a box this close to an inner band edge was cut by it
UNION_SHARE = 0.3  # boxes sharing this much of the smaller one are one text
SAME_LINE = 0.7  # overlapping boxes sharing this much of the shorter height are pieces of one line
LEFTOVER_GROW = 9  # the second pass also takes the glow round what is left
LETTER_FILL = 0.2  # ink share of a letter's bounding box; outlines and hairlines fall below it
SPLIT_GAIN = 1.5  # a dark or light split must find this much more letter area than the border-colour one
FAR_SHARE = 0.5  # share of the border-colour threshold a dark or light letter must still stand out by
SPECK_SHARE = 0.4  # in a dark or light split, pieces shorter than this share of the letters are art
INK_TOLERANCE = 40  # how much fainter than the letters, on the 0-255 split score, a joined piece may be
HALO_REACH = 1.0  # how far, in letter heights, glow may spread from a letter
HALO_REACH_MAX = 64  # and never further than this many pixels
HALO_RING = 8  # width of the band past that reach where the background colour is read
HALO_MARGIN = 4.0  # Lab distance past the band's own spread that still counts as glow
HOLE_AREA = 4.0  # largest enclosed hole filled, in squared letter heights (white fill inside an outline)


def letter_mask(image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Letters of ``box`` including ones the box cuts at its edge; returns the grown box and its mask."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    # Lines can run well past a box's side, so the crop reaches further sideways than up and down.
    side = max(EDGE_GROW, min(LINE_REACH_MAX, 3 * (y2 - y1)))
    gx1, gy1, gx2, gy2 = max(0, x1 - side), max(0, y1 - EDGE_GROW), min(w, x2 + side), min(h, y2 + EDGE_GROW)
    crop = image[gy1:gy2, gx1:gx2]
    ix1, iy1, ix2, iy2 = x1 - gx1, y1 - gy1, x2 - gx1, y2 - gy1
    if min(iy2 - iy1, ix2 - ix1) < 8:
        return box, np.zeros((y2 - y1, x2 - x1), bool)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = lab[iy1:iy2, ix1:ix2]
    # The background colour comes from the detector box's own border, as before.
    ring = np.concatenate([inner[:3].reshape(-1, 3), inner[-3:].reshape(-1, 3),
                           inner[:, :3].reshape(-1, 3), inner[:, -3:].reshape(-1, 3)])
    diff = np.linalg.norm(lab - np.median(ring, axis=0), axis=2)
    scale = 255.0 / max(1.0, float(diff[iy1:iy2, ix1:ix2].max()))
    lightness = lab[..., 0].astype(np.uint8)
    open_top, open_bottom, open_left, open_right = gy1 == 0, gy2 == h, gx1 == 0, gx2 == w
    ch, cw = crop.shape[:2]

    def touches_edge(x, y, bw, bh):
        return ((not open_top and y == 0) or (not open_bottom and y + bh >= ch)
                or (not open_left and x == 0) or (not open_right and x + bw >= cw))

    def split(score, letters_only=False):
        # The threshold is chosen inside the box and applied to the grown crop.
        level = cv2.threshold(score[iy1:iy2, ix1:ix2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        count, labels, stats, _ = cv2.connectedComponentsWithStats((score > level).astype(np.uint8))
        inside = [label for label in range(1, count) if stats[label][0] >= ix1 and stats[label][1] >= iy1
                  and stats[label][0] + stats[label][2] <= ix2 and stats[label][1] + stats[label][3] <= iy2
                  and not touches_edge(*stats[label][:4])]
        if letters_only and inside:
            # Gaps of background between glowing letters are dark too; letters stand out from the border colour.
            far = np.bincount(labels.ravel(), weights=base[0].ravel(), minlength=count) / np.maximum(1, stats[:, 4])
            inside = [label for label in inside if far[label] >= FAR_SHARE * base[1]]
        if letters_only and inside:
            # Art specks (windows, leaves) are many but small; letters hold most of the ink.
            tall = _ink_height(stats, inside)
            inside = [label for label in inside if stats[label][3] >= SPECK_SHARE * tall]
        return score, level, count, labels, stats, inside, not letters_only

    # Far from the border colour first; on a border crossing sky and trees, dark or light letters instead.
    base = split(np.clip(diff * scale, 0, 255).astype(np.uint8))
    candidates = [base, split(255 - lightness, letters_only=True), split(lightness, letters_only=True)]
    area = [int(sum(c[4][label][4] for label in c[5])) for c in candidates]
    best = max(range(1, len(candidates)), key=lambda i: area[i])
    diff, level, count, labels, stats, inside, glow = candidates[best if area[best] > SPLIT_GAIN * area[0] else 0]
    # Letters share one ink; art of the same height (trees, windows) is fainter in the split.
    means = np.bincount(labels.ravel(), weights=diff.ravel(), minlength=count) / np.maximum(1, stats[:, 4])
    ink = float(np.median(means[inside])) if inside else 0.0
    if not glow:  # a dark or light split also holds art strokes inside the box; letters are the deepest ink
        inside = [label for label in inside if means[label] >= ink - INK_TOLERANCE]
    keep = np.zeros(count, bool)
    keep[inside] = True
    letter = _ink_height(stats, inside)

    def letter_like(label) -> bool:
        x, y, bw, bh, area = stats[label]
        # Letter sized and solid, not a thin outline or art running out of the crop.
        return (bool(letter) and not touches_edge(x, y, bw, bh) and bh <= 1.5 * letter and bw <= 3 * letter
                and area >= LETTER_FILL * bw * bh and means[label] >= ink - INK_TOLERANCE)

    for label in range(1, count):
        x, y, bw, bh, _area = stats[label]
        if not keep[label] and x < ix2 and y < iy2 and x + bw > ix1 and y + bh > iy1 and letter_like(label):
            keep[label] = True  # a letter the box cut in half
    # Letters past the box's side that continue a kept line, one letter gap at a time.
    grew = bool(letter)
    while grew:
        grew = False
        kept = np.flatnonzero(keep)
        for label in range(1, count):
            if keep[label] or not letter_like(label):
                continue
            x, y, bw, bh, _area = stats[label]
            for other in kept:
                ox, oy, ow, oh, _ = stats[other]
                same_line = min(y + bh, oy + oh) - max(y, oy) >= LINE_SHARE * min(bh, oh)
                gap = max(x - (ox + ow), ox - (x + bw))
                if same_line and gap <= LETTER_GAP * letter:
                    keep[label] = grew = True
                    break
    part = keep[labels]
    if letter:
        if glow:  # a dark or light split is chosen over busy art, where there is no glow to follow
            part = _with_halo(lab, part, letter)
        part = _fill_holes(part, letter)
    part = part.astype(np.uint8)
    part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL,) * 2))
    # Outlines and glow round big lettering are wider than round small text.
    grow = min(GROW_MAX, max(GROW_KERNEL, int(OUTLINE_SHARE * letter) | 1))
    part = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow,) * 2)) > 0
    ys, xs = np.nonzero(part)
    bx1, by1 = min(ix1, int(xs.min())) if len(xs) else ix1, min(iy1, int(ys.min())) if len(ys) else iy1
    bx2, by2 = max(ix2, int(xs.max()) + 1) if len(xs) else ix2, max(iy2, int(ys.max()) + 1) if len(ys) else iy2
    return (gx1 + bx1, gy1 + by1, gx1 + bx2, gy1 + by2), part[by1:by2, bx1:bx2]


def _ink_height(stats: np.ndarray, labels: list[int]) -> float:
    """Letter height: the height holding the middle of the ink, so specks and stars do not shrink it."""
    if not labels:
        return 0.0
    order = sorted(labels, key=lambda label: stats[label][3])
    ink = np.cumsum([stats[label][4] for label in order])
    return float(stats[order[int(np.searchsorted(ink, ink[-1] / 2))]][3])


def _with_halo(lab: np.ndarray, part: np.ndarray, letter: float) -> np.ndarray:
    """Letters plus the glow joined to them, out to where the colour settles to the background past it."""
    reach = max(3, min(HALO_REACH_MAX, int(HALO_REACH * letter)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
    near = cv2.dilate(part.astype(np.uint8), kernel) > 0
    outer = (cv2.dilate(near.astype(np.uint8), np.ones((2 * HALO_RING + 1,) * 2, np.uint8)) > 0) & ~near
    if int(outer.sum()) < 50:
        return part
    # The background is read beyond the glow, not on the box border that may sit inside it.
    dist = np.linalg.norm(lab - np.median(lab[outer], axis=0), axis=2)
    # Busy art past the glow raises the bar, so only light well above it is taken.
    noise = float(np.percentile(dist[outer], 90))
    faint = ((dist > noise + HALO_MARGIN) & near) | part
    _count, labels = cv2.connectedComponents(faint.astype(np.uint8))
    joined = np.unique(labels[part])
    return np.isin(labels, joined[joined > 0])


def _fill_holes(part: np.ndarray, letter: float) -> np.ndarray:
    """Enclosed holes up to HOLE_AREA letter squares, such as white fill inside a dark outline."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~part).astype(np.uint8), connectivity=4)
    h, w = part.shape
    holes = [label for label in range(1, count) if stats[label][4] <= HOLE_AREA * letter * letter
             and stats[label][0] > 0 and stats[label][1] > 0
             and stats[label][0] + stats[label][2] < w and stats[label][1] + stats[label][3] < h]
    return part | np.isin(labels, holes) if holes else part


def _text_box(x1, y1, x2, y2, score, mask, source_model) -> BubbleBox:
    return BubbleBox(
        x1, y1, x2, y2, score, mask,
        source_model=source_model, class_name="text", semantic_type="free_text",
        mask_source="text_segmenter", safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
    )


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
        for b in boxes:
            if b.x1 >= x1 and b.y1 >= y1 and b.x2 <= x2 and b.y2 <= y2:
                view = mask[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1]
                np.maximum(view, b.mask, out=view)
        joined.append(_text_box(x1, y1, x2, y2, score, mask, boxes[0].source_model))
    return joined


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
        self._metrics = threading.local()
        self.source_model = Path(str(model_path)).name

    def _input_size(self, height: int, width: int) -> tuple[int, int]:
        if self.fixed is not None:
            return self.fixed
        return int(math.ceil(height / STRIDE) * STRIDE), int(math.ceil(width / STRIDE) * STRIDE)

    def raw_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Boxes (x1, y1, x2, y2, score) in ``image`` pixels from one letterboxed pass."""
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
        """Padded text boxes for a slice: one coarse pass plus near-native bands, pieces of one text merged."""
        h, w = image.shape[:2]
        found = _union_overlapping(self._coarse_boxes(image) + self.band_boxes(image))
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
                x += rw + HALVES_GAP
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

    def text_boxes(self, image: np.ndarray) -> list[BubbleBox]:
        """Detected text blocks with letter masks, ready for inpainting and OCR."""
        boxes = []
        for x1, y1, x2, y2, score in self.detect_slice(image):
            (x1, y1, x2, y2), mask = letter_mask(image, (x1, y1, x2, y2))
            if mask.any():
                boxes.append(_text_box(x1, y1, x2, y2, score, mask.astype(np.uint8) * 255, self.source_model))
        return _join_grown(boxes)

    def leftover_boxes(self, clean: np.ndarray, targets) -> list[BubbleBox]:
        """Text still seen inside a first-pass box after inpainting, masked by what is left of its strokes."""
        boxes = []
        for x1, y1, x2, y2, score in self.detect_slice(clean):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(t.x1 <= cx <= t.x2 and t.y1 <= cy <= t.y2 for t in targets):
                (x1, y1, x2, y2), mask = letter_mask(clean, (x1, y1, x2, y2))
                if mask.any():
                    # Glow and outlines left round erased letters; a rectangle would leave a flat patch on art.
                    mask = cv2.dilate(mask.astype(np.uint8) * 255, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (LEFTOVER_GROW,) * 2))
                else:
                    mask = np.full((y2 - y1, x2 - x1), 255, np.uint8)
                boxes.append(_text_box(x1, y1, x2, y2, score, mask, self.source_model))
        return boxes

    def detect(self, image: np.ndarray, *, parallel: bool = False) -> list[BubbleBox]:
        """Text boxes for the pipeline, timing kept for ``last_metrics``."""
        started = time.perf_counter()
        boxes = self.text_boxes(image)
        self._metrics.value = {"text_model_ms": (time.perf_counter() - started) * 1000.0, "result_boxes": len(boxes)}
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._metrics, "value", {}) or {})
