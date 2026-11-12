from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.detector import ctd_mask

JOIN_PX = 15
MIN_BLOCK_PX = 150
PAD_PX = 12
GHOST = 0.1
LEFT_SHARE, LEFT_MIN_PX = 0.02, 40
GHOST_SHARE, GHOST_MIN_PX = 0.03, 200
MIN_PIECES, PIECE_PX = 3, 20
UNCHANGED_TOL = 12
UNCHANGED_SHARE = 0.85
UNCHANGED_MIN_STD = 10


@dataclass(frozen=True)
class Leftover:
    box: tuple[int, int, int, int]
    kind: str
    px: int
    share: float


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for box in boxes:
        if box.mask is not None and box.mask.shape == (box.y2 - box.y1, box.x2 - box.x1):
            mask[box.y1:box.y2, box.x1:box.x2] |= box.mask > 127
    return mask


def _blocks(letters: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(letters.astype(np.uint8), np.ones((JOIN_PX, JOIN_PX), np.uint8)))
    blocks = []
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        block = (labels[y:y + h, x:x + w] == i) & letters[y:y + h, x:x + w]
        if block.sum() >= MIN_BLOCK_PX:
            blocks.append((block, (x, y, x + w, y + h)))
    return blocks


def _kept(box: tuple[int, int, int, int], keep) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(k[0] <= cx < k[2] and k[1] <= cy < k[3] for k in keep)


def _covered_by(rect: tuple[int, int, int, int], rects) -> bool:
    x1, y1, x2, y2 = rect
    area = max(1, (x2 - x1) * (y2 - y1))
    for ax1, ay1, ax2, ay2 in rects:
        ix = max(0, min(x2, ax2) - max(x1, ax1))
        iy = max(0, min(y2, ay2) - max(y1, ay1))
        if ix * iy >= 0.25 * area:
            return True
    return False


def leftovers(original: np.ndarray, clean: np.ndarray, letters: np.ndarray, keep=(), raw_boxes=()) -> list[Leftover]:
    h, w = letters.shape
    found = []
    rects = []
    for block, (x1, y1, x2, y2) in _blocks(letters):
        rects.append((x1, y1, x2, y2))
        if _kept((x1, y1, x2, y2), keep):
            continue
        bx1, by1, bx2, by2 = max(0, x1 - PAD_PX), max(0, y1 - PAD_PX), min(w, x2 + PAD_PX), min(h, y2 + PAD_PX)
        area = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
        area[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1] = block
        area = cv2.dilate(area, np.ones((2 * PAD_PX + 1,) * 2, np.uint8)) > 0
        before = ctd_mask.probability(np.ascontiguousarray(original[by1:by2, bx1:bx2]))
        after = ctd_mask.probability(np.ascontiguousarray(clean[by1:by2, bx1:bx2]))
        inked = (before > ctd_mask.THRESHOLD) & area
        _, _, stats, _ = cv2.connectedComponentsWithStats(inked.astype(np.uint8))
        if int((stats[1:, cv2.CC_STAT_AREA] >= PIECE_PX).sum()) < MIN_PIECES:
            continue
        ink = max(1, int(inked.sum()))
        left = int(((after > ctd_mask.THRESHOLD) & area).sum())
        faint = int(((after > GHOST) & area).sum())
        if left >= max(LEFT_MIN_PX, LEFT_SHARE * ink):
            found.append(Leftover((x1, y1, x2, y2), "text", left, round(left / ink, 3)))
        elif faint >= max(GHOST_MIN_PX, GHOST_SHARE * ink):
            found.append(Leftover((x1, y1, x2, y2), "ghost", faint, round(faint / ink, 3)))
    for box in raw_boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8 or _kept((x1, y1, x2, y2), keep) or _covered_by((x1, y1, x2, y2), rects):
            continue
        before = original[y1:y2, x1:x2]
        after = clean[y1:y2, x1:x2]
        if before.shape != after.shape or before.size == 0:
            continue
        grey = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
        if float(grey.std()) < UNCHANGED_MIN_STD:
            continue
        same = np.abs(before.astype(np.int16) - after.astype(np.int16))
        same = same.max(axis=2) if same.ndim == 3 else same
        same = same <= UNCHANGED_TOL
        share = float(same.mean())
        if share >= UNCHANGED_SHARE:
            found.append(Leftover((x1, y1, x2, y2), "text", int(same.sum()), round(share, 3)))
    return found
