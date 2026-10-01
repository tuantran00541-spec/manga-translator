"""Local checks of a cleaned slice with the detector models: text or ghosts an erase left behind."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.detector import ctd_mask

JOIN_PX = 15  # letters closer than this belong to one block of text
MIN_BLOCK_PX = 150  # smaller letter groups are specks, not text
PAD_PX = 12  # room round a block in which leftover marks are counted
GHOST = 0.1  # letter probability of a faint trace: a soft shadow or a redrawn outline
LEFT_SHARE, LEFT_MIN_PX = 0.02, 40  # ink still read, as a share of the block's original ink
GHOST_SHARE, GHOST_MIN_PX = 0.03, 200  # faint traces, as a share of the block's original ink
MIN_PIECES, PIECE_PX = 3, 20  # text is several separate letters; a balloon outline scrap is one or two strokes


@dataclass(frozen=True)
class Leftover:
    box: tuple[int, int, int, int]
    kind: str  # "text" still reads as letters, "ghost" is a faint trace of them
    px: int
    share: float


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    """The letter pixels of detected text boxes on one image."""
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


def leftovers(original: np.ndarray, clean: np.ndarray, letters: np.ndarray, keep=()) -> list[Leftover]:
    """Blocks of the original's letters that still read as text, or as a faint ghost, on the clean image.

    ``letters`` marks the original's letters; blocks centred in a ``keep`` box (logos, art) are left alone.
    """
    h, w = letters.shape
    found = []
    for block, (x1, y1, x2, y2) in _blocks(letters):
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
    return found
