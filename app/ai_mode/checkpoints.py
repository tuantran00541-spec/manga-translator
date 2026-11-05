"""Checkpoint 3: raw | clean crop pairs round every erased place, packed onto a few sheets, checked for residue."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.ai_mode.vision_json import request_vision_json
from app.region_policy import overlap_share

OVERLAP = 0.3  # of the smaller box; two lettered objects overlapping this much garble each other
MIN_SIDE_PX = 10
MAX_AREA_RATIO = 0.4  # a box over most of a slice is a misread
MARGIN_PX = 6
CROP_PAD_PX = 24  # background shown round an erased box, where a ghost or smear would sit
SHEET_SIDE = 2048
TAG_PX = 30  # strip above each crop carrying its id
GAP_PX = 8
TILE_OVERLAP = 128
CLEAN_REVIEW_EFFORT = "low"

CLEAN_REVIEW_PROMPT = (
    "IMAGE: a sheet of crop pairs from a manga chapter. In each pair LEFT is the raw crop with its text and RIGHT "
    "is the same place after the text was erased; the pair's id is in the black tag above it.\n"
    "QUESTION: compared with LEFT, does RIGHT still show any of the erased text: letter fragments, ghost outlines, "
    "smears or blotches?\n"
    "Text that looks the same in LEFT and RIGHT was kept on purpose; clean background, art and balloon outlines "
    "are not residue.\n"
    'ANSWER with JSON only: {"residue":["<id>"]} listing the ids whose RIGHT must be erased again, [] when all '
    "are clean."
)


@dataclass(frozen=True)
class Crop:
    id: str
    page_index: int
    rect: tuple[int, int, int, int]  # the erased box on the slice
    image: np.ndarray


def crops_for(page_index: int, original: np.ndarray, clean: np.ndarray, boxes) -> list[Crop]:
    """One raw | clean pair round each erased box."""
    h, w = clean.shape[:2]
    out = []
    def starts(length, side):
        values = [0]
        while values[-1] + side < length:
            values.append(min(values[-1] + side - TILE_OVERLAP, length - side))
        return values

    for x1, y1, x2, y2 in boxes:
        tile_w, tile_h = (SHEET_SIDE - GAP_PX) // 2, SHEET_SIDE - TAG_PX
        # Do not split a region that already fits merely to add its surrounding padding.
        pad_x = min(CROP_PAD_PX, max(0, (tile_w - (x2 - x1)) // 2)) if x2 - x1 <= tile_w else CROP_PAD_PX
        pad_y = min(CROP_PAD_PX, max(0, (tile_h - (y2 - y1)) // 2)) if y2 - y1 <= tile_h else CROP_PAD_PX
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if cx2 - cx1 < MIN_SIDE_PX or cy2 - cy1 < MIN_SIDE_PX:
            continue
        for ty in starts(cy2 - cy1, tile_h):
            for tx in starts(cx2 - cx1, tile_w):
                ax, ay = cx1 + tx, cy1 + ty
                bx, by = min(cx2, ax + tile_w), min(cy2, ay + tile_h)
                rect = (max(x1, ax), max(y1, ay), min(x2, bx), min(y2, by))
                if rect[0] >= rect[2] or rect[1] >= rect[3]:
                    continue
                before, after = original[ay:by, ax:bx], clean[ay:by, ax:bx]
                crop = np.hstack([before, np.full((before.shape[0], GAP_PX, 3), 255, np.uint8), after])
                out.append(Crop(f"{page_index + 1}.{len(out) + 1}", page_index, rect, crop))
    return out


def pack(crops: list[Crop]) -> list[tuple[np.ndarray, list[Crop]]]:
    """Crops in rows on white sheets no bigger than SHEET_SIDE, each under a black tag with its id."""
    sheets, placed, rows = [], [], []
    x = y = row_h = 0

    def flush():
        nonlocal placed, rows
        if placed:
            height = max(top + c.image.shape[0] + TAG_PX for _, top, c in rows)
            width = max(left + max(c.image.shape[1], 70) for left, _, c in rows)
            sheet = np.full((height, width, 3), 255, np.uint8)
            for left, top, crop in rows:
                ch, cw = crop.image.shape[:2]
                cv2.rectangle(sheet, (left, top), (left + max(cw, 70), top + TAG_PX - 2), (0, 0, 0), -1)
                cv2.putText(sheet, crop.id, (left + 4, top + TAG_PX - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                sheet[top + TAG_PX:top + TAG_PX + ch, left:left + cw] = crop.image
            sheets.append((sheet, placed))
        placed, rows = [], []

    for crop in sorted(crops, key=lambda c: -c.image.shape[0]):
        ch, cw = crop.image.shape[0] + TAG_PX, max(crop.image.shape[1], 70)
        if x and x + cw > SHEET_SIDE:
            x, y, row_h = 0, y + row_h + GAP_PX, 0
        if y and y + ch > SHEET_SIDE:
            flush()
            x = y = row_h = 0
        rows.append((x, y, crop))
        placed.append(crop)
        x, row_h = x + cw + GAP_PX, max(row_h, ch)
    flush()
    return sheets


def _covered(box, others) -> bool:
    """True when ``box`` and one of ``others`` share at least OVERLAP of the smaller one's area."""
    return any(overlap_share(box, other) >= OVERLAP for other in others)


def review_sheet(provider, model: str, api_key: str, sheet: np.ndarray, crops: list[Crop]) -> tuple[list[Crop], float | None]:
    """Checkpoint 3: one request per sheet; returns the crops with residue left."""
    # This check thinks (low effort) so it looks at every crop.
    result = request_vision_json(provider, model, api_key, CLEAN_REVIEW_PROMPT, [("SHEET OF CROPS", sheet)],
                                 max_tokens=800, reasoning_effort=CLEAN_REVIEW_EFFORT, stage="review")
    flagged = {str(item) for item in result.data.get("residue") or [] if isinstance(item, (str, int, float))}
    return [crop for crop in crops if crop.id in flagged], result.estimated_cost_usd
