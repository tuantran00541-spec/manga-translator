"""Checkpoint 3: crops round every erased place, packed onto a few sheets, checked for residue."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.ai_mode.vision_json import request_vision_json

OVERLAP = 0.3  # of the smaller box; two lettered objects overlapping this much garble each other
MIN_SIDE_PX = 10
MAX_AREA_RATIO = 0.4  # a box over most of a slice is a misread
MARGIN_PX = 6
CROP_PAD_PX = 24  # background shown round an erased box, where a ghost or smear would sit
MAX_CROP_SIDE = 900  # a bigger crop is shrunk to this; letter fragments still show
SHEET_SIDE = 2048
TAG_PX = 30  # strip above each crop carrying its id
GAP_PX = 8
CLEAN_REVIEW_EFFORT = "low"

CLEAN_REVIEW_PROMPT = (
    "You do the final check of an automatic manga text cleanup. The image is a sheet of crops; each crop shows "
    "one place on a slice where text was erased, with its id in the black tag above it. For each crop decide "
    "whether fragments of the erased letters, ghost outlines, smears or blotches are left. Clean backgrounds, "
    "drawn art, balloon outlines, sound effects and text on a sign that was never erased are not residue. "
    'Return JSON only: {"residue":["<id>"]} with the ids that need another erase, or an empty list.'
)


@dataclass(frozen=True)
class Crop:
    id: str
    page_index: int
    rect: tuple[int, int, int, int]  # the erased box on the slice
    image: np.ndarray


def crops_for(page_index: int, clean: np.ndarray, boxes) -> list[Crop]:
    """One crop of the clean slice round each erased box."""
    h, w = clean.shape[:2]
    out = []
    for n, (x1, y1, x2, y2) in enumerate(boxes):
        cx1, cy1 = max(0, x1 - CROP_PAD_PX), max(0, y1 - CROP_PAD_PX)
        cx2, cy2 = min(w, x2 + CROP_PAD_PX), min(h, y2 + CROP_PAD_PX)
        if cx2 - cx1 < MIN_SIDE_PX or cy2 - cy1 < MIN_SIDE_PX:
            continue
        crop = clean[cy1:cy2, cx1:cx2]
        scale = min(1.0, MAX_CROP_SIDE / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(Crop(f"{page_index + 1}.{n + 1}", page_index, (x1, y1, x2, y2), crop))
    return out


def pack(crops: list[Crop]) -> list[tuple[np.ndarray, list[Crop]]]:
    """Crops in rows on white sheets no bigger than SHEET_SIDE, each under a black tag with its id."""
    sheets, placed, rows = [], [], []
    x = y = row_h = 0

    def flush():
        nonlocal placed, rows
        if placed:
            height = max(top + c.image.shape[0] + TAG_PX for _, top, c in rows)
            width = max(left + c.image.shape[1] for left, _, c in rows)
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
    area = (box[2] - box[0]) * (box[3] - box[1])
    for other in others:
        ix = min(box[2], other[2]) - max(box[0], other[0])
        iy = min(box[3], other[3]) - max(box[1], other[1])
        smaller = max(1, min(area, (other[2] - other[0]) * (other[3] - other[1])))
        if ix > 0 and iy > 0 and ix * iy >= OVERLAP * smaller:
            return True
    return False


def review_sheet(provider, model: str, api_key: str, sheet: np.ndarray, crops: list[Crop]) -> tuple[list[Crop], float | None]:
    """Checkpoint 3: one request per sheet; returns the crops with residue left."""
    # This check thinks (low effort) so it looks at every crop.
    result = request_vision_json(provider, model, api_key, CLEAN_REVIEW_PROMPT, [("SHEET OF CROPS", sheet)],
                                 max_tokens=800, reasoning_effort=CLEAN_REVIEW_EFFORT, stage="review")
    flagged = {str(item) for item in result.data.get("residue") or [] if isinstance(item, (str, int, float))}
    return [crop for crop in crops if crop.id in flagged], result.estimated_cost_usd
