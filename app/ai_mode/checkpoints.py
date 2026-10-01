"""Checkpoint 3 (review of the cleaned slice): prompt and parsing."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai_mode.vision_json import request_vision_json
from app.box_format import BOX_RULE, scaled_box

MIN_CONFIDENCE = 0.6
RESTORE_CONFIDENCE = 0.8  # a wrong restore leaves source text on the page
OVERLAP = 0.3  # of the smaller box; two lettered objects overlapping this much garble each other
MIN_SIDE_PX = 10
MAX_AREA_RATIO = 0.4  # a box over most of a slice is a misread
MARGIN_PX = 6
MAX_BOXES = 12
CLEAN_REVIEW_EFFORT = "low"

_BOX = '{"x1":0,"y1":0,"x2":0,"y2":0,"confidence":0.0}'

CLEAN_REVIEW_PROMPT = (
    "You do the final check of an automatic manga text cleanup. The image is one slice after its text was "
    "erased; each green box marks a place where text was erased. Look in and around every green box for "
    "fragments of erased letters, ghost outlines, smears and blotches, and anywhere for leftover scanlator "
    "watermarks or credits. They will be erased; nothing is translated. Text that was never erased is found by "
    "the translator, and sound effects and the art stay unreported. "
    + BOX_RULE
    + "Keep each box tight around the problem. A clean slice returns an empty list. Return JSON only: "
    f'{{"residue":[{_BOX}]}}'
)
ERASED_COLOR = (0, 170, 0)  # BGR green


def mark_erased(clean: np.ndarray, boxes) -> np.ndarray:
    """The clean slice with a thin green outline round each place where text was erased."""
    marked = clean.copy()
    thickness = max(2, clean.shape[1] // 500)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(marked, (int(x1), int(y1)), (int(x2), int(y2)), ERASED_COLOR, thickness)
    return marked


@dataclass(frozen=True)
class CleanReview:
    page_index: int
    missed: tuple[tuple[int, int, int, int], ...] = ()
    residue: tuple[tuple[int, int, int, int], ...] = ()
    restore: tuple[tuple[int, int, int, int], ...] = ()


def _box(raw, width: int, height: int, *, min_confidence: float = MIN_CONFIDENCE) -> tuple[int, int, int, int] | None:
    """Pixel box from a named 0-1000 box, or None when unsure, tiny or implausibly large."""
    if not isinstance(raw, dict):
        return None
    try:
        if float(raw.get("confidence", 1.0)) < min_confidence:
            return None
    except (TypeError, ValueError):
        return None
    scaled = scaled_box(raw, width, height)
    if scaled is None:
        return None
    x1, y1, x2, y2 = scaled
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX or (x2 - x1) * (y2 - y1) > MAX_AREA_RATIO * width * height:
        return None
    return (max(0, math.floor(x1) - MARGIN_PX), max(0, math.floor(y1) - MARGIN_PX),
            min(width, math.ceil(x2) + MARGIN_PX), min(height, math.ceil(y2) + MARGIN_PX))


def _boxes(items, width: int, height: int, min_confidence: float = MIN_CONFIDENCE) -> tuple[tuple[int, int, int, int], ...]:
    items = items if isinstance(items, list) else []
    return tuple(box for raw in items[:MAX_BOXES]
                 if (box := _box(raw, width, height, min_confidence=min_confidence)) is not None)


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


def settle_clean_review(review: CleanReview, existing: list[tuple[int, int, int, int]]) -> CleanReview:
    """Missed text over an existing box is re-erased, not added twice; restore never brings back text."""
    missed = [box for box in review.missed if not _covered(box, existing)]
    residue = list(review.residue) + [box for box in review.missed if box not in missed]
    erase = missed + residue
    # Restoring over a detected text box puts the source lettering back untranslated.
    restore = [box for box in review.restore if not _covered(box, erase + list(existing))]
    return CleanReview(review.page_index, tuple(missed), tuple(residue), tuple(restore))


def parse_clean_review(data: dict, page_index: int, width: int, height: int) -> CleanReview:
    data = data if isinstance(data, dict) else {}
    return CleanReview(
        page_index,
        missed=_boxes(data.get("missed"), width, height),
        residue=_boxes(data.get("residue"), width, height),
        restore=_boxes(data.get("restore"), width, height, RESTORE_CONFIDENCE),
    )


def review_clean(provider, model: str, api_key: str, page_index: int,
                 original: np.ndarray, clean: np.ndarray, boxes=()) -> tuple[CleanReview, float | None]:
    """Checkpoint 3: one request with the cleaned slice and its erased places outlined (the raw slice never led to a restore)."""
    # This check thinks (low effort) so it looks at every part of the slice.
    result = request_vision_json(provider, model, api_key, CLEAN_REVIEW_PROMPT,
                                 [("CLEAN SLICE", mark_erased(clean, boxes))], max_tokens=1200,
                                 reasoning_effort=CLEAN_REVIEW_EFFORT, stage="review")
    height, width = original.shape[:2]
    return parse_clean_review(result.data, page_index, width, height), result.estimated_cost_usd
