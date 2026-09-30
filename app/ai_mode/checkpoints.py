"""Checkpoint 3 (raw vs clean review): prompt and parsing."""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    "You do the final check of an automatic manga text cleanup. IMAGE 1 is the ORIGINAL slice; IMAGE 2 is the "
    "same slice after the text was erased (CLEAN). Scan CLEAN top to bottom and edge to edge, inside and around "
    "every bubble, caption box, dark area, gradient, panel border, screen and the art, and compare each spot "
    "with IMAGE 1. Report three kinds of problem:\n"
    "- missed: readable text left in CLEAN that someone says, thinks or narrates (dialogue, narration, captions, "
    "system messages, comments on a screen, titles), in any language, even when only partly erased. It will be "
    "erased and translated, so report every such line.\n"
    "- residue: fragments of erased letters, ghost outlines, smears, blotches and leftover scanlator watermarks "
    "or credits. They will be erased; nothing is translated.\n"
    "- restore: artwork CLEAN damaged that holds no words: ornaments, patterns, drawn objects, the series logo. "
    "The original pixels are put back untranslated, so list only artwork, never text a reader reads, whatever its "
    "font, size or colour.\n"
    "Leave sound effects drawn into the art unreported. "
    + BOX_RULE
    + "Keep each box tight around the problem. "
    "A list with nothing to report stays empty; a clean slice returns three empty lists. Return JSON only: "
    f'{{"missed":[{_BOX}],"residue":[{_BOX}],"restore":[{_BOX}]}}'
)

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
                 original: np.ndarray, clean: np.ndarray) -> tuple[CleanReview, float | None]:
    """Checkpoint 3: one request with the raw and the cleaned slice."""
    # This check thinks (low effort) so it looks at every part of the slice.
    result = request_vision_json(provider, model, api_key, CLEAN_REVIEW_PROMPT,
                                 [("IMAGE 1: ORIGINAL", original), ("IMAGE 2: CLEAN", clean)], max_tokens=1200,
                                 reasoning_effort=CLEAN_REVIEW_EFFORT, stage="review")
    height, width = original.shape[:2]
    return parse_clean_review(result.data, page_index, width, height), result.estimated_cost_usd
