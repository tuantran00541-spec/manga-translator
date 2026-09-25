"""Ask a vision model which slices are credit pages and where series logos are.

Runs on the ORIGINAL slices before cleanup, so credit slices can be skipped
(no cleanup time spent on them) and logo artwork can be stored as preserve
regions before inpainting touches it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai_mode.vision_json import request_vision_json

# Only confident answers change the chapter; everything else stays as is.
CREDIT_MIN_CONFIDENCE = 0.75
LOGO_MIN_CONFIDENCE = 0.6
# A "logo" covering most of a slice is a misread, not a logo.
LOGO_MAX_AREA_RATIO = 0.5
LOGO_MIN_SIDE_PX = 12
LOGO_MARGIN_PX = 8
SCAN_MAX_SIDE = 1280

SCAN_SCHEMA = {
    "type": "object",
    "properties": {
        "slices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slice": {"type": "integer"},
                    "is_credit": {"type": "boolean"},
                    "credit_confidence": {"type": "number"},
                    "logos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "box_2d": {"type": "array", "items": {"type": "number"}},
                                "confidence": {"type": "number"},
                            },
                            "required": ["box_2d", "confidence"],
                        },
                    },
                    "reason": {"type": "string"},
                },
                "required": ["slice", "is_credit", "credit_confidence", "logos"],
            },
        }
    },
    "required": ["slices"],
}

SCAN_PROMPT = (
    "You are preparing manga/manhwa/webtoon slices for automatic translation. "
    "Each image below is one vertical slice of a chapter, labelled with its slice number. "
    "For EVERY slice decide:\n"
    "1. is_credit: true only when the WHOLE slice is non-story material added by the "
    "uploader or scanlation group: credit/staff pages, recruitment ads, Discord/Patreon/"
    "donation promotions, 'read at <site>' banners, end-of-chapter notices. A story slice "
    "with a small watermark is NOT a credit slice. Blank or near-blank slices are NOT credit.\n"
    "2. logos: boxes around the SERIES TITLE LOGO (stylised title artwork, usually near the "
    "start of the chapter) and publisher/studio logos drawn as artwork. These are kept "
    "untouched. Never box speech bubbles, captions, narration, sound effects or plain text.\n"
    "Boxes are [ymin, xmin, ymax, xmax] normalised to 0-1000 inside that slice's image. "
    "Confidences are 0-1. Return JSON only: "
    '{"slices":[{"slice":<number>,"is_credit":false,"credit_confidence":0.0,'
    '"logos":[{"box_2d":[0,0,0,0],"confidence":0.0}],"reason":"short"}]}'
)


@dataclass(frozen=True)
class SliceScan:
    page_index: int
    is_credit: bool
    credit_confidence: float
    logos: tuple[tuple[int, int, int, int], ...]
    reason: str = ""


def _confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def _logo_box(raw, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, dict) or _confidence(raw.get("confidence")) < LOGO_MIN_CONFIDENCE:
        return None
    box = raw.get("box_2d")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (ymin, xmin, ymax, xmax)) or ymax <= ymin or xmax <= xmin:
        return None
    x1, x2 = xmin / 1000.0 * width, xmax / 1000.0 * width
    y1, y2 = ymin / 1000.0 * height, ymax / 1000.0 * height
    # Size limits apply to what the model saw, before the safety margin.
    if x2 - x1 < LOGO_MIN_SIDE_PX or y2 - y1 < LOGO_MIN_SIDE_PX:
        return None
    if (x2 - x1) * (y2 - y1) > LOGO_MAX_AREA_RATIO * width * height:
        return None
    return (
        max(0, int(math.floor(x1)) - LOGO_MARGIN_PX),
        max(0, int(math.floor(y1)) - LOGO_MARGIN_PX),
        min(width, int(math.ceil(x2)) + LOGO_MARGIN_PX),
        min(height, int(math.ceil(y2)) + LOGO_MARGIN_PX),
    )


def parse_scan(data: dict, sizes: dict[int, tuple[int, int]]) -> list[SliceScan]:
    """Map the model answer onto the requested slices; unknown slices are ignored.

    ``sizes`` maps page index to the ORIGINAL slice (width, height). A slice the
    model did not answer for is reported as neither credit nor logo-bearing.
    """
    answers: dict[int, dict] = {}
    for entry in (data or {}).get("slices") or []:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("slice"))
        except (TypeError, ValueError):
            continue
        if index in sizes and index not in answers:
            answers[index] = entry

    scans = []
    for index, (width, height) in sizes.items():
        entry = answers.get(index, {})
        confidence = _confidence(entry.get("credit_confidence"))
        is_credit = entry.get("is_credit") is True and confidence >= CREDIT_MIN_CONFIDENCE
        logos = tuple(
            box for raw in (entry.get("logos") or [])
            if (box := _logo_box(raw, width, height)) is not None
        )
        scans.append(SliceScan(
            page_index=index,
            is_credit=is_credit,
            credit_confidence=confidence,
            # A credit slice is skipped whole; its logos are irrelevant.
            logos=() if is_credit else logos,
            reason=str(entry.get("reason") or "")[:200],
        ))
    return scans


def _thumbnail(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, SCAN_MAX_SIDE / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def scan_slices(
    provider,
    model: str,
    api_key: str,
    slices: list[tuple[int, np.ndarray]],
) -> tuple[list[SliceScan], float | None]:
    """Scan one batch of (page_index, original image) with a single request."""
    if not slices:
        return [], 0.0
    sizes = {index: (image.shape[1], image.shape[0]) for index, image in slices}
    images = [(f"SLICE {index}", _thumbnail(image)) for index, image in slices]
    result = request_vision_json(
        provider, model, api_key, SCAN_PROMPT, images,
        schema=SCAN_SCHEMA, max_tokens=min(4096, 400 + 220 * len(slices)),
    )
    return parse_scan(result.data, sizes), result.estimated_cost_usd
