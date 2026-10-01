"""Ask a vision model which slices are credit pages or textless and where series logos are.

Runs on the ORIGINAL slices before cleanup, so credit and textless slices can be
skipped (no cleanup time spent on them; they are exported as the original) and
logo artwork can be stored as preserve regions before inpainting touches it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai_mode.vision_json import request_vision_json
from app.box_format import BOX_KEYS, BOX_RULE, scaled_box

# Only confident answers change the chapter; everything else stays as is.
CREDIT_MIN_CONFIDENCE = 0.75
# Skipping a slice that does hold text leaves it untranslated, so this is strict.
TEXTLESS_MIN_CONFIDENCE = 0.85
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
                    "no_text": {"type": "boolean"},
                    "no_text_confidence": {"type": "number"},
                    "logos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                **{key: {"type": "number"} for key in BOX_KEYS},
                                "confidence": {"type": "number"},
                            },
                            "required": [*BOX_KEYS, "confidence"],
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
    "IMAGES: each image is one slice of a manga, manhwa or webtoon chapter, labelled SLICE <number>.\n"
    "QUESTION: for each slice, is it a credit slice, does it have no lettering at all, and where are logos drawn "
    "as artwork?\n"
    "- is_credit: true only when the whole slice was added by the uploader or scanlation group (credits, staff "
    "list, recruitment, Discord or Patreon ads, 'read at <site>' banners, end notices). A story slice with a small "
    "watermark is not one, nor is a blank slice.\n"
    "- no_text: true only when the slice has no lettering at all, not even a sound effect.\n"
    "- logos: boxes round the series title logo or a publisher or studio logo drawn as artwork; they are kept. "
    "Never box bubbles, captions, sound effects, plain text, watermarks, group logos or site names.\n"
    + BOX_RULE.replace("the image", "that slice's image")
    + "Confidences are 0-1.\nANSWER with JSON only: "
    '{"slices":[{"slice":<number>,"is_credit":false,"credit_confidence":0.0,"no_text":false,"no_text_confidence":0.0,'
    '"logos":[{"x1":0,"y1":0,"x2":0,"y2":0,"confidence":0.0}],"reason":"short"}]}'
)


@dataclass(frozen=True)
class SliceScan:
    page_index: int
    is_credit: bool
    credit_confidence: float
    logos: tuple[tuple[int, int, int, int], ...]
    reason: str = ""
    no_text: bool = False


def _confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def _logo_box(raw, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, dict) or _confidence(raw.get("confidence")) < LOGO_MIN_CONFIDENCE:
        return None
    scaled = scaled_box(raw, width, height)
    if scaled is None:
        return None
    x1, y1, x2, y2 = scaled
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
        no_text = (not is_credit and entry.get("no_text") is True
                   and _confidence(entry.get("no_text_confidence")) >= TEXTLESS_MIN_CONFIDENCE)
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
            no_text=no_text and not logos,
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
        schema=SCAN_SCHEMA, max_tokens=min(4096, 400 + 220 * len(slices)), stage="scan",
    )
    return parse_scan(result.data, sizes), result.estimated_cost_usd
