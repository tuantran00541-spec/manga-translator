"""Checkpoint 3 (raw vs clean review) and checkpoint 5 (final review) prompts and parsing."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from app.ai_mode.vision_json import request_vision_json

MIN_CONFIDENCE = 0.6
MIN_SIDE_PX = 10
MAX_AREA_RATIO = 0.4  # a box over most of a slice is a misread
MARGIN_PX = 6
MAX_BOXES = 12
FINAL_ACTIONS = frozenset({"repaint", "retranslate", "restore"})

_BOX = '{"box_2d":[ymin,xmin,ymax,xmax],"confidence":0.0}'

CLEAN_REVIEW_PROMPT = (
    "You check an automatic manga/manhwa text cleanup. IMAGE 1 is the ORIGINAL slice, IMAGE 2 is the "
    "same slice after the text was erased (CLEAN). Compare them and report only clear problems in IMAGE 2:\n"
    "- missed: story text a reader must read (dialogue, narration, captions, system windows, titles) that is "
    "still fully readable in CLEAN. It will be erased and translated.\n"
    "- residue: fragments, ghost outlines, smears or blotches left where text was erased, and leftover "
    "scanlator watermarks or credits. They will be erased; nothing is translated.\n"
    "- restore: things CLEAN erased that must stay as drawn: artwork, ornaments, patterns, series logos, "
    "writing that belongs to the drawing. The original pixels will be put back.\n"
    "Boxes are [ymin, xmin, ymax, xmax] normalised to 0-1000 on the slice, tight around the problem. "
    "Leave a list empty when nothing applies; a clean slice returns three empty lists. Return JSON only: "
    f'{{"missed":[{_BOX}],"residue":[{_BOX}],"restore":[{_BOX}]}}'
)

FINAL_REVIEW_PROMPT = (
    "You do the final check of a translated manga/manhwa slice before it is exported. IMAGE 1 is the ORIGINAL "
    "slice, IMAGE 2 is the FINAL slice with the translation lettered in. OBJECTS lists every lettered region: "
    "its id, its box as [ymin, xmin, ymax, xmax] normalised to 0-1000, and its translation.\n"
    "Decide verdict \"ok\" or \"fix\". Ask for a fix only for clear defects a reader would notice:\n"
    "- repaint (box_2d): source-language text or erase marks still visible in FINAL outside the lettering.\n"
    "- retranslate (id): a translation that is wrong, misspelled, missing words, or does not fit the scene.\n"
    "- restore (id): lettering that should not be there (art, logo, sound effect drawn as art); the original "
    "pixels are put back.\n"
    "Do not ask for style changes. Return JSON only: "
    '{"verdict":"ok","fixes":[{"action":"repaint","box_2d":[0,0,0,0]},{"action":"retranslate","id":"<id>"},'
    '{"action":"restore","id":"<id>"}],"reason":"short"}'
)


@dataclass(frozen=True)
class CleanReview:
    page_index: int
    missed: tuple[tuple[int, int, int, int], ...] = ()
    residue: tuple[tuple[int, int, int, int], ...] = ()
    restore: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class FinalReview:
    page_index: int
    ok: bool
    repaint: tuple[tuple[int, int, int, int], ...] = ()
    retranslate: tuple[str, ...] = ()
    restore: tuple[str, ...] = ()
    reason: str = ""


def _box(raw, width: int, height: int, *, need_confidence: bool = True) -> tuple[int, int, int, int] | None:
    """Pixel box from a 0-1000 ``box_2d``, or None when unsure, tiny or implausibly large."""
    if not isinstance(raw, dict):
        return None
    if need_confidence:
        try:
            if float(raw.get("confidence", 1.0)) < MIN_CONFIDENCE:
                return None
        except (TypeError, ValueError):
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
    x1, x2, y1, y2 = xmin * width / 1000, xmax * width / 1000, ymin * height / 1000, ymax * height / 1000
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX or (x2 - x1) * (y2 - y1) > MAX_AREA_RATIO * width * height:
        return None
    return (max(0, math.floor(x1) - MARGIN_PX), max(0, math.floor(y1) - MARGIN_PX),
            min(width, math.ceil(x2) + MARGIN_PX), min(height, math.ceil(y2) + MARGIN_PX))


def _boxes(items, width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    found = [box for raw in (items or [])[:MAX_BOXES] if (box := _box(raw, width, height)) is not None]
    return tuple(found)


def parse_clean_review(data: dict, page_index: int, width: int, height: int) -> CleanReview:
    data = data if isinstance(data, dict) else {}
    return CleanReview(
        page_index,
        missed=_boxes(data.get("missed"), width, height),
        residue=_boxes(data.get("residue"), width, height),
        restore=_boxes(data.get("restore"), width, height),
    )


def parse_final_review(data: dict, page_index: int, width: int, height: int, ids: set[str]) -> FinalReview:
    data = data if isinstance(data, dict) else {}
    repaint, retranslate, restore = [], [], []
    for fix in (data.get("fixes") or [])[:MAX_BOXES]:
        if not isinstance(fix, dict) or fix.get("action") not in FINAL_ACTIONS:
            continue
        if fix["action"] == "repaint":
            if (box := _box(fix, width, height, need_confidence=False)) is not None:
                repaint.append(box)
        elif str(fix.get("id")) in ids:
            (retranslate if fix["action"] == "retranslate" else restore).append(str(fix["id"]))
    # A retranslated object that is also restored is just restored.
    retranslate = [obj for obj in dict.fromkeys(retranslate) if obj not in restore]
    ok = str(data.get("verdict") or "").lower() != "fix" or not (repaint or retranslate or restore)
    return FinalReview(page_index, ok, tuple(repaint), tuple(retranslate), tuple(dict.fromkeys(restore)),
                       str(data.get("reason") or "")[:200])


def review_clean(provider, model: str, api_key: str, page_index: int,
                 original: np.ndarray, clean: np.ndarray) -> tuple[CleanReview, float | None]:
    """Checkpoint 3: one request with the raw and the cleaned slice."""
    result = request_vision_json(provider, model, api_key, CLEAN_REVIEW_PROMPT,
                                 [("IMAGE 1: ORIGINAL", original), ("IMAGE 2: CLEAN", clean)], max_tokens=1200)
    height, width = original.shape[:2]
    return parse_clean_review(result.data, page_index, width, height), result.estimated_cost_usd


def review_final(provider, model: str, api_key: str, page_index: int, original: np.ndarray,
                 final: np.ndarray, objects: list[dict]) -> tuple[FinalReview, float | None]:
    """Checkpoint 5: one request with the raw slice, the lettered slice and its objects."""
    height, width = original.shape[:2]
    listed = []
    for obj in objects:
        region = obj.get("region") or {}
        try:
            box = [round(1000 * int(region["y1"]) / height), round(1000 * int(region["x1"]) / width),
                   round(1000 * int(region["y2"]) / height), round(1000 * int(region["x2"]) / width)]
        except (KeyError, TypeError, ValueError):
            continue
        listed.append({"id": str(obj["id"]), "box_2d": box, "translation": str(obj.get("translation") or "")})
    prompt = FINAL_REVIEW_PROMPT + "\nOBJECTS " + json.dumps(listed, ensure_ascii=False, separators=(",", ":"))
    result = request_vision_json(provider, model, api_key, prompt,
                                 [("IMAGE 1: ORIGINAL", original), ("IMAGE 2: FINAL", final)], max_tokens=1200)
    review = parse_final_review(result.data, page_index, width, height, {item["id"] for item in listed})
    return review, result.estimated_cost_usd
