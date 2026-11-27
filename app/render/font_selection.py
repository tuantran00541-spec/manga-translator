from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.mask_store import decode_mask_value
from app.render.font_catalog import FontNotFoundError, resolve_font_id
from app.render.font_guide import font_for

__all__ = ["resolve_object_font"]


def _valid_font_id(value: str | None) -> bool:
    try:
        resolve_font_id(value)
        return True
    except (FontNotFoundError, OSError, ValueError):
        return False


def _style_font(obj: dict) -> str:
    style = obj.get("style") or {}
    return str(style.get("font") or "default").strip() or "default"


def _title_width_ratio(obj: dict) -> float | None:
    if not (obj.get("verified_balloon") and obj.get("container") == "box" and obj.get("typography_role") == "title"):
        return None
    cap = obj.get("source_cap_px")
    if not isinstance(cap, (float, int)) or cap < 8:
        return None
    letters = decode_mask_value(obj.get("letters_mask"))
    if letters is None or letters.ndim != 2 or letters.size == 0:
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats((letters > 0).astype(np.uint8))
    _, _, widths, heights, areas = stats[1:].T
    glyphs = ((heights >= .65 * cap) & (heights <= 1.3 * cap)
              & (widths <= 1.35 * heights) & (areas >= .24 * widths * heights))
    if np.count_nonzero(glyphs) < 6:
        return None
    return float(np.median(widths[glyphs] / heights[glyphs]))


def resolve_object_font(
    obj: dict,
    *,
    ai_font_id: str | None = None,
    ai_font_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    style_font = _style_font(obj)
    selection_mode = str(obj.get("font_selection_mode") or "").lower()

    if selection_mode == "user":
        if _valid_font_id(style_font):
            return style_font, "user", {"reason": "explicit_user_selection"}
        return "default", "default", {"reason": "user_selection_rejected", "requested": style_font}

    if str(ai_font_id or "").strip().lower() == "auto":
        ai_font_id = None
        style_font = "auto"

    if (obj.get("auto_generated") and obj.get("container") == "spiky"
            and obj.get("typography_role") in ("dialogue", "whisper", "thought")
            and ai_font_id == "emphasis.bangers"):
        ai_font_id = None

    if ai_font_id and (ai_font_mode in (None, "", "ai")):
        if _valid_font_id(ai_font_id):
            requested = str(ai_font_id)
            return requested, "ai", {"reason": "ai_selection", "font_id": requested}
        return "default", "default", {"reason": "ai_selection_rejected", "requested": str(ai_font_id)}

    if style_font.lower() == "auto" or selection_mode == "auto" or bool(obj.get("auto_generated")):
        width_ratio = _title_width_ratio(obj)
        full_width = "dialogue.be-vietnam-pro"
        if width_ratio is not None and width_ratio >= .73 and _valid_font_id(full_width):
            return full_width, "auto", {"reason": "source_proportions", "glyph_width_ratio": round(width_ratio, 2)}
        guided = font_for(obj.get("typography_role"), obj.get("container"))
        if _valid_font_id(guided):
            return guided, "auto", {"reason": "auto_guide",
                                    "role": obj.get("typography_role"),
                                    "container": obj.get("container")}
        return "default", "default", {"reason": "auto_default"}

    if _valid_font_id(style_font) and style_font.lower() != "default":
        return style_font, "legacy", {"reason": "legacy_style"}
    return "default", "default", {"reason": "default_fallback"}
