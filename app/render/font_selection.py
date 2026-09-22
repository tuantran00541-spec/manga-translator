"""Resolve font ownership and precedence for a render snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.render.font_catalog import FontNotFoundError, load_font_catalog, resolve_font_id
from app.render.font_matcher import match_fonts


def _valid_font_id(value: str | None) -> bool:
    try:
        resolve_font_id(value)
        return True
    except (FontNotFoundError, OSError, ValueError):
        return False


def _style_font(obj: dict) -> str:
    style = obj.get("style") or {}
    return str(style.get("font") or "default").strip() or "default"


def resolve_object_font(
    obj: dict,
    *,
    source_image: Image.Image | None,
    region: tuple[int, int, int, int] | None,
    source_text: str | None,
    ai_font_id: str | None = None,
    ai_font_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(font_id, selection_mode, metadata)`` with safe precedence.

    A legacy/default style is considered implicit until the object explicitly
    carries ``font_selection_mode=user``. This lets AI and automatic matching
    operate on old manifests without taking away a user's deliberate choice.
    """

    style_font = _style_font(obj)
    selection_mode = str(obj.get("font_selection_mode") or "").lower()

    if str(ai_font_id or "").strip().lower() == "auto":
        # An AI may explicitly delegate the choice to the visual matcher.
        ai_font_id = None
        style_font = "auto"

    if selection_mode == "user":
        if _valid_font_id(style_font):
            return style_font, "user", {"reason": "explicit_user_selection"}
        return "default", "default", {"reason": "user_selection_rejected", "requested": style_font}

    if ai_font_id and (ai_font_mode in (None, "", "ai")):
        if _valid_font_id(ai_font_id):
            return str(ai_font_id), "ai", {"reason": "ai_selection", "font_id": str(ai_font_id)}
        return "default", "default", {"reason": "ai_selection_rejected", "requested": str(ai_font_id)}

    should_auto_match = style_font.lower() == "auto" or style_font.lower() == "default"
    if should_auto_match and source_image is not None and region is not None:
        matches = match_fonts(source_image, region, source_text, top_k=3)
        if matches:
            best = matches[0]
            return best.font_id, "auto", {"reason": "visual_match", "matches": [item.as_dict() for item in matches]}

    if _valid_font_id(style_font) and style_font.lower() != "default":
        # A pre-existing non-default ID without an ownership marker is kept for
        # backwards compatibility, but the metadata makes the implicit origin
        # visible to API clients.
        return style_font, "legacy", {"reason": "legacy_style"}
    return "default", "default", {"reason": "default_fallback"}


def font_path_for_id(font_id: str) -> Path:
    return load_font_catalog().resolve(font_id)
