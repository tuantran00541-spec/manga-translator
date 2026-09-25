"""Resolve font ownership and precedence for a render snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.render.font_catalog import FontNotFoundError, load_font_catalog, resolve_font_id

__all__ = ["font_path_for_id", "resolve_object_font"]


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
    ai_font_id: str | None = None,
    ai_font_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(font_id, selection_mode, metadata)`` with safe precedence.

    Precedence: an explicit user choice, then a validated AI choice, then the
    catalog default. ``auto`` means "let the AI pick"; without an AI choice it
    renders with ``default``. A legacy/default style is considered implicit
    until the object explicitly carries ``font_selection_mode=user``.
    """

    style_font = _style_font(obj)
    selection_mode = str(obj.get("font_selection_mode") or "").lower()

    if str(ai_font_id or "").strip().lower() == "auto":
        ai_font_id = None
        style_font = "auto"

    if selection_mode == "user":
        if _valid_font_id(style_font):
            return style_font, "user", {"reason": "explicit_user_selection"}
        return "default", "default", {"reason": "user_selection_rejected", "requested": style_font}

    if ai_font_id and (ai_font_mode in (None, "", "ai")):
        if _valid_font_id(ai_font_id):
            requested = str(ai_font_id)
            return requested, "ai", {"reason": "ai_selection", "font_id": requested}
        return "default", "default", {"reason": "ai_selection_rejected", "requested": str(ai_font_id)}

    if style_font.lower() == "auto" or selection_mode == "auto" or bool(obj.get("auto_generated")):
        return "default", "default", {"reason": "auto_default"}

    if _valid_font_id(style_font) and style_font.lower() != "default":
        # A pre-existing non-default ID without an ownership marker is kept for
        # backwards compatibility, but the metadata makes the implicit origin
        # visible to API clients.
        return style_font, "legacy", {"reason": "legacy_style"}
    return "default", "default", {"reason": "default_fallback"}


def font_path_for_id(font_id: str) -> Path:
    return load_font_catalog().resolve(font_id)
