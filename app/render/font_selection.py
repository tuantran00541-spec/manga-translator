"""Resolve font ownership and precedence for a render snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.render.font_catalog import FontNotFoundError, load_font_catalog, resolve_font_id
from app.render.font_matcher import match_fonts

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


def _match_metadata(matches: list) -> dict[str, Any]:
    if not matches:
        return {}
    return {"matches": [item.as_dict() for item in matches]}


def _has_actionable_match(matches: list) -> bool:
    """Return whether the top match has enough evidence to drive a render."""

    if not matches or str((matches[0].evidence or {}).get("reason") or "") == "source_crop_unavailable":
        return False
    return str(matches[0].confidence or "").lower() in {"high", "medium"}


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

    # Always run the local matcher when a source crop is available. Its result
    # is used both to power automatic selection and to validate explicit
    # user/AI choices, so there is no separate suggestion action in the UI.
    matches = []
    if source_image is not None and region is not None:
        try:
            matches = match_fonts(source_image, region, source_text, top_k=3)
        except (OSError, TypeError, ValueError):
            # A malformed or unavailable crop must not make a render fail.
            matches = []
    match_data = _match_metadata(matches)
    actionable_match = _has_actionable_match(matches)
    candidate_ids = {item.font_id for item in matches}
    best = matches[0] if actionable_match else None

    if selection_mode == "user":
        if _valid_font_id(style_font):
            if style_font.lower() == "default" or not actionable_match or style_font in candidate_ids:
                return style_font, "user", {"reason": "explicit_user_selection", **match_data}
            return best.font_id, "auto", {
                "reason": "user_selection_not_suitable",
                "requested": style_font,
                **match_data,
            }
        if best:
            return best.font_id, "auto", {
                "reason": "user_selection_not_suitable",
                "requested": style_font,
                **match_data,
            }
        return "default", "default", {"reason": "user_selection_rejected", "requested": style_font, **match_data}

    if ai_font_id and (ai_font_mode in (None, "", "ai")):
        if _valid_font_id(ai_font_id):
            requested = str(ai_font_id)
            if not actionable_match or requested in candidate_ids:
                return requested, "ai", {"reason": "ai_selection", "font_id": requested, **match_data}
            return best.font_id, "auto", {
                "reason": "ai_selection_not_suitable",
                "requested": requested,
                **match_data,
            }
        if best:
            return best.font_id, "auto", {
                "reason": "ai_selection_not_suitable",
                "requested": str(ai_font_id),
                **match_data,
            }
        return "default", "default", {"reason": "ai_selection_rejected", "requested": str(ai_font_id), **match_data}

    should_auto_match = (
        style_font.lower() == "auto"
        or selection_mode == "auto"
        or bool(obj.get("auto_generated"))
    )
    if should_auto_match and best:
        return best.font_id, "auto", {"reason": "visual_match", **match_data}
    if should_auto_match and matches:
        return "default", "default", {"reason": "match_confidence_too_low", **match_data}

    if _valid_font_id(style_font) and style_font.lower() != "default":
        # A pre-existing non-default ID without an ownership marker is kept for
        # backwards compatibility, but the metadata makes the implicit origin
        # visible to API clients.
        return style_font, "legacy", {"reason": "legacy_style", **match_data}
    if matches:
        # Old manifests without a selection marker retain their default style,
        # while still exposing the automatic candidates to callers and logs.
        return "default", "default", {"reason": "legacy_default_match_recorded", **match_data}
    return "default", "default", {"reason": "default_fallback"}


def font_path_for_id(font_id: str) -> Path:
    return load_font_catalog().resolve(font_id)
