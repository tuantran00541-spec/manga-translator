from __future__ import annotations

from difflib import SequenceMatcher


from app.parameters import (
    OCR_BOX_CROP_PADDING,
    OCR_CENTERED_SINGLE_LINE_ASPECT,
)


_OCR_COMPLETENESS_REASONS = frozenset(
    {"crop-edge-text", "incomplete-coverage"}
)



_OCR_RETRY_NON_STORY_CLASSES = frozenset(
    {
        "sfx",
        "sound_effect",
        "credit",
        "credits",
        "artwork",
        "text_recovery",
        "focus_deferred",
    }
)



def is_story_retry_candidate(box: dict) -> bool:
    class_name = str(box.get("class_name") or "").strip().lower()
    if class_name in _OCR_RETRY_NON_STORY_CLASSES:
        return False
    semantic = str(box.get("semantic_type") or "").strip().lower()
    source_role = str(box.get("source_role") or "").strip().lower()
    return bool(
        semantic
        in {
            "speech_bubble",
            "dialogue",
            "thought",
            "narration",
            "free_text",
            "text",
            "review_region",
        }
        or source_role == "text_segmenter"
    )



def _result_center_hits_target(
    result: object,
    crop_bounds: tuple[int, int, int, int],
    box: dict,
) -> bool:
    text_bounds = getattr(result, "text_bounds", None)
    input_shape = getattr(result, "input_shape", None)
    if not text_bounds or not input_shape:
        return False
    try:
        prepared_h, prepared_w = float(input_shape[0]), float(input_shape[1])
        tx1, ty1, tx2, ty2 = (float(value) for value in text_bounds)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
        bx1, by1, bx2, by2 = (
            float(box["x1"]),
            float(box["y1"]),
            float(box["x2"]),
            float(box["y2"]),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    crop_w = max(1.0, float(crop_x2 - crop_x1))
    crop_h = max(1.0, float(crop_y2 - crop_y1))
    center_x = crop_x1 + ((tx1 + tx2) * 0.5) * crop_w / max(1.0, prepared_w)
    center_y = crop_y1 + ((ty1 + ty2) * 0.5) * crop_h / max(1.0, prepared_h)
    guard = max(4.0, float(OCR_BOX_CROP_PADDING) * 0.5)
    return bool(
        bx1 - guard <= center_x <= bx2 + guard
        and by1 - guard <= center_y <= by2 + guard
    )



def _context_retry_score(
    *,
    quality: str,
    reason: str | None,
    coverage: float | None,
    confidence: float | None,
) -> tuple[int, int, float, float]:
    quality_rank = {"reject": 0, "unknown": 0, "review": 1, "good": 2}
    try:
        coverage_score = -1.0 if coverage is None else float(coverage)
    except (TypeError, ValueError):
        coverage_score = -1.0
    try:
        confidence_score = -1.0 if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence_score = -1.0
    return (
        quality_rank.get(str(quality or "unknown"), 0),
        0 if str(reason or "") in _OCR_COMPLETENESS_REASONS else 1,
        coverage_score,
        confidence_score,
    )



def prefer_context_retry(
    *,
    base_text: str,
    base_quality: str,
    base_reason: str | None,
    base_coverage: float | None,
    base_confidence: float | None,
    expanded_text: str,
    expanded_quality: str,
    expanded_reason: str | None,
    expanded_coverage: float | None,
    expanded_confidence: float | None,
    expanded_result: object,
    expanded_bounds: tuple[int, int, int, int],
    box: dict,
) -> bool:
    if not str(expanded_text or "").strip():
        return False
    if not _result_center_hits_target(expanded_result, expanded_bounds, box):
        return False
    if not str(base_text or "").strip():
        return str(expanded_quality or "") in {"review", "good"}
    return _context_retry_score(
        quality=expanded_quality,
        reason=expanded_reason,
        coverage=expanded_coverage,
        confidence=expanded_confidence,
    ) > _context_retry_score(
        quality=base_quality,
        reason=base_reason,
        coverage=base_coverage,
        confidence=base_confidence,
    )



def ocr_target_mode_for_box(box: dict) -> str:
    explicit = str(box.get("ocr_target_mode") or "").strip().lower()
    if explicit in {"all", "centered"}:
        return explicit

    semantic_type = str(box.get("semantic_type") or "").strip().lower()
    source_role = str(box.get("source_role") or "").strip().lower()
    if semantic_type in {"speech_bubble", "narration", "free_text", "review_region"}:
        return "all"
    if source_role in {"text_segmenter", "recovery", "scheduler"}:
        return "all"
    try:
        line_count = int(box.get("line_count") or 0)
    except (TypeError, ValueError):
        line_count = 0
    if line_count > 1 or bool(box.get("grouped")):
        return "all"

    try:
        width = max(1.0, float(box["x2"]) - float(box["x1"]))
        height = max(1.0, float(box["y2"]) - float(box["y1"]))
    except (KeyError, TypeError, ValueError):
        return "all"
    if semantic_type == "text" and width / height >= OCR_CENTERED_SINGLE_LINE_ASPECT:
        return "centered"
    return "all"



def ocr_target_skip_reason(box: dict) -> str | None:
    source_model = str(box.get("source_model") or "").strip().lower()
    source_role = str(box.get("source_role") or "").strip().lower()
    class_name = str(box.get("class_name") or "").strip().lower()
    deferred_reason = str(box.get("deferred_reason") or "").strip()
    if source_role == "scheduler":
        return "deferred-review-region"
    if deferred_reason and source_role != "text_segmenter":
        return "deferred-review-region"
    if source_model == "opencv_mser" and source_role != "text_segmenter":
        return "unverified-recovery-proposal"
    if class_name in {
        "text_recovery", "focus_deferred", "sfx", "sound_effect",
        "credit", "credits", "artwork",
    }:
        return "non-dialogue-target"
    if box.get("needs_review") and not box.get("safe_to_inpaint") and source_role != "text_segmenter":
        return "unverified-review-proposal"
    return None



def _normalized_ocr_lines(text: str) -> tuple[str, ...]:
    lines = [
        " ".join(part.upper().split())
        for part in str(text or "").splitlines()
        if part.strip()
    ]
    return tuple(sorted(lines))



def _compact_ocr_text(text: str) -> str:
    return "".join(ch for ch in str(text or "").upper() if ch.isalnum())



def prefer_edge_recrop(
    *,
    base_text: str,
    base_quality: str,
    base_reason: str | None,
    base_confidence: float | None,
    base_region_count: int,
    expanded_text: str,
    expanded_quality: str,
    expanded_reason: str | None,
    expanded_confidence: float | None,
    expanded_region_count: int,
) -> bool:
    if str(base_reason or "") != "crop-edge-text":
        return False
    if not str(expanded_text or "").strip():
        return False

    base_regions = max(0, int(base_region_count or 0))
    expanded_regions = max(0, int(expanded_region_count or 0))
    if base_regions and expanded_regions > base_regions:
        return False

    rank = {"reject": 0, "unknown": 0, "review": 1, "good": 2}
    base_rank = rank.get(str(base_quality or "unknown"), 0)
    expanded_rank = rank.get(str(expanded_quality or "unknown"), 0)
    base_compact = _compact_ocr_text(base_text)
    expanded_compact = _compact_ocr_text(expanded_text)
    similarity = SequenceMatcher(None, base_compact, expanded_compact).ratio()

    if expanded_rank > base_rank:
        return similarity >= 0.45

    if (
        expanded_rank == base_rank
        and str(expanded_reason or "") == "crop-edge-text"
        and _normalized_ocr_lines(base_text) == _normalized_ocr_lines(expanded_text)
        and str(base_text or "").strip() != str(expanded_text or "").strip()
    ):
        try:
            base_conf = float(base_confidence) if base_confidence is not None else 0.0
            expanded_conf = (
                float(expanded_confidence) if expanded_confidence is not None else 0.0
            )
        except (TypeError, ValueError):
            return False
        return expanded_conf + 0.05 >= base_conf

    return False
