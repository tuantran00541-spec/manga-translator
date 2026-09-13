from __future__ import annotations

import cv2
import numpy as np

from app.mask_store import decode_mask_value
from app.ocr.quality import classify_ocr_quality
from app.ocr.service import OCRService, ocr_target_mode_for_box
from app.parameters import (
    OCR_BOX_CROP_PADDING,
    OCR_MASK_CROP_PADDING,
    OCR_MASK_EDGE_CONTEXT_TRIGGER,
    OCR_MASK_PAGE_CONTEXT_PADDING,
)

_CONTEXT_RETRY_PADDING = max(
    32,
    int(OCR_BOX_CROP_PADDING) + int(OCR_MASK_PAGE_CONTEXT_PADDING),
)
_CRITICAL_COMPLETENESS_REASONS = {"crop-edge-text", "incomplete-coverage"}
_NON_STORY_CLASSES = {
    "sfx",
    "sound_effect",
    "credit",
    "credits",
    "artwork",
    "text_recovery",
    "focus_deferred",
}
_DIALOGUE_PUNCTUATION = frozenset(
    ".…!?！？,，。;；:：~〜-—–_()（）[]【】{}「」『』\"'“”‘’"
)


def _clamped_detector_bounds(
    image_shape: tuple[int, ...],
    box: dict,
    *,
    padding: int,
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    try:
        bx1, by1, bx2, by2 = map(
            int, (box["x1"], box["y1"], box["x2"], box["y2"])
        )
    except (KeyError, TypeError, ValueError):
        return 0, 0, 0, 0
    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
    bx2, by2 = max(bx1, min(w, bx2)), max(by1, min(h, by2))
    if bx2 <= bx1 or by2 <= by1:
        return 0, 0, 0, 0
    pad = max(0, int(padding))
    return (
        max(0, bx1 - pad),
        max(0, by1 - pad),
        min(w, bx2 + pad),
        min(h, by2 + pad),
    )


def recognition_safe_crop_bounds(
    image_shape: tuple[int, ...],
    box: dict,
) -> tuple[int, int, int, int]:
    """Build OCR bounds from detector context; masks may expand but never shrink it.

    Inpaint masks answer "what may be erased", not "what must be recognized".
    OCR therefore always keeps detector bbox context first. Segmenter support is
    allowed to add bounded context when it reaches detector edges, but can never
    reduce the recognition crop below the detector-derived bounds.
    """

    base = _clamped_detector_bounds(
        image_shape,
        box,
        padding=OCR_BOX_CROP_PADDING,
    )
    if base == (0, 0, 0, 0):
        return base

    h, w = image_shape[:2]
    try:
        bx1, by1, bx2, by2 = map(
            int, (box["x1"], box["y1"], box["x2"], box["y2"])
        )
    except (KeyError, TypeError, ValueError):
        return base
    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
    bx2, by2 = max(bx1, min(w, bx2)), max(by1, min(h, by2))

    raw_mask = box.get("mask")
    mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
    expected_shape = (by2 - by1, bx2 - bx1)
    if mask is None or mask.shape != expected_shape:
        return base

    ys, xs = np.nonzero(mask > 127)
    if not xs.size or not ys.size:
        return base

    pad = int(OCR_MASK_CROP_PADDING)
    trigger = int(OCR_MASK_EDGE_CONTEXT_TRIGGER)
    context = int(OCR_MASK_PAGE_CONTEXT_PADDING)
    left = context if int(xs.min()) <= trigger else 0
    top = context if int(ys.min()) <= trigger else 0
    right = context if int(xs.max()) >= mask.shape[1] - 1 - trigger else 0
    bottom = context if int(ys.max()) >= mask.shape[0] - 1 - trigger else 0

    mask_bounds = (
        max(0, bx1 + int(xs.min()) - pad - left),
        max(0, by1 + int(ys.min()) - pad - top),
        min(w, bx1 + int(xs.max()) + 1 + pad + right),
        min(h, by1 + int(ys.max()) + 1 + pad + bottom),
    )
    if mask_bounds[2] <= mask_bounds[0] or mask_bounds[3] <= mask_bounds[1]:
        return base

    return (
        min(base[0], mask_bounds[0]),
        min(base[1], mask_bounds[1]),
        max(base[2], mask_bounds[2]),
        max(base[3], mask_bounds[3]),
    )


def expanded_context_crop_bounds(
    image_shape: tuple[int, ...],
    box: dict,
) -> tuple[int, int, int, int]:
    """Return one bounded recognition retry crop around the detector geometry."""

    return _clamped_detector_bounds(
        image_shape,
        box,
        padding=_CONTEXT_RETRY_PADDING,
    )


def _is_story_candidate(box: dict) -> bool:
    class_name = str(box.get("class_name") or "").strip().lower()
    if class_name in _NON_STORY_CLASSES:
        return False
    semantic = str(box.get("semantic_type") or "").strip().lower()
    source_role = str(box.get("source_role") or "").strip().lower()
    return (
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


def _looks_like_dialogue_punctuation(text: str) -> bool:
    value = "".join(ch for ch in str(text or "").strip() if not ch.isspace())
    if not value or len(value) > 16:
        return False
    return all(ch in _DIALOGUE_PUNCTUATION for ch in value) and any(
        ch in value for ch in ".…!?！？—-"
    )


def _metadata_score(metadata: dict) -> tuple[int, int, float, float]:
    quality_rank = {"reject": 0, "unknown": 0, "review": 1, "good": 2}
    status = str(metadata.get("quality") or "unknown").strip().lower()
    reason = str(metadata.get("quality_reason") or "").strip().lower()
    issue_free = 0 if reason in _CRITICAL_COMPLETENESS_REASONS else 1
    coverage = metadata.get("coverage")
    confidence = metadata.get("confidence")
    try:
        coverage_score = -1.0 if coverage is None else float(coverage)
    except (TypeError, ValueError):
        coverage_score = -1.0
    try:
        confidence_score = -1.0 if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence_score = -1.0
    return (
        quality_rank.get(status, 0),
        issue_free,
        coverage_score,
        confidence_score,
    )


def _result_center_hits_target(
    result: object,
    crop_bounds: tuple[int, int, int, int],
    box: dict,
) -> bool:
    """Reject expanded-crop text whose OCR span is centred outside the target."""

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
    return (
        bx1 - guard <= center_x <= bx2 + guard
        and by1 - guard <= center_y <= by2 + guard
    )


class ConsistencyOCRService(OCRService):
    """OCR service with recognition-safe crops and one completeness recovery pass."""

    def _read_detailed_candidate(
        self,
        image: np.ndarray,
        crop_bounds: tuple[int, int, int, int],
        box_snapshot: dict,
        lang: str,
        *,
        target_mode: str,
    ) -> tuple[str, dict, object | None]:
        x1, y1, x2, y2 = crop_bounds
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            return (
                "",
                {
                    "confidence": None,
                    "model": "none",
                    "orientation": "unknown",
                    "region_count": 0,
                    "quality": "reject",
                    "quality_reason": "empty-crop",
                    "coverage": None,
                    "target_mode": target_mode,
                    "retry_applied": False,
                },
                None,
            )

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        detailed_reader = getattr(self.ocr, "read_detailed", None)
        if not callable(detailed_reader):
            text = str(self.ocr.read(rgb, lang) or "").strip()
            quality = classify_ocr_quality(text, lang, confidence=None)
            metadata = {
                "confidence": None,
                "model": "legacy-reader",
                "orientation": "unknown",
                "region_count": 1 if text else 0,
                "quality": quality.status,
                "quality_reason": quality.reason,
                "coverage": None,
                "target_mode": target_mode,
                "retry_applied": False,
            }
            self._preserve_punctuation_only_story_text(
                text, metadata, box_snapshot
            )
            return text, metadata, None

        try:
            result = detailed_reader(rgb, lang, target_mode=target_mode)
        except TypeError:
            result = detailed_reader(rgb, lang)

        text = str(getattr(result, "text", "") or "").strip()
        coverage = self._mask_text_coverage(box_snapshot, crop_bounds, result)
        checked = classify_ocr_quality(
            text,
            lang,
            confidence=getattr(result, "confidence", None),
            coverage=coverage,
        )
        quality, quality_reason = self._conservative_quality(
            str(getattr(result, "quality", "unknown") or "unknown"),
            getattr(result, "quality_reason", None),
            checked.status,
            checked.reason,
        )
        metadata = {
            "confidence": getattr(result, "confidence", None),
            "model": str(getattr(result, "model", "") or ""),
            "orientation": str(
                getattr(result, "orientation", "unknown") or "unknown"
            ),
            "region_count": int(getattr(result, "region_count", 0) or 0),
            "quality": quality,
            "quality_reason": quality_reason,
            "coverage": (
                coverage
                if coverage is not None
                else getattr(result, "coverage", None)
            ),
            "target_mode": str(
                getattr(result, "target_mode", target_mode) or target_mode
            ),
            "retry_applied": bool(getattr(result, "retry_applied", False)),
        }
        self._preserve_punctuation_only_story_text(text, metadata, box_snapshot)
        return text, metadata, result

    @staticmethod
    def _preserve_punctuation_only_story_text(
        text: str,
        metadata: dict,
        box_snapshot: dict,
    ) -> None:
        if (
            metadata.get("quality") == "reject"
            and metadata.get("quality_reason") == "no-content"
            and _is_story_candidate(box_snapshot)
            and _looks_like_dialogue_punctuation(text)
        ):
            metadata["quality"] = "review"
            metadata["quality_reason"] = "punctuation-only"

    @staticmethod
    def _needs_context_retry(
        text: str,
        metadata: dict,
        box_snapshot: dict,
    ) -> bool:
        if not _is_story_candidate(box_snapshot):
            return False
        reason = str(metadata.get("quality_reason") or "").strip().lower()
        if reason in _CRITICAL_COMPLETENESS_REASONS:
            return True
        return not str(text or "").strip()

    @staticmethod
    def _retry_is_better(
        base_text: str,
        base_metadata: dict,
        retry_text: str,
        retry_metadata: dict,
        retry_result: object | None,
        retry_bounds: tuple[int, int, int, int],
        box_snapshot: dict,
    ) -> bool:
        if not str(retry_text or "").strip():
            return False
        if retry_result is not None and not _result_center_hits_target(
            retry_result, retry_bounds, box_snapshot
        ):
            return False
        if not str(base_text or "").strip():
            return str(retry_metadata.get("quality") or "") in {"review", "good"}
        return _metadata_score(retry_metadata) > _metadata_score(base_metadata)

    def _read_box_text(self, original_path, box_snapshot: dict, lang: str) -> str:
        image = self._cached_source_image(original_path)
        base_bounds = recognition_safe_crop_bounds(image.shape, box_snapshot)
        target_mode = ocr_target_mode_for_box(box_snapshot)
        text, metadata, _result = self._read_detailed_candidate(
            image,
            base_bounds,
            box_snapshot,
            lang,
            target_mode=target_mode,
        )
        metadata["recognition_crop_policy"] = "detector-context-mask-expand-only-v1"
        metadata["context_retry_applied"] = False

        if self._needs_context_retry(text, metadata, box_snapshot):
            retry_bounds = expanded_context_crop_bounds(image.shape, box_snapshot)
            if retry_bounds != base_bounds:
                retry_text, retry_metadata, retry_result = self._read_detailed_candidate(
                    image,
                    retry_bounds,
                    box_snapshot,
                    lang,
                    target_mode=target_mode,
                )
                if self._retry_is_better(
                    text,
                    metadata,
                    retry_text,
                    retry_metadata,
                    retry_result,
                    retry_bounds,
                    box_snapshot,
                ):
                    text = retry_text
                    retry_metadata["retry_applied"] = True
                    retry_metadata["context_retry_applied"] = True
                    retry_metadata[
                        "recognition_crop_policy"
                    ] = "detector-context-expanded-retry-v1"
                    metadata = retry_metadata
                else:
                    metadata["context_retry_applied"] = True
                    metadata["retry_applied"] = True

        self._result_local.metadata = metadata
        return text
