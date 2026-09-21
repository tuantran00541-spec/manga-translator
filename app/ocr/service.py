from __future__ import annotations

from collections import OrderedDict
import copy
from difflib import SequenceMatcher
import threading
from pathlib import Path

import cv2
import numpy as np

from app.logging_config import logger
from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
)
from app.mask_store import decode_mask_value
from app.ocr.identity import (
    engine_identity,
    file_revision,
    machine_cache_valid,
    ocr_crop_signature,
    stamp_machine_cache,
)
from app.ocr.quality import classify_ocr_quality
from app.parameters import (
    OCR_BOX_CROP_PADDING,
    OCR_CENTERED_SINGLE_LINE_ASPECT,
    OCR_IMAGE_CACHE_MB,
    OCR_MASK_EDGE_CONTEXT_TRIGGER,
    OCR_MASK_CROP_PADDING,
    OCR_MASK_PAGE_CONTEXT_PADDING,
)
from app.image_io import read_image
from app.region_policy import geometry_center_in_regions, page_preserve_regions, text_object_in_preserve_region
from app.security import validate_chapter_id
from app.text_objects import (
    invalidate_stale_machine_translation,
    sync_existing_auto_text_object,
)

OCR_CANCELLED_MESSAGE = "OCR job was cancelled"


class OCRResultStale(RuntimeError):
    pass


class OCRCancelled(RuntimeError):
    pass


def _cache_budget_bytes() -> int:
    return int(OCR_IMAGE_CACHE_MB) * 1024 * 1024


def _region_overlaps_box(region: dict, box: dict) -> bool:
    if not box:
        return False
    bx = (box.get("x1", 0) + box.get("x2", 0)) / 2
    by = (box.get("y1", 0) + box.get("y2", 0)) / 2
    rx1, ry1 = region.get("x1", 0), region.get("y1", 0)
    rx2, ry2 = region.get("x2", 0), region.get("y2", 0)
    if rx1 <= bx <= rx2 and ry1 <= by <= ry2:
        return True
    ix1 = max(rx1, box.get("x1", 0))
    iy1 = max(ry1, box.get("y1", 0))
    ix2 = min(rx2, box.get("x2", 0))
    iy2 = min(ry2, box.get("y2", 0))
    return ix2 > ix1 and iy2 > iy1


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


def _visual_cache_complete(box: dict) -> bool:
    if not str(box.get("ocr_text") or "").strip():
        return True
    return bool(
        box.get("ocr_text_color")
        and box.get("ocr_font_size")
        and isinstance(box.get("ocr_text_region"), dict)
    )


def _box_region(box: dict) -> dict | None:
    try:
        x1, y1, x2, y2 = (
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _ocr_text_region(
    image_shape: tuple[int, ...],
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    box: dict,
) -> dict | None:
    h, w = image_shape[:2]
    text_bounds = getattr(result, "text_bounds", None) if result is not None else None
    input_shape = getattr(result, "input_shape", None) if result is not None else None
    if text_bounds and input_shape:
        try:
            prepared_h = max(1.0, float(input_shape[0]))
            prepared_w = max(1.0, float(input_shape[1]))
            tx1, ty1, tx2, ty2 = (float(value) for value in text_bounds)
            cx1, cy1, cx2, cy2 = crop_bounds
            crop_w = max(1.0, float(cx2 - cx1))
            crop_h = max(1.0, float(cy2 - cy1))
            x1 = int(round(cx1 + tx1 * crop_w / prepared_w))
            y1 = int(round(cy1 + ty1 * crop_h / prepared_h))
            x2 = int(round(cx1 + tx2 * crop_w / prepared_w))
            y2 = int(round(cy1 + ty2 * crop_h / prepared_h))
            x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
            x2, y2 = max(x1, min(w, x2)), max(y1, min(h, y2))
            if x1 < x2 and y1 < y2:
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        except (TypeError, ValueError, IndexError):
            pass
    region = _box_region(box)
    if region is None:
        return None
    region["x1"] = max(0, min(w, region["x1"]))
    region["y1"] = max(0, min(h, region["y1"]))
    region["x2"] = max(region["x1"], min(w, region["x2"]))
    region["y2"] = max(region["y1"], min(h, region["y2"]))
    return region if region["x1"] < region["x2"] and region["y1"] < region["y2"] else None


def _rgb_hex_from_bgr(values: np.ndarray) -> str | None:
    if values.size == 0:
        return None
    median = np.median(values.reshape(-1, 3), axis=0)
    b, g, r = (int(np.clip(round(float(value)), 0, 255)) for value in median)
    return f"#{r:02x}{g:02x}{b:02x}"


def _sample_source_text_color(
    image: np.ndarray,
    box: dict,
    text_region: dict | None,
) -> str | None:
    if image is None or image.size == 0:
        return None

    # Verified text-segmenter masks are the strongest colour authority because
    # they isolate glyph pixels instead of bubble/background pixels.
    if str(box.get("source_role") or "").strip().lower() == "text_segmenter":
        region = _box_region(box)
        raw_mask = box.get("mask")
        mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
        if region is not None and mask is not None:
            expected = (region["y2"] - region["y1"], region["x2"] - region["x1"])
            if mask.shape == expected:
                patch = image[
                    region["y1"]:region["y2"],
                    region["x1"]:region["x2"],
                ]
                if patch.shape[:2] == mask.shape:
                    pixels = patch[mask > 127]
                    if pixels.ndim == 2 and pixels.shape[0] >= 8:
                        sampled = _rgb_hex_from_bgr(pixels)
                        if sampled:
                            return sampled

    if not isinstance(text_region, dict):
        return None
    x1, y1, x2, y2 = (
        int(text_region["x1"]),
        int(text_region["y1"]),
        int(text_region["x2"]),
        int(text_region["y2"]),
    )
    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 2:
        return None

    # Fallback for OCR engines without a glyph mask: estimate the surrounding
    # background from the region edge, then keep pixels most different from it.
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = max(1, min(roi.shape[:2]) // 8)
    border = np.concatenate(
        [
            lab[:edge].reshape(-1, 3),
            lab[-edge:].reshape(-1, 3),
            lab[:, :edge].reshape(-1, 3),
            lab[:, -edge:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    percentile = float(np.percentile(distance, 70.0))
    threshold = max(12.0, percentile)
    foreground = roi[distance >= threshold]
    if foreground.ndim != 2 or foreground.shape[0] < 8:
        flat = distance.reshape(-1)
        count = min(flat.size, max(8, flat.size // 4))
        if count <= 0:
            return None
        indices = np.argpartition(flat, -count)[-count:]
        foreground = roi.reshape(-1, 3)[indices]
    return _rgb_hex_from_bgr(foreground)


def _source_font_size(
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    text_region: dict | None,
    region_count: int,
) -> int | None:
    if result is not None:
        hint = getattr(result, "font_size_hint", None)
        input_shape = getattr(result, "input_shape", None)
        if hint is not None and input_shape:
            try:
                cx1, cy1, cx2, cy2 = crop_bounds
                prepared_h = max(1.0, float(input_shape[0]))
                prepared_w = max(1.0, float(input_shape[1]))
                orientation = str(getattr(result, "orientation", "horizontal") or "horizontal").lower()
                scale = (
                    max(1.0, float(cx2 - cx1)) / prepared_w
                    if orientation == "vertical"
                    else max(1.0, float(cy2 - cy1)) / prepared_h
                )
                size = int(round(float(hint) * scale))
                if size > 0:
                    return max(4, min(512, size))
            except (TypeError, ValueError, IndexError):
                pass

    if isinstance(text_region, dict):
        height = max(1, int(text_region["y2"]) - int(text_region["y1"]))
        lines = max(1, int(region_count or 1))
        return max(4, min(512, int(round(height / lines))))
    return None


def _visual_text_metadata(
    image: np.ndarray,
    box: dict,
    crop_bounds: tuple[int, int, int, int],
    result: object | None,
    *,
    text: str,
    region_count: int,
) -> dict:
    if not str(text or "").strip():
        return {}
    text_region = _ocr_text_region(image.shape, crop_bounds, result, box)
    color = _sample_source_text_color(image, box, text_region)
    font_size = _source_font_size(crop_bounds, result, text_region, region_count)
    metadata: dict = {}
    if text_region is not None:
        metadata["text_region"] = text_region
    if color:
        metadata["text_color"] = color
    if font_size is not None:
        metadata["font_size"] = font_size
    return metadata


def _ocr_crop_bounds(
    image_shape: tuple[int, ...], box: dict
) -> tuple[int, int, int, int]:
    """Keep detector context as OCR minimum; masks may expand but never shrink it."""
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


_OCR_CONTEXT_RETRY_PADDING = max(
    32,
    int(OCR_BOX_CROP_PADDING) + int(OCR_MASK_PAGE_CONTEXT_PADDING),
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


def _expanded_context_crop_bounds(
    image_shape: tuple[int, ...],
    box: dict,
) -> tuple[int, int, int, int]:
    return _clamped_detector_bounds(
        image_shape,
        box,
        padding=_OCR_CONTEXT_RETRY_PADDING,
    )


def _is_story_retry_candidate(box: dict) -> bool:
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


def _prefer_context_retry(
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
    """Use centered selection only when the detector target is truly a line."""
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
    """Skip known review/noise proposals from batch OCR, retain them in manifest."""
    source_model = str(box.get("source_model") or "").strip().lower()
    source_role = str(box.get("source_role") or "").strip().lower()
    class_name = str(box.get("class_name") or "").strip().lower()
    deferred_reason = str(box.get("deferred_reason") or "").strip()
    if source_role == "scheduler":
        return "deferred-review-region"
    # Geometry/safety deferral controls destructive cleanup, not OCR. A
    # text-segmenter target remains a valid non-destructive OCR target even when
    # its bbox is too wide/large/aspect-outlying for automatic erase.
    if deferred_reason and source_role != "text_segmenter":
        return "deferred-review-region"
    # Raw MSER proposals are review evidence, not OCR authority. A proposal
    # promoted through focused segmentation changes source_role to
    # text_segmenter and is eligible again.
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


def _active_overlap_signatures(
    page: dict, region: dict
) -> list[tuple[str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    preserve_regions = page_preserve_regions(page)
    for box in page.get("boxes", []) or []:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        if box.get("ocr_eligible") is False:
            continue
        if geometry_center_in_regions(box, preserve_regions):
            continue
        box_id = box.get("id")
        if not box_id or not _region_overlaps_box(region, box):
            continue
        matches.append(
            (
                int(box.get("y1", 0)),
                int(box.get("x1", 0)),
                str(box_id),
                ocr_crop_signature(box),
            )
        )
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(box_id, geometry) for _, _, box_id, geometry in matches]


def _find_box(page: dict, box_id: str) -> dict | None:
    return next(
        (
            item
            for item in (page.get("boxes", []) or [])
            if isinstance(item, dict) and str(item.get("id")) == str(box_id)
        ),
        None,
    )


def _sync_group_visual_metadata(
    obj: dict,
    page: dict,
    source_box_ids: list[str],
) -> None:
    source_ids = {str(value) for value in source_box_ids}
    source_boxes = [
        box
        for box in (page.get("boxes") or [])
        if isinstance(box, dict) and str(box.get("id") or "") in source_ids
    ]

    colors = [
        str(box.get("ocr_text_color") or "")
        for box in source_boxes
        if str(box.get("ocr_text_color") or "")
    ]
    if colors:
        counts: dict[str, int] = {}
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
        obj["ocr_text_color"] = max(counts, key=counts.get)
    else:
        obj.pop("ocr_text_color", None)

    sizes: list[int] = []
    for box in source_boxes:
        try:
            size = int(box.get("ocr_font_size") or 0)
        except (TypeError, ValueError):
            continue
        if size > 0:
            sizes.append(size)
    if sizes:
        obj["ocr_font_size"] = int(round(float(np.median(sizes))))
    else:
        obj.pop("ocr_font_size", None)

    regions = [
        box.get("ocr_text_region")
        for box in source_boxes
        if isinstance(box.get("ocr_text_region"), dict)
    ]
    valid_regions = [region for region in regions if _box_region(region) is not None]
    if valid_regions:
        obj["ocr_text_region"] = {
            "x1": min(int(region["x1"]) for region in valid_regions),
            "y1": min(int(region["y1"]) for region in valid_regions),
            "x2": max(int(region["x2"]) for region in valid_regions),
            "y2": max(int(region["y2"]) for region in valid_regions),
        }
    else:
        obj.pop("ocr_text_region", None)


def _find_text_object(page: dict, text_object_id: str) -> dict | None:
    return next(
        (
            item
            for item in (page.get("text_objects", []) or [])
            if isinstance(item, dict) and item.get("id") == text_object_id
        ),
        None,
    )


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OCRCancelled(OCR_CANCELLED_MESSAGE)


OCR_EDGE_RECROP_EXTRA_PADDINGS = (24, 48, 96)


def _expand_ocr_crop_bounds(
    image_shape: tuple[int, ...],
    crop_bounds: tuple[int, int, int, int],
    extra_padding: int = OCR_EDGE_RECROP_EXTRA_PADDINGS[0],
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = crop_bounds
    pad = max(0, int(extra_padding))
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(w, int(x2) + pad),
        min(h, int(y2) + pad),
    )


def _edge_recrop_bounds_sequence(
    image_shape: tuple[int, ...],
    crop_bounds: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    # Grow only after an edge-truncation signal; each retry stays anchored
    # to the original crop so padding never compounds accidentally.
    seen = {tuple(int(v) for v in crop_bounds)}
    bounds = []
    for extra_padding in OCR_EDGE_RECROP_EXTRA_PADDINGS:
        expanded = _expand_ocr_crop_bounds(
            image_shape, crop_bounds, extra_padding=extra_padding
        )
        if expanded in seen:
            continue
        seen.add(expanded)
        bounds.append(expanded)
    return tuple(bounds)


def _normalized_ocr_lines(text: str) -> tuple[str, ...]:
    lines = [
        " ".join(part.upper().split())
        for part in str(text or "").splitlines()
        if part.strip()
    ]
    return tuple(sorted(lines))


def _compact_ocr_text(text: str) -> str:
    return "".join(ch for ch in str(text or "").upper() if ch.isalnum())


def _prefer_edge_recrop(
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


class OCRService:
    def __init__(self, ocr_engine, pipeline):
        self.ocr = ocr_engine
        self.pipeline = pipeline
        self._image_cache: OrderedDict[
            tuple[str, tuple[int, int, int]], np.ndarray
        ] = OrderedDict()
        self._image_cache_bytes = 0
        self._image_cache_budget = _cache_budget_bytes()
        self._image_cache_lock = threading.RLock()
        # Detailed OCR metadata is transient per call. One OCRService instance can
        # serve concurrent jobs, so it must never live directly on the instance.
        self._result_local = threading.local()

    def plan_chapter(self, chapter_id: str) -> list[tuple[int, str]]:
        validate_chapter_id(chapter_id)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            items: list[tuple[int, str]] = []
            for page_index, page in enumerate(manifest.get("pages", [])):
                if page.get("skipped") or page.get("process_required"):
                    continue
                preserve_regions = page_preserve_regions(page)
                for box in page.get("boxes", []) or []:
                    if not isinstance(box, dict) or box.get("removed"):
                        continue
                    if box.get("ocr_eligible") is False:
                        continue
                    if geometry_center_in_regions(box, preserve_regions):
                        continue
                    if ocr_target_skip_reason(box):
                        continue
                    box_id = box.get("id")
                    if box_id:
                        items.append((page_index, str(box_id)))
            return items

    def box_id_at_index(self, chapter_id: str, page_index: int, box_index: int) -> str:
        validate_chapter_id(chapter_id)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page_index: {page_index}")
            boxes = pages[page_index].get("boxes", []) or []
            if box_index < 0 or box_index >= len(boxes):
                raise ValueError(f"Invalid box_index: {box_index}")
            box_id = boxes[box_index].get("id")
            if not box_id:
                raise ValueError("OCR target box has no stable id")
            return str(box_id)

    def inspect_box_index(
        self,
        chapter_id: str,
        page_index: int,
        box_index: int,
        lang: str,
        *,
        force: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        box_id = self.box_id_at_index(chapter_id, page_index, box_index)
        return self.inspect_box_id(
            chapter_id,
            page_index,
            box_id,
            lang,
            force=force,
            cancel_event=cancel_event,
        )

    def inspect_box_id(
        self,
        chapter_id: str,
        page_index: int,
        box_id: str,
        lang: str,
        *,
        force: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        self._result_local.metadata = None
        try:
            validate_chapter_id(chapter_id)
            engine = engine_identity(lang)
            box_snapshot, original_value, source_revision = self._snapshot_box(
                chapter_id, page_index, box_id
            )
            original_path, original_revision = self._source_identity(original_value)

            if (
                not force
                and machine_cache_valid(
                    box_snapshot,
                    lang=lang,
                    engine=engine,
                    source_revision=source_revision,
                    original_revision=original_revision,
                )
                and _visual_cache_complete(box_snapshot)
            ):
                return self._cached_box_result(
                    page_index, box_id, box_snapshot, lang, engine
                )

            _check_cancelled(cancel_event)
            text = self._read_box_text(original_path, box_snapshot, lang)
            _check_cancelled(cancel_event)
            self._commit_box_result(
                chapter_id,
                page_index,
                box_id,
                box_snapshot=box_snapshot,
                original_value=original_value,
                source_revision=source_revision,
                original_revision=original_revision,
                text=text,
                lang=lang,
                engine=engine,
                cancel_event=cancel_event,
            )
            result = {
                "page_index": page_index,
                "box_id": str(box_id),
                "text": text or "",
                "lang": lang,
                "engine": engine,
                "cached": False,
                "committed": True,
                "stale": False,
            }
            metadata = getattr(self._result_local, "metadata", None)
            if metadata:
                result.update(metadata)
            return result
        finally:
            self._result_local.metadata = None

    def _snapshot_box(
        self, chapter_id: str, page_index: int, box_id: str
    ) -> tuple[dict, str, int]:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page_index: {page_index}")
            page = pages[page_index]
            if page.get("skipped"):
                raise ValueError("Cannot OCR a skipped page")
            if page.get("process_required"):
                raise ValueError("Cannot OCR a page that requires processing")
            box = _find_box(page, box_id)
            if box is None or box.get("removed"):
                raise ValueError(f"OCR target box not found: {box_id}")
            if box.get("ocr_eligible") is False:
                raise ValueError(f"OCR target box is not eligible: {box_id}")
            if geometry_center_in_regions(box, page_preserve_regions(page)):
                raise ValueError(f"OCR target box is inside a preserve region: {box_id}")
            original_value = page.get("original")
            if not original_value:
                raise FileNotFoundError("Original page image is not configured")
            box_snapshot = copy.deepcopy(box)
            box_snapshot["_ocr_snapshot_crop_signature"] = ocr_crop_signature(box)
            return (
                box_snapshot,
                str(original_value),
                int(page.get("source_revision") or 0),
            )

    @staticmethod
    def _source_identity(original_value: str) -> tuple[Path, tuple[int, int, int]]:
        original_path = Path(original_value)
        if not original_path.is_file():
            raise FileNotFoundError("Original page image not found")
        return original_path, file_revision(original_path)

    @staticmethod
    def _cached_box_result(
        page_index: int, box_id: str, box_snapshot: dict, lang: str, engine: str
    ) -> dict:
        return {
            "page_index": page_index,
            "box_id": str(box_id),
            "text": str(box_snapshot.get("ocr_text") or ""),
            "lang": lang,
            "engine": engine,
            "cached": True,
            "committed": True,
            "stale": False,
            "confidence": box_snapshot.get("ocr_confidence"),
            "model": str(box_snapshot.get("ocr_model") or ""),
            "orientation": str(box_snapshot.get("ocr_orientation") or "unknown"),
            "region_count": int(box_snapshot.get("ocr_region_count") or 0),
            "quality": str(box_snapshot.get("ocr_quality") or "unknown"),
            "quality_reason": box_snapshot.get("ocr_quality_reason"),
            "coverage": box_snapshot.get("ocr_coverage"),
            "target_mode": str(box_snapshot.get("ocr_target_mode") or "all"),
            "retry_applied": bool(box_snapshot.get("ocr_retry_applied")),
            "text_color": box_snapshot.get("ocr_text_color"),
            "font_size": box_snapshot.get("ocr_font_size"),
            "text_region": copy.deepcopy(box_snapshot.get("ocr_text_region")),
        }

    def _cached_source_image(self, original_path: Path) -> np.ndarray:
        revision = file_revision(original_path)
        key = (str(original_path), revision)
        with self._image_cache_lock:
            cached = self._image_cache.pop(key, None)
            if cached is not None:
                self._image_cache[key] = cached
                return cached

        image = read_image(original_path)
        image_bytes = int(image.nbytes)
        if self._image_cache_budget <= 0 or image_bytes > self._image_cache_budget:
            return image

        with self._image_cache_lock:
            cached = self._image_cache.pop(key, None)
            if cached is not None:
                self._image_cache[key] = cached
                return cached

            while (
                self._image_cache
                and self._image_cache_bytes + image_bytes > self._image_cache_budget
            ):
                _old_key, old_image = self._image_cache.popitem(last=False)
                self._image_cache_bytes -= int(old_image.nbytes)

            self._image_cache[key] = image
            self._image_cache_bytes += image_bytes
        return image

    def _read_box_text(self, original_path: Path, box_snapshot: dict, lang: str) -> str:
        image = self._cached_source_image(original_path)
        crop_bounds = _ocr_crop_bounds(image.shape, box_snapshot)
        x1, y1, x2, y2 = crop_bounds
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            self._result_local.metadata = {
                "confidence": None,
                "model": "none",
                "orientation": "unknown",
                "region_count": 0,
                "quality": "reject",
                "quality_reason": "empty-crop",
                "context_retry_applied": False,
            }
            return ""

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        detailed_reader = getattr(self.ocr, "read_detailed", None)
        if callable(detailed_reader):
            target_mode = ocr_target_mode_for_box(box_snapshot)

            def read_candidate(
                bounds: tuple[int, int, int, int],
            ) -> tuple[str, dict, object | None]:
                cx1, cy1, cx2, cy2 = bounds
                candidate_crop = image[cy1:cy2, cx1:cx2]
                if not candidate_crop.size:
                    return "", {
                        "confidence": None,
                        "model": "none",
                        "orientation": "unknown",
                        "region_count": 0,
                        "quality": "reject",
                        "quality_reason": "empty-crop",
                        "coverage": None,
                        "target_mode": target_mode,
                        "reader_retry_applied": False,
                    }, None
                candidate_rgb = cv2.cvtColor(candidate_crop, cv2.COLOR_BGR2RGB)
                try:
                    candidate_result = detailed_reader(
                        candidate_rgb, lang, target_mode=target_mode
                    )
                except TypeError:
                    candidate_result = detailed_reader(candidate_rgb, lang)
                candidate_text = str(
                    getattr(candidate_result, "text", "") or ""
                ).strip()
                candidate_coverage = self._mask_text_coverage(
                    box_snapshot, bounds, candidate_result
                )
                checked = classify_ocr_quality(
                    candidate_text,
                    lang,
                    confidence=getattr(candidate_result, "confidence", None),
                    coverage=candidate_coverage,
                )
                candidate_quality, candidate_reason = self._conservative_quality(
                    str(
                        getattr(candidate_result, "quality", "unknown")
                        or "unknown"
                    ),
                    getattr(candidate_result, "quality_reason", None),
                    checked.status,
                    checked.reason,
                )
                return candidate_text, {
                    "confidence": getattr(candidate_result, "confidence", None),
                    "model": str(getattr(candidate_result, "model", "") or ""),
                    "orientation": str(
                        getattr(candidate_result, "orientation", "unknown")
                        or "unknown"
                    ),
                    "region_count": int(
                        getattr(candidate_result, "region_count", 0) or 0
                    ),
                    "quality": candidate_quality,
                    "quality_reason": candidate_reason,
                    "coverage": (
                        candidate_coverage
                        if candidate_coverage is not None
                        else getattr(candidate_result, "coverage", None)
                    ),
                    "target_mode": str(
                        getattr(candidate_result, "target_mode", target_mode)
                        or target_mode
                    ),
                    "reader_retry_applied": bool(
                        getattr(candidate_result, "retry_applied", False)
                    ),
                }, candidate_result

            text, metadata, result = read_candidate(crop_bounds)
            recrop_attempted = False
            context_retry_applied = False
            recognition_only = metadata.get("target_mode") == "recognition-only"

            if not recognition_only and metadata["quality_reason"] == "crop-edge-text":
                initial_crop_bounds = crop_bounds
                for expanded_bounds in _edge_recrop_bounds_sequence(
                    image.shape, initial_crop_bounds
                ):
                    recrop_attempted = True
                    expanded_text, expanded_metadata, expanded_result = read_candidate(
                        expanded_bounds
                    )
                    if expanded_result is None:
                        continue
                    if not _prefer_edge_recrop(
                        base_text=text,
                        base_quality=metadata["quality"],
                        base_reason=metadata["quality_reason"],
                        base_confidence=metadata["confidence"],
                        base_region_count=metadata["region_count"],
                        expanded_text=expanded_text,
                        expanded_quality=expanded_metadata["quality"],
                        expanded_reason=expanded_metadata["quality_reason"],
                        expanded_confidence=expanded_metadata["confidence"],
                        expanded_region_count=expanded_metadata["region_count"],
                    ):
                        continue
                    text = expanded_text
                    metadata = expanded_metadata
                    result = expanded_result
                    crop_bounds = expanded_bounds
                    if metadata["quality_reason"] != "crop-edge-text":
                        break
            elif not recognition_only and (
                metadata["quality_reason"] == "incomplete-coverage"
                or (
                    not text
                    and _is_story_retry_candidate(box_snapshot)
                )
            ):
                expanded_bounds = _expanded_context_crop_bounds(
                    image.shape, box_snapshot
                )
                if expanded_bounds != crop_bounds:
                    recrop_attempted = True
                    context_retry_applied = True
                    expanded_text, expanded_metadata, expanded_result = read_candidate(
                        expanded_bounds
                    )
                    if (
                        expanded_result is not None
                        and _prefer_context_retry(
                            base_text=text,
                            base_quality=metadata["quality"],
                            base_reason=metadata["quality_reason"],
                            base_coverage=metadata["coverage"],
                            base_confidence=metadata["confidence"],
                            expanded_text=expanded_text,
                            expanded_quality=expanded_metadata["quality"],
                            expanded_reason=expanded_metadata["quality_reason"],
                            expanded_coverage=expanded_metadata["coverage"],
                            expanded_confidence=expanded_metadata["confidence"],
                            expanded_result=expanded_result,
                            expanded_bounds=expanded_bounds,
                            box=box_snapshot,
                        )
                    ):
                        text = expanded_text
                        metadata = expanded_metadata
                        result = expanded_result
                        crop_bounds = expanded_bounds

            reader_retry_applied = bool(
                metadata.pop("reader_retry_applied", False)
            )
            metadata["retry_applied"] = bool(
                reader_retry_applied or recrop_attempted
            )
            metadata["context_retry_applied"] = context_retry_applied
            metadata.update(
                _visual_text_metadata(
                    image,
                    box_snapshot,
                    crop_bounds,
                    result,
                    text=text,
                    region_count=int(metadata.get("region_count") or 0),
                )
            )
            self._result_local.metadata = metadata
            return text

        text = str(self.ocr.read(rgb, lang) or "").strip()
        quality = classify_ocr_quality(text, lang, confidence=None)
        metadata = {
            "confidence": None,
            "model": "legacy-reader",
            "orientation": "unknown",
            "region_count": 1 if text else 0,
            "quality": quality.status,
            "quality_reason": quality.reason,
            "context_retry_applied": False,
        }
        metadata.update(
            _visual_text_metadata(
                image,
                box_snapshot,
                crop_bounds,
                None,
                text=text,
                region_count=1 if text else 0,
            )
        )
        self._result_local.metadata = metadata
        return text

    @staticmethod
    def _conservative_quality(
        base_status: str,
        base_reason: str | None,
        checked_status: str,
        checked_reason: str | None,
    ) -> tuple[str, str | None]:
        rank = {"reject": 2, "review": 1, "good": 0, "unknown": 0}
        base = base_status if base_status in rank else "unknown"
        checked = checked_status if checked_status in rank else "unknown"
        if rank[base] >= rank[checked]:
            return base, base_reason
        return checked, checked_reason

    @staticmethod
    def _mask_text_coverage(
        box: dict,
        crop_bounds: tuple[int, int, int, int],
        result: object,
    ) -> float | None:
        """Compare OCR text span to segmented text support when it exists.

        This detects the important failure mode where a confident OCR line is
        only a subset of a multi-line segmenter mask. It is intentionally
        neutral for boxes without a trustworthy mask.
        """
        raw_mask = box.get("mask")
        mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
        if mask is None:
            return None
        try:
            bx1, by1, bx2, by2 = map(
                int, (box["x1"], box["y1"], box["x2"], box["y2"])
            )
        except (KeyError, TypeError, ValueError):
            return None
        if mask.shape != (max(0, by2 - by1), max(0, bx2 - bx1)):
            return None
        ys, xs = np.nonzero(mask > 127)
        text_bounds = getattr(result, "text_bounds", None)
        input_shape = getattr(result, "input_shape", None)
        if not xs.size or not ys.size or not text_bounds or not input_shape:
            return None
        try:
            prepared_h, prepared_w = float(input_shape[0]), float(input_shape[1])
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
            predicted_x1, predicted_y1, predicted_x2, predicted_y2 = (
                float(value) for value in text_bounds
            )
        except (TypeError, ValueError, IndexError):
            return None
        crop_w, crop_h = max(1, crop_x2 - crop_x1), max(1, crop_y2 - crop_y1)
        expected_x1, expected_x2 = bx1 + int(xs.min()), bx1 + int(xs.max()) + 1
        expected_y1, expected_y2 = by1 + int(ys.min()), by1 + int(ys.max()) + 1
        observed_x1 = crop_x1 + predicted_x1 * crop_w / max(1.0, prepared_w)
        observed_x2 = crop_x1 + predicted_x2 * crop_w / max(1.0, prepared_w)
        observed_y1 = crop_y1 + predicted_y1 * crop_h / max(1.0, prepared_h)
        observed_y2 = crop_y1 + predicted_y2 * crop_h / max(1.0, prepared_h)
        orientation = str(getattr(result, "orientation", "horizontal") or "horizontal")
        if orientation == "vertical":
            expected_start, expected_end = expected_x1, expected_x2
            observed_start, observed_end = observed_x1, observed_x2
        else:
            expected_start, expected_end = expected_y1, expected_y2
            observed_start, observed_end = observed_y1, observed_y2
        expected_span = max(1.0, float(expected_end - expected_start))
        overlap = max(
            0.0,
            min(float(expected_end), observed_end) - max(float(expected_start), observed_start),
        )
        return max(0.0, min(1.0, overlap / expected_span))

    def _commit_box_result(
        self,
        chapter_id: str,
        page_index: int,
        box_id: str,
        *,
        box_snapshot: dict,
        original_value: str,
        source_revision: int,
        original_revision: tuple[int, int, int],
        text: str,
        lang: str,
        engine: str,
        cancel_event: threading.Event | None,
    ) -> None:
        metadata = getattr(self._result_local, "metadata", None)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            page = self._current_box_page(
                manifest,
                page_index,
                original_value=original_value,
                source_revision=source_revision,
                original_revision=original_revision,
            )
            if page.get("skipped") or page.get("process_required"):
                raise OCRResultStale("Page became unavailable while OCR was running")
            target = _find_box(page, box_id)
            if target is not None and geometry_center_in_regions(target, page_preserve_regions(page)):
                raise OCRResultStale("OCR target entered a preserve region while OCR was running")
            if self._box_changed(target, box_snapshot):
                raise OCRResultStale("OCR target box changed while OCR was running")
            _check_cancelled(cancel_event)
            stamp_machine_cache(
                target,
                text=text,
                lang=lang,
                engine=engine,
                source_revision=source_revision,
                original_revision=original_revision,
                metadata=metadata,
            )
            # Existing auto-generated objects track committed machine OCR in the
            # same manifest transaction, including translation ownership rules.
            sync_existing_auto_text_object(page, target)
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])

    @staticmethod
    def _current_box_page(
        manifest: dict,
        page_index: int,
        *,
        original_value: str,
        source_revision: int,
        original_revision: tuple[int, int, int],
    ) -> dict:
        pages = manifest.get("pages", [])
        if page_index < 0 or page_index >= len(pages):
            raise OCRResultStale("Page disappeared while OCR was running")
        page = pages[page_index]
        current_original = page.get("original")
        current_source_revision = int(page.get("source_revision") or 0)
        if current_original != original_value or current_source_revision != source_revision:
            raise OCRResultStale("Original page changed while OCR was running")
        try:
            current_file_revision = file_revision(Path(str(current_original)))
        except OSError as exc:
            raise OCRResultStale("Original page changed while OCR was running") from exc
        if current_file_revision != original_revision:
            raise OCRResultStale("Original page file changed while OCR was running")
        return page

    @staticmethod
    def _box_changed(target: dict | None, box_snapshot: dict) -> bool:
        return (
            target is None
            or bool(target.get("removed"))
            or target.get("ocr_eligible") is False
            or ocr_crop_signature(target)
            != box_snapshot.get(
                "_ocr_snapshot_crop_signature",
                ocr_crop_signature(box_snapshot),
            )
        )

    def group_text_object(
        self,
        chapter_id: str,
        page_index: int,
        text_object_id: str,
        lang: str,
    ) -> dict:
        validate_chapter_id(chapter_id)
        engine = engine_identity(lang)
        snapshot = self._snapshot_group(chapter_id, page_index, text_object_id)
        original_path, original_revision = self._source_identity(snapshot["original_value"])

        source_box_ids, combined = self._collect_group_text(
            chapter_id,
            page_index,
            snapshot["signatures"],
            lang,
        )
        if not combined:
            # A manually drawn rectangle/ellipse is itself explicit OCR
            # authority. Detector boxes improve grouping when available, but a
            # complete detector miss must not turn the user's region into a
            # silent empty result (common for HUD/system-panel/free text).
            combined = self._read_box_text(
                original_path,
                snapshot["region"],
                lang,
            )
            self._result_local.metadata = None
        return self._commit_group_text(
            chapter_id,
            page_index,
            text_object_id,
            lang=lang,
            engine=engine,
            source_box_ids=source_box_ids,
            combined=combined,
            snapshot=snapshot,
            original_revision=original_revision,
        )

    def _snapshot_group(
        self, chapter_id: str, page_index: int, text_object_id: str
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page index {page_index}")
            page = pages[page_index]
            obj = _find_text_object(page, text_object_id)
            if obj is None:
                raise ValueError(f"Text object not found {text_object_id!r}")
            if page.get("skipped") or page.get("process_required"):
                raise ValueError("Cannot OCR this page before processing")
            if text_object_in_preserve_region(page, obj):
                raise ValueError("Cannot OCR a text object inside a preserve region")
            original_value = page.get("original")
            if not original_value:
                raise FileNotFoundError("Original page image is not configured")
            region = copy.deepcopy(obj.get("region") or {})
            return {
                "region": region,
                "text": obj.get("ocr_text") or "",
                "original_value": str(original_value),
                "source_revision": int(page.get("source_revision") or 0),
                "signatures": _active_overlap_signatures(page, region),
            }

    def _collect_group_text(
        self,
        chapter_id: str,
        page_index: int,
        signatures: list[tuple[str, str]],
        lang: str,
    ) -> tuple[list[str], str]:
        texts: list[str] = []
        source_box_ids: list[str] = []
        for box_id, _geometry in signatures:
            try:
                result = self.inspect_box_id(
                    chapter_id,
                    page_index,
                    box_id,
                    lang,
                    force=False,
                )
            except OCRResultStale:
                logger.warning(
                    "Chapter {} page {} box {} became stale during grouped OCR",
                    chapter_id,
                    page_index,
                    box_id,
                )
                continue
            source_box_ids.append(box_id)
            text = str(result.get("text") or "")
            if text:
                texts.append(text)
        return source_box_ids, "\n".join(texts)

    def _commit_group_text(
        self,
        chapter_id: str,
        page_index: int,
        text_object_id: str,
        *,
        lang: str,
        engine: str,
        source_box_ids: list[str],
        combined: str,
        snapshot: dict,
        original_revision: tuple[int, int, int],
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                return manifest
            page = pages[page_index]
            obj = _find_text_object(page, text_object_id)
            if obj is None:
                return manifest
            if page.get("skipped") or page.get("process_required") or text_object_in_preserve_region(page, obj):
                return manifest
            if self._group_result_stale(page, obj, snapshot, original_revision):
                logger.warning(
                    "Chapter {} page {} object {}: OCR result became stale; keeping newer state",
                    chapter_id,
                    page_index,
                    text_object_id,
                )
                return manifest

            self._stamp_group_object(
                obj,
                source_box_ids=source_box_ids,
                combined=combined,
                lang=lang,
                engine=engine,
                source_revision=snapshot["source_revision"],
                original_revision=original_revision,
                region=snapshot["region"],
            )
            _sync_group_visual_metadata(obj, page, source_box_ids)
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    @staticmethod
    def _group_result_stale(
        page: dict,
        obj: dict,
        snapshot: dict,
        original_revision: tuple[int, int, int],
    ) -> bool:
        current_original = page.get("original")
        try:
            current_file_revision = file_revision(Path(str(current_original)))
        except OSError:
            current_file_revision = None
        return (
            current_original != snapshot["original_value"]
            or int(page.get("source_revision") or 0) != snapshot["source_revision"]
            or current_file_revision != original_revision
            or obj.get("region") != snapshot["region"]
            or (obj.get("ocr_text") or "") != snapshot["text"]
            or _active_overlap_signatures(page, snapshot["region"])
            != snapshot["signatures"]
        )

    @staticmethod
    def _stamp_group_object(
        obj: dict,
        *,
        source_box_ids: list[str],
        combined: str,
        lang: str,
        engine: str,
        source_revision: int,
        original_revision: tuple[int, int, int],
        region: dict,
    ) -> None:
        # Grouped OCR writes directly to a text object, so apply the same
        # translation-ownership rule before replacing its machine-owned source.
        invalidate_stale_machine_translation(obj, combined)
        obj["source_boxes"] = source_box_ids
        obj["ocr_text"] = combined
        obj["ocr_source"] = "machine"
        obj["ocr_lang"] = lang
        obj["ocr_engine"] = engine
        obj["ocr_source_revision"] = source_revision
        obj["ocr_file_revision"] = list(original_revision)
        obj["ocr_region"] = [
            int(region.get(key, 0)) for key in ("x1", "y1", "x2", "y2")
        ]
        obj["ocr_quality"] = "review"
        obj["ocr_quality_reason"] = "grouped-machine-ocr"
