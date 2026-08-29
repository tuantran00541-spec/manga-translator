import os

import numpy as np
import cv2
from app.detector.bubble_detector import BubbleBox, DETECTOR_CONFIDENCE_MAX
from app.detector.stroke_refinement import refine_stroke_mask
from app.logging_config import logger

from app.config import MASK_DILATE_KERNEL_SIZE

MASK_EXPAND = 8
MANUAL_CONFIDENCE_SENTINEL = 1.0


def _stroke_refinement_enabled() -> bool:
    return os.getenv("MANGA_STROKE_MASK_REFINEMENT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def adaptive_dilate_mask(mask: np.ndarray, crop_img: np.ndarray | None = None) -> np.ndarray:
    if not np.any(mask > 127):
        return mask

    if _stroke_refinement_enabled():
        refined, stats = refine_stroke_mask(
            mask,
            crop_img,
            min_radius=1,
            max_radius=6,
            complexity_guard=True,
        )
        logger.debug(
            "Stroke mask refinement: components=%d growth=%.3f max_radius=%d mean_radius=%.2f",
            stats.components,
            stats.growth_ratio,
            stats.max_radius_used,
            stats.mean_radius_used,
        )
        return refined

    initial_k = MASK_DILATE_KERNEL_SIZE if MASK_DILATE_KERNEL_SIZE % 2 == 1 else 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (initial_k, initial_k))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    if crop_img is not None and crop_img.ndim >= 2:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
        border_bool = (cv2.dilate(dilated, np.ones((3, 3), np.uint8)) > 127) & (dilated <= 127)
        if np.any(border_bool):
            border_std = float(gray[border_bool].std())
            if border_std > 18.0:
                expanded_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                dilated = cv2.dilate(mask, expanded_kernel, iterations=1)

    return dilated


def _rectangle_fallback_allowed(box: BubbleBox) -> bool:
    """Allow destructive rectangle masks only for explicit/manual intent.

    Callers may opt in with ``allow_rectangle_fallback``. The persisted v0.1
    manual-box format predates that attribute and uses confidence=1.0. Detector
    confidence is capped at ``DETECTOR_CONFIDENCE_MAX`` strictly below 1.0, so
    the legacy sentinel is now collision-free rather than heuristic.
    """
    explicit = getattr(box, "allow_rectangle_fallback", None)
    if explicit is not None:
        return bool(explicit)
    return (
        DETECTOR_CONFIDENCE_MAX < MANUAL_CONFIDENCE_SENTINEL
        and float(box.confidence) >= MANUAL_CONFIDENCE_SENTINEL
    )


def build_mask(image_shape: tuple[int, int], boxes: list[BubbleBox], crop_img: np.ndarray | None = None) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        box_w = box.x2 - box.x1
        box_h = box.y2 - box.y1
        if box_w <= 0 or box_h <= 0:
            continue

        if box.mask is not None:
            if box.mask.shape != (box_h, box_w):
                logger.warning(
                    "Resizing mismatched box mask from %s to (%d, %d) at (%d, %d, %d, %d)",
                    box.mask.shape,
                    box_h,
                    box_w,
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                )
                try:
                    box.mask = cv2.resize(box.mask, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                except Exception as exc:
                    logger.error("Failed to resize box mask at (%d, %d, %d, %d): %s", box.x1, box.y1, box.x2, box.y2, exc)
                    box.mask = None

        if box.mask is not None:
            x1 = max(0, box.x1)
            y1 = max(0, box.y1)
            x2 = min(w, box.x2)
            y2 = min(h, box.y2)
            if x2 <= x1 or y2 <= y1:
                continue
            src = box.mask[y1 - box.y1:y2 - box.y1, x1 - box.x1:x2 - box.x1]
            dest = mask[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = np.maximum(dest, src)
        else:
            if not _rectangle_fallback_allowed(box):
                logger.warning(
                    "Skipping unsafe rectangle fallback for detector box (%d, %d, %d, %d): segmentation mask is missing",
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                )
                continue
            logger.warning(
                "Using explicit rectangle fallback mask for manual box (%d, %d, %d, %d)",
                box.x1,
                box.y1,
                box.x2,
                box.y2,
            )
            x1 = max(0, box.x1 - MASK_EXPAND)
            y1 = max(0, box.y1 - MASK_EXPAND)
            x2 = min(w, box.x2 + MASK_EXPAND)
            y2 = min(h, box.y2 + MASK_EXPAND)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return adaptive_dilate_mask(mask, crop_img)
