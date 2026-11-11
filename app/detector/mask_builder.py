import numpy as np
import cv2

from app.detector.boxes import BubbleBox
from app.parameters import DETECTOR_CONFIDENCE_MAX
from app.logging_config import logger
from app.parameters import (
    MANUAL_CONFIDENCE_SENTINEL,
    MASK_ADAPTIVE_BORDER_STD_THRESHOLD,
    MASK_ADAPTIVE_DILATE_KERNEL_SIZE,
    MASK_DILATE_KERNEL_SIZE,
    MASK_EXPAND,
)

AUTO_DESTRUCTIVE_MASK_SOURCES = frozenset(
    {"text_segmenter", "bubble_flat_contrast"}
)


def adaptive_dilate_mask(
    mask: np.ndarray,
    crop_img: np.ndarray | None = None,
    *,
    force_max_dilation: bool = False,
) -> np.ndarray:
    if not np.any(mask > 127):
        return mask

    initial_k = (
        MASK_ADAPTIVE_DILATE_KERNEL_SIZE
        if force_max_dilation
        else MASK_DILATE_KERNEL_SIZE
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (initial_k, initial_k))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    if force_max_dilation:
        return dilated

    if crop_img is not None and crop_img.ndim >= 2:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
        border_bool = (cv2.dilate(dilated, np.ones((3, 3), np.uint8)) > 127) & (dilated <= 127)
        if np.any(border_bool):
            border_std = float(gray[border_bool].std())
            if border_std > MASK_ADAPTIVE_BORDER_STD_THRESHOLD:
                expanded_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        MASK_ADAPTIVE_DILATE_KERNEL_SIZE,
                        MASK_ADAPTIVE_DILATE_KERNEL_SIZE,
                    ),
                )
                dilated = cv2.dilate(mask, expanded_kernel, iterations=1)

    return dilated


def _repair_mask_from_ink(
    mask: np.ndarray,
    crop_img: np.ndarray | None,
    box_rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Union the ink the segmenter mask only partly covers.

    A letter model reads the middle of a stroke and misses its ends,
    so the mask covers a giant SFX at a third and the clean pass leaves
    most of it smeared on the page (Mount Hua p075: 47% of the SFX ink
    survived). Dark-ink components that touch the mask are the same
    text the mask partly covers, so they join it, dilated the same
    ~10px. A component joins only when it is a box's own ink: inside
    the box bar a small cut the box model made, and no bigger than the
    box — a shape bigger than the box, or sticking out of it by more
    than a few pixels (a hat, a bubble outline), is art, not text,
    and stays out.
    """
    if crop_img is None:
        # No image to read ink from: the repair cannot run, and the
        # segmenter mask stands as-is (Mount Hua p075's smear stays).
        logger.warning("Skipping mask repair from ink: no crop image")
        return mask
    if not box_rects or not np.any(mask > 127):
        return mask
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
    if gray.shape[:2] != mask.shape[:2]:
        return mask
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size / 2:
        ink = 255 - ink  # letters are the minority class
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    near_mask = cv2.dilate((mask > 127).astype(np.uint8), kernel) > 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    repair = np.zeros_like(mask)
    for label, (x, y, w, h, area) in enumerate(
            (tuple(int(v) for v in row) for row in stats[1:]), start=1):
        if area < 12:
            continue
        # The component belongs to a box when it sits inside it: a
        # stroke the box cut by a few pixels still counts (the box model
        # draws tight), but a shape that sticks out further — a hat, a
        # bubble outline — is art continuing past the box. A component
        # wider or taller than the box itself is likewise a shape, not
        # a letter, however little it sticks out.
        if not any(_component_in_box(x, y, w, h, rect) for rect in box_rects):
            continue
        component = labels[y:y + h, x:x + w] == label
        if not np.any(component & near_mask[y:y + h, x:x + w]):
            continue
        repair[y:y + h, x:x + w] = np.maximum(
            repair[y:y + h, x:x + w], component.astype(np.uint8) * 255)
    if np.any(repair > 127):
        mask = np.maximum(mask, cv2.dilate(repair, kernel, iterations=1))
    return mask


def _component_in_box(x: int, y: int, w: int, h: int,
                      rect: tuple[int, int, int, int]) -> bool:
    """True when a component is a box's own ink: inside the box bar a
    small cut (3% of the box's shorter side), and no bigger than the box
    in either dimension."""
    bx1, by1, bx2, by2 = rect
    margin = 0.03 * min(bx2 - bx1, by2 - by1)
    return (w <= bx2 - bx1 and h <= by2 - by1
            and x >= bx1 - margin and y >= by1 - margin
            and x + w <= bx2 + margin and y + h <= by2 + margin)


def _rectangle_fallback_allowed(box: BubbleBox) -> bool:
    explicit = getattr(box, "allow_rectangle_fallback", None)
    if explicit is not None:
        return bool(explicit)
    return (
        DETECTOR_CONFIDENCE_MAX < MANUAL_CONFIDENCE_SENTINEL
        and float(box.confidence) >= MANUAL_CONFIDENCE_SENTINEL
    )


def is_destructive_box_authorized(box: BubbleBox) -> bool:
    return bool(
        getattr(box, "safe_to_inpaint", False)
        or _rectangle_fallback_allowed(box)
    )


def build_mask(
    image_shape: tuple[int, int],
    boxes: list[BubbleBox],
    crop_img: np.ndarray | None = None,
    *,
    force_max_dilation: bool = False,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    box_rects: list[tuple[int, int, int, int]] = []

    for box in boxes:
        box_w = box.x2 - box.x1
        box_h = box.y2 - box.y1
        if box_w <= 0 or box_h <= 0:
            continue

        if not is_destructive_box_authorized(box):
            # A skipped box with a mask loses the mask too, not just the
            # rectangle fallback; today manual boxes carry confidence 1.0
            # so the sentinel passes them, but a config change that lowers
            # it would silently drop their masks.
            logger.warning(
                "Skipping non-authorized destructive mask for box ({}, {}, {}, {}){}",
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                " — its segmentation mask is dropped with it" if box.mask is not None else "",
            )
            continue

        if box.mask is not None:
            if box.mask.shape != (box_h, box_w):
                logger.warning(
                    "Resizing mismatched box mask from {} to ({}, {}) at ({}, {}, {}, {})",
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
                    logger.error("Failed to resize box mask at ({}, {}, {}, {}): {}", box.x1, box.y1, box.x2, box.y2, exc)
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
            box_rects.append((max(0, box.x1), max(0, box.y1), min(w, box.x2), min(h, box.y2)))
        else:
            if not _rectangle_fallback_allowed(box):
                logger.warning(
                    "Skipping unsafe rectangle fallback for detector box ({}, {}, {}, {}): segmentation mask is missing",
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                )
                continue
            logger.warning(
                "Using explicit rectangle fallback mask for manual box ({}, {}, {}, {})",
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
            box_rects.append((x1, y1, x2, y2))

    mask = _repair_mask_from_ink(mask, crop_img, box_rects)

    return adaptive_dilate_mask(
        mask,
        crop_img,
        force_max_dilation=force_max_dilation,
    )
