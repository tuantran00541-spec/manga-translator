import numpy as np
import cv2
from app.detector.bubble_detector import BubbleBox
from app.config import MASK_DILATE_KERNEL_SIZE

MASK_EXPAND = 8


def adaptive_dilate_mask(mask: np.ndarray, crop_img: np.ndarray | None = None) -> np.ndarray:
    """Dynamically dilates text mask from 5px to 9px based on border color variance."""
    if not np.any(mask > 127):
        return mask

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


def build_mask(image_shape: tuple[int, int], boxes: list[BubbleBox], crop_img: np.ndarray | None = None) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        box_w = box.x2 - box.x1
        box_h = box.y2 - box.y1
        if box_w <= 0 or box_h <= 0:
            continue

        if box.mask is not None and box.mask.shape == (box_h, box_w):
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
            x1 = max(0, box.x1 - MASK_EXPAND)
            y1 = max(0, box.y1 - MASK_EXPAND)
            x2 = min(w, box.x2 + MASK_EXPAND)
            y2 = min(h, box.y2 + MASK_EXPAND)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return adaptive_dilate_mask(mask, crop_img)
