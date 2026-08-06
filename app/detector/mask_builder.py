import numpy as np
import cv2
from app.detector.bubble_detector import BubbleBox

MASK_EXPAND = 8


def build_mask(image_shape: tuple[int, int], boxes: list[BubbleBox]) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    has_stroke_masks = False

    for box in boxes:
        box_w = box.x2 - box.x1
        box_h = box.y2 - box.y1
        if box_w <= 0 or box_h <= 0:
            continue

        if box.mask is not None and box.mask.shape == (box_h, box_w):
            has_stroke_masks = True
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

    # Dilate text stroke masks with 9x9 ellipse kernel (4px expansion) to cover antialiased edges,
    # stroke outlines, and shadows without spilling across bubble boundaries.
    dilation_kernel_size = (9, 9) if has_stroke_masks else (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel_size)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
