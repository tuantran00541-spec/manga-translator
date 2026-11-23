from __future__ import annotations

import cv2
import numpy as np


LOCAL_INK_MASK = "local_ink"
CONTRAST_ARTWORK_MASK = "contrast_artwork"
PRECISE_INK_MASK_SOURCES = frozenset({LOCAL_INK_MASK, CONTRAST_ARTWORK_MASK})


def fills_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.size and np.count_nonzero(mask) >= 0.96 * mask.size)


def expand_local_ink(ink: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate((ink > 0).astype(np.uint8) * 255, kernel)


def flat_background_ink(crop: np.ndarray) -> np.ndarray | None:
    """Recover bounded foreground strokes from a corroborated flat background."""
    if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 8:
        return None
    height, width = crop.shape[:2]
    band = min(8, max(2, min(height, width) // 8))
    border = np.concatenate((
        crop[:band].reshape(-1, 3), crop[-band:].reshape(-1, 3),
        crop[:, :band].reshape(-1, 3), crop[:, -band:].reshape(-1, 3),
    ))
    color = np.rint(np.median(border, axis=0)).astype(np.uint8)
    border_distance = np.abs(border.astype(np.int16) - color).max(axis=1)
    if np.mean(border_distance <= 12) < 0.92:
        return None
    distance = cv2.absdiff(crop, tuple(float(v) for v in color) + (0.0,)).max(axis=2)
    if np.mean(distance <= 12) < 0.62:
        return None

    foreground = distance >= 24
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    x, y, w, h, area = stats.T
    keep = (
        (w >= 2) & (h >= 3) & (w <= 4 * h) & (area >= 8)
        & (area <= 0.20 * foreground.size)
        & (x > 0) & (y > 0) & (x + w < width) & (y + h < height)
    )
    keep[0] = False
    if np.count_nonzero(keep) < 2 or int(area[keep].sum()) < 0.65 * np.count_nonzero(foreground):
        return None
    return keep[labels]
