from __future__ import annotations

import cv2
import numpy as np


LOCAL_INK_MASK = "local_ink"
CONTRAST_ARTWORK_MASK = "contrast_artwork"
LARGE_TEXT_MASK_AREA = 24000
PAPER_INK_MASK = "isolated_paper_ink"
PANEL_INK_MASK = "panel_ink"
PRECISE_INK_MASK_SOURCES = frozenset({LOCAL_INK_MASK, CONTRAST_ARTWORK_MASK, PAPER_INK_MASK, PANEL_INK_MASK})


def fills_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.size and np.count_nonzero(mask) >= 0.96 * mask.size)


def expand_local_ink(ink: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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


def _ellipse(radius: int) -> np.ndarray:
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def isolated_paper_ink(gray: np.ndarray, grown: np.ndarray, letters: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Find dark lettering on an isolated light surface surrounded by dark artwork."""
    paper = gray >= 225
    grown_area = np.count_nonzero(grown)
    paper_overlap = np.count_nonzero(paper & grown)
    if not grown_area or paper_overlap < .25 * grown_area:
        return None
    if np.mean(gray < 100) < .20:
        return None

    radius = min(24, max(4, round(min(gray.shape) * .055)))
    closed = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(radius))
    _, labels = cv2.connectedComponents(closed)
    overlap = np.bincount(labels[grown])
    overlap[0] = 0
    if overlap.max() < .30 * paper_overlap:
        return None
    face = (labels == overlap.argmax()).astype(np.uint8)
    if face[0].any() or face[-1].any() or face[:, 0].any() or face[:, -1].any():
        return None
    outlines, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face = cv2.drawContours(face, outlines, -1, 1, cv2.FILLED).astype(bool)
    if np.mean(face[grown]) < .65 or np.mean(paper[face]) < .48:
        return None

    outside = (cv2.dilate(face.astype(np.uint8), _ellipse(7)) > 0) & ~face
    if not outside.any() or np.mean(gray[outside] < 180) < .76:
        return None

    interior = cv2.erode(face.astype(np.uint8), _ellipse(2), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    ink = letters & interior & (gray < 145)
    if np.count_nonzero(ink) < max(40, .08 * np.count_nonzero(face)):
        return None
    near = cv2.dilate(ink.astype(np.uint8), _ellipse(8)) > 0
    enclosed = cv2.erode(interior.astype(np.uint8), _ellipse(6)) > 0
    paper_level = np.percentile(gray[paper & face], 80)
    fringe = near & interior & (gray < paper_level)
    enclosed_ink = enclosed & grown & (gray < paper_level - 4)
    return fringe | enclosed_ink, ink


def protect_free_text_art(gray: np.ndarray, grown: np.ndarray, letters: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Reject seeds continuing into line art and bound expansion by the detected stroke widths."""
    ink_area = np.count_nonzero(letters)
    if ink_area < 40:
        return None
    # A varied surround indicates that expansion can cross an artwork boundary.
    surround = gray[~grown]
    if surround.size < 100 or np.ptp(np.percentile(surround, (10, 90))) < 50:
        return None

    # Count how much of each dark component continues beyond the old inpaint mask.
    count, dark_labels = cv2.connectedComponents((gray < 80).astype(np.uint8))
    outside = np.bincount(dark_labels[~grown], minlength=count)
    inside = np.bincount(dark_labels[grown], minlength=count)
    continuing = (outside > 2 * inside) & (outside >= 250)
    continuing[0] = False
    _, seed_labels, stats, _ = cv2.connectedComponentsWithStats(letters.astype(np.uint8))
    contact = np.bincount(seed_labels[continuing[dark_labels]], minlength=len(stats))
    discard = (contact >= 16) & (contact >= .18 * stats[:, cv2.CC_STAT_AREA])
    discard[0] = False

    if not discard.any() and np.count_nonzero(grown) <= 1.55 * ink_area:
        return None
    retained = letters & ~discard[seed_labels]
    if not retained.any():
        return retained, retained
    thickness = cv2.distanceTransform(retained.astype(np.uint8), cv2.DIST_L2, 3)
    radius = max(3., min(12., .8 * float(np.percentile(thickness[retained], 90))))
    distance = cv2.distanceTransform((~retained).astype(np.uint8), cv2.DIST_L2, 3)
    return (grown & (distance <= radius) & ~discard[seed_labels]) | retained, retained
