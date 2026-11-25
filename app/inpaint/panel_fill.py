from __future__ import annotations

import cv2
import numpy as np

from app.inpaint.smart_fill import quadratic_design


def _cell_surface(
    crop: np.ndarray, interior: np.ndarray, letters: np.ndarray, collar: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * collar + 1,) * 2)
    ink = cv2.dilate(letters.astype(np.uint8), kernel) > 0
    known = interior & ~ink
    if np.count_nonzero(known) < .22 * np.count_nonzero(interior):
        return None

    height, width = known.shape
    ys, xs = np.nonzero(known)
    if min(np.ptp(xs) / width, np.ptp(ys) / height) < .48:
        return None
    design = quadratic_design(xs, ys, width, height)
    values = crop[known].astype(np.float32)
    keep = np.ones(len(values), bool)
    for _ in range(3):
        if np.count_nonzero(keep) < .60 * len(values):
            return None
        coeff, _, rank, _ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        if rank != design.shape[1]:
            return None
        residual = np.max(np.abs(design @ coeff - values), axis=1)
        keep = residual <= max(7, np.percentile(residual, 75) + 2)

    median, upper = np.percentile(residual, (50, 90))
    if median > 7 or upper > 16:
        return None
    fringe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (collar + 4) + 1,) * 2)
    candidate = interior & (cv2.dilate(letters.astype(np.uint8), fringe) > 0)
    ys, xs = np.nonzero(candidate)
    estimate = quadratic_design(xs, ys, width, height) @ coeff
    low, high = np.percentile(values[keep], (1, 99), axis=0)
    estimate = np.clip(estimate, low - 4, high + 4)
    write = ink[candidate] | (np.max(np.abs(estimate - crop[candidate]), axis=1) > max(5, 2 * median))
    return ys[write], xs[write], np.clip(np.rint(estimate[write]), 0, 255).astype(np.uint8)


def fit_panel_grid(crop: np.ndarray, letters: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct lettered interiors when several separate frames and their surfaces agree."""
    if crop.ndim != 3 or min(letters.shape) < 48 or letters.size > 750_000:
        return None
    letters = letters > 0
    ink_pixels = np.count_nonzero(letters)
    if ink_pixels < 250:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .6), 60, 140, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    painted = np.zeros_like(crop)
    write = np.zeros(letters.shape, np.uint8)
    frames = np.zeros(letters.shape, np.uint8)
    covered = 0
    frame_count = 0
    reference_size = None
    for contour in sorted(contours, key=cv2.contourArea):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if min(w, h) < 42 or not .6 <= h / w <= 1.65 or not .70 <= area / (w * h) <= .95:
            continue
        if area < .96 * cv2.contourArea(cv2.convexHull(contour)):
            continue
        if reference_size is not None:
            frame_w, frame_h = reference_size
            if not (.82 <= w / frame_w <= 1.22 and .82 <= h / frame_h <= 1.22):
                continue
        bounds = np.s_[y:y + h, x:x + w]
        if write[bounds].any():
            continue

        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [contour - [x, y]], -1, 1, cv2.FILLED)
        margin = max(4, round(min(w, h) * .07))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
        interior = cv2.erode(local, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        ink = letters[bounds] & interior
        count = np.count_nonzero(ink)
        interior_area = np.count_nonzero(interior)
        if not max(25, .07 * interior_area) <= count <= .55 * interior_area:
            continue
        fit = _cell_surface(crop[bounds], interior, ink, max(3, min(8, round(min(w, h) * .06))))
        if fit is None:
            continue
        ys, xs, colors = fit
        write[y + ys, x + xs] = 255
        painted[y + ys, x + xs] = colors
        frames[bounds] |= local
        if reference_size is None:
            reference_size = (w, h)
        frame_count += 1
        covered += count

    if frame_count < 3 or reference_size is None or covered < .87 * ink_pixels:
        return None
    rim = min(16, max(5, round(min(reference_size) * .13)))
    near_frames = cv2.dilate(
        frames, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1,) * 2),
    ) > 0
    outside = letters & ~near_frames
    _, _, stats, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8))
    if np.any(stats[1:, cv2.CC_STAT_AREA] >= max(40, .003 * ink_pixels)):
        return None
    return write, painted
