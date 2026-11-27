from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.image_io import read_image, write_image
from app.parameters import (
    SLICE_BACKGROUND_DISTANCE,
    SLICE_CLOSE_KERNEL_SIZE,
    SLICE_CONTENT_CANNY_HIGH,
    SLICE_CONTENT_CANNY_LOW,
    SLICE_CONTENT_SCORE_WEIGHT,
    SLICE_CONTOUR_MIN_HEIGHT,
    SLICE_CONTOUR_MIN_WIDTH,
    SLICE_CONTOUR_PAD_Y,
    SLICE_FALLBACK_BAND as FALLBACK_BAND,
    SLICE_FALLBACK_TOLERANCE_RATIO,
    SLICE_MAX_HEIGHT,
    SLICE_MIN_HEIGHT,
    SLICE_SAFE_CUT_BAND as SAFE_CUT_BAND,
    SLICE_SEARCH_WINDOW,
    SLICE_TARGET_HEIGHT,
)


def _horizontal_kernel_width(width: int) -> int:
    if width <= 1:
        return 1
    max_odd = width if width % 2 == 1 else width - 1
    return max(1, min(max_odd, max(3, (width // 12) | 1)))


def slice_image(
    image_path: Path, out_dir: Path, prefix: str, *, return_metadata: bool = False
):
    try:
        image = read_image(image_path)
    except (OSError, ValueError):
        logger.warning("Skipping undecodable image {}", image_path)
        return []

    h, w = image.shape[:2]
    ext = ".png"
    out_dir.mkdir(parents=True, exist_ok=True)

    if h <= SLICE_MAX_HEIGHT:
        out_path = out_dir / f"{prefix}_00{ext}"
        write_image(out_path, image)
        if not return_metadata:
            return [out_path]
        return [
            {
                "path": out_path,
                "width": w,
                "height": h,
                "source_y1": 0,
                "source_y2": h,
                "core_y1": 0,
                "core_y2": h,
                "core_source_y1": 0,
                "core_source_y2": h,
                "unsafe_before": False,
                "unsafe_after": False,
                "source_height": h,
            }
        ]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    occupied = _get_content_row_mask(gray, h, w, color=image)
    cut_rows = _find_cut_rows(gray, h, w, unsafe_rows=occupied)
    del gray, occupied

    boundaries = [0] + cut_rows + [h]
    paths = []
    meta = []
    for i in range(len(boundaries) - 1):
        core_start, core_end = int(boundaries[i]), int(boundaries[i + 1])
        if core_end <= core_start:
            continue
        segment = image[core_start:core_end, :]
        out_path = out_dir / f"{prefix}_{i:02d}{ext}"
        write_image(out_path, segment)
        paths.append(out_path)
        meta.append(
            {
                "path": out_path,
                "width": w,
                "height": core_end - core_start,
                "source_y1": core_start,
                "source_y2": core_end,
                "core_y1": 0,
                "core_y2": core_end - core_start,
                "core_source_y1": core_start,
                "core_source_y2": core_end,
                "unsafe_before": False,
                "unsafe_after": False,
                "source_height": h,
            }
        )
    return meta if return_metadata else paths


def _find_cut_rows(
    gray: np.ndarray, h: int, w: int, *, unsafe_rows: np.ndarray | None = None
) -> list[int]:
    if h <= SLICE_MAX_HEIGHT:
        return []

    if unsafe_rows is None:
        unsafe_rows = _get_content_row_mask(gray, h, w)
    safe_rows = _safe_cut_rows(unsafe_rows)
    if not safe_rows.size:
        return []
    scores = _get_row_scores(gray)

    cuts: list[int] = []
    y = 0

    while h - y > SLICE_MAX_HEIGHT:
        remaining = h - y
        min_height = min(SLICE_MIN_HEIGHT, remaining // 2)
        allowed_lo = y + min_height
        allowed_hi = min(y + SLICE_MAX_HEIGHT, h - min_height)
        if allowed_lo > allowed_hi:
            break
        target = max(allowed_lo, min(allowed_hi, y + SLICE_TARGET_HEIGHT))

        lo = max(allowed_lo, target - SLICE_SEARCH_WINDOW)
        hi = min(allowed_hi, target + SLICE_SEARCH_WINDOW)
        cut = _find_safe_cut(safe_rows, scores, lo, hi, target)

        if cut is None:
            cut = _find_safe_cut(
                safe_rows, scores, allowed_lo, allowed_hi, target
            )

        if cut is None:
            # Keep the panel together when the next safe gutter is farther away.
            cut = _find_safe_cut(
                safe_rows, scores, allowed_hi + 1, h - min_height, target
            )

        if cut is None or cut <= y:
            break

        cuts.append(int(cut))
        y = int(cut)

    return cuts


def _content(gray: np.ndarray) -> np.ndarray:
    """Detect pixels that stand out from their local background (supports white, black, and colored gutters)."""
    if gray.ndim != 2 or gray.shape[1] < 2:
        return np.zeros_like(gray, dtype=np.uint8)
    kernel = _horizontal_kernel_width(gray.shape[1])
    background = cv2.blur(gray, (kernel, 1))
    dev = cv2.absdiff(gray, background)
    return np.where(dev > SLICE_BACKGROUND_DISTANCE, np.uint8(255), np.uint8(0))


def _row_std(gray: np.ndarray, chunk: int = 1024) -> np.ndarray:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(h, dtype=np.float32)
    sums = np.empty(h, dtype=np.int64)
    squares = np.empty(h, dtype=np.int64)
    for top in range(0, h, chunk):
        part = gray[top : top + chunk].astype(np.int32)
        sums[top : top + chunk] = part.sum(axis=1, dtype=np.int64)
        squares[top : top + chunk] = (part * part).sum(axis=1, dtype=np.int64)
    variance = np.maximum(0.0, (w * squares - sums * sums) / float(w * w))
    return np.sqrt(variance).astype(np.float32)


def _row_transitions(
    gray: np.ndarray, threshold: int = 28, chunk: int = 1024
) -> np.ndarray:
    """Count horizontal high-contrast stroke transitions per row (peaks sharply on text lines)."""
    h, w = gray.shape[:2]
    if h == 0 or w < 2:
        return np.zeros(h, dtype=np.float32)
    counts = np.empty(h, dtype=np.float32)
    for top in range(0, h, chunk):
        part = gray[top : top + chunk].astype(np.int16)
        counts[top : top + chunk] = np.count_nonzero(
            np.abs(np.diff(part, axis=1)) > threshold, axis=1
        )
    return counts


def _get_row_scores(gray: np.ndarray) -> np.ndarray:
    row_std = _row_std(gray)
    content_count_per_row = np.count_nonzero(_content(gray), axis=1).astype(np.float32)
    text_transitions = _row_transitions(gray)
    return (
        row_std
        + content_count_per_row * SLICE_CONTENT_SCORE_WEIGHT
        + text_transitions * 3.0
    )


def _safe_cut_rows(unsafe_rows: np.ndarray) -> np.ndarray:
    window = 2 * SAFE_CUT_BAND + 1
    if unsafe_rows.size < window:
        return np.empty(0, dtype=np.intp)
    integral = np.r_[0, np.cumsum(unsafe_rows.astype(bool), dtype=np.int32)]
    return np.flatnonzero(integral[window:] == integral[:-window]) + SAFE_CUT_BAND


def _find_safe_cut(
    safe_rows: np.ndarray,
    scores: np.ndarray,
    lo: int,
    hi: int,
    target: int,
) -> int | None:
    if lo > hi or safe_rows.size == 0:
        return None

    start, end = np.searchsorted(safe_rows, (lo, hi + 1))
    if start >= end:
        return None

    # Expand to the enclosing contiguous safe runs so gap width/midpoint are not
    # skewed when a wide gutter straddles lo or hi.
    run_start = int(start)
    while run_start > 0 and safe_rows[run_start] - safe_rows[run_start - 1] == 1:
        run_start -= 1
    run_end = int(end)
    n_safe = int(safe_rows.size)
    while run_end < n_safe and safe_rows[run_end] - safe_rows[run_end - 1] == 1:
        run_end += 1

    enclosing = safe_rows[run_start:run_end]
    breaks = np.flatnonzero(np.diff(enclosing) > 1) + 1
    runs = np.split(enclosing, breaks)

    best_cut: int | None = None
    best_rank: tuple[float, float] | None = None
    for run in runs:
        r_lo, r_hi = int(run[0]), int(run[-1])
        clamped_lo = max(r_lo, int(lo))
        clamped_hi = min(r_hi, int(hi))
        if clamped_lo > clamped_hi:
            continue
        width = r_hi - r_lo + 1
        if r_lo <= target <= r_hi:
            margin = min(64, width // 2)
            ideal = max(r_lo + margin, min(r_hi - margin, int(target)))
        else:
            ideal = (r_lo + r_hi) // 2
        cut = max(clamped_lo, min(clamped_hi, ideal))
        dist = abs(cut - int(target))
        clearance_bonus = min(width, 160) * 0.35
        rank = (dist - clearance_bonus, float(scores[cut]))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_cut = int(cut)

    return best_cut


def _find_low_content_cut(
    scores: np.ndarray,
    lo: int,
    hi: int,
    target: int,
    unsafe_rows: np.ndarray | None = None,
) -> int | None:
    if lo > hi or len(scores) == 0:
        return None

    radius = FALLBACK_BAND
    lo = max(radius, int(lo))
    hi = min(len(scores) - radius - 1, int(hi))
    if lo > hi:
        return None

    win_start = lo - radius
    win_end = hi + radius + 1
    local = np.asarray(scores[win_start:win_end], dtype=np.float64)
    integral = np.concatenate(([0.0], np.cumsum(local)))
    rows = np.arange(lo, hi + 1)
    width = 2 * radius + 1
    band_score = (integral[width:] - integral[:-width]) / float(width)

    best = float(np.min(band_score))
    tolerance = max(1.0, abs(best) * SLICE_FALLBACK_TOLERANCE_RATIO)
    low_content = rows[band_score <= best + tolerance]
    if unsafe_rows is not None and low_content.size:
        unsafe_bool = np.asarray(unsafe_rows, dtype=bool)
        safe_candidates = low_content[~unsafe_bool[low_content]]
        if safe_candidates.size:
            low_content = safe_candidates
    if low_content.size == 0:
        return int(rows[int(np.argmin(band_score))])

    distance = np.abs(low_content - int(target))
    nearest = low_content[distance == distance.min()]
    if nearest.size == 1:
        return int(nearest[0])

    idx = np.searchsorted(rows, nearest)
    return int(nearest[int(np.argmin(band_score[idx]))])


def _get_content_row_mask(
    gray: np.ndarray, h: int, w: int, *, color: np.ndarray | None = None
) -> np.ndarray:
    full_edges = cv2.Canny(gray, SLICE_CONTENT_CANNY_LOW, SLICE_CONTENT_CANNY_HIGH)
    mask = np.any(full_edges, axis=1)

    scale = 2 if (w >= 400 and h >= 1600) else 1
    if scale == 2:
        del full_edges
        sh, sw = max(1, h // 2), max(1, w // 2)
        work = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(work, SLICE_CONTENT_CANNY_LOW, SLICE_CONTENT_CANNY_HIGH)
    else:
        sh, sw = h, w
        work = gray
        edges = full_edges

    combined = cv2.bitwise_or(edges, _content(work))

    k_size = max(3, (SLICE_CLOSE_KERNEL_SIZE // scale) | 1)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k_size, k_size),
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_w = max(4, SLICE_CONTOUR_MIN_WIDTH // scale)
    min_h = max(4, SLICE_CONTOUR_MIN_HEIGHT // scale)
    for c in contours:
        _x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        if w_box >= min_w and h_box >= min_h:
            y_start = max(0, y_box * scale - SLICE_CONTOUR_PAD_Y)
            y_end = min(h, (y_box + h_box) * scale + SLICE_CONTOUR_PAD_Y)
            mask[y_start:y_end] = True

    if color is not None and color.ndim == 3 and color.shape[0] == h and color.shape[1] >= 2:
        kernel = _horizontal_kernel_width(color.shape[1])
        chunk = 1024
        for y0 in range(0, h, chunk):
            y1 = min(h, y0 + chunk)
            strip = color[y0:y1]
            color_bg = cv2.blur(strip, (kernel, 1))
            color_dev = cv2.absdiff(strip, color_bg)
            mask[y0:y1] |= np.any(color_dev >= SLICE_BACKGROUND_DISTANCE, axis=(1, 2))

    return mask
