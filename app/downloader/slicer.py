from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np
from loguru import logger

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
    SLICE_OVERLAP_CONTEXT as OVERLAP_CONTEXT,
    SLICE_SAFE_CUT_BAND as SAFE_CUT_BAND,
    SLICE_SEARCH_WINDOW,
    SLICE_TARGET_HEIGHT,
)


def slice_image(image_path: Path, out_dir: Path, prefix: str, *, return_metadata: bool = False):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            with image_path.open("rb") as source_file:
                os.posix_fadvise(source_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    if image is None:
        logger.warning("Skipping undecodable image {}", image_path)
        return []

    h, w = image.shape[:2]
    ext = ".png"

    def save_segment(path: Path, seg: np.ndarray):
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if ext == ".png" else []
        succ, buf = cv2.imencode(ext, seg, params)
        if succ:
            with path.open("wb") as out_file:
                out_file.write(buf.tobytes())
                out_file.flush()
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                    try:
                        os.posix_fadvise(out_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
        else:
            if not cv2.imwrite(str(path), seg, params):
                raise ValueError(f"Could not write image segment at {path}")

    if h <= SLICE_MAX_HEIGHT:
        out_path = out_dir / f"{prefix}_00{ext}"
        save_segment(out_path, image)
        if not return_metadata:
            return [out_path]
        return [{
            "path": out_path, "width": w, "height": h,
            "source_y1": 0, "source_y2": h,
            "core_y1": 0, "core_y2": h, "core_source_y1": 0, "core_source_y2": h,
            "unsafe_before": False, "unsafe_after": False, "source_height": h,
        }]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    unsafe_rows = _get_content_row_mask(gray, h, w)
    cut_rows = _find_cut_rows(gray, h, w, unsafe_rows=unsafe_rows)

    def boundary_unsafe(y: int) -> bool:
        lo = max(0, int(y) - SAFE_CUT_BAND)
        hi = min(h, int(y) + SAFE_CUT_BAND + 1)
        return bool(np.any(unsafe_rows[lo:hi]))

    boundaries = [0] + cut_rows + [h]
    flags = {int(y): boundary_unsafe(int(y)) for y in cut_rows}
    paths = []
    meta = []
    for i in range(len(boundaries) - 1):
        core_start, core_end = int(boundaries[i]), int(boundaries[i + 1])
        unsafe_before = bool(flags.get(core_start, False))
        unsafe_after = bool(flags.get(core_end, False))
        context_start = max(0, core_start - (OVERLAP_CONTEXT if unsafe_before else 0))
        context_end = min(h, core_end + (OVERLAP_CONTEXT if unsafe_after else 0))
        segment = image[context_start:context_end, :]
        out_path = out_dir / f"{prefix}_{i:02d}{ext}"
        save_segment(out_path, segment)
        paths.append(out_path)
        meta.append({
            "path": out_path, "width": w, "height": context_end - context_start,
            "source_y1": context_start, "source_y2": context_end,
            "core_y1": core_start - context_start, "core_y2": core_end - context_start,
            "core_source_y1": core_start, "core_source_y2": core_end,
            "unsafe_before": unsafe_before, "unsafe_after": unsafe_after,
            "source_height": h,
        })
    return meta if return_metadata else paths


def _find_cut_rows(
    gray: np.ndarray, h: int, w: int, *, unsafe_rows: np.ndarray | None = None
) -> list[int]:
    if h <= SLICE_MAX_HEIGHT:
        return []

    if unsafe_rows is None:
        unsafe_rows = _get_content_row_mask(gray, h, w)
    scores = _get_row_scores(gray)

    cuts: list[int] = []
    y = 0

    while h - y > SLICE_MAX_HEIGHT:
        remaining = h - y
        min_height = min(SLICE_MIN_HEIGHT, remaining // 2)
        allowed_lo = y + min_height
        allowed_hi = min(y + SLICE_MAX_HEIGHT, h - min_height)
        target = max(allowed_lo, min(allowed_hi, y + SLICE_TARGET_HEIGHT))

        lo = max(allowed_lo, target - SLICE_SEARCH_WINDOW)
        hi = min(allowed_hi, target + SLICE_SEARCH_WINDOW)
        cut = _find_safe_cut(unsafe_rows, scores, lo, hi, target)

        if cut is None:
            cut = _find_safe_cut(
                unsafe_rows, scores, allowed_lo, allowed_hi, target
            )

        if cut is None:
            cut = _find_low_content_cut(
                scores, allowed_lo, allowed_hi, target, unsafe_rows
            )

        if cut is None or cut <= y or cut >= h:
            cut = min(h - 1, y + SLICE_MAX_HEIGHT)
            if cut <= y:
                break

        cuts.append(int(cut))
        y = int(cut)

    return cuts


def _content(gray: np.ndarray) -> np.ndarray:
    """Detect pixels that stand out from their local background (supports white, black, and colored gutters)."""
    kernel = max(3, (gray.shape[1] // 12) | 1)
    background = cv2.blur(gray, (kernel, 1))
    dev = np.abs(gray.astype(np.int16) - background.astype(np.int16))
    return np.where(dev > SLICE_BACKGROUND_DISTANCE, np.uint8(255), np.uint8(0))


def _row_std(gray: np.ndarray, chunk: int = 1024) -> np.ndarray:
    h, w = gray.shape
    sums = np.empty(h, np.int64)
    squares = np.empty(h, np.int64)
    for top in range(0, h, chunk):
        part = gray[top:top + chunk].astype(np.int32)
        sums[top:top + chunk] = part.sum(axis=1)
        squares[top:top + chunk] = (part * part).sum(axis=1)
    return np.sqrt((w * squares - sums * sums) / float(w * w)).astype(np.float32)


def _row_transitions(gray: np.ndarray, threshold: int = 28, chunk: int = 1024) -> np.ndarray:
    """Count horizontal high-contrast stroke transitions per row (peaks sharply on text lines)."""
    h, w = gray.shape
    if w < 2:
        return np.zeros(h, dtype=np.float32)
    counts = np.empty(h, dtype=np.float32)
    for top in range(0, h, chunk):
        part = gray[top:top + chunk].astype(np.int16)
        counts[top:top + chunk] = np.count_nonzero(np.abs(np.diff(part, axis=1)) > threshold, axis=1)
    return counts


def _get_row_scores(gray: np.ndarray) -> np.ndarray:
    row_std = _row_std(gray)
    content_count_per_row = np.count_nonzero(_content(gray), axis=1).astype(np.float32)
    text_transitions = _row_transitions(gray)
    return row_std + content_count_per_row * SLICE_CONTENT_SCORE_WEIGHT + text_transitions * 3.0


def _find_safe_cut(
    unsafe_rows: np.ndarray,
    scores: np.ndarray,
    lo: int,
    hi: int,
    target: int,
) -> int | None:
    if lo > hi:
        return None

    start = max(SAFE_CUT_BAND, lo)
    end = min(len(unsafe_rows) - SAFE_CUT_BAND, hi + 1)
    if start >= end:
        return None

    safe = ~unsafe_rows
    kernel = np.ones(2 * SAFE_CUT_BAND + 1, dtype=np.int32)
    safe_band = np.convolve(safe.astype(np.int32), kernel, mode="same")
    required = 2 * SAFE_CUT_BAND + 1

    candidates = np.arange(start, end)
    candidates = candidates[safe_band[candidates] == required]
    if candidates.size == 0:
        return None

    # Group contiguous safe rows into runs and prefer the center of a wide safe gap
    # instead of hugging the 12px boundary next to a speech bubble or panel border.
    breaks = np.flatnonzero(np.diff(candidates) > 1) + 1
    runs = np.split(candidates, breaks)

    best_cut: int | None = None
    best_rank: tuple[float, float] | None = None
    for run in runs:
        r_lo, r_hi = int(run[0]), int(run[-1])
        width = r_hi - r_lo + 1
        if r_lo <= target <= r_hi:
            # Target already inside this safe gap: stay near target while keeping generous clearance.
            margin = min(64, width // 2)
            cut = max(r_lo + margin, min(r_hi - margin, int(target)))
        else:
            # Cut at the midpoint of this safe gap so both adjacent slices keep maximum clean margin.
            cut = (r_lo + r_hi) // 2
        dist = abs(cut - int(target))
        # Modest preference for wider safe gaps (up to 160px bonus) when two gaps are comparably close.
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

    lo = max(FALLBACK_BAND, int(lo))
    hi = min(len(scores) - FALLBACK_BAND - 1, int(hi))
    if lo > hi:
        return None

    radius = FALLBACK_BAND
    padded = np.pad(scores.astype(np.float64), (1, 0), mode="constant")
    integral = np.cumsum(padded)
    rows = np.arange(lo, hi + 1)
    starts = rows - radius
    ends = rows + radius + 1
    band_score = (integral[ends] - integral[starts]) / float(2 * radius + 1)

    best = float(np.min(band_score))
    tolerance = max(1.0, abs(best) * SLICE_FALLBACK_TOLERANCE_RATIO)
    low_content = rows[band_score <= best + tolerance]
    if unsafe_rows is not None and low_content.size:
        safe_candidates = low_content[~unsafe_rows[low_content]]
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


def _get_content_row_mask(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros(h, dtype=bool)

    # Downsample tall webtoon strips by 2x for morphological contour analysis (~5x faster, same row bounds).
    scale = 2 if (w >= 400 and h >= 1600) else 1
    if scale == 2:
        sh, sw = max(1, h // 2), max(1, w // 2)
        work = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sh, sw = h, w
        work = gray

    edges = cv2.Canny(
        work, SLICE_CONTENT_CANNY_LOW, SLICE_CONTENT_CANNY_HIGH
    )

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

    return mask
