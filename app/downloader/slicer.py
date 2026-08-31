from __future__ import annotations

from pathlib import Path
import math
import os

import cv2
import numpy as np

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
    SLICE_MAX_SAFE_SEARCH_EXPANSION as MAX_SAFE_SEARCH_EXPANSION,
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
        return [image_path] if not return_metadata else [{"path": image_path}]

    h, w = image.shape[:2]
    ext = ".png"

    def save_segment(path: Path, seg: np.ndarray):
        succ, buf = cv2.imencode(ext, seg)
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
            cv2.imwrite(str(path), seg)

    if h <= SLICE_MAX_HEIGHT:
        out_path = out_dir / f"{prefix}_00{ext}"
        save_segment(out_path, image)
        if not return_metadata:
            return [out_path]
        return [{
            "path": out_path, "source_y1": 0, "source_y2": h,
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
            "path": out_path,
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
    """Return cuts that prefer blank gutters but always bound slice height.

    The old slicer stopped entirely when it could not find a perfectly safe
    band. On dense webtoon pages that could leave a very tall image intact,
    forcing each detector to perform many overlapping internal passes.

    This version keeps the existing safe-cut preference. If there is no safe
    band, it chooses the lowest-content band within a constrained range. The
    constraints guarantee every produced segment is <= SLICE_MAX_HEIGHT while
    avoiding tiny trailing fragments whenever possible.
    """
    if h <= SLICE_MAX_HEIGHT:
        return []

    if unsafe_rows is None:
        unsafe_rows = _get_content_row_mask(gray, h, w)
    scores = _get_row_scores(gray)

    cuts: list[int] = []
    y = 0

    while h - y > SLICE_MAX_HEIGHT:
        remaining = h - y
        chunks = max(2, math.ceil(remaining / SLICE_MAX_HEIGHT))
        chunks_after = chunks - 1

        min_len = max(
            SLICE_MIN_HEIGHT,
            remaining - chunks_after * SLICE_MAX_HEIGHT,
        )
        max_len = min(
            SLICE_MAX_HEIGHT,
            remaining - chunks_after * SLICE_MIN_HEIGHT,
        )
        if min_len > max_len:
            min_len = min(SLICE_MIN_HEIGHT, SLICE_MAX_HEIGHT)
            max_len = SLICE_MAX_HEIGHT

        ideal_len = int(round(remaining / chunks))
        preferred_len = int(round((ideal_len + SLICE_TARGET_HEIGHT) / 2.0))
        preferred_len = max(min_len, min(max_len, preferred_len))

        allowed_lo = y + min_len
        allowed_hi = y + max_len
        target = y + preferred_len

        lo = max(allowed_lo, target - SLICE_SEARCH_WINDOW)
        hi = min(allowed_hi, target + SLICE_SEARCH_WINDOW)
        cut = _find_safe_cut(unsafe_rows, scores, lo, hi, target)

        expanded_lo = lo
        expanded_hi = hi
        if cut is None:
            expanded_lo = max(
                allowed_lo,
                target - SLICE_SEARCH_WINDOW - MAX_SAFE_SEARCH_EXPANSION,
            )
            expanded_hi = min(
                allowed_hi,
                target + SLICE_SEARCH_WINDOW + MAX_SAFE_SEARCH_EXPANSION,
            )
            cut = _find_safe_cut(
                unsafe_rows, scores, expanded_lo, expanded_hi, target
            )

        if cut is None:
            cut = _find_low_content_cut(
                scores, expanded_lo, expanded_hi, target, unsafe_rows
            )

        if cut is None or cut <= y or cut >= h:
            cut = min(h - 1, y + SLICE_MAX_HEIGHT)
            if cut <= y:
                break

        cuts.append(int(cut))
        y = int(cut)

    return cuts


def _get_row_scores(gray: np.ndarray) -> np.ndarray:
    row_std = gray.std(axis=1).astype(np.float32)

    white_diff = cv2.absdiff(gray, 255)
    black_diff = cv2.absdiff(gray, 0)
    non_bg_pixels = (
        (white_diff > SLICE_BACKGROUND_DISTANCE)
        & (black_diff > SLICE_BACKGROUND_DISTANCE)
    )
    content_count_per_row = non_bg_pixels.sum(axis=1).astype(np.float32)

    return row_std + content_count_per_row * SLICE_CONTENT_SCORE_WEIGHT


def _find_safe_cut(
    unsafe_rows: np.ndarray,
    scores: np.ndarray,
    lo: int,
    hi: int,
    target: int,
) -> int | None:
    if lo >= hi:
        return None

    start = max(0, lo + SAFE_CUT_BAND)
    end = min(len(unsafe_rows), hi - SAFE_CUT_BAND + 1)
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

    distance = np.abs(candidates - target)
    min_distance = distance.min()
    nearest = candidates[distance == min_distance]

    if nearest.size == 1:
        return int(nearest[0])

    return int(nearest[np.argmin(scores[nearest])])


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

    edges = cv2.Canny(
        gray, SLICE_CONTENT_CANNY_LOW, SLICE_CONTENT_CANNY_HIGH
    )

    white_diff = cv2.absdiff(gray, 255)
    black_diff = cv2.absdiff(gray, 0)
    content_binary = (
        (white_diff > SLICE_BACKGROUND_DISTANCE)
        & (black_diff > SLICE_BACKGROUND_DISTANCE)
    ).astype(np.uint8) * 255

    combined = cv2.bitwise_or(edges, content_binary)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SLICE_CLOSE_KERNEL_SIZE, SLICE_CLOSE_KERNEL_SIZE),
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        _x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        if (
            w_box >= SLICE_CONTOUR_MIN_WIDTH
            and h_box >= SLICE_CONTOUR_MIN_HEIGHT
        ):
            y_start = max(0, y_box - SLICE_CONTOUR_PAD_Y)
            y_end = min(h, y_box + h_box + SLICE_CONTOUR_PAD_Y)
            mask[y_start:y_end] = True

    return mask
