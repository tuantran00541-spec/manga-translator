from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np

from app.config import (
    SLICE_TARGET_HEIGHT,
    SLICE_SEARCH_WINDOW,
    SLICE_MIN_HEIGHT,
    SLICE_MAX_HEIGHT,
)


SAFE_CUT_BAND = 12
MAX_SAFE_SEARCH_EXPANSION = 360
FALLBACK_BAND = 18


def slice_image(image_path: Path, out_dir: Path, prefix: str) -> list[Path]:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return [image_path]

    h, w = image.shape[:2]
    ext = image_path.suffix or ".jpg"

    def save_segment(path: Path, seg: np.ndarray):
        succ, buf = cv2.imencode(ext, seg)
        if succ:
            buf.tofile(str(path))
        else:
            cv2.imwrite(str(path), seg)

    # A page below the detector's single-pass limit does not need slicing.
    if h <= SLICE_MAX_HEIGHT:
        out_path = out_dir / f"{prefix}_00{ext}"
        save_segment(out_path, image)
        return [out_path]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cut_rows = _find_cut_rows(gray, h, w)

    paths = []
    y_start = 0
    for i, y_end in enumerate(cut_rows + [h]):
        segment = image[y_start:y_end, :]
        out_path = out_dir / f"{prefix}_{i:02d}{ext}"
        save_segment(out_path, segment)
        paths.append(out_path)
        y_start = y_end

    return paths


def _find_cut_rows(gray: np.ndarray, h: int, w: int) -> list[int]:
    """Return cuts that prefer blank gutters but always bound slice height.

    The old slicer stopped entirely when it could not find a perfectly safe
    25-row band. On dense webtoon pages that could leave a 16k image intact,
    forcing each detector to perform ~20 overlapping internal passes.

    This version keeps the existing safe-cut preference. If there is no safe
    band, it chooses the lowest-content band within a constrained range. The
    constraints guarantee every produced segment is <= SLICE_MAX_HEIGHT while
    avoiding tiny trailing fragments whenever possible.
    """
    if h <= SLICE_MAX_HEIGHT:
        return []

    unsafe_rows = _get_content_row_mask(gray, h, w)
    scores = _get_row_scores(gray)

    cuts: list[int] = []
    y = 0

    while h - y > SLICE_MAX_HEIGHT:
        remaining = h - y
        chunks = max(2, math.ceil(remaining / SLICE_MAX_HEIGHT))
        chunks_after = chunks - 1

        # Current segment must leave enough (and not too much) height for the
        # remaining chunks. This is what prevents a final 7k tail.
        min_len = max(
            SLICE_MIN_HEIGHT,
            remaining - chunks_after * SLICE_MAX_HEIGHT,
        )
        max_len = min(
            SLICE_MAX_HEIGHT,
            remaining - chunks_after * SLICE_MIN_HEIGHT,
        )
        if min_len > max_len:
            # Defensive fallback for unusual user-configured constants.
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
            # Dense artwork can legitimately have no completely blank band.
            # Pick the least-content local band instead of giving up and
            # passing the entire long page into the detector.
            cut = _find_low_content_cut(scores, expanded_lo, expanded_hi, target)

        if cut is None or cut <= y or cut >= h:
            # Should be unreachable with sane constants, but avoids a loop if
            # configuration is malformed.
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
    non_bg_pixels = (white_diff > 18) & (black_diff > 18)
    content_count_per_row = non_bg_pixels.sum(axis=1).astype(np.float32)

    return row_std + content_count_per_row * 2.0


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
) -> int | None:
    if lo > hi or len(scores) == 0:
        return None

    lo = max(FALLBACK_BAND, int(lo))
    hi = min(len(scores) - FALLBACK_BAND - 1, int(hi))
    if lo > hi:
        return None

    # Mean content score in a small vertical band. Using an integral sum keeps
    # this O(number of rows), even for 16k+ webtoons.
    radius = FALLBACK_BAND
    padded = np.pad(scores.astype(np.float64), (1, 0), mode="constant")
    integral = np.cumsum(padded)
    rows = np.arange(lo, hi + 1)
    starts = rows - radius
    ends = rows + radius + 1
    band_score = (integral[ends] - integral[starts]) / float(2 * radius + 1)

    # First choose genuinely low-content candidates, then prefer the one near
    # the balanced target. This prevents a tiny visual-score improvement from
    # creating very uneven chunks.
    best = float(np.min(band_score))
    tolerance = max(1.0, abs(best) * 0.08)
    low_content = rows[band_score <= best + tolerance]
    if low_content.size == 0:
        return int(rows[int(np.argmin(band_score))])

    distance = np.abs(low_content - int(target))
    nearest = low_content[distance == distance.min()]
    if nearest.size == 1:
        return int(nearest[0])

    # Tie-break by exact local score.
    idx = np.searchsorted(rows, nearest)
    return int(nearest[int(np.argmin(band_score[idx]))])


def _get_content_row_mask(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros(h, dtype=bool)

    edges = cv2.Canny(gray, 30, 120)

    white_diff = cv2.absdiff(gray, 255)
    black_diff = cv2.absdiff(gray, 0)
    content_binary = ((white_diff > 18) & (black_diff > 18)).astype(np.uint8) * 255

    combined = cv2.bitwise_or(edges, content_binary)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pad_y = 40
    for c in contours:
        _x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        if w_box >= 15 and h_box >= 15:
            y_start = max(0, y_box - pad_y)
            y_end = min(h, y_box + h_box + pad_y)
            mask[y_start:y_end] = True

    return mask
