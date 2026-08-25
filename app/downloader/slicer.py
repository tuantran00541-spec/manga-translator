from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SliceLayout:
    path: Path
    source_y_start: int
    source_y_end: int


def slice_image(image_path: Path, out_dir: Path, prefix: str) -> list[Path]:
    return [layout.path for layout in slice_image_with_layout(image_path, out_dir, prefix)]


def slice_image_with_layout(
    image_path: Path,
    out_dir: Path,
    prefix: str,
) -> list[SliceLayout]:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return [SliceLayout(image_path, 0, 0)]

    h, w = image.shape[:2]
    ext = image_path.suffix or ".jpg"

    def save_segment(path: Path, seg: np.ndarray):
        succ, buf = cv2.imencode(ext, seg)
        if succ:
            buf.tofile(str(path))
        else:
            cv2.imwrite(str(path), seg)

    if h <= SLICE_MAX_HEIGHT:
        out_path = out_dir / f"{prefix}_00{ext}"
        save_segment(out_path, image)
        return [SliceLayout(out_path, 0, h)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cut_rows = _find_cut_rows(gray, h, w)

    layouts = []
    y_start = 0
    for i, y_end in enumerate(cut_rows + [h]):
        segment = image[y_start:y_end, :]
        out_path = out_dir / f"{prefix}_{i:02d}{ext}"
        save_segment(out_path, segment)
        layouts.append(SliceLayout(out_path, y_start, y_end))
        y_start = y_end

    return layouts


def _find_cut_rows(
    gray: np.ndarray,
    h: int,
    w: int,
    unsafe_rows: np.ndarray | None = None,
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
            cut = _find_low_content_cut(scores, expanded_lo, expanded_hi, target)

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

    radius = FALLBACK_BAND
    padded = np.pad(scores.astype(np.float64), (1, 0), mode="constant")
    integral = np.cumsum(padded)
    rows = np.arange(lo, hi + 1)
    starts = rows - radius
    ends = rows + radius + 1
    band_score = (integral[ends] - integral[starts]) / float(2 * radius + 1)

    best = float(np.min(band_score))
    tolerance = max(1.0, abs(best) * 0.08)
    low_content = rows[band_score <= best + tolerance]
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
