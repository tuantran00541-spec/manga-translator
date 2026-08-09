from pathlib import Path
import cv2
import numpy as np
from app.config import SLICE_TARGET_HEIGHT, SLICE_SEARCH_WINDOW, SLICE_MIN_HEIGHT


SAFE_CUT_BAND = 12
MAX_SAFE_SEARCH_EXPANSION = 800


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

    if h <= SLICE_TARGET_HEIGHT + SLICE_SEARCH_WINDOW:
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
    unsafe_rows = _get_content_row_mask(gray, h, w)

    row_std = gray.std(axis=1).astype(np.float32)

    white_diff = cv2.absdiff(gray, 255)
    black_diff = cv2.absdiff(gray, 0)
    non_bg_pixels = (white_diff > 18) & (black_diff > 18)
    content_count_per_row = non_bg_pixels.sum(axis=1).astype(np.float32)

    scores = row_std + content_count_per_row * 2.0

    cuts = []
    y = 0

    while h - y > SLICE_TARGET_HEIGHT + SLICE_MIN_HEIGHT:
        target = y + SLICE_TARGET_HEIGHT
        lo = max(y + SLICE_MIN_HEIGHT, target - SLICE_SEARCH_WINDOW)
        hi = min(h - SLICE_MIN_HEIGHT, target + SLICE_SEARCH_WINDOW)

        cut = _find_safe_cut(unsafe_rows, scores, lo, hi, target)

        if cut is None:
            expanded_lo = max(
                y + SLICE_MIN_HEIGHT,
                target - SLICE_SEARCH_WINDOW - MAX_SAFE_SEARCH_EXPANSION,
            )
            expanded_hi = min(
                h - SLICE_MIN_HEIGHT,
                target + SLICE_SEARCH_WINDOW + MAX_SAFE_SEARCH_EXPANSION,
            )
            cut = _find_safe_cut(
                unsafe_rows, scores, expanded_lo, expanded_hi, target
            )

        if cut is None:
            break

        cuts.append(cut)
        y = cut

    return cuts


def _find_safe_cut(
    unsafe_rows: np.ndarray,
    scores: np.ndarray,
    lo: int,
    hi: int,
    target: int,
) -> int | None:
    """Return a cut inside a genuinely empty vertical band, or None.

    `hi` is treated as an exclusive upper bound. A cut is safe only when
    SAFE_CUT_BAND rows on both sides are outside all detected content masks.
    """
    if lo >= hi:
        return None

    start = max(0, lo + SAFE_CUT_BAND)
    end = min(len(unsafe_rows), hi - SAFE_CUT_BAND)
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
        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        if w_box >= 15 and h_box >= 15:
            y_start = max(0, y_box - pad_y)
            y_end = min(h, y_box + h_box + pad_y)
            mask[y_start:y_end] = True

    return mask
