from pathlib import Path
import cv2
import numpy as np
from app.config import SLICE_TARGET_HEIGHT, SLICE_SEARCH_WINDOW, SLICE_MIN_HEIGHT


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
    scores[unsafe_rows] += 100000.0

    cuts = []
    y = 0

    while h - y > SLICE_TARGET_HEIGHT + SLICE_MIN_HEIGHT:
        target = y + SLICE_TARGET_HEIGHT
        lo = max(y + SLICE_MIN_HEIGHT, target - SLICE_SEARCH_WINDOW)
        hi = min(h - SLICE_MIN_HEIGHT, target + SLICE_SEARCH_WINDOW)

        if lo >= hi:
            cut = target
        else:
            window = scores[lo:hi]
            min_idx = int(np.argmin(window))

            if window[min_idx] >= 50000.0:
                exp_lo = max(y + SLICE_MIN_HEIGHT, target - SLICE_SEARCH_WINDOW - 350)
                exp_hi = min(h - SLICE_MIN_HEIGHT, target + SLICE_SEARCH_WINDOW + 350)
                exp_window = scores[exp_lo:exp_hi]
                exp_min_idx = int(np.argmin(exp_window))
                cut = exp_lo + exp_min_idx
            else:
                cut = lo + min_idx

        cuts.append(cut)
        y = cut

    return cuts


def _get_content_row_mask(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros(h, dtype=bool)

    edges = cv2.Canny(gray, 30, 120)

    white_diff = cv2.absdiff(gray, 255)
    black_diff = cv2.absdiff(gray, 0)
    content_binary = ((white_diff > 18) & (black_diff > 18)).astype(np.uint8) * 255

    combined = cv2.bitwise_or(edges, content_binary)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pad_y = 35
    for c in contours:
        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        if w_box >= 10 and h_box >= 10:
            y_start = max(0, y_box - pad_y)
            y_end = min(h, y_box + h_box + pad_y)
            mask[y_start:y_end] = True

    return mask
