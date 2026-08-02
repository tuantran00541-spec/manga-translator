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
    cut_rows = _find_cut_rows(gray, h)

    paths = []
    y_start = 0
    for i, y_end in enumerate(cut_rows + [h]):
        segment = image[y_start:y_end, :]
        out_path = out_dir / f"{prefix}_{i:02d}{ext}"
        save_segment(out_path, segment)
        paths.append(out_path)
        y_start = y_end

    return paths


def _find_cut_rows(gray: np.ndarray, h: int) -> list[int]:
    row_score = gray.std(axis=1).astype(np.float32)
    in_bubble_mask = _get_bubble_row_mask(gray)

    row_score[in_bubble_mask] += 10000.0

    cuts = []
    y = 0

    while h - y > SLICE_TARGET_HEIGHT + SLICE_MIN_HEIGHT:
        target = y + SLICE_TARGET_HEIGHT
        lo = max(y + SLICE_MIN_HEIGHT, target - SLICE_SEARCH_WINDOW)
        hi = min(h - SLICE_MIN_HEIGHT, target + SLICE_SEARCH_WINDOW)

        if lo >= hi:
            cut = target
        else:
            window = row_score[lo:hi]
            min_idx = int(np.argmin(window))

            if window[min_idx] >= 5000.0:
                expanded_lo = max(y + SLICE_MIN_HEIGHT, target - SLICE_SEARCH_WINDOW - 150)
                expanded_hi = min(h - SLICE_MIN_HEIGHT, target + SLICE_SEARCH_WINDOW + 150)
                exp_window = row_score[expanded_lo:expanded_hi]
                exp_min_idx = int(np.argmin(exp_window))
                cut = expanded_lo + exp_min_idx
            else:
                cut = lo + min_idx

        cuts.append(cut)
        y = cut

    return cuts


def _get_bubble_row_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros(h, dtype=bool)

    edges = cv2.Canny(gray, 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        if w_box > 25 and h_box > 18 and w_box < int(w * 0.98):
            pad_y = 6
            y_start = max(0, y_box - pad_y)
            y_end = min(h, y_box + h_box + pad_y)
            mask[y_start:y_end] = True

    return mask
