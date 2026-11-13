from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.parameters import RENDER_AUTO_STROKE_WIDTH, RENDER_DEFAULT_PADDING, RENDER_PADDING_RATIO_MAX

SIZE_SLACK = 1.15
BUDGET_SCALE = 0.85
_SAMPLE = "Nhưng tôi không thể bỏ cuộc dễ dàng như vậy"


def _letter_pieces(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width = ink.shape[:2]
    x, y, w, h, area = (stats[:, i] for i in range(5))
    keep = ((area >= 12) & (h >= 5) & (h <= 0.9 * height) & (w <= 3 * h)
            & (x > 0) & (y > 0) & (x + w < width) & (y + h < height))
    keep[0] = False
    return labels, np.where(keep[:, None], stats, -1)


def _fg_score(candidate: np.ndarray) -> tuple[int, int]:
    labels, stats = _letter_pieces(candidate)
    areas = stats[:, 4]
    valid = (areas > 0) & (areas <= 0.25 * candidate.size)
    valid[0] = False
    kept_mask = valid[labels].astype(np.uint8)
    eroded = cv2.erode(kept_mask, np.ones((3, 3), np.uint8)) > 0
    surviving = np.unique(labels[eroded])
    return (int(np.count_nonzero(eroded)), int(np.count_nonzero(surviving > 0)))


def _foreground_ink(gray: np.ndarray) -> np.ndarray:
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = 255 - bright
    border = np.concatenate([bright[0, :], bright[-1, :], bright[:, 0], bright[:, -1]])
    bright_share = float(np.count_nonzero(border)) / max(1, border.size)
    if bright_share >= 0.60:
        return dark
    if bright_share <= 0.40:
        return bright
    return bright if _fg_score(bright) > _fg_score(dark) else dark


def source_cap_px(raw: np.ndarray, rect, min_pieces: int = 3, bounds=None) -> int | None:
    x1, y1, x2, y2 = (int(v) for v in rect)
    if bounds is not None:
        bx1, by1, bx2, by2 = (int(v) for v in bounds)
        x1, y1 = max(x1, bx1), max(y1, by1)
        x2, y2 = min(x2, bx2), min(y2, by2)
    crop = raw[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = _foreground_ink(gray)
    _labels, stats = _letter_pieces(ink)
    pieces = sorted((int(h), int(area)) for _x, _y, _w, h, area in stats if area > 0)
    if len(pieces) < max(1, min_pieces):
        return None
    while len(pieces) > max(3, min_pieces):
        rest = pieces[:-1]
        rest_ink = np.cumsum([area for _h, area in rest])
        rest_med = rest[int(np.searchsorted(rest_ink, rest_ink[-1] / 2))][0]
        if pieces[-1][0] <= 3 * max(1, rest_med):
            break
        pieces = rest
    if len(pieces) > 1:
        groups: list[list[tuple[int, int]]] = [[pieces[0]]]
        for piece in pieces[1:]:
            if piece[0] > 1.5 * max(1, groups[-1][-1][0]):
                groups.append([piece])
            else:
                groups[-1].append(piece)

        def _median_height(group: list[tuple[int, int]]) -> int:
            heights = sorted(h for h, _a in group)
            return heights[len(heights) // 2]

        tallest = max(_median_height(group) for group in groups)
        groups = [group for group in groups if _median_height(group) * 8 >= tallest]
        if groups:
            group = max(groups, key=lambda g: (len(g), sum(a for _h, a in g)))
            pieces = group
    ink = np.cumsum([area for _h, area in pieces])
    return pieces[int(np.searchsorted(ink, ink[-1] / 2))][0]


def source_ink_hex(raw: np.ndarray, rect) -> str | None:
    x1, y1, x2, y2 = (int(v) for v in rect)
    crop = raw[max(0, y1):y2, max(0, x1):x2]
    if crop.ndim != 3 or crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = _foreground_ink(gray)
    labels, stats = _letter_pieces(ink)
    letters = (stats[:, 4] > 0)[labels].astype(np.uint8)
    body = cv2.erode(letters, np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(body) < 20:
        return None
    b, g, r = (int(np.median(crop[..., channel][body])) for channel in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


@lru_cache(maxsize=32)
def _metrics(font_path: str) -> tuple[float, float, float]:
    from app.render.text_renderer import _calc_line_height, get_font_object

    font = get_font_object(font_path, 100)
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    top, bottom = draw.textbbox((0, 0), "H", font=font)[1::2]
    width = draw.textbbox((0, 0), _SAMPLE, font=font)[2] / len(_SAMPLE)
    return (bottom - top) / 100, width / 100, _calc_line_height(draw, font, stroke_w=RENDER_AUTO_STROKE_WIDTH) / 100


def matching_font_px(font_path, cap_px: int) -> int:
    return max(1, round(cap_px / max(0.3, _metrics(str(font_path))[0])))


def char_budget(font_path, rect, cap_px: int) -> int:
    x1, y1, x2, y2 = (int(v) for v in rect)
    raw_w, raw_h = x2 - x1, y2 - y1
    pad = max(2, min(RENDER_DEFAULT_PADDING, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
    size = matching_font_px(font_path, cap_px) * BUDGET_SCALE
    _cap, char_w, line_h = _metrics(str(font_path))
    per_line = int((raw_w - 2 * pad) / (char_w * size))
    lines = int((raw_h - 2 * pad) / (line_h * size))
    return max(4, per_line * max(1, lines))
