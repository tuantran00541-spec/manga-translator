"""Letter size of the source lettering, and how much translated text fits a box at that size."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.parameters import RENDER_AUTO_STROKE_WIDTH, RENDER_DEFAULT_PADDING, RENDER_PADDING_RATIO_MAX

SIZE_SLACK = 1.15  # rendered capitals may be this much taller than the source ones
BUDGET_SCALE = 0.85  # the character budget assumes lettering slightly smaller than the source
_SAMPLE = "Nhưng tôi không thể bỏ cuộc dễ dàng như vậy"


def source_cap_px(raw: np.ndarray, rect) -> int | None:
    """Median letter height in ``rect`` of the raw slice, or None when no letters stand out."""
    x1, y1, x2, y2 = (int(v) for v in rect)
    crop = raw[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size / 2:
        ink = 255 - ink  # letters are the minority class
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width = crop.shape[:2]
    # Bubble outlines and art reach the crop edge; letters of a one-line shout can fill most of its height.
    heights = [int(h) for x, y, w, h, area in stats[1:count]
               if area >= 12 and 5 <= h <= 0.9 * height and w <= 3 * h
               and x > 0 and y > 0 and x + w < width and y + h < height]
    return int(np.median(heights)) if len(heights) >= 3 else None


def source_ink_hex(raw: np.ndarray, rect) -> str | None:
    """Median colour of the source letters in ``rect`` as #rrggbb, or None when no letters stand out."""
    x1, y1, x2, y2 = (int(v) for v in rect)
    crop = raw[max(0, y1):y2, max(0, x1):x2]
    if crop.ndim != 3 or crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size / 2:
        ink = 255 - ink
    # Skip the anti-aliased rim so the colour comes from the letter bodies.
    body = cv2.erode(ink, np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(body) < 20:
        return None
    b, g, r = (int(np.median(crop[..., channel][body])) for channel in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


@lru_cache(maxsize=32)
def _metrics(font_path: str) -> tuple[float, float, float]:
    """Cap height, mean character width and line height of a font, per pixel of font size."""
    from app.render.text_renderer import _calc_line_height, get_font_object

    font = get_font_object(font_path, 100)
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    top, bottom = draw.textbbox((0, 0), "H", font=font)[1::2]
    width = draw.textbbox((0, 0), _SAMPLE, font=font)[2] / len(_SAMPLE)
    return (bottom - top) / 100, width / 100, _calc_line_height(draw, font, stroke_w=RENDER_AUTO_STROKE_WIDTH) / 100


def matching_font_px(font_path, cap_px: int) -> int:
    """Font size whose capitals are as tall as ``cap_px``."""
    return max(1, round(cap_px / max(0.3, _metrics(str(font_path))[0])))


def char_budget(font_path, rect, cap_px: int) -> int:
    """Characters that fit the box when lettered at the source size."""
    x1, y1, x2, y2 = (int(v) for v in rect)
    raw_w, raw_h = x2 - x1, y2 - y1
    pad = max(2, min(RENDER_DEFAULT_PADDING, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
    size = matching_font_px(font_path, cap_px) * BUDGET_SCALE
    _cap, char_w, line_h = _metrics(str(font_path))
    per_line = int((raw_w - 2 * pad) / (char_w * size))
    lines = int((raw_h - 2 * pad) / (line_h * size))
    return max(4, per_line * max(1, lines))
