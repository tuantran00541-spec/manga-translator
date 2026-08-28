from __future__ import annotations

from PIL import Image, ImageDraw

from app.config import MAX_FONT_SIZE, MIN_FONT_SIZE
from app.render.text_renderer import _fits, get_font_path


def measure_text_layout(
    text: str,
    box: tuple[int, int, int, int],
    *,
    font_size: int | str = "auto",
    font_name: str = "default",
    stroke_width: int | str = "auto",
    padding: int = 6,
) -> dict:
    x1, y1, x2, y2 = (int(value) for value in box)
    raw_w = max(0, x2 - x1)
    raw_h = max(0, y2 - y1)
    if raw_w <= 0 or raw_h <= 0 or not str(text or "").strip():
        return {"fits": False, "font_size": 0, "lines": []}

    pad = max(2, min(padding, int(min(raw_w, raw_h) * 0.08)))
    box_w = raw_w - pad * 2
    box_h = raw_h - pad * 2
    if box_w <= 0 or box_h <= 0:
        return {"fits": False, "font_size": 0, "lines": []}

    if stroke_width is None or stroke_width == "auto" or stroke_width == "":
        stroke_w = 2
    else:
        try:
            stroke_w = max(0, min(12, int(stroke_width)))
        except (TypeError, ValueError):
            stroke_w = 2

    target_size = None
    if isinstance(font_size, (int, float)) and font_size > 0:
        target_size = int(font_size)
    elif isinstance(font_size, str) and font_size.isdigit() and int(font_size) > 0:
        target_size = int(font_size)

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    font_path = str(get_font_path(font_name))
    content = str(text).strip()

    if target_size is not None:
        fits, lines = _fits(draw, content, box_w, box_h, font_path, target_size, stroke_w)
        return {"fits": bool(fits), "font_size": target_size, "lines": lines}

    lo, hi = MIN_FONT_SIZE, MAX_FONT_SIZE
    best_size = 0
    best_lines: list[str] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        fits, lines = _fits(draw, content, box_w, box_h, font_path, mid, stroke_w)
        if fits:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    if best_size > 0:
        return {"fits": True, "font_size": best_size, "lines": best_lines}

    _fits_at_min, min_lines = _fits(
        draw, content, box_w, box_h, font_path, MIN_FONT_SIZE, stroke_w
    )
    return {"fits": False, "font_size": MIN_FONT_SIZE, "lines": min_lines}
