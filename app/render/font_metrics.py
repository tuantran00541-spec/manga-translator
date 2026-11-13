from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

from app.config import DEFAULT_FONT
from app.parameters import (
    FONT_CACHE_SIZE,
    RENDER_AUTO_TEXT_DARK_BG_THRESHOLD,
    RENDER_LINE_HEIGHT_FACTOR,
)
from app.render.font_catalog import list_font_records, resolve_font_id


@lru_cache(maxsize=FONT_CACHE_SIZE)
def get_font_object(font_path_str: str, size: int) -> ImageFont.FreeTypeFont:
    size = max(1, int(size))
    try:
        return ImageFont.truetype(font_path_str, size)
    except OSError as e:
        raise OSError(f"Cannot load font '{font_path_str}' size={size}: {e}") from e


_PROBE_SIZE = 32
_UNASSIGNED = "\U0010FFFD"


@lru_cache(maxsize=FONT_CACHE_SIZE)
def _missing_glyph_mask(font_path_str: str) -> tuple:
    mask = get_font_object(font_path_str, _PROBE_SIZE).getmask(_UNASSIGNED)
    return mask.size, bytes(mask)


@lru_cache(maxsize=8192)
def _draws_char(font_path_str: str, char: str) -> bool:
    mask = get_font_object(font_path_str, _PROBE_SIZE).getmask(char)
    return (mask.size, bytes(mask)) != _missing_glyph_mask(font_path_str)


_PUNCTUATION_FALLBACKS = {
    "\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2022": "·", "\u00ab": '"', "\u00bb": '"',
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u27e8": "<", "\u27e9": ">", "\u3008": "<", "\u3009": ">", "\u300a": "<", "\u300b": ">",
    "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
}


def _plain_punctuation(font_path, text: str) -> str:
    font_path_str = str(font_path)
    return "".join(
        _PUNCTUATION_FALLBACKS[char] if char in _PUNCTUATION_FALLBACKS and not _draws_char(font_path_str, char) else char
        for char in text
    )


def drawable(font_path, text: str) -> tuple[Path, str]:
    font_path = Path(font_path)
    if font_draws_text(font_path, text):
        return font_path, text
    plain = _plain_punctuation(font_path, text)
    if font_draws_text(font_path, plain):
        return font_path, plain
    if font_draws_text(DEFAULT_FONT, text):
        return DEFAULT_FONT, text
    return font_path, text


def font_draws_text(font_path, text: str) -> bool:
    font_path_str = str(font_path)
    return all(_draws_char(font_path_str, char) for char in set(str(text or "")) if not char.isspace())


def parse_color(color_input, default=(0, 0, 0)) -> tuple[int, int, int]:
    if not color_input:
        return default
    if isinstance(color_input, (tuple, list)) and len(color_input) >= 3:
        try:
            return (int(color_input[0]), int(color_input[1]), int(color_input[2]))
        except (ValueError, TypeError):
            return default
    if isinstance(color_input, str):
        color_str = color_input.strip().lstrip("#")
        if color_str == "auto":
            return default
        if len(color_str) == 3:
            color_str = "".join([c * 2 for c in color_str])
        if len(color_str) == 6:
            try:
                return (
                    int(color_str[0:2], 16),
                    int(color_str[2:4], 16),
                    int(color_str[4:6], 16),
                )
            except ValueError:
                return default
    return default


def auto_detect_text_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x1, y1, x2, y2 = box
    crop = image.crop((x1, y1, x2, y2)).convert("L")
    arr = np.array(crop)
    if arr.size == 0:
        return (0, 0, 0)
    mean_bg = float(arr.mean())
    if mean_bg < RENDER_AUTO_TEXT_DARK_BG_THRESHOLD:
        return (255, 255, 255)
    return (0, 0, 0)


LOW_CONTRAST_RATIO = 3.0
LOW_CONTRAST_STROKE_WIDTH = 3


def _luminance(rgb) -> float:
    def channel(value: float) -> float:
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(float(v)) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _light_background_grey(image: Image.Image, box) -> float:
    grey = np.asarray(image.convert("L").crop(tuple(int(v) for v in box)))
    if grey.size == 0:
        return 255.0
    return float(np.percentile(grey, 75))


def get_font_path(font_name: str = "default") -> Path:
    try:
        return resolve_font_id(font_name)
    except (OSError, ValueError):
        font_dir = DEFAULT_FONT.parent
        for f in font_dir.glob("*.[tT][tT][fF]"):
            if f.stem.lower() == str(font_name).lower() or f.name.lower() == str(font_name).lower():
                return f.resolve()
    win_fonts = Path("C:/Windows/Fonts")
    if win_fonts.exists():
        win_fonts_resolved = win_fonts.resolve()
        sys_candidate = win_fonts / f"{font_name}.ttf"
        if sys_candidate.exists():
            if not sys_candidate.resolve().is_relative_to(win_fonts_resolved):
                return DEFAULT_FONT
            return sys_candidate
    return DEFAULT_FONT


def list_available_fonts() -> list[dict[str, str]]:
    fonts = [{"id": "default", "name": "Mặc định (Comic)", "category": "default", "tags": []}]
    seen = {"default"}
    for item in list_font_records():
        if item["id"] in seen:
            continue
        fonts.append(
            {
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "tags": item["tags"],
                "vietnamese": item["vietnamese"],
                "license": item["license"],
                "source_url": item["source_url"],
            }
        )
        seen.add(item["id"])

    win_fonts = Path("C:/Windows/Fonts")
    popular = [
        ("comic", "Comic Sans MS"),
        ("arial", "Arial"),
        ("calibri", "Calibri"),
        ("tahoma", "Tahoma"),
        ("times", "Times New Roman"),
        ("impact", "Impact"),
        ("segoeui", "Segoe UI"),
    ]
    if win_fonts.exists():
        for fname, dname in popular:
            if fname not in seen and (win_fonts / f"{fname}.ttf").exists():
                fonts.append({"id": fname, "name": dname, "category": "system", "tags": []})
                seen.add(fname)

    return fonts


def _calc_line_height(draw, font, stroke_w: int = 0) -> int:
    try:
        ascent, descent = font.getmetrics()
        base_h = ascent + descent
    except Exception:
        bbox = draw.textbbox((0, 0), "ÅgỶệJqỹ", font=font)
        base_h = bbox[3] - bbox[1]
    return int(max(base_h, 8) * RENDER_LINE_HEIGHT_FACTOR) + stroke_w * 2


def _line_start_x(
    draw, line, font, *, x1, pad, box_w, h_align, centred, curved,
    line_centres, draw_spans, i, line_count,
):
    line_w = draw.textbbox((0, 0), line, font=font)[2]
    if h_align == "left":
        return x1 + pad, line_w
    if h_align == "right":
        return x1 + pad + max(0, box_w - line_w), line_w
    if centred and curved and line_centres and len(line_centres) == line_count:
        start_x = x1 + pad + int(line_centres[i] - line_w / 2)
        if draw_spans is not None and i < len(draw_spans):
            lo, hi = draw_spans[i]
            return max(x1 + pad + lo, min(start_x, x1 + pad + max(lo, hi - line_w))), line_w
        return max(x1 + pad, min(start_x, x1 + pad + max(0, box_w - line_w))), line_w
    return x1 + pad + max(0, (box_w - line_w) // 2), line_w
