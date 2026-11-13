import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import DEFAULT_FONT
from app.detector.bubble_mask import (
    derive_bubble_mask,
    find_waist,
    mask_inscribed_rect,
    mask_row_spans,
    split_bubble_mask,
)
from app.render.font_catalog import list_font_records, resolve_font_id
from app.parameters import (
    FONT_CACHE_SIZE,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    RENDER_AUTO_STROKE_WIDTH,
    RENDER_AUTO_TEXT_DARK_BG_THRESHOLD,
    RENDER_DEFAULT_PADDING,
    RENDER_LINE_HEIGHT_FACTOR,
    RENDER_MIN_READABLE_FONT_SIZE,
    RENDER_PADDING_RATIO_MAX,
    RENDER_SAFE_CAPTION_EXPANSION,
    RENDER_SAFE_CAPTION_EXPANSION_RATIO,
    RENDER_STROKE_WIDTH_MAX,
)


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


_PUNCTUATION_FALLBACKS = {"\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2022": "·", "\u00ab": '"', "\u00bb": '"',
                          "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                          "\u27e8": "<", "\u27e9": ">", "\u3008": "<", "\u3009": ">", "\u300a": "<", "\u300b": ">",
                          "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"'}


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


def _line_start_x(draw, line, font, *, x1, pad, box_w, h_align, centred, curved,
                  line_centres, draw_spans, i, line_count):
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


def render_text_in_box(
    image: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    font_path=None,
    padding: int = RENDER_DEFAULT_PADDING,
    fill=None,
    font_size: int | str = "auto",
    is_bold: bool = False,
    font_name: str = "default",
    stroke_width: int | str = "auto",
    stroke_color: str = "auto",
    bg_color: str = "transparent",
    corner_radius: int = 0,
    shape: str = "rectangle",
    horizontal_align: str = "center",
    vertical_align: str = "middle",
    source_cap_px: int | None = None,
    enlarge: bool = False,
    min_source_share: float = 0.0,
    container: str | None = None,
    source_letter_box: tuple[int, int, int, int] | None = None,
    stats: dict | None = None,
    size_band_hi_cap: int | None = None,
    bubble_mask: dict | None = None,
    letters=None,
    letters_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    raw_w = x2 - x1
    raw_h = y2 - y1
    if raw_w <= 0 or raw_h <= 0 or not text or not str(text).strip():
        return image
    text = str(text).strip()

    pad = max(
        2,
        min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)),
    )
    box_w = raw_w - pad * 2
    box_h = raw_h - pad * 2

    h_align = str(horizontal_align or "").lower()
    if h_align not in ("left", "center", "right"):
        h_align = "center"
    v_align = str(vertical_align or "").lower()
    if v_align not in ("top", "middle", "bottom"):
        v_align = "middle"

    if font_path is None:
        font_path = get_font_path(font_name)
    else:
        font_path = Path(font_path)
    font_path, text = drawable(font_path, text)

    if fill is None or fill == "auto" or fill == "":
        bg_rgb = (
            parse_color(bg_color, default=None)
            if bg_color and str(bg_color).lower() not in ("transparent", "none", "")
            else None
        )
        if bg_rgb is not None:
            text_color = (255, 255, 255) if _luminance(bg_rgb) < 0.179 else (0, 0, 0)
        else:
            text_color = auto_detect_text_color(image, box)
        explicit_fill = False
    else:
        text_color = parse_color(fill, default=(0, 0, 0))
        explicit_fill = True

    if stroke_width is None or stroke_width == "auto" or stroke_width == "":
        stroke_w = RENDER_AUTO_STROKE_WIDTH if font_path else 1
        stroke_w_auto = True
    else:
        stroke_w_auto = False
        try:
            stroke_w = max(0, min(RENDER_STROKE_WIDTH_MAX, int(stroke_width)))
        except (ValueError, TypeError):
            stroke_w = RENDER_AUTO_STROKE_WIDTH
            stroke_w_auto = True

    if stroke_w_auto:
        if bg_color and bg_color not in ("transparent", "none", ""):
            bg_ref = parse_color(bg_color, default=(255, 255, 255))
        else:
            bg_grey = _light_background_grey(image, (x1, y1, x2, y2))
            bg_ref = (bg_grey, bg_grey, bg_grey)
        if _contrast(text_color, bg_ref) < LOW_CONTRAST_RATIO:
            stroke_w = max(stroke_w, LOW_CONTRAST_STROKE_WIDTH)

    if stroke_color is None or stroke_color == "auto" or stroke_color == "":
        stroke_auto = True
        luminance = (text_color[0] * 299 + text_color[1] * 587 + text_color[2] * 114) / 1000
        stroke_c = (0, 0, 0) if luminance > 128 else (255, 255, 255)
    else:
        stroke_auto = False
        stroke_c = parse_color(stroke_color, default=(0, 0, 0))

    draw = ImageDraw.Draw(image)

    target_size = None
    if isinstance(font_size, (int, float)) and font_size > 0:
        target_size = int(font_size)
    elif isinstance(font_size, str) and font_size.isdigit() and int(font_size) > 0:
        target_size = int(font_size)

    font_path_str = str(font_path)

    _mask, _mask_source = _resolve_mask(image, (x1, y1, x2, y2), container,
                                        source_letter_box, bubble_mask,
                                        letters, letters_box)
    curved = str(container or "").lower() in _CURVED_CONTAINERS or str(shape or "").lower() == "ellipse" \
        or (_mask is not None and _mask.get("kind") == "bubble")
    centred = h_align == "center" and v_align == "middle"

    _fixed_height_fallback = False
    if target_size is not None:
        _readable_fixed = int(RENDER_MIN_READABLE_FONT_SIZE
                              * max(1.0, image.size[0] / REFERENCE_PAGE_WIDTH))
        if target_size < _readable_fixed:
            target_size = _readable_fixed

    if target_size is not None and curved and centred:
        _fixed_font = get_font_object(font_path_str, target_size)
        _fixed_lines = _wrap_text(draw, text, _fixed_font, box_w, balance=True)
        _bold_extra = 1 if is_bold else 0
        if _mask is not None and _mask.get("kind") == "bubble":
            _bands = _mask_bands(_mask, (x1 + pad, y1 + pad, x2 - pad, y2 - pad))
            if _bands is not None:
                _widest = max((b - a for band in _bands for a, b in band), default=0)
                if _widest >= 0.4 * box_w:
                    _lh = _calc_line_height(draw, _fixed_font, stroke_w=stroke_w)
                    _anchor = int(_mask["cy"]) - (y1 + pad)
                    _top = _anchored_top(box_h, _lh * len(_fixed_lines), _anchor)
                    _limits = _band_line_widths(box_w, box_h, len(_fixed_lines), _lh, _bands, top=_top)
                    if _limits is None or len(_limits) != len(_fixed_lines) or not all(
                            draw.textbbox((0, 0), line, font=_fixed_font)[2] + 2 * stroke_w + _bold_extra <= limit
                            for line, limit in zip(_fixed_lines, _limits)):
                        target_size = None
                        _fixed_height_fallback = True
        elif not _lines_fit_ellipse(draw, _fixed_lines, _fixed_font, box_w, box_h, stroke_w):
            target_size = None
            _fixed_height_fallback = True

    if target_size is not None:
        _fixed_font = get_font_object(font_path_str, target_size)
        _fixed_lines = _wrap_text(draw, text, _fixed_font, box_w, balance=True)
        _bold_extra = 1 if is_bold else 0
        _whole_words = [w for ln in _fixed_lines for w in ln.split()] == text.split()
        _max_line_w = max((draw.textbbox((0, 0), line, font=_fixed_font)[2] for line in _fixed_lines), default=0)
        if (
            not _fixed_lines
            or not _whole_words
            or _calc_line_height(draw, _fixed_font, stroke_w=stroke_w) * len(_fixed_lines) > box_h
            or _max_line_w + 2 * stroke_w + _bold_extra > box_w
        ):
            target_size = None
            _fixed_height_fallback = True

    block_top = None
    if target_size is not None:
        actual_size = target_size
        font = get_font_object(font_path_str, actual_size)
        lines = _wrap_text(draw, text, font, box_w, balance=True)
        line_centres = None
        lines_source = "fixed_size"
        draw_spans = None
        fit_stats = {}
    else:
        opaque_caption = bool(bg_color and bg_color not in ("transparent", "none", ""))
        line_centres = []
        spans_holder: list = []
        fit_stats = {}
        actual_size, fitted_lines, fits_readably, (x1, y1, x2, y2) = auto_fit(
            draw, text, (x1, y1, x2, y2), image.size, font_path_str, padding=padding, stroke_w=stroke_w,
            source_cap_px=source_cap_px, enlarge=enlarge, opaque_caption=opaque_caption,
            curved=curved, image=image, centered=centred, container=container,
            source_letter_box=source_letter_box, centres_out=line_centres,
            stats_out=fit_stats, min_source_share=min_source_share,
            size_ceiling_px=size_band_hi_cap, bubble_mask=_mask, spans_out=spans_holder,
            letters=letters, letters_box=letters_box)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        if not fits_readably:
            raise ValueError(
                "Translation does not fit at the minimum readable font size; "
                "shorten the translation or enlarge the text region."
            )
        fit_stats["mask_source"] = _mask_source
        block_top = fit_stats.get("block_top")
        font = get_font_object(font_path_str, actual_size)
        lines = _wrap_text(draw, text, font, box_w, balance=True)
        lines_source = "balanced"
        draw_spans = spans_holder if spans_holder else None
        _lh_check = _calc_line_height(draw, font, stroke_w=stroke_w)
        _bold_extra = 1 if is_bold else 0
        if fitted_lines and (
            _lh_check * len(lines) > box_h
            or max((draw.textbbox((0, 0), ln, font=font)[2] for ln in lines), default=0) + 2 * stroke_w + _bold_extra > box_w
        ):
            lines = fitted_lines
            lines_source = "fitted"
        if centred and curved:
            if draw_spans is not None:
                if len(lines) != len(draw_spans):
                    lines = fitted_lines
                    lines_source = "fitted"
                elif (not _lines_fit_spans(draw, lines, font, draw_spans, stroke_w)
                        and len(fitted_lines) == len(draw_spans)
                        and _lines_fit_spans(draw, fitted_lines, font, draw_spans, stroke_w)):
                    lines = fitted_lines
                    lines_source = "fitted"
            elif not _lines_fit_ellipse(draw, lines, font, box_w, box_h, stroke_w):
                if _lines_fit_ellipse(draw, fitted_lines, font, box_w, box_h, stroke_w):
                    lines = fitted_lines
                    lines_source = "fitted"
    if stats is not None:
        stats["lines"] = len(lines)
        stats["size"] = actual_size
        stats["fit_box"] = (x1, y1, x2, y2)
        stats["fixed_size_height_fallback"] = _fixed_height_fallback
        if h_align == "center":
            stats["line_centring"] = "per_line" if line_centres else "box_centre"
        if target_size is None:
            stats["frame"] = fit_stats.get("frame")
            stats["frame_source"] = fit_stats.get("frame_source")
            stats["lobe"] = fit_stats.get("lobe")
            stats["fallbacks"] = fit_stats.get("fallbacks", [])
            stats["stages_skipped"] = fit_stats.get("stages_skipped", [])
            stats["band_narrowing"] = fit_stats.get("band_narrowing", {})
            stats["cap_relaxed"] = fit_stats.get("cap_relaxed", False)
            stats["size_ceiling_px"] = fit_stats.get("size_ceiling_px")
            stats["mask_source"] = fit_stats.get("mask_source")
            stats["floor_px"] = fit_stats.get("floor_px")
            stats["lines_source"] = lines_source

    shape = str(shape or "").lower()
    if shape not in ("rectangle", "ellipse"):
        shape = "rectangle"
    if bg_color and bg_color not in ("transparent", "none", ""):
        box_bg_c = parse_color(bg_color, default=(255, 255, 255))
        bg_bbox = (x1, y1, x2 - 1, y2 - 1)
        if shape == "ellipse":
            draw.ellipse(bg_bbox, fill=box_bg_c)
        else:
            r = max(0, min(int(corner_radius), int((min(raw_w, raw_h) - 1) // 2)))
            draw.rounded_rectangle(bg_bbox, radius=r, fill=box_bg_c)

    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)

    if v_align == "top":
        start_y = y1 + pad
    elif v_align == "bottom":
        start_y = y1 + pad + max(0, box_h - total_h)
    elif block_top is not None:
        start_y = int(block_top)
    else:
        start_y = y1 + pad + max(0, (box_h - total_h) // 2)
    if lines:
        first_top = draw.textbbox((0, 0), lines[0], font=font)[1] - stroke_w
        if start_y + first_top < y1 + 1:
            shift_down = min(
                max(0, (y2 - pad) - (start_y + total_h)),
                (y1 + 1) - (start_y + first_top),
            )
            start_y += shift_down

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)] if is_bold else [(0, 0)]

    fill_fallback = None
    if explicit_fill:
        if bg_color and bg_color not in ("transparent", "none", ""):
            local_bg = parse_color(bg_color, default=(255, 255, 255))
            if _contrast(text_color, local_bg) < LOW_CONTRAST_RATIO:
                bg_lum = (local_bg[0] * 299 + local_bg[1] * 587 + local_bg[2] * 114) / 1000
                text_color = (0, 0, 0) if bg_lum > 128 else (255, 255, 255)
                fill_fallback = "pale_to_auto"
        else:
            iw, ih = image.size
            grey = np.asarray(image.convert("L"))
            ink_rect = np.zeros((ih, iw), dtype=bool)
            ring_rect = np.zeros((ih, iw), dtype=bool)
            for i, line in enumerate(lines):
                sx, lw = _line_start_x(draw, line, font, x1=x1, pad=pad, box_w=box_w,
                                       h_align=h_align, centred=centred, curved=curved,
                                       line_centres=line_centres, draw_spans=draw_spans,
                                       i=i, line_count=len(lines))
                cy = start_y + i * line_height
                ink_rect[max(0, cy):min(ih, cy + line_height),
                         max(0, sx):min(iw, sx + lw)] = True
                ring_rect[max(0, cy - 8):min(ih, cy + line_height + 8),
                          max(0, sx - 8):min(iw, sx + lw + 8)] = True
            ring = ring_rect & ~ink_rect
            vals = grey[ring]
            if vals.size >= 50:
                bg_grey = float(np.percentile(vals, 75))
                if _contrast(text_color, (bg_grey, bg_grey, bg_grey)) < LOW_CONTRAST_RATIO:
                    text_color = auto_detect_text_color(image, (x1, y1, x2, y2))
                    fill_fallback = "pale_to_auto"
        if fill_fallback is not None and stroke_auto:
            luminance = (text_color[0] * 299 + text_color[1] * 587 + text_color[2] * 114) / 1000
            stroke_c = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        if stats is not None and fill_fallback is not None:
            stats["fill_fallback"] = fill_fallback

    ink_box = None
    for i, line in enumerate(lines):
        start_x, line_w = _line_start_x(draw, line, font, x1=x1, pad=pad, box_w=box_w,
                                        h_align=h_align, centred=centred, curved=curved,
                                        line_centres=line_centres, draw_spans=draw_spans,
                                        i=i, line_count=len(lines))
        cur_y = start_y + i * line_height
        if line_w > 0:
            bold = 1 if is_bold else 0
            gx0, gy0, gx1, gy1 = draw.textbbox(
                (start_x, cur_y), line, font=font, stroke_width=stroke_w)
            line_box = (
                min(start_x, gx0),
                min(cur_y, gy0),
                max(start_x + line_w, gx1) + bold,
                max(cur_y + line_height, gy1) + bold,
            )
            ink_box = line_box if ink_box is None else (
                min(ink_box[0], line_box[0]), min(ink_box[1], line_box[1]),
                max(ink_box[2], line_box[2]), max(ink_box[3], line_box[3]))
        for dx, dy in offsets:
            draw.text(
                (start_x + dx, cur_y + dy),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_w,
                stroke_fill=stroke_c,
            )
    if stats is not None and ink_box is not None:
        stats["ink_box"] = ink_box

    return image


MASK_BAND_PAD = 4


def _mask_bands(mask, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    bands = []
    for i in range(24):
        mid = y1 + ((2 * i + 1) * (y2 - y1)) // 48
        spans = mask_row_spans(mask, mid)
        bands.append([(a - x1 + MASK_BAND_PAD, b - x1 - MASK_BAND_PAD)
                      for a, b in spans if b - a > 2 * MASK_BAND_PAD])
    return bands


def _lines_fit_spans(draw, lines, font, spans, stroke_w: int) -> bool:
    if not lines:
        return True
    if spans is None or len(spans) != len(lines):
        return False
    return all(
        draw.textbbox((0, 0), line, font=font)[2] + 2 * stroke_w <= b - a
        for line, (a, b) in zip(lines, spans))


def _mask_centroid(mask):
    try:
        x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
        rows = mask["rows"]
    except (KeyError, TypeError, ValueError):
        return None
    height = max(1, y2 - y1)
    total = 0
    sum_x = 0.0
    sum_y = 0.0
    for i, row in enumerate(rows):
        band_y1 = y1 + (i * height) // len(rows)
        band_y2 = y1 + ((i + 1) * height) // len(rows)
        cy = (band_y1 + band_y2) / 2
        for a, b in row:
            w = int(b) - int(a)
            if w <= 0:
                continue
            sum_x += (x1 + (int(a) + int(b)) / 2) * w
            sum_y += cy * w
            total += w
    if total <= 0:
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    return (int(sum_x / total), int(sum_y / total))


def _trim_mask_to_letters(mask, box, letter_box, letters=None, letters_box=None):
    if not isinstance(mask, dict) or mask.get("kind") != "bubble":
        return mask
    if letter_box is None:
        return mask
    waist = None
    if letters is not None and letters_box is not None:
        arr = np.asarray(letters)
        lb = tuple(int(v) for v in letters_box)
        if (arr.ndim == 2 and arr.shape == (lb[3] - lb[1], lb[2] - lb[0])
                and arr.any()):
            waist = find_waist(mask, letters=arr, box=lb)
    if waist is None:
        waist = find_waist(mask)
    if waist is None:
        return mask
    pieces = split_bubble_mask(mask, waist)
    if pieces is None:
        return mask
    anchor = letter_box if letter_box is not None else box
    cx = (int(anchor[0]) + int(anchor[2])) // 2
    cy = (int(anchor[1]) + int(anchor[3])) // 2
    for piece in pieces:
        px1, py1, px2, py2 = (int(v) for v in piece["bbox"])
        if not (px1 <= cx < px2 and py1 <= cy < py2):
            continue
        if letter_box is not None:
            lx1, ly1, lx2, ly2 = (int(v) for v in letter_box)
            margin = 6
            if not (
                px1 - margin <= lx1
                and ly1 >= py1 - margin
                and lx2 <= px2 + margin
                and ly2 <= py2 + margin
            ):
                continue
        centroid = _mask_centroid(piece)
        if centroid is not None:
            piece["cx"], piece["cy"] = centroid
        return piece
    return mask


_DERIVED_MASK_CACHE: dict = {}
_DERIVED_MASK_CACHE_MAX = 64


def _derived_mask(image, box, letter_box):
    import hashlib

    arr = np.asarray(image)
    key = (tuple(int(v) for v in box),
           tuple(int(v) for v in letter_box) if letter_box is not None else None,
           arr.shape, hashlib.md5(arr.tobytes()).hexdigest())
    hit = _DERIVED_MASK_CACHE.get(key)
    if hit is not None:
        return hit
    bgr = arr[:, :, ::-1].copy() if arr.ndim == 3 else arr
    seed = None
    if letter_box is not None:
        seed = ((int(letter_box[0]) + int(letter_box[2])) // 2,
                (int(letter_box[1]) + int(letter_box[3])) // 2)
    mask = derive_bubble_mask(bgr, tuple(int(v) for v in box), seed=seed)
    if len(_DERIVED_MASK_CACHE) >= _DERIVED_MASK_CACHE_MAX:
        _DERIVED_MASK_CACHE.pop(next(iter(_DERIVED_MASK_CACHE), None), None)
    _DERIVED_MASK_CACHE[key] = mask
    return mask


def _resolve_mask(image, box, container, letter_box, bubble_mask,
                  letters=None, letters_box=None):
    mask = bubble_mask if isinstance(bubble_mask, dict) else None
    if mask is not None:
        return _trim_mask_to_letters(mask, box, letter_box, letters, letters_box), "object"
    if image is None or str(container or "").lower() not in ("bubble", "spiky"):
        return None, "none"
    derive_box = box
    if letter_box is not None:
        lx1, ly1, lx2, ly2 = (int(v) for v in letter_box)
        if box[0] <= lx1 and box[1] <= ly1 and lx2 <= box[2] and ly2 <= box[3]:
            derive_box = (lx1, ly1, lx2, ly2)
    derived = _derived_mask(image, derive_box, letter_box)
    if not isinstance(derived, dict) or derived.get("kind") != "bubble":
        return None, "none"
    return _trim_mask_to_letters(derived, box, letter_box, letters, letters_box), "derived"


def auto_fit(draw, text: str, box, image_size, font_path_str: str, *, padding: int = RENDER_DEFAULT_PADDING,
             stroke_w: int = RENDER_AUTO_STROKE_WIDTH, source_cap_px: int | None = None, enlarge: bool = False,
             opaque_caption: bool = False, curved: bool = False, image=None, centered: bool = True,
             container: str | None = None, source_letter_box: tuple[int, int, int, int] | None = None,
             centres_out: list | None = None, stats_out: dict | None = None,
             min_source_share: float = 0.0, size_ceiling_px: int | None = None,
             bubble_mask: dict | None = None, spans_out: list | None = None,
             letters=None,
             letters_box: tuple[int, int, int, int] | None = None):
    x1, y1, x2, y2 = (int(v) for v in box)
    raw_w, raw_h = x2 - x1, y2 - y1
    pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
    box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    attempt_box = (x1, y1, x2, y2)

    scale = max(1.0, image_size[0] / REFERENCE_PAGE_WIDTH)
    readable = int(RENDER_MIN_READABLE_FONT_SIZE * scale)
    maximum_size = int(MAX_FONT_SIZE * scale)
    enlarged_min = int(ENLARGED_MIN_FONT_SIZE * scale)

    source_match = None
    if source_cap_px:
        from app.render.source_size import matching_font_px
        source_match = matching_font_px(font_path_str, int(source_cap_px))
    if enlarge:
        floor = max(readable, enlarged_min)
    else:
        floor = readable

    mask, mask_source = _resolve_mask(image, (x1, y1, x2, y2), container,
                                      source_letter_box, bubble_mask,
                                      letters, letters_box)
    frame = (x1, y1, x2, y2)
    frame_source = "region_no_frame"
    fallbacks: list[str] = []
    use_mask = False
    if mask is not None and mask.get("kind") == "bubble":
        mx1, my1, mx2, my2 = (int(v) for v in mask["bbox"])
        mx1, my1 = max(0, mx1), max(0, my1)
        mx2, my2 = min(image_size[0], mx2), min(image_size[1], my2)
        if mx2 - mx1 >= 16 and my2 - my1 >= 16:
            frame = (mx1, my1, mx2, my2)
            frame_source = "mask"
            x1, y1, x2, y2 = frame
            use_mask = True
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2

    if not use_mask and str(container or "").lower() == "screen":
        inset = max(24, int(0.05 * min(raw_w, raw_h)))
        ix1, iy1, ix2, iy2 = x1 + inset, y1 + inset, x2 - inset, y2 - inset
        if image is not None:
            mfx1, mfx2, mfy1, mfy2, mx, my = _screen_frame(image, (x1, y1, x2, y2))
            if mx:
                ix1, ix2 = mfx1, mfx2
            if my:
                iy1, iy2 = mfy1, mfy2
            frame_source = ("screen_measured" if mx and my else
                            "screen_partial" if mx or my else "screen_inset")
        else:
            frame_source = "screen_inset"
        x1, y1, x2, y2 = ix1, iy1, ix2, iy2
        if x2 - x1 < 16 or y2 - y1 < 16:
            x1, y1, x2, y2 = (int(v) for v in box)
            frame_source = "screen_inset"
        else:
            frame = (x1, y1, x2, y2)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2

    if not use_mask and str(container or "").lower() == "box" and not curved and image is not None:
        measured = _box_frame(image, (x1, y1, x2, y2))
        if measured is not None:
            frame = measured
            frame_source = "box_rules"
            x1, y1, x2, y2 = measured
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        else:
            fallbacks.append("frame_unmeasured")

    def clamp_growth(bx1, by1, bx2, by2):
        if frame_source != "region_no_frame":
            fx1, fy1, fx2, fy2 = frame
            return (max(bx1, fx1), max(by1, fy1), min(bx2, fx2), min(by2, fy2))
        if curved or str(container or "").lower() == "box":
            ax1, ay1, ax2, ay2 = attempt_box
            return (max(bx1, ax1), max(by1, ay1), min(bx2, ax2), min(by2, ay2))
        return (bx1, by1, bx2, by2)

    if enlarge:
        grow_x, grow_y = int(round(raw_w * ENLARGE_GROW_RATIO)), int(round(raw_h * ENLARGE_GROW_RATIO))
        image_w, image_h = image_size
        x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
        x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
        x1, y1, x2, y2 = clamp_growth(x1, y1, x2, y2)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2

    def search(fx1, fy1, fx2, fy2):
        fw, fh = fx2 - fx1, fy2 - fy1
        fpad = max(2, min(padding, int(min(fw, fh) * RENDER_PADDING_RATIO_MAX)))
        fbw, fbh = fw - fpad * 2, fh - fpad * 2
        padded = (fx1 + fpad, fy1 + fpad, fx2 - fpad, fy2 - fpad)
        if fbw <= 0 or fbh <= 0:
            return floor, [], False, None, None, padded, floor, floor
        bands = None
        anchor_y = None
        if use_mask and mask is not None:
            bands = _mask_bands(mask, padded)
            anchor_y = int(mask["cy"]) - padded[1]
            if bands is not None:
                widest = max((b - a for band in bands for a, b in band), default=0)
                if widest < 0.4 * fbw:
                    bands = None
                    fallbacks.append("mask_degenerate_rectangle")
        probe = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                          minimum_size=readable, maximum_size=maximum_size,
                          curved=bool(bands is not None or curved), bands=bands)[0]
        fit_floor = floor if enlarge else max(floor, int(SOURCE_FLOOR_RATIO * probe))
        if enlarge:
            ceiling = maximum_size
        elif source_match:
            ceiling = max(source_match, int(SOURCE_FLOOR_RATIO * probe))
        else:
            ceiling = maximum_size
        ceiling = max(fit_floor, min(ceiling, SOURCE_MATCH_MAX_FONT_SIZE))
        if bands is not None:
            size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                          minimum_size=fit_floor, maximum_size=ceiling,
                                          curved=True, bands=bands, anchor_y=anchor_y)
            if not fits and fit_floor > readable:
                size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                              minimum_size=readable, maximum_size=fit_floor,
                                              curved=True, bands=bands, anchor_y=anchor_y)
        elif curved:
            size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                          minimum_size=fit_floor, maximum_size=ceiling,
                                          curved=True, bands=None)
            if not fits and fit_floor > readable:
                size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                              minimum_size=readable, maximum_size=fit_floor,
                                              curved=True, bands=None)
        else:
            size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                          minimum_size=fit_floor, maximum_size=ceiling,
                                          curved=False, bands=None)
            if not fits and fit_floor > readable:
                size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                              minimum_size=readable, maximum_size=fit_floor,
                                              curved=False, bands=None)
        return size, lines, fits, bands, anchor_y, padded, ceiling, fit_floor

    size, lines, fits_readably, bands, anchor_y, padded, ceiling, floor = search(x1, y1, x2, y2)
    if not fits_readably and opaque_caption and RENDER_SAFE_CAPTION_EXPANSION:
        grow_x = int(round(raw_w * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
        grow_y = int(round(raw_h * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
        image_w, image_h = image_size
        gx1, gy1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
        gx2, gy2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
        gx1, gy1, gx2, gy2 = clamp_growth(gx1, gy1, gx2, gy2)
        gsize, glines, gfits, gbands, ganchor, gpadded, gceiling, gfloor = search(gx1, gy1, gx2, gy2)
        if gfits:
            size, lines, fits_readably, bands, anchor_y, padded, ceiling, floor = \
                gsize, glines, gfits, gbands, ganchor, gpadded, gceiling, gfloor
            x1, y1, x2, y2 = gx1, gy1, gx2, gy2
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    if not fits_readably:
        if use_mask and mask is not None:
            rx1, ry1, rx2, ry2 = mask_inscribed_rect(mask)
        else:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        rw, rh = rx2 - rx1, ry2 - ry1
        rpad = max(2, min(padding, int(min(rw, rh) * RENDER_PADDING_RATIO_MAX)))
        rbw, rbh = rw - rpad * 2, rh - rpad * 2
        if rbw > 0 and rbh > 0:
            ok, flines = _fits(draw, text, rbw, rbh, font_path_str, floor, stroke_w,
                               curved=False, bands=None)
            rsize = floor
            if not ok and floor > readable:
                rsize, flines, ok = _fit_text(
                    draw, text, rbw, rbh, font_path_str, stroke_w=stroke_w,
                    minimum_size=readable, maximum_size=floor, curved=False, bands=None)
            if ok:
                size, lines, fits_readably = rsize, flines, True
                x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                bands, anchor_y = None, None
                padded = (rx1 + rpad, ry1 + rpad, rx2 - rpad, ry2 - rpad)
                raw_w, raw_h = x2 - x1, y2 - y1
                pad = rpad
                box_w, box_h = rbw, rbh
                fallbacks.append("outline_fit_failed_floor_rect")
    block_top = None
    if fits_readably and lines and bands is not None:
        font_c = get_font_object(font_path_str, size)
        line_height_c = _calc_line_height(draw, font_c, stroke_w=stroke_w)
        total_c = line_height_c * len(lines)
        if total_c <= box_h:
            top_c = _anchored_top(box_h, total_c, anchor_y)
            centres = _band_line_centres(box_w, box_h, len(lines), line_height_c, bands, top=top_c)
            spans = _band_line_spans(box_w, box_h, len(lines), line_height_c, bands, top=top_c)
            if centres is not None and spans is not None:
                if centres_out is not None:
                    centres_out.extend(centres)
                if spans_out is not None:
                    spans_out.extend(spans)
                block_top = y1 + pad + top_c
    if stats_out is not None:
        share_ok = True
        if source_match and not enlarge:
            share_ok = size >= int(SOURCE_FLOOR_RATIO * source_match)
        stats_out.update({
            "frame": frame,
            "frame_source": frame_source,
            "below_readable_floor": not fits_readably,
            "readable_floor_px": readable,
            "bands_used": bands is not None,
            "lobe": "mask" if use_mask and bands is not None else None,
            "fallbacks": list(fallbacks),
            "stages_skipped": [],
            "band_narrowing": {},
            "cap_ceiling": ceiling,
            "cap_relaxed": False,
            "size_ceiling_px": None,
            "share_ok": share_ok,
            "below_floor_ratio": False,
            "mask_source": mask_source,
            "floor_px": floor,
            "block_top": block_top,
        })
    return size, lines, fits_readably, (x1, y1, x2, y2)
ENLARGE_GROW_RATIO = 0.25
ENLARGED_MIN_FONT_SIZE = 22
SOURCE_FLOOR_RATIO = 0.6
REFERENCE_PAGE_WIDTH = 800
SOURCE_MATCH_MAX_FONT_SIZE = 200


def _ellipse_line_widths(box_w: int, box_h: int, line_count: int, line_height: int) -> list[float]:
    if line_count <= 0 or box_w <= 0 or box_h <= 0 or line_height <= 0:
        return []
    import math

    total_h = line_height * line_count
    top = max(0, (box_h - total_h) // 2)
    widths = []
    for i in range(line_count):
        a = top + i * line_height
        b = a + line_height
        edge = a if abs(a - box_h / 2) >= abs(b - box_h / 2) else b
        edge = min(max(edge, 0), box_h)
        widths.append(box_w * math.sqrt(max(0.0, 1.0 - (2 * edge / box_h - 1) ** 2)))
    return widths


_CURVED_CONTAINERS = frozenset({"bubble", "spiky"})


def _lines_fit_ellipse(draw, lines, font, box_w: int, box_h: int, stroke_w: int) -> bool:
    if not lines:
        return True
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    limits = _ellipse_line_widths(box_w, box_h, len(lines), line_height)
    if len(limits) != len(lines):
        return False
    return all(
        draw.textbbox((0, 0), line, font=font)[2] + 2 * stroke_w <= limit
        for line, limit in zip(lines, limits))


BOX_RULE_EXPAND = 64


def _box_frame(image: Image.Image, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    width, height = image.size
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if cx2 - cx1 < 40 or cy2 - cy1 < 40:
        return None
    grey = np.asarray(image.convert("L"))[cy1:cy2, cx1:cx2]
    if grey.size == 0:
        return None
    dark = grey < 110
    box_w, box_h = cx2 - cx1, cy2 - cy1

    def rule_groups(share, edge=2):
        groups = []
        start = None
        for i, v in enumerate(share):
            if v >= 0.8 and start is None:
                start = i
            elif v < 0.8 and start is not None:
                groups.append((start, i))
                start = None
        if start is not None:
            groups.append((start, len(share)))
        return [(a, b) for a, b in groups if b - a <= 16 and a >= edge and b <= len(share) - edge]

    col_share = dark.mean(axis=0)
    col_groups = rule_groups(col_share)
    if len(col_groups) < 2:
        col_groups = rule_groups(col_share, edge=0)
    fx1, fx2 = cx1, cx2
    if len(col_groups) >= 2:
        left, right = col_groups[0], col_groups[-1]
        if left[1] < box_w // 2 < right[0] and right[0] - left[1] >= max(40, box_w // 2):
            fx1, fx2 = cx1 + left[1], cx1 + right[0]
    row_share = dark.mean(axis=1)
    row_groups = rule_groups(row_share)
    if len(row_groups) < 2:
        row_groups = rule_groups(row_share, edge=0)
    fy1, fy2 = cy1, cy2
    if len(row_groups) >= 2:
        top, bottom = row_groups[0], row_groups[-1]
        if top[1] < box_h // 2 < bottom[0] and bottom[0] - top[1] >= max(40, box_h // 2):
            fy1, fy2 = cy1 + top[1], cy1 + bottom[0]
    first_fx = (fx1, fx2)
    first_fy = (fy1, fy2)
    full_grey = np.asarray(image.convert("L"))
    for axis in ("x", "y"):
        if axis == "x":
            ex1, ex2 = max(0, x1 - BOX_RULE_EXPAND), min(width, x2 + BOX_RULE_EXPAND)
            crop = full_grey[cy1:cy2, ex1:ex2] < 110
            share = crop.mean(axis=0) if crop.size else np.array([])
            lo, hi, extent = cx1 - ex1, cx2 - ex1, box_w
            strips = lambda a, b: full_grey[cy1:cy2, ex1 + a:ex1 + b]
            first_groups = col_groups
            unmeasured = first_fx == (cx1, cx2)
        else:
            ey1, ey2 = max(0, y1 - BOX_RULE_EXPAND), min(height, y2 + BOX_RULE_EXPAND)
            crop = full_grey[ey1:ey2, cx1:cx2] < 110
            share = crop.mean(axis=1) if crop.size else np.array([])
            lo, hi, extent = cy1 - ey1, cy2 - ey1, box_h
            strips = lambda a, b: full_grey[ey1 + a:ey1 + b, cx1:cx2]
            first_groups = row_groups
            unmeasured = first_fy == (cy1, cy2)
        groups = rule_groups(share) if share.size else []
        before = [g for g in groups if (g[0] + g[1]) // 2 < lo]
        after = [g for g in groups if (g[0] + g[1]) // 2 > hi]
        if unmeasured and not before and first_groups:
            g0 = first_groups[0]
            if g0[1] + lo <= (lo + hi) // 2:
                before = [(g0[0] + lo, g0[1] + lo)]
        if unmeasured and not after and first_groups:
            g1 = first_groups[-1]
            if g1[0] + lo >= (lo + hi) // 2:
                after = [(g1[0] + lo, g1[1] + lo)]
        if not before or not after:
            continue
        outer_lo, outer_hi = before[-1], after[0]
        if not (outer_lo[1] < (lo + hi) // 2 < outer_hi[0]):
            continue
        if outer_hi[0] - outer_lo[1] < max(40, extent // 2):
            continue
        strip_a = strips(outer_lo[1], lo) if lo > outer_lo[1] else None
        strip_b = strips(hi, outer_hi[0]) if outer_hi[0] > hi else None
        paper_ok = True
        for strip, trim_left in ((strip_a, True), (strip_b, False)):
            if strip is None:
                continue
            if axis == "x":
                strip = strip[:, 1:] if trim_left else strip[:, :-1]
            else:
                strip = strip[1:, :] if trim_left else strip[:-1, :]
            if strip.size == 0 or strip.shape[0] < 2 or strip.shape[1] < 2:
                continue
            if float((strip >= 180).mean()) < 0.8:
                paper_ok = False
                break
        if not paper_ok:
            continue
        if axis == "x":
            fx1, fx2 = ex1 + outer_lo[1], ex1 + outer_hi[0]
        else:
            fy1, fy2 = ey1 + outer_lo[1], ey1 + outer_hi[0]
    if (fx1, fy1, fx2, fy2) == (cx1, cy1, cx2, cy2):
        return None
    return (fx1, fy1, fx2, fy2)


def _screen_frame(image: Image.Image, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    width, height = image.size
    grey = np.asarray(image.convert("L"))
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 40 or box_h < 40:
        return (x1, x2, y1, y2, False, False)
    my1, my2 = y1 + box_h // 5, y2 - box_h // 5
    mx1, mx2 = x1 + box_w // 5, x2 - box_w // 5

    def groups(share):
        out = []
        start = None
        for i, v in enumerate(share):
            if v >= 0.6 and start is None:
                start = i
            elif v < 0.6 and start is not None:
                out.append((start, i))
                start = None
        if start is not None:
            out.append((start, len(share)))
        return [(a, b) for a, b in out if b - a <= 24]

    def side(share, lo, edge: int, from_start: bool):
        found = None
        for a, b in groups(share):
            ga, gb = lo + a, lo + b
            if from_start:
                if edge < gb <= edge + 64:
                    found = gb
                    break
            elif edge - 64 <= ga < edge:
                found = ga
        return found

    left = side((grey[my1:my2, max(0, x1 - 16):min(width, x1 + 64)] > 180).mean(axis=0),
                max(0, x1 - 16), x1, True)
    rx0 = max(0, x2 - 64)
    right = side((grey[my1:my2, rx0:min(width, x2 + 16)] > 180).mean(axis=0),
                 rx0, x2, False)
    top = side((grey[max(0, y1 - 16):min(height, y1 + 64), mx1:mx2] > 180).mean(axis=1),
               max(0, y1 - 16), y1, True)
    by0 = max(0, y2 - 64)
    bottom = side((grey[by0:min(height, y2 + 16), mx1:mx2] > 180).mean(axis=1),
                  by0, y2, False)
    fx1, fx2, measured_x = x1, x2, False
    if left is not None and right is not None and left < (x1 + x2) // 2 < right \
            and right - left >= 40:
        fx1, fx2, measured_x = left, right, True
    fy1, fy2, measured_y = y1, y2, False
    if top is not None and bottom is not None and top < (y1 + y2) // 2 < bottom \
            and bottom - top >= 40:
        fy1, fy2, measured_y = top, bottom, True
    return (fx1, fx2, fy1, fy2, measured_x, measured_y)


def _band_line_widths(box_w: int, box_h: int, line_count: int, line_height: int, bands, top: int | None = None) -> list[int] | None:
    spans = _band_line_spans(box_w, box_h, line_count, line_height, bands, top=top)
    if spans is None:
        return None
    return [b - a for a, b in spans]


def _band_line_spans(box_w: int, box_h: int, line_count: int, line_height: int, bands, top: int | None = None) -> list[tuple[int, int]] | None:
    if line_count <= 0 or box_w <= 0 or box_h <= 0 or line_height <= 0 or not bands:
        return None
    total_h = line_height * line_count
    if top is None:
        top = max(0, (box_h - total_h) // 2)
    count = len(bands)
    centre = box_w / 2
    spans = []
    for i in range(line_count):
        a = top + i * line_height
        b = a + line_height
        first = max(0, min(count - 1, a * count // box_h))
        last = max(0, min(count - 1, (b - 1) * count // box_h))
        span = None
        for band in bands[first:last + 1]:
            if not band:
                continue
            if span is None:
                holding = [iv for iv in band if iv[0] <= centre <= iv[1]]
                span = max(holding, key=lambda iv: iv[1] - iv[0]) if holding else max(band, key=lambda iv: iv[1] - iv[0])
                continue
            overlaps = [iv for iv in band if iv[0] < span[1] and span[0] < iv[1]]
            if not overlaps:
                return None
            pick = max(overlaps, key=lambda iv: min(iv[1], span[1]) - max(iv[0], span[0]))
            span = (max(span[0], pick[0]), min(span[1], pick[1]))
            if span[1] <= span[0]:
                return None
        if span is None:
            return None
        spans.append(span)
    return spans


def _band_line_centres(box_w: int, box_h: int, line_count: int, line_height: int, bands, top: int | None = None) -> list[float] | None:
    spans = _band_line_spans(box_w, box_h, line_count, line_height, bands, top=top)
    if spans is None:
        return None
    return [(a + b) / 2 for a, b in spans]


def _anchored_top(box_h: int, total_h: int, anchor_y: int | None) -> int:
    if anchor_y is None:
        return max(0, (box_h - total_h) // 2)
    return max(0, min(box_h - total_h, anchor_y - total_h // 2))


def _wrap_to_spans(draw, text: str, font, spans: list[tuple[int, int]],
                   stroke_w: int = 0) -> list[str] | None:
    raw_lines = [raw for raw in text.splitlines() if raw.strip()]
    if not raw_lines:
        return None
    out: list[str] = []
    span_idx = 0
    for raw_line in raw_lines:
        words = raw_line.split()
        idx = 0
        while idx < len(words):
            if span_idx >= len(spans):
                return None
            a, b = spans[span_idx]
            span_w = int(b) - int(a) - 2 * stroke_w
            if span_w <= 0:
                return None
            cur = words[idx]
            if draw.textbbox((0, 0), cur, font=font)[2] > span_w:
                return None
            idx += 1
            while idx < len(words):
                cand = f"{cur} {words[idx]}"
                if draw.textbbox((0, 0), cand, font=font)[2] <= span_w:
                    cur = cand
                    idx += 1
                else:
                    break
            out.append(cur)
            span_idx += 1
    if [w for ln in out for w in ln.split()] != text.split():
        return None
    return out


def _fits(draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int,
          curved: bool = False, bands=None, anchor_y: int | None = None) -> tuple[bool, list[str]]:
    font = get_font_object(font_path_str, size)
    lines = _wrap_text(draw, text, font, box_w)
    if not lines:
        return False, []
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)
    widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    whole = [word for line in lines for word in line.split()] == text.split()
    fits = whole and total_h <= box_h and max(widths) + 2 * stroke_w <= box_w
    if fits:
        if bands is not None:
            top0 = _anchored_top(box_h, line_height * len(lines), anchor_y)
            spans0 = _band_line_spans(box_w, box_h, len(lines), line_height, bands, top=top0)
            if spans0 is not None and all(
                w + 2 * stroke_w <= b - a for w, (a, b) in zip(widths, spans0)
            ):
                pass
            else:
                fitted_to_bands = None
                max_lines = box_h // line_height
                for k in range(len(lines), min(max_lines, len(lines) + 3) + 1):
                    top = _anchored_top(box_h, line_height * k, anchor_y)
                    spans = _band_line_spans(box_w, box_h, k, line_height, bands, top=top)
                    if spans is None:
                        continue
                    cand = _wrap_to_spans(draw, text, font, spans, stroke_w)
                    if cand is None or line_height * len(cand) > box_h:
                        continue
                    if len(cand) == k:
                        fitted_to_bands = cand
                        break
                    ctop = _anchored_top(box_h, line_height * len(cand), anchor_y)
                    cspans = _band_line_spans(box_w, box_h, len(cand), line_height, bands, top=ctop)
                    if cspans is not None and all(
                        draw.textbbox((0, 0), ln, font=font)[2] + 2 * stroke_w <= cb - ca
                        for ln, (ca, cb) in zip(cand, cspans)
                    ):
                        fitted_to_bands = cand
                        break
                if fitted_to_bands is not None:
                    lines = fitted_to_bands
                else:
                    fits = False
        elif curved:
            limits = _ellipse_line_widths(box_w, box_h, len(lines), line_height)
            fits = len(limits) == len(lines) and all(
                w + 2 * stroke_w <= limit for w, limit in zip(widths, limits))
    return fits, lines


def _scan_sizes(hi: int, lo: int, limit: int = 200):
    span = hi - lo
    if span <= limit:
        yield from range(hi, lo, -1)
        return
    step = max(2, -(-span // limit))
    last = None
    for size in range(hi, lo, -step):
        last = size
        yield size
    if last != lo + 1:
        yield lo + 1


def _fit_text(
    draw,
    text: str,
    box_w: int,
    box_h: int,
    font_path_str: str,
    stroke_w: int = RENDER_AUTO_STROKE_WIDTH,
    minimum_size: int = RENDER_MIN_READABLE_FONT_SIZE,
    maximum_size: int = MAX_FONT_SIZE,
    curved: bool = False,
    bands=None,
    anchor_y: int | None = None,
) -> tuple[int, list[str], bool]:
    minimum_size = max(MIN_FONT_SIZE, int(minimum_size))
    lo, hi = minimum_size, max(minimum_size, min(SOURCE_MATCH_MAX_FONT_SIZE, int(maximum_size)))
    best_size = minimum_size
    best_lines: list[str] = []

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, lines = _fits(draw, text, box_w, box_h, font_path_str, mid, stroke_w)
        if ok:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    hi_max = max(minimum_size, min(SOURCE_MATCH_MAX_FONT_SIZE, int(maximum_size)))
    for size in range(best_size + 1, hi_max + 1):
        ok, lines = _fits(draw, text, box_w, box_h, font_path_str, size, stroke_w)
        if ok:
            best_size = size
            best_lines = lines

    if (curved or bands is not None) and best_lines:
        for size in _scan_sizes(best_size, minimum_size - 1):
            ok, lines = _fits(draw, text, box_w, box_h, font_path_str, size, stroke_w,
                              curved=curved, bands=bands, anchor_y=anchor_y)
            if ok:
                return size, lines, True
        min_font = get_font_object(font_path_str, minimum_size)
        return minimum_size, _wrap_text(draw, text, min_font, box_w), False

    if not best_lines:
        min_font = get_font_object(font_path_str, minimum_size)
        best_lines = _wrap_text(draw, text, min_font, box_w)
    fits = bool(best_lines) and _fits(
        draw, text, box_w, box_h, font_path_str, best_size, stroke_w
    )[0]
    return best_size, best_lines, fits


def _greedy_groups(draw, words: list[str], font, width: int) -> list[list[str]]:
    groups, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current.append(word)
        else:
            groups.append(current)
            current = [word]
    if current:
        groups.append(current)
    return groups


def _greedy_lines(draw, words: list[str], font, width: int) -> list[str]:
    return [" ".join(group) for group in _greedy_groups(draw, words, font, width)]


_VI_WORDS = Path(__file__).with_name("vi_words.txt")
_SYLLABLE = re.compile(r"(\W*)(\w+)(\W*)")


@lru_cache(maxsize=1)
def _vi_words() -> frozenset[str]:
    try:
        return frozenset(line for line in _VI_WORDS.read_text(encoding="utf-8").splitlines() if line)
    except OSError:
        return frozenset()


def _word_units(words: list[str]) -> list[str]:
    known, units, i = _vi_words(), [], 0
    while i < len(words):
        for n in (4, 3, 2, 1):
            part = words[i:i + n]
            if len(part) < n:
                continue
            if n == 1:
                break
            parsed = [_SYLLABLE.fullmatch(word) for word in part]
            if all(parsed) and all(not m.group(1) for m in parsed[1:]) and all(not m.group(3) for m in parsed[:-1]) \
                    and " ".join(m.group(2) for m in parsed).lower() in known:
                break
        units.append(" ".join(words[i:i + n]))
        i += n
    return units


def _balanced_lines(draw, words: list[str], font, box_w: int) -> list[str]:
    lines = _greedy_lines(draw, words, font, box_w)
    if len(lines) < 2:
        return lines
    units = _word_units(words)
    if len(units) < len(words) and all(draw.textbbox((0, 0), unit, font=font)[2] <= box_w for unit in units) \
            and len(_greedy_lines(draw, units, font, box_w)) == len(lines):
        words = units
    lo = max(draw.textbbox((0, 0), word, font=font)[2] for word in words)
    hi = box_w
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_greedy_lines(draw, words, font, mid)) <= len(lines):
            hi = mid
        else:
            lo = mid + 1
    groups = _greedy_groups(draw, words, font, hi)
    if len(groups) >= 2 and len(groups[-1]) == 1 and len(groups[-2]) >= 2:
        moved = groups[-2][-1]
        new_last = " ".join([moved] + groups[-1])
        if draw.textbbox((0, 0), new_last, font=font)[2] <= box_w:
            groups[-2] = groups[-2][:-1]
            groups[-1] = [moved] + groups[-1]
    return [" ".join(group) for group in groups]


def _wrap_text(draw, text: str, font, box_w: int, *, balance: bool = False,
               join_lines: bool = False) -> list[str]:
    if join_lines:
        text = " ".join(text.split())
    raw_lines = text.splitlines()
    all_wrapped = []
    for raw_line in raw_lines:
        if not raw_line.strip():
            continue
        words = raw_line.split()
        if balance and all(draw.textbbox((0, 0), word, font=font)[2] <= box_w for word in words):
            all_wrapped.extend(_balanced_lines(draw, words, font, box_w))
            continue
        current = ""
        for word in words:
            if draw.textbbox((0, 0), word, font=font)[2] > box_w:
                if current:
                    all_wrapped.append(current)
                    current = ""
                for char in word:
                    candidate = f"{current}{char}"
                    if draw.textbbox((0, 0), candidate, font=font)[2] <= box_w or not current:
                        current = candidate
                    else:
                        all_wrapped.append(current)
                        current = char
            else:
                candidate = f"{current} {word}".strip()
                w = draw.textbbox((0, 0), candidate, font=font)[2]
                if w <= box_w or not current:
                    current = candidate
                else:
                    all_wrapped.append(current)
                    current = word
        if current:
            all_wrapped.append(current)
    return all_wrapped
