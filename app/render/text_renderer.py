import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import DEFAULT_FONT
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


# Typographic punctuation many comic fonts lack, with the plain form they do have.
_PUNCTUATION_FALLBACKS = {"\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2022": "·", "\u00ab": '"', "\u00bb": '"',
                          "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                          # Brackets round names and skills in webtoons and CJK sources.
                          "\u27e8": "<", "\u27e9": ">", "\u3008": "<", "\u3009": ">", "\u300a": "<", "\u300b": ">",
                          "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"'}


def _plain_punctuation(font_path, text: str) -> str:
    font_path_str = str(font_path)
    return "".join(
        _PUNCTUATION_FALLBACKS[char] if char in _PUNCTUATION_FALLBACKS and not _draws_char(font_path_str, char) else char
        for char in text
    )


def drawable(font_path, text: str) -> tuple[Path, str]:
    """The font and text that are actually drawn: punctuation the font lacks is replaced, else the default font."""
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
    """False when some character would come out as the font's missing-glyph box."""
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


LOW_CONTRAST_RATIO = 3.0  # below this text-to-background contrast the text gets an outline
LOW_CONTRAST_STROKE_WIDTH = 3


def _luminance(rgb) -> float:
    def channel(value: float) -> float:
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(float(v)) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    """WCAG contrast ratio between two RGB colours."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _mean_color(image: Image.Image, box) -> tuple[int, int, int]:
    crop = np.asarray(image.convert("RGB").crop(tuple(int(v) for v in box)))
    if crop.size == 0:
        return (255, 255, 255)
    return tuple(int(v) for v in crop.reshape(-1, 3).mean(axis=0))


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
    size_is_hint: bool = False,
    taper: bool = False,
    _lobe_run=None,
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
        text_color = auto_detect_text_color(image, box)
    else:
        text_color = parse_color(fill, default=(0, 0, 0))

    if stroke_width is None or stroke_width == "auto" or stroke_width == "":
        stroke_w = RENDER_AUTO_STROKE_WIDTH if font_path else 1
    else:
        try:
            stroke_w = max(0, min(RENDER_STROKE_WIDTH_MAX, int(stroke_width)))
        except (ValueError, TypeError):
            stroke_w = RENDER_AUTO_STROKE_WIDTH

    if _contrast(text_color, _mean_color(image, (x1, y1, x2, y2))) < LOW_CONTRAST_RATIO:
        # Letters close to the background colour get a thick contrasting outline.
        stroke_w = max(stroke_w, LOW_CONTRAST_STROKE_WIDTH)

    if stroke_color is None or stroke_color == "auto" or stroke_color == "":
        luminance = (text_color[0] * 299 + text_color[1] * 587 + text_color[2] * 114) / 1000
        stroke_c = (0, 0, 0) if luminance > 128 else (255, 255, 255)
    else:
        stroke_c = parse_color(stroke_color, default=(0, 0, 0))

    draw = ImageDraw.Draw(image)

    # Tapered lines follow the interior at their own height; without a bright
    # interior the ellipse stays in charge. A lobe render already knows its run.
    run_at = _lobe_run if _lobe_run is not None else (
        _interior_run_at(image, (x1, y1, x2, y2)) if taper else None)

    target_size = None
    if isinstance(font_size, (int, float)) and font_size > 0:
        target_size = int(font_size)
    elif isinstance(font_size, str) and font_size.isdigit() and int(font_size) > 0:
        target_size = int(font_size)

    font_path_str = str(font_path)

    # A double balloon with one paragraph per lobe letters each paragraph in
    # its own lobe, at one shared size; anything else keeps the single block.
    paras = text.split("\n")
    if (taper and _lobe_run is None and (not bg_color or bg_color in ("transparent", "none", ""))
            and len(paras) >= 2 and all(p.strip() for p in paras)):
        lobes = _detect_lobes(image, (x1, y1, x2, y2))
        if lobes and len(lobes) == len(paras):
            size, lobes = _fit_lobes(draw, paras, lobes, font_path_str, stroke_w, padding,
                                     image.size[0])
            if size is not None and (target_size is None or target_size <= size):
                if target_size is not None:
                    size = target_size
                elif source_cap_px and not enlarge:
                    from app.render.source_size import SIZE_SLACK, matching_font_px
                    readable = int(RENDER_MIN_READABLE_FONT_SIZE * max(1.0, image.width / REFERENCE_PAGE_WIDTH))
                    source_size = matching_font_px(font_path_str, int(source_cap_px))
                    size = min(size, max(readable, int(SOURCE_FLOOR_RATIO * size), int(source_size * SIZE_SLACK)))
                    if min_source_share and size < min_source_share * source_size:
                        raise ValueError("Translation fits only well below the source letter size")
                for para, (lbox, lrun) in zip(paras, lobes):
                    render_text_in_box(image, para, lbox, font_path=font_path, padding=padding,
                                       fill=fill, font_size=size, is_bold=is_bold, font_name=font_name,
                                       stroke_width=stroke_width, stroke_color=stroke_color,
                                       bg_color=bg_color, corner_radius=corner_radius, shape=shape,
                                       horizontal_align=horizontal_align, vertical_align="middle",
                                       source_cap_px=None, enlarge=False, min_source_share=0.0,
                                       size_is_hint=False, taper=True, _lobe_run=lrun)
                return image

    if target_size is not None:
        actual_size = target_size
        widths_for = (lambda count, lh: _run_widths(run_at, y1 + pad, box_w, box_h, lh, count)) \
            if run_at else None
        fits, fit_lines = _fits(draw, text, box_w, box_h, font_path_str, actual_size, stroke_w, taper=taper,
                                widths_for=widths_for)
        if not fits and size_is_hint:
            readable = int(RENDER_MIN_READABLE_FONT_SIZE * max(1.0, image.width / REFERENCE_PAGE_WIDTH))
            actual_size, fit_lines, fits = _fit_text(draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
                                                     minimum_size=readable, maximum_size=target_size, taper=taper)
        if not fits:
            raise ValueError("Translation does not fit at the requested font size")
        if size_is_hint and min_source_share and actual_size < min_source_share * target_size:
            raise ValueError("Translation fits only well below the OCR font size")
        font = get_font_object(font_path_str, actual_size)
        # Tapered lines stay as fitted: a full-width re-wrap would undo the narrowing.
        lines = fit_lines if taper else _wrap_text(draw, text, font, max(1, box_w - 2 * stroke_w), balance=True)
    else:
        opaque_caption = bool(bg_color and bg_color not in ("transparent", "none", ""))
        actual_size, lines, fits_readably, (x1, y1, x2, y2) = auto_fit(
            draw, text, (x1, y1, x2, y2), image.size, font_path_str, padding=padding, stroke_w=stroke_w,
            source_cap_px=source_cap_px, enlarge=enlarge, opaque_caption=opaque_caption, taper=taper,
            run_at=run_at)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        if not fits_readably:
            raise ValueError(
                "Translation does not fit at the minimum readable font size; "
                "shorten the translation or enlarge the text region."
            )
        if source_cap_px and not enlarge and min_source_share:
            from app.render.source_size import matching_font_px
            if actual_size < min_source_share * matching_font_px(font_path_str, int(source_cap_px)):
                raise ValueError("Translation fits only well below the source letter size")
        font = get_font_object(font_path_str, actual_size)
        # The same number of lines, evened out so no word is left alone.
        # Tapered lines stay as fitted: a full-width re-wrap would undo the narrowing.
        if not taper:
            lines = _wrap_text(draw, text, font, max(1, box_w - 2 * stroke_w), balance=True)

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
    bounds = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w) for line in lines]
    bold_extra = 1 if is_bold else 0
    top = min(i * line_height + bbox[1] for i, bbox in enumerate(bounds))
    bottom = max(i * line_height + bbox[3] for i, bbox in enumerate(bounds)) + bold_extra
    total_h = bottom - top

    if v_align == "top":
        start_y = y1 + pad - top
    elif v_align == "bottom":
        start_y = y1 + pad + max(0, box_h - total_h) - top
    else:
        start_y = y1 + pad + max(0, (box_h - total_h) // 2) - top

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)] if is_bold else [(0, 0)]

    for i, line in enumerate(lines):
        left, _, right, _ = bounds[i]
        line_w = right - left + bold_extra
        cur_y = start_y + i * line_height
        if h_align == "left":
            start_x = x1 + pad - left
        elif h_align == "right":
            start_x = x1 + pad + max(0, box_w - line_w) - left
        else:
            start_x = x1 + pad + max(0, (box_w - line_w) // 2) - left
            if run_at:
                # A line sits on the interior run at its own height, not the box
                # centre: a double balloon's lobes sit off-centre, and the box
                # centre would push their lines across the outline.
                run = run_at(int(round(cur_y + (bounds[i][1] + bounds[i][3]) / 2)))
                if run and line_w <= run[1] - run[0] + 1:
                    start_x = (run[0] + run[1]) // 2 - line_w // 2 - left
        for dx, dy in offsets:
            draw.text(
                (start_x + dx, cur_y + dy),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_w,
                stroke_fill=stroke_c,
            )

    return image


def auto_fit(draw, text: str, box, image_size, font_path_str: str, *, padding: int = RENDER_DEFAULT_PADDING,
             stroke_w: int = RENDER_AUTO_STROKE_WIDTH, source_cap_px: int | None = None, enlarge: bool = False,
             opaque_caption: bool = False, taper: bool = False, run_at=None):
    """The auto font size, wrapped lines, whether it fits readably and the box it was fitted in, as the renderer letters it."""
    x1, y1, x2, y2 = (int(v) for v in box)
    raw_w, raw_h = x2 - x1, y2 - y1
    pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
    box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    # Line widths follow the interior at each line's height when the image offers
    # one; the widths are rebuilt whenever the box grows below.
    widths_for = (lambda bw, bh, oy: (lambda count, lh: _run_widths(run_at, oy, bw, bh, lh, count))) \
        if run_at else None
    # Size limits are set for an 800 px wide page and grow with wider pages.
    scale = max(1.0, image_size[0] / REFERENCE_PAGE_WIDTH)
    readable, maximum_size = int(RENDER_MIN_READABLE_FONT_SIZE * scale), int(MAX_FONT_SIZE * scale)
    enlarged_min = int(ENLARGED_MIN_FONT_SIZE * scale)
    # Auto size never letters bigger than the source did.
    if enlarge:
        # Text flagged too small to read grows its area and may exceed the source size.
        grow_x, grow_y = int(round(raw_w * ENLARGE_GROW_RATIO)), int(round(raw_h * ENLARGE_GROW_RATIO))
        image_w, image_h = image_size
        x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
        x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    elif source_cap_px:
        from app.render.source_size import SIZE_SLACK, matching_font_px
        # A misread source size must not shrink the text far below what the box holds.
        box_fit = _fit_text(draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
                            minimum_size=readable, maximum_size=maximum_size, taper=taper,
                            widths_for=widths_for(box_w, box_h, y1 + pad) if widths_for else None)[0]
        maximum_size = max(readable, int(SOURCE_FLOOR_RATIO * box_fit),
                           int(matching_font_px(font_path_str, int(source_cap_px)) * SIZE_SLACK))
    actual_size, lines, fits_readably = _fit_text(
        draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
        minimum_size=enlarged_min if enlarge else readable, maximum_size=maximum_size, taper=taper,
        widths_for=widths_for(box_w, box_h, y1 + pad) if widths_for else None,
    )
    if enlarge and not fits_readably:
        actual_size, lines, fits_readably = _fit_text(
            draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
            minimum_size=readable, maximum_size=maximum_size, taper=taper,
            widths_for=widths_for(box_w, box_h, y1 + pad) if widths_for else None,
        )
    if not fits_readably and opaque_caption and RENDER_SAFE_CAPTION_EXPANSION:
        grow_x = int(round(raw_w * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
        grow_y = int(round(raw_h * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
        image_w, image_h = image_size
        x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
        x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        actual_size, lines, fits_readably = _fit_text(
            draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
            minimum_size=readable, maximum_size=maximum_size, taper=taper,
            widths_for=widths_for(box_w, box_h, y1 + pad) if widths_for else None,
        )
    return actual_size, lines, fits_readably, (x1, y1, x2, y2)


ENLARGE_GROW_RATIO = 0.25  # each side of a region flagged enlarge grows by this share
ENLARGED_MIN_FONT_SIZE = 22
SOURCE_FLOOR_RATIO = 0.6  # source-matched text never drops below this share of the box-fit size
REFERENCE_PAGE_WIDTH = 800  # font size limits are for a page this wide
SOURCE_MATCH_MAX_FONT_SIZE = 200  # large source lettering may be matched past MAX_FONT_SIZE


TAPER_SAFETY = 0.95  # a curved outline's chord is measured, then kept this far inside it


def _ellipse_chord(box_w: int, box_h: int, line_height: int, count: int, i: int) -> int:
    """Line ``i``'s width on an ellipse inscribed in the box; lines near the poles get narrower."""
    if count <= 0 or box_h <= 0:
        return 1
    total_h = line_height * count
    top = (box_h - total_h) / 2
    center = top + (i + 0.5) * line_height
    t = max(-1.0, min(1.0, (center - box_h / 2) / (box_h / 2)))
    return max(1, int(box_w * TAPER_SAFETY * math.sqrt(max(0.0, 1 - t * t))))


def _chord_widths(count: int, box_w: int, box_h: int, line_height: int) -> list[int]:
    """Each line's width on an ellipse inscribed in the box; lines near the poles get narrower."""
    return [_ellipse_chord(box_w, box_h, line_height, count, i) for i in range(count)]


def _interior_run_at(image, box):
    """Bright interior run at a row, for tapered text, or None when the box holds no bright balloon.

    The run holding the box centre, else the widest bounded run in the box: a
    double balloon's lobes sit off-centre, so a line belongs to the run at its
    own height, not to the box centre. A run counts only when an outline bounds
    it on both sides; page background runs to the image edge and is ignored, and
    a box whose centre column is mostly dark is a dark balloon, not a bright one.
    """
    import numpy as np

    x1, y1, x2, y2 = (int(v) for v in box)
    w, _h = image.size
    cx = (x1 + x2) // 2
    top = max(0, y1)
    try:
        gray = np.asarray(image.crop((0, top, w, y2)).convert("L"))
    except Exception:
        return None
    if gray.shape[0] == 0 or not (0 <= cx < w):
        return None
    if float((gray[:, cx] > 210).mean()) < 0.5:
        return None

    def run_at(y):
        row_i = y - top
        if row_i < 0 or row_i >= gray.shape[0]:
            return None
        row = gray[row_i] > 210
        runs, start, last = [], None, None
        for x, on in enumerate(row):
            if on:
                if start is None:
                    start = x
                last = x
            elif start is not None and x - last > 3:
                runs.append((start, last))
                start = None
        if start is not None:
            runs.append((start, last))
        bounded = [(a, b) for a, b in runs if a > 0 and b < len(row) - 1]
        for a, b in bounded:
            if a <= cx <= b:
                return (a, b)
        inside = [(a, b) for a, b in bounded if b >= x1 and a <= x2]
        return max(inside, key=lambda r: r[1] - r[0]) if inside else None

    return run_at


def _run_widths(run_at, origin_y, box_w, box_h, line_height, count):
    """Each line's width from the interior run at its height, else the ellipse chord."""
    if count <= 0:
        return []
    total_h = line_height * count
    top = (box_h - total_h) / 2
    widths = []
    for i in range(count):
        run = run_at(int(round(origin_y + top + (i + 0.5) * line_height))) if run_at else None
        if run:
            widths.append(max(1, int((run[1] - run[0] + 1) * TAPER_SAFETY)))
        else:
            widths.append(_ellipse_chord(box_w, box_h, line_height, count, i))
    return widths


LOBE_SECOND_PEAK = 0.55  # a second peak this small is a bump on one balloon, not a lobe
LOBE_NECK = 0.80  # the pinch between lobes sits this far inside the smaller lobe's radius
LOBE_MIN_RADIUS = 20  # a lobe smaller than a readable line holds no paragraph
LOBE_MIN_SIZE = 40  # a lobe box smaller than this is not a place to letter


def _bright_component(image, box):
    """The enclosed bright region overlapping ``box``, or None.

    Page background is bright too, but it touches the image edge; a balloon
    interior is fenced in by its outline, so it does not.
    """
    import cv2

    gray = np.asarray(image.convert("L"))
    bright = (gray > 210).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    x1, y1, x2, y2 = (int(v) for v in box)
    img_h, img_w = gray.shape[:2]
    best, best_count = None, 0
    for lab in range(1, len(stats)):
        bx, by, bw, bh, _ = (int(v) for v in stats[lab])
        if bx <= 0 or by <= 0 or bx + bw >= img_w or by + bh >= img_h:
            continue
        comp = labels == lab
        count = int(comp[y1:y2, x1:x2].sum())
        if count > best_count:
            best, best_count = comp, count
    if best is None or best_count == 0:
        return None
    flood = np.zeros((best.shape[0] + 2, best.shape[1] + 2), dtype=np.uint8)
    canvas = (1 - best.astype(np.uint8)) * 255
    cv2.floodFill(canvas, flood, (0, 0), 0)
    return best | (canvas == 255)


def _detect_lobes(image, box):
    """A double balloon's lobes, as (box, run) pairs top to bottom, or None.

    The lobes are the interior's two large inscribed circles, joined by a
    pinch: the distance transform peaks twice, and the saddle on the line
    between the peaks sits well inside the smaller peak's radius. A single
    balloon peaks once and falls through, as does a shape with no real pinch.
    """
    import cv2

    try:
        comp = _bright_component(image, box)
    except Exception:
        return None
    if comp is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    sub = dist[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    py, px = np.unravel_index(np.argmax(sub), sub.shape)
    p1 = (int(px + x1), int(py + y1))
    d1 = float(sub[py, px])
    if d1 < LOBE_MIN_RADIUS:
        return None
    suppressed = dist.copy()
    yy, xx = np.ogrid[:dist.shape[0], :dist.shape[1]]
    suppressed[(xx - p1[0]) ** 2 + (yy - p1[1]) ** 2 <= d1 ** 2] = 0
    sub2 = suppressed[y1:y2, x1:x2]
    py2, px2 = np.unravel_index(np.argmax(sub2), sub2.shape)
    p2 = (int(px2 + x1), int(py2 + y1))
    d2 = float(sub2[py2, px2])
    if d2 < LOBE_SECOND_PEAK * d1:
        return None
    saddle = min(float(dist[int(round(p1[1] + (p2[1] - p1[1]) * t)),
                            int(round(p1[0] + (p2[0] - p1[0]) * t))])
                 for t in np.linspace(0, 1, 101))
    if saddle > LOBE_NECK * min(d1, d2):
        return None
    peaks = sorted(((p1, d1), (p2, d2)), key=lambda pd: pd[0][1])
    if abs(peaks[1][0][1] - peaks[0][0][1]) < 0.25 * math.hypot(p2[0] - p1[0], p2[1] - p1[1]):
        return None
    ys, xs = np.where(comp)
    near = [np.hypot(xs - pk[0], ys - pk[1]) for pk, _ in peaks]
    lobes = []
    for i, (pk, _d) in enumerate(peaks):
        own = near[i] <= near[1 - i]
        ax, ay = xs[own], ys[own]
        if not len(ax):
            return None
        lx1, ly1 = max(x1, int(ax.min())), max(y1, int(ay.min()))
        lx2, ly2 = min(x2, int(ax.max()) + 1), min(y2, int(ay.max()) + 1)
        if lx2 - lx1 < LOBE_MIN_SIZE or ly2 - ly1 < LOBE_MIN_SIZE:
            return None
        order = np.argsort(ay, kind="stable")
        say, sax = ay[order], ax[order]
        uniq, starts = np.unique(say, return_index=True)
        runs = {}
        for j, row in enumerate(uniq):
            seg = sax[starts[j]:starts[j + 1] if j + 1 < len(uniq) else len(sax)]
            runs[int(row)] = (int(seg.min()), int(seg.max()))
        lobes.append(((lx1, ly1, lx2, ly2), (lambda r: (lambda y: r.get(y)))(runs)))
    return lobes


def _windowed_run(run, half):
    """The run shared by every row a line covers either side of its centre.

    A line's glyphs reach above and below its centre row, where a neck
    narrows fast; centring on the centre row alone lets descenders cross it.
    """
    def at(y):
        lo = hi = None
        yy = y - half
        while yy <= y + half:
            r = run(yy)
            if r is None:
                return None
            lo = r[0] if lo is None else max(lo, r[0])
            hi = r[1] if hi is None else min(hi, r[1])
            if hi < lo:
                return None
            yy += 4
        r = run(y)
        if r is None:
            return None
        lo, hi = max(lo, r[0]), min(hi, r[1])
        return (lo, hi) if hi >= lo else None

    return at


def _lobe_widths(run, origin_y, box_w, box_h, line_height, count):
    """Each line's width from its lobe's run at its height, else too narrow to fit.

    Unlike the ellipse fallback, a missing run here must not widen the line:
    no run means the line has left its lobe.
    """
    if count <= 0:
        return []
    total_h = line_height * count
    top = (box_h - total_h) / 2
    widths = []
    for i in range(count):
        r = run(int(round(origin_y + top + (i + 0.5) * line_height))) if run else None
        widths.append(max(1, int((r[1] - r[0] + 1) * TAPER_SAFETY)) if r else 1)
    return widths


def _lobe_common_size(draw, paras, lobes, font_path_str, stroke_w, padding, image_w):
    """Largest size fitting every paragraph in its own lobe, or None.

    One size for all paragraphs: lobes lettered at different sizes would read
    as two voices, not one block split across lobes.
    """
    scale = max(1.0, image_w / REFERENCE_PAGE_WIDTH)
    readable = int(RENDER_MIN_READABLE_FONT_SIZE * scale)
    maximum = int(MAX_FONT_SIZE * scale)
    lo, hi, best = readable, maximum, None
    while lo <= hi:
        mid = (lo + hi) // 2
        ok = True
        for para, (lbox, lrun) in zip(paras, lobes):
            lx1, ly1, lx2, ly2 = lbox
            pad = max(2, min(padding, int(min(lx2 - lx1, ly2 - ly1) * RENDER_PADDING_RATIO_MAX)))
            bw, bh = (lx2 - lx1) - pad * 2, (ly2 - ly1) - pad * 2
            widths_for = (lambda count, lh, _run=lrun, _oy=ly1 + pad, _bw=bw, _bh=bh:
                          _lobe_widths(_run, _oy, _bw, _bh, lh, count))
            if not _fits(draw, para, bw, bh, font_path_str, mid, stroke_w,
                         taper=True, widths_for=widths_for)[0]:
                ok = False
                break
        if ok:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _fit_lobes(draw, paras, lobes, font_path_str, stroke_w, padding, image_w):
    """A shared size and windowed runs for per-lobe lettering, or (None, None).

    The window follows the size's line height, and the size follows the
    windowed fit, so the loop shrinks until a size passes at its own line
    height — or gives up and the caller keeps the single block.
    """
    size = _lobe_common_size(draw, paras, lobes, font_path_str, stroke_w, padding, image_w)
    if size is None:
        return None, None
    for _ in range(4):
        lh = _calc_line_height(draw, get_font_object(font_path_str, size), stroke_w)
        windowed = [(box, _windowed_run(run, lh // 2)) for box, run in lobes]
        again = _lobe_common_size(draw, paras, windowed, font_path_str, stroke_w, padding, image_w)
        if again is None:
            return None, None
        if again >= size:
            return size, windowed
        size = again
    return None, None


def _tapered_wrap(draw, text: str, font, box_w: int, box_h: int, line_height: int, stroke_w: int,
                  widths_for=None):
    """Greedy wrap where each line fits the chord at its height; returns lines and their widths."""
    lines = _wrap_text(draw, text, font, max(1, box_w - 2 * stroke_w))
    widths_for = widths_for or (lambda count, lh: _chord_widths(count, box_w, box_h, lh))
    for _ in range(4):
        widths = widths_for(len(lines), line_height)
        wrapped, idx = [], 0
        for raw in text.splitlines():
            if not raw.strip():
                wrapped.append("")
                idx += 1
                continue
            current = ""
            for word in raw.split():
                candidate = f"{current} {word}".strip()
                width = widths[idx] - 2 * stroke_w if idx < len(widths) else widths[-1] - 2 * stroke_w
                if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= max(1, width):
                    current = candidate
                else:
                    wrapped.append(current)
                    idx += 1
                    current = word
            if current:
                wrapped.append(current)
                idx += 1
        if len(wrapped) == len(lines):
            lines = wrapped
            break
        lines = wrapped
    return lines, widths_for(len(lines), line_height)


def _fits(draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int,
          taper: bool = False, widths_for=None) -> tuple[bool, list[str]]:
    font = get_font_object(font_path_str, size)
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    if taper:
        lines, widths = _tapered_wrap(draw, text, font, box_w, box_h, line_height, stroke_w,
                                      widths_for=widths_for)
    else:
        lines = _wrap_text(draw, text, font, max(1, box_w - 2 * stroke_w))
        widths = [box_w] * len(lines)
    if not lines:
        return False, []
    total_h = line_height * len(lines)
    bounds = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w) for line in lines]
    # A word broken across lines ("Khôn/g...") does not fit; a smaller size is tried first.
    whole = [word for line in lines for word in line.split()] == text.split()
    fits_w = all(right - left <= widths[i] for i, (left, _, right, _) in enumerate(bounds))
    return whole and total_h <= box_h and fits_w, lines


def _fit_text(
    draw,
    text: str,
    box_w: int,
    box_h: int,
    font_path_str: str,
    stroke_w: int = RENDER_AUTO_STROKE_WIDTH,
    minimum_size: int = MIN_FONT_SIZE,
    maximum_size: int = MAX_FONT_SIZE,
    taper: bool = False,
    widths_for=None,
) -> tuple[int, list[str], bool]:
    minimum_size = max(MIN_FONT_SIZE, int(minimum_size))
    lo, hi = minimum_size, max(minimum_size, min(SOURCE_MATCH_MAX_FONT_SIZE, int(maximum_size)))
    best_size = minimum_size
    best_lines: list[str] = []

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, lines = _fits(draw, text, box_w, box_h, font_path_str, mid, stroke_w, taper=taper,
                          widths_for=widths_for)
        if ok:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    if not best_lines:
        min_font = get_font_object(font_path_str, minimum_size)
        if taper:
            line_height = _calc_line_height(draw, min_font, stroke_w=stroke_w)
            best_lines, _ = _tapered_wrap(draw, text, min_font, box_w, box_h, line_height, stroke_w,
                                          widths_for=widths_for)
        else:
            best_lines = _wrap_text(draw, text, min_font, box_w)
    fits = bool(best_lines) and _fits(
        draw, text, box_w, box_h, font_path_str, best_size, stroke_w, taper=taper,
        widths_for=widths_for,
    )[0]
    return best_size, best_lines, fits


def _greedy_lines(draw, words: list[str], font, width: int) -> list[str]:
    """Words filled into lines no wider than ``width``."""
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


_VI_WORDS = Path(__file__).with_name("vi_words.txt")  # pyvi's word list (MIT, vi_words.LICENSE.txt), 2-4 syllables
_SYLLABLE = re.compile(r"(\W*)(\w+)(\W*)")


@lru_cache(maxsize=1)
def _vi_words() -> frozenset[str]:
    """Vietnamese words of two to four syllables, lower case."""
    try:
        return frozenset(line for line in _VI_WORDS.read_text(encoding="utf-8").splitlines() if line)
    except OSError:
        return frozenset()


def _word_units(words: list[str]) -> list[str]:
    """Syllables joined into the longest known Vietnamese words, so a word such as "huấn luyện" stays on one line."""
    known, units, i = _vi_words(), [], 0
    while i < len(words):
        for n in (4, 3, 2, 1):
            part = words[i:i + n]
            if len(part) < n:
                continue
            if n == 1:
                break
            parsed = [_SYLLABLE.fullmatch(word) for word in part]
            # Punctuation may open the first syllable and close the last, never sit inside the word.
            if all(parsed) and all(not m.group(1) for m in parsed[1:]) and all(not m.group(3) for m in parsed[:-1]) \
                    and " ".join(m.group(2) for m in parsed).lower() in known:
                break
        units.append(" ".join(words[i:i + n]))
        i += n
    return units


def _balanced_lines(draw, words: list[str], font, box_w: int) -> list[str]:
    """As few lines as fit ``box_w``, made as even as those lines allow, so no word is left alone on the last."""
    lines = _greedy_lines(draw, words, font, box_w)
    if len(lines) < 2:
        return lines
    units = _word_units(words)
    if len(units) < len(words) and all(draw.textbbox((0, 0), unit, font=font)[2] <= box_w for unit in units) \
            and len(_greedy_lines(draw, units, font, box_w)) == len(lines):
        words = units  # the same number of lines with no word split, so the size still fits
    lo = max(draw.textbbox((0, 0), word, font=font)[2] for word in words)
    hi = box_w
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_greedy_lines(draw, words, font, mid)) <= len(lines):
            hi = mid
        else:
            lo = mid + 1
    return _greedy_lines(draw, words, font, hi)


def _wrap_text(draw, text: str, font, box_w: int, *, balance: bool = False) -> list[str]:
    """Lines of ``text`` no wider than ``box_w``; ``balance`` evens each wrapped line at the size finally drawn."""
    raw_lines = text.splitlines()
    all_wrapped = []
    for raw_line in raw_lines:
        if not raw_line.strip():
            all_wrapped.append("")
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
