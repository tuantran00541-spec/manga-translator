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
                          "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}


def _plain_punctuation(font_path, text: str) -> str:
    font_path_str = str(font_path)
    return "".join(
        _PUNCTUATION_FALLBACKS[char] if char in _PUNCTUATION_FALLBACKS and not _draws_char(font_path_str, char) else char
        for char in text
    )


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
    if not font_draws_text(font_path, text):
        # Replace glyphs the font lacks; fall back to the default font for missing letters.
        plain = _plain_punctuation(font_path, text)
        if font_draws_text(font_path, plain):
            text = plain
        elif font_draws_text(DEFAULT_FONT, text):
            font_path = DEFAULT_FONT

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

    target_size = None
    if isinstance(font_size, (int, float)) and font_size > 0:
        target_size = int(font_size)
    elif isinstance(font_size, str) and font_size.isdigit() and int(font_size) > 0:
        target_size = int(font_size)

    font_path_str = str(font_path)

    if target_size is not None:
        actual_size = target_size
        font = get_font_object(font_path_str, actual_size)
        lines = _wrap_text(draw, text, font, box_w, balance=True)
    else:
        # Size limits are set for an 800 px wide page and grow with wider pages.
        scale = max(1.0, image.size[0] / REFERENCE_PAGE_WIDTH)
        readable, maximum_size = int(RENDER_MIN_READABLE_FONT_SIZE * scale), int(MAX_FONT_SIZE * scale)
        enlarged_min = int(ENLARGED_MIN_FONT_SIZE * scale)
        # Auto size never letters bigger than the source did.
        if enlarge:
            # Text flagged too small to read grows its area and may exceed the source size.
            grow_x, grow_y = int(round(raw_w * ENLARGE_GROW_RATIO)), int(round(raw_h * ENLARGE_GROW_RATIO))
            image_w, image_h = image.size
            x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
            x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        elif source_cap_px:
            from app.render.source_size import SIZE_SLACK, matching_font_px
            # A misread source size must not shrink the text far below what the box holds.
            box_fit = _fit_text(draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
                                minimum_size=readable, maximum_size=maximum_size)[0]
            maximum_size = max(readable, int(SOURCE_FLOOR_RATIO * box_fit),
                               int(matching_font_px(font_path_str, int(source_cap_px)) * SIZE_SLACK))
        actual_size, lines, fits_readably = _fit_text(
            draw,
            text,
            box_w,
            box_h,
            font_path_str,
            stroke_w=stroke_w,
            minimum_size=enlarged_min if enlarge else readable,
            maximum_size=maximum_size,
        )
        if enlarge and not fits_readably:
            actual_size, lines, fits_readably = _fit_text(
                draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
                minimum_size=readable, maximum_size=maximum_size,
            )
        opaque_caption = bool(bg_color and bg_color not in ("transparent", "none", ""))
        if not fits_readably and opaque_caption and RENDER_SAFE_CAPTION_EXPANSION:
            grow_x = int(round(raw_w * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
            grow_y = int(round(raw_h * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
            image_w, image_h = image.size
            x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
            x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
            actual_size, lines, fits_readably = _fit_text(
                draw,
                text,
                box_w,
                box_h,
                font_path_str,
                stroke_w=stroke_w,
                minimum_size=readable,
                maximum_size=maximum_size,
            )
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
        lines = _wrap_text(draw, text, font, box_w, balance=True)

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
    else:
        start_y = y1 + pad + max(0, (box_h - total_h) // 2)

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)] if is_bold else [(0, 0)]

    for i, line in enumerate(lines):
        line_w = draw.textbbox((0, 0), line, font=font)[2]
        if h_align == "left":
            start_x = x1 + pad
        elif h_align == "right":
            start_x = x1 + pad + max(0, box_w - line_w)
        else:
            start_x = x1 + pad + max(0, (box_w - line_w) // 2)
        cur_y = start_y + i * line_height
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


ENLARGE_GROW_RATIO = 0.25  # each side of a region flagged enlarge grows by this share
ENLARGED_MIN_FONT_SIZE = 22
SOURCE_FLOOR_RATIO = 0.6  # source-matched text never drops below this share of the box-fit size
REFERENCE_PAGE_WIDTH = 800  # font size limits are for a page this wide
SOURCE_MATCH_MAX_FONT_SIZE = 200  # large source lettering may be matched past MAX_FONT_SIZE


def _fits(draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int) -> tuple[bool, list[str]]:
    font = get_font_object(font_path_str, size)
    lines = _wrap_text(draw, text, font, box_w)
    if not lines:
        return False, []
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)
    max_line_w = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    return total_h <= box_h and max_line_w <= box_w, lines


def _fit_text(
    draw,
    text: str,
    box_w: int,
    box_h: int,
    font_path_str: str,
    stroke_w: int = RENDER_AUTO_STROKE_WIDTH,
    minimum_size: int = MIN_FONT_SIZE,
    maximum_size: int = MAX_FONT_SIZE,
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

    if not best_lines:
        min_font = get_font_object(font_path_str, minimum_size)
        best_lines = _wrap_text(draw, text, min_font, box_w)
    fits = bool(best_lines) and _fits(
        draw, text, box_w, box_h, font_path_str, best_size, stroke_w
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
