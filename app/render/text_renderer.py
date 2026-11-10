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


def _light_background_grey(image: Image.Image, box) -> float:
    """75th-percentile grey of the box: the paper the glyphs will touch.

    A box mean mixes the bubble paper with dark art around it, so black
    text on a light bubble reads low-contrast and gets a thick outline
    it does not need. The lightest quarter is the paper, not the art.
    """
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
    """Left edge of ``line`` as drawn, on its own band when it has one.

    The band interval, not the box edge, is the clamp: a line whose
    centre sits near the box edge would otherwise be pulled back onto
    the outline the band centre was avoiding.
    Per-line centres exist only for curved, centred, auto-fit text
    that fits; otherwise the line centres on the box, recorded in
    render stats as ``line_centring`` (``per_line`` or ``box_centre``).
    """
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

    # A chapter band caps the size a short block may grow to in a roomy
    # bubble. The ceiling is in font px, from the band's cap through this
    # object's own font, so a narrow font keeps its own proportions.
    size_ceiling_px = None
    if size_band_hi_cap:
        from app.render.source_size import SIZE_SLACK, matching_font_px
        size_ceiling_px = int(matching_font_px(str(font_path), int(size_band_hi_cap)) * SIZE_SLACK)

    if fill is None or fill == "auto" or fill == "":
        text_color = auto_detect_text_color(image, box)
        explicit_fill = False
    else:
        text_color = parse_color(fill, default=(0, 0, 0))
        explicit_fill = True

    if stroke_width is None or stroke_width == "auto" or stroke_width == "":
        stroke_w = RENDER_AUTO_STROKE_WIDTH if font_path else 1
    else:
        try:
            stroke_w = max(0, min(RENDER_STROKE_WIDTH_MAX, int(stroke_width)))
        except (ValueError, TypeError):
            stroke_w = RENDER_AUTO_STROKE_WIDTH

    bg_grey = _light_background_grey(image, (x1, y1, x2, y2))
    if _contrast(text_color, (bg_grey, bg_grey, bg_grey)) < LOW_CONTRAST_RATIO:
        # Letters close to the background colour get a thick contrasting outline.
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

    # A bubble or spiky outline narrows toward its top and bottom, so its lines
    # must fit the measured interior, not the full rectangle. Spiky shares the
    # bubble band machinery, but when bands fail the fallback is the inscribed
    # ellipse, and spiky valleys cut inside it, so the fallback can cross the
    # outline; nothing here validates spiky points or valleys.
    curved = str(container or "").lower() in _CURVED_CONTAINERS or str(shape or "").lower() == "ellipse"
    centred = h_align == "center" and v_align == "middle"

    if target_size is not None:
        actual_size = target_size
        font = get_font_object(font_path_str, actual_size)
        lines = _wrap_text(draw, text, font, box_w, balance=True)
        line_centres = None
        lines_source = "fixed_size"
        draw_spans = None
    else:
        opaque_caption = bool(bg_color and bg_color not in ("transparent", "none", ""))
        line_centres = []
        retry_adopted = False
        fit_stats: dict = {}
        actual_size, fitted_lines, fits_readably, (x1, y1, x2, y2) = auto_fit(
            draw, text, (x1, y1, x2, y2), image.size, font_path_str, padding=padding, stroke_w=stroke_w,
            source_cap_px=source_cap_px, enlarge=enlarge, opaque_caption=opaque_caption,
            curved=curved, image=image, centered=centred, container=container,
            source_letter_box=source_letter_box, centres_out=line_centres,
            stats_out=fit_stats, min_source_share=min_source_share,
            size_ceiling_px=size_ceiling_px)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        # A band fit that fails — outright, or below the source floor —
        # while the rectangle fit holds it is a degenerate-bands failure,
        # not a too-long-text failure; retry without bands before refusing.
        # Slice 81's cleaned outline trips it on the readable floor (the
        # waist check fires for the region box but not the trimmed one),
        # so the retry covers both failures, not just the floor ratio.
        # The retry is skipped for a roomy fallback attempt when a tight
        # letter box exists: a rectangle fit there lands the text on the
        # region, not the bubble, and the ranking prefers it over the
        # tight attempt that placed the text correctly (slice 25's region
        # attempt at 200px outranking the tight 28px one).
        _letter_area = ((source_letter_box[2] - source_letter_box[0])
                        * (source_letter_box[3] - source_letter_box[1])
                        if source_letter_box is not None else None)
        _retry_roomy = (_letter_area is not None
                        and (x2 - x1) * (y2 - y1) > 4 * _letter_area)
        if (curved and fit_stats.get("bands_used") and not _retry_roomy
                and (not fits_readably
                     or (source_cap_px and not enlarge and fit_stats.get("below_floor_ratio")))):
            retry_stats: dict = {}
            retry_centres: list = []
            r_size, r_lines, r_fits, (rx1, ry1, rx2, ry2) = auto_fit(
                draw, text, (x1, y1, x2, y2), image.size, font_path_str, padding=padding,
                stroke_w=stroke_w, source_cap_px=source_cap_px, enlarge=enlarge,
                opaque_caption=opaque_caption, curved=False, image=image, centered=centred,
                container=container, source_letter_box=source_letter_box,
                centres_out=retry_centres, stats_out=retry_stats,
                min_source_share=min_source_share, size_ceiling_px=size_ceiling_px)
            if r_fits and not retry_stats.get("below_floor_ratio"):
                actual_size, fitted_lines, fits_readably = r_size, r_lines, r_fits
                x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                line_centres = retry_centres
                fit_stats = retry_stats
                retry_adopted = True
                raw_w, raw_h = x2 - x1, y2 - y1
                pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
                box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        if not fits_readably:
            raise ValueError(
                "Translation does not fit at the minimum readable font size; "
                "shorten the translation or enlarge the text region."
            )
        if source_cap_px and not enlarge and min_source_share and not fit_stats.get("share_ok", True):
            raise ValueError("Translation fits only well below the source letter size")
        if source_cap_px and not enlarge and fit_stats.get("below_floor_ratio"):
            raise ValueError("Translation fits only below the source floor share of its box")
        font = get_font_object(font_path_str, actual_size)
        # The same number of lines, evened out so no word is left alone.
        # Which lines are drawn is recorded: the fit sized the greedy
        # lines, so evened lines that leave the measured bands give way
        # to them instead of being drawn over an outline.
        lines = _wrap_text(draw, text, font, box_w, balance=True)
        lines_source = "balanced"
        draw_spans = None
        if centred and curved and not retry_adopted:
            bands = _bubble_bands(image, (x1 + pad, y1 + pad, x2 - pad, y2 - pad),
                                  letter_box=source_letter_box)
            if bands is not None:
                # Evened lines can push a long line into a narrow band; the fitted lines stay inside.
                if not _lines_fit_bands(draw, lines, font, box_w, box_h, stroke_w, bands):
                    if _lines_fit_bands(draw, fitted_lines, font, box_w, box_h, stroke_w, bands):
                        lines = fitted_lines
                        lines_source = "fitted"
                draw_spans = _band_line_spans(
                    box_w, box_h, len(lines), _calc_line_height(draw, font, stroke_w=stroke_w), bands)
            elif curved and not _lines_fit_ellipse(draw, lines, font, box_w, box_h, stroke_w):
                if _lines_fit_ellipse(draw, fitted_lines, font, box_w, box_h, stroke_w):
                    lines = fitted_lines
                    lines_source = "fitted"

    if stats is not None:
        stats["lines"] = len(lines)
        stats["size"] = actual_size
        stats["fit_box"] = (x1, y1, x2, y2)
        if h_align == "center":
            # Per-line centres only exist for a curved auto fit; every
            # other centred path centres on the box, and the flag says
            # which one the drawing used.
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

    if target_size is not None:
        # A named size is still lettering in a box: below the readable
        # floor, or taller than the box, it leaves the object for review
        # instead of painting small or past the frame.
        readable_fixed = int(RENDER_MIN_READABLE_FONT_SIZE
                             * max(1.0, image.size[0] / REFERENCE_PAGE_WIDTH))
        if target_size < readable_fixed:
            raise ValueError(
                "Translation does not fit at the minimum readable font size; "
                "shorten the translation or enlarge the text region."
            )
        if total_h > box_h:
            raise ValueError(
                "Translation does not fit in its box at the requested font size; "
                "shorten the translation or enlarge the text region."
            )

    if v_align == "top":
        start_y = y1 + pad
    elif v_align == "bottom":
        start_y = y1 + pad + max(0, box_h - total_h)
    else:
        start_y = y1 + pad + max(0, (box_h - total_h) // 2)

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)] if is_bold else [(0, 0)]

    fill_fallback = None
    if explicit_fill:
        # A stored fill is a suggestion, not a verdict: a pale fill on
        # matching paper reads pale however thick its outline. The
        # background it is judged against is the ring of pixels the
        # glyphs will actually touch — box means and medians mix paper,
        # outlines and art, which is what let pale fills pass before.
        # Mottled art is judged on its lightest patches, since those are
        # where a pale fill vanishes.
        if bg_color and bg_color not in ("transparent", "none", ""):
            local_bg = parse_color(bg_color, default=(255, 255, 255))
            if _contrast(text_color, local_bg) < LOW_CONTRAST_RATIO:
                text_color = auto_detect_text_color(image, (x1, y1, x2, y2))
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

    # Where the drawn lines actually land, as their advance boxes: the
    # ranking judges placement on ink, not on the box the fit started
    # from, and a line's advance box slightly overestimates its glyphs,
    # so a candidate inside the letter box is truly inside it.
    ink_box = None
    for i, line in enumerate(lines):
        start_x, line_w = _line_start_x(draw, line, font, x1=x1, pad=pad, box_w=box_w,
                                        h_align=h_align, centred=centred, curved=curved,
                                        line_centres=line_centres, draw_spans=draw_spans,
                                        i=i, line_count=len(lines))
        cur_y = start_y + i * line_height
        if line_w > 0:
            bold = 1 if is_bold else 0
            line_box = (start_x, cur_y, start_x + line_w + bold,
                        cur_y + line_height + bold)
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


def auto_fit(draw, text: str, box, image_size, font_path_str: str, *, padding: int = RENDER_DEFAULT_PADDING,
             stroke_w: int = RENDER_AUTO_STROKE_WIDTH, source_cap_px: int | None = None, enlarge: bool = False,
             opaque_caption: bool = False, curved: bool = False, image=None, centered: bool = True,
             container: str | None = None, source_letter_box: tuple[int, int, int, int] | None = None,
             centres_out: list | None = None, stats_out: dict | None = None,
             min_source_share: float = 0.0, size_ceiling_px: int | None = None):
    """The auto font size, wrapped lines, whether it fits readably and the box it was fitted in, as the renderer letters it.

    One decision order, each stage feeding the next: the frame interior is
    measured first and every later move stays inside it; the fit sizes
    inside that frame; the caller ranks attempts and separates them. A
    stage that cannot hold its constraint says so in ``stats_out``
    instead of quietly re-deciding for an earlier stage.
    The frame is a hard box only when measurement succeeds: on
    ``frame_unmeasured``, ``bands_none_rectangle`` or ``region_no_frame``
    the fit proceeds as a rectangle fit on a bubble, recorded in
    ``fallbacks``, and the text can cross the outline.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    raw_w, raw_h = x2 - x1, y2 - y1
    pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
    box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    # The frame interior is the hard box for everything below: the fit,
    # the growths and the regrow move all stay inside it. What it is
    # depends on the container; an unmeasured frame is recorded, not
    # silently replaced by the region.
    frame = (x1, y1, x2, y2)
    frame_source = "region_no_frame"
    fallbacks: list[str] = []
    stages_skipped: list[str] = []
    lobe_decision = None
    band_verdict: dict = {}
    cap_ceiling = None
    cap_relaxed = False
    box_fit = None

    def clamp_to_frame(bx1, by1, bx2, by2):
        fx1, fy1, fx2, fy2 = frame
        return (max(bx1, fx1), max(by1, fy1), min(bx2, fx2), min(by2, fy2))

    attempt_box = (x1, y1, x2, y2)

    def clamp_growth(bx1, by1, bx2, by2):
        # Growth stays inside the measured frame; where no frame was
        # measured, a container with an outline still bounds growth to
        # the box it was given, since that box is the outline's guess.
        # Only frameless text has nothing to cross, so it grows freely.
        if frame_source != "region_no_frame":
            return clamp_to_frame(bx1, by1, bx2, by2)
        if curved or str(container or "").lower() == "box":
            ax1, ay1, ax2, ay2 = attempt_box
            return (max(bx1, ax1), max(by1, ay1), min(bx2, ax2), min(by2, ay2))
        return (bx1, by1, bx2, by2)

    if str(container or "").lower() == "screen":
        # A system window's detection region is its outer extent, so the
        # frame itself sits inside the fit box; lettering to the region
        # edge paints over the border. Hold the text inside the frame.
        # Bright border lines are measured where both sides show; any
        # dimension without them keeps the inset guess, recorded as such.
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

    if str(container or "").lower() == "box" and not curved and image is not None:
        # A narration box has no curved measurement, so its fit box used
        # to be the detector region — wider than the frame it sits in.
        # Shrink the fit to the measured rules when they show.
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

    split_bubble = False
    if curved and image is not None and centered:
        # The region box is a guess at the bubble; lettering in the bubble
        # itself, or its lobe, is what keeps text off outlines and panel
        # edges. A rejected guess leaves the region box untouched.
        mag = _edge_magnitude(image)
        grey = np.asarray(image.convert("L"))
        cut = _divider_cut(mag, (x1, y1, x2, y2))
        import cv2

        dist = cv2.distanceTransform((mag <= RAY_EDGE_THRESHOLD).astype(np.uint8),
                                     cv2.DIST_L2, 5)
        ray = None
        cut_frame = False
        if cut != (x1, y1, x2, y2):
            # A bubble drawn over a panel edge is one text over two lobes,
            # not a bubble on one side of a divider; rays on the uncut box
            # recover it whole, while a box of art measures nothing.
            ray = _ray_boxes(mag, grey, (x1, y1, x2, y2),
                             _seed_points(mag, dist, (x1, y1, x2, y2)))
            if ray is not None:
                # A ray from inside one lobe stops at that lobe's outline:
                # when the cut side's own flood proves two-lobed interior
                # outside the ray frame, the ray is a lobe, not the bubble,
                # so the cut box itself is the frame and its two-lobe
                # handling below runs instead of the split path.
                cut_bands = _bubble_bands(image, (cut[0] + pad, cut[1] + pad,
                                                  cut[2] - pad, cut[3] - pad),
                                            letter_box=source_letter_box)
                if _ray_loses_lobe(cut_bands, cut, pad, ray):
                    ray = None
                    cut_frame = True
                    frame = cut
                    frame_source = "divider_cut"
                    x1, y1, x2, y2 = cut
                else:
                    split_bubble = True
                    frame = ray
                    frame_source = "ray_outline"
                    lobe_decision = "split_whole_bubble"
                    stages_skipped.append("trim:whole_bubble_measured")
                    stages_skipped.append("regrow:whole_bubble_measured")
                    x1, y1, x2, y2 = ray
            else:
                x1, y1, x2, y2 = cut
        if ray is None and not cut_frame:
            ray = _ray_boxes(mag, grey, (x1, y1, x2, y2),
                             _seed_points(mag, dist, (x1, y1, x2, y2)))
        if ray is not None and not split_bubble:
            frame = ray
            frame_source = "ray_outline"
            x1, y1, x2, y2 = ray
        if not split_bubble and ray is None and not cut_frame:
            fallbacks.append("frame_unmeasured")
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        if not split_bubble:
            # Lobe selection is an explicit verdict, never a silent trim:
            # when the measured bands hold two wide lobes, the block is
            # fitted to the whole interior and split across them by its
            # per-line centres, instead of one lobe being discarded.
            probe = _bubble_bands(image, (x1 + pad, y1 + pad, x2 - pad, y2 - pad),
                                  verdict=band_verdict, letter_box=source_letter_box)
            if probe is not None and _two_lobes(probe, box_w,
                                                frame_measured=(frame_source != "region_no_frame")):
                lobe_decision = "two_lobes_full"
                stages_skipped.append("trim:two_lobes_kept")
            else:
                x1, y1, x2, y2 = _trim_bubble_box(image, (x1, y1, x2, y2),
                                                  letter_box=source_letter_box)
                lobe_decision = "single_lobe_trim"
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        # The region box can be wider than the bubble body it holds: a
        # bent balloon's lobes sit off the box centre, so lines centred on
        # the box cross an edge long before they fill the lobe. Narrow the
        # box to the middle band's interior when that interior is clearly
        # narrower than the box yet still most of the widest band; a waist
        # between two lobes is not a body, so it never earns the shrink.
        # Its gate already refuses a waist, so it runs on the split path
        # too; only the trim stays split-only.
        centre_bands = _bubble_bands(image, (x1 + pad, y1 + pad, x2 - pad, y2 - pad),
                                     verdict=band_verdict, letter_box=source_letter_box)
        # A two-lobe fit already centres each line on its own lobe's band,
        # so narrowing to the middle band would throw away the lobe
        # interior the two-lobe verdict just kept.
        if centre_bands and lobe_decision != "two_lobes_full":
            mid = centre_bands[len(centre_bands) // 2]
            centre = box_w / 2
            holding = [iv for iv in mid if iv[0] <= centre <= iv[1]]
            widest = max((iv[1] - iv[0] for band in centre_bands for iv in band), default=0)
            if holding:
                a, b = max(holding, key=lambda iv: iv[1] - iv[0])
                if b - a < 0.9 * box_w and b - a >= 0.5 * widest and b - a >= 100:
                    x1, x2 = x1 + a, x1 + b + 2 * pad
                    raw_w, raw_h = x2 - x1, y2 - y1
                    pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
                    box_w, box_h = raw_w - pad * 2, raw_h - pad * 2

    def bands_for():
        # Bands are measured on the box the lines are actually drawn in, so a grown
        # box re-measures; only a curved outline is measured, so free text on art
        # keeps the rectangle fit instead of a flood fill through the artwork.
        if image is None or not centered or not curved:
            return None
        if frame_source == "region_no_frame" and source_letter_box is None:
            # No measured outline and no lettering to anchor the flood: the
            # narrowed bands would be unanchored, so a rectangle fit is the
            # honest fallback.
            return None
        # The frame is only the gate here, not an input: the flood seeds
        # from the lettering itself, and its gap-cut, vote and fragmented
        # checks reject a leak the same way. A ray that fails on a
        # LaMa-softened outline must not force the ellipse guess when the
        # interior can still be measured.
        bands = _bubble_bands(image, (x1 + pad, y1 + pad, x2 - pad, y2 - pad),
                              verdict=band_verdict, letter_box=source_letter_box)
        if bands is not None:
            if source_letter_box is not None:
                ly1, ly2 = source_letter_box[1], source_letter_box[3]
            else:
                ly1, ly2 = y1 + pad + 0.25 * box_h, y1 + pad + 0.75 * box_h
            n = len(bands)
            rows = [band for i, band in enumerate(bands)
                    if y1 + pad + (i + 1) * box_h / n > ly1 and y1 + pad + i * box_h / n < ly2]
            widest = max((b - a for band in (rows or bands) for a, b in band), default=0)
            if widest < 0.4 * box_w:
                # A fragmented flood on the lettering rows is shading, not an interior; trusting it collapses the fit.
                band_verdict["bands_fragmented"] = True
                return None
        return bands

    # Size limits are set for an 800 px wide page and grow with wider pages.
    scale = max(1.0, image_size[0] / REFERENCE_PAGE_WIDTH)
    readable, maximum_size = int(RENDER_MIN_READABLE_FONT_SIZE * scale), int(MAX_FONT_SIZE * scale)
    enlarged_min = int(ENLARGED_MIN_FONT_SIZE * scale)
    # Auto size never letters bigger than the source did.
    if enlarge:
        # Text flagged too small to read grows its area and may exceed the source size.
        # Growth stays inside the frame: a framed text grows into its bubble,
        # never onto the art around it. Free text has no frame, so it grows freely.
        grow_x, grow_y = int(round(raw_w * ENLARGE_GROW_RATIO)), int(round(raw_h * ENLARGE_GROW_RATIO))
        image_w, image_h = image_size
        x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
        x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
        x1, y1, x2, y2 = clamp_growth(x1, y1, x2, y2)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    elif source_cap_px:
        from app.render.source_size import SIZE_SLACK, matching_font_px
        # A misread source size must not shrink the text far below what the box holds.
        box_fit = _fit_text(draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
                            minimum_size=readable, maximum_size=maximum_size, curved=curved,
                            bands=bands_for())[0]
        cap_ceiling = int(matching_font_px(font_path_str, int(source_cap_px)) * SIZE_SLACK)
        maximum_size = max(readable, int(SOURCE_FLOOR_RATIO * box_fit), cap_ceiling)
        if curved and source_letter_box is not None:
            # Small source lettering in a much taller bubble: the cap alone
            # strands a short block in a tall balloon. When the fitted box
            # stands at least half again as tall as the source letters, the
            # ceiling may rise to one and a half times itself, still bounded
            # by what the box holds. This is the only relaxation of the cap.
            letter_h = int(source_letter_box[3]) - int(source_letter_box[1])
            if letter_h > 0 and box_h >= 1.5 * letter_h:
                uncapped = _fit_text(draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
                                     minimum_size=readable, maximum_size=SOURCE_MATCH_MAX_FONT_SIZE,
                                     curved=curved, bands=bands_for())[0]
                maximum_size = max(maximum_size, min(uncapped, int(cap_ceiling * 1.5)))
                cap_relaxed = maximum_size > max(readable, int(SOURCE_FLOOR_RATIO * box_fit), cap_ceiling)
    if size_ceiling_px:
        # The band is a ceiling on growth, never a floor: a long text in
        # a small box still shrinks below it to fit.
        maximum_size = max(readable, min(maximum_size, int(size_ceiling_px)))
    if not curved and frame_source == "region_no_frame" and box_h > 0:
        # A frameless box shorter than one line at the readable floor can
        # never hold text: the padding alone eats the line, so the insert
        # fails and the original stays. Frameless text has no outline to
        # keep off, so the margin yields to readability — shrink the pad
        # until one floor line fits, instead of refusing the text.
        floor_line = _calc_line_height(
            draw, get_font_object(font_path_str, readable), stroke_w=stroke_w)
        if box_h < floor_line:
            pad = max(0, (raw_h - floor_line) // 2)
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
    fit_bands = bands_for()
    if curved and centered and image is not None and fit_bands is None:
        # The outline measurement failed, so the fit below is a rectangle
        # fit on a bubble: recorded, not passed off as outline-aware.
        fallbacks.append("bands_none_rectangle")
    actual_size, lines, fits_readably = _fit_text(
        draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
        minimum_size=enlarged_min if enlarge else readable, maximum_size=maximum_size, curved=curved,
        bands=fit_bands,
    )
    if enlarge and not fits_readably:
        actual_size, lines, fits_readably = _fit_text(
            draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
            minimum_size=readable, maximum_size=maximum_size, curved=curved,
            bands=bands_for(),
        )
    if not fits_readably and opaque_caption and RENDER_SAFE_CAPTION_EXPANSION:
        grow_x = int(round(raw_w * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
        grow_y = int(round(raw_h * RENDER_SAFE_CAPTION_EXPANSION_RATIO))
        image_w, image_h = image_size
        x1, y1 = max(0, x1 - grow_x), max(0, y1 - grow_y)
        x2, y2 = min(image_w, x2 + grow_x), min(image_h, y2 + grow_y)
        x1, y1, x2, y2 = clamp_growth(x1, y1, x2, y2)
        raw_w, raw_h = x2 - x1, y2 - y1
        pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
        box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        actual_size, lines, fits_readably = _fit_text(
            draw, text, box_w, box_h, font_path_str, stroke_w=stroke_w,
            minimum_size=readable, maximum_size=maximum_size, curved=curved,
            bands=bands_for(),
        )
    # The recovery move stays available without a frame: a centred fit
    # that fits nowhere is rescued by moving to fitting rows, and there
    # is no centred placement to preserve. Only the regrow below is
    # gated on a measured frame.
    if (curved and centered and image is not None and not fits_readably and fit_bands is not None
            and not split_bubble):
        # Centred lines can land on rows where the region box holds no bubble
        # interior at all — a tall region over a short bubble, or a bubble
        # over the next panel — while wider rows above or below hold them.
        # Move the block to the closest fitting rows instead of failing.
        for size in _scan_sizes(maximum_size, readable - 1):
            font_try = get_font_object(font_path_str, size)
            lines_try = _wrap_text(draw, text, font_try, box_w)
            if not lines_try or [w for line in lines_try for w in line.split()] != text.split():
                continue
            line_height_try = _calc_line_height(draw, font_try, stroke_w=stroke_w)
            total_h_try = line_height_try * len(lines_try)
            if total_h_try > box_h:
                continue
            top_try = _best_band_top(draw, lines_try, font_try, box_w, box_h, stroke_w, fit_bands)
            if top_try is None:
                continue
            centred_top = max(0, (box_h - total_h_try) // 2)
            delta = top_try - centred_top
            new_y1, new_y2 = y1 + delta, y2 + delta
            if frame_source != "region_no_frame":
                new_y1 = max(new_y1, frame[1])
                new_y2 = min(new_y2, frame[3])
            if new_y1 < 0:
                new_y2, new_y1 = new_y2 - new_y1, 0
            if new_y2 > image_size[1]:
                new_y1, new_y2 = new_y1 - (new_y2 - image_size[1]), image_size[1]
            if new_y1 < 0:
                new_y1, new_y2 = 0, new_y2 - new_y1
            # A clamped move no longer lands on the rows the size was
            # validated for, so it is refused and a smaller size is tried.
            if (new_y1, new_y2) != (y1 + delta, y2 + delta):
                continue
            actual_size, lines = size, lines_try
            fits_readably = True
            y1, y2 = new_y1, new_y2
            fallbacks.append("band_placement_move")
            break
    # Without a measured frame the regrow must not move the block onto
    # the widest measured rows — those may be leaked paper, not bubble.
    if (curved and centered and image is not None and fits_readably and actual_size < maximum_size
            and not split_bubble and frame_source != "region_no_frame"):
        bands = bands_for()
        if bands is not None:
            # Centred placement can strand a short block on a narrow waist
            # while the lobe above or below holds far wider rows; regrow
            # only when a larger size fits somewhere in the measured bands,
            # and move the box so centred drawing lands on that spot.
            # The move-and-rewrap form stays off the split path:
            # re-compacting a whole bubble's block to one best band
            # undoes the split it was kept whole for. The split path
            # gets its own growth below, without a move or a re-wrap.
            # The move stays inside the measured frame, never just on
            # the canvas.
            for size in _scan_sizes(maximum_size, actual_size):
                font_try = get_font_object(font_path_str, size)
                lines_try = _wrap_text(draw, text, font_try, box_w)
                if not lines_try or [w for line in lines_try for w in line.split()] != text.split():
                    continue
                line_height_try = _calc_line_height(draw, font_try, stroke_w=stroke_w)
                total_h_try = line_height_try * len(lines_try)
                if total_h_try > box_h:
                    continue
                top_try = _best_band_top(draw, lines_try, font_try, box_w, box_h, stroke_w, bands)
                if top_try is None:
                    continue
                # The block itself must stay inside the region it belongs
                # to: bands measured past the region edge are paper, not
                # bubble, and growing into them strands text over art.
                if y1 + top_try < attempt_box[1] or y1 + top_try + total_h_try > attempt_box[3]:
                    continue
                centred_top = max(0, (box_h - total_h_try) // 2)
                delta = top_try - centred_top
                new_y1, new_y2 = y1 + delta, y2 + delta
                if frame_source != "region_no_frame":
                    new_y1 = max(new_y1, frame[1])
                    new_y2 = min(new_y2, frame[3])
                if new_y2 - new_y1 < y2 - y1:
                    # The frame cannot hold the moved box whole, so this
                    # size is refused and a smaller one is tried.
                    continue
                if new_y1 < 0:
                    new_y2, new_y1 = new_y2 - new_y1, 0
                if new_y2 > image_size[1]:
                    new_y1, new_y2 = new_y1 - (new_y2 - image_size[1]), image_size[1]
                if new_y1 < 0:
                    new_y1, new_y2 = 0, new_y2 - new_y1
                if (new_y1, new_y2) != (y1 + delta, y2 + delta):
                    continue
                actual_size, lines = size, lines_try
                y1, y2 = new_y1, new_y2
                break
    if curved and centered and image is not None and fits_readably and actual_size < maximum_size and split_bubble:
        bands = bands_for()
        if bands is not None:
            # A split block grows only as itself: re-wrapping it at a
            # larger size compacts it into one lobe, which is the defect
            # the split exists to avoid, and moving it to one best band
            # does the same. Growth counts only when the split's own
            # lines still fit their bands, at their centred height, at
            # the larger size; otherwise the fitted size stands.
            for size in range(maximum_size, actual_size, -1):
                font_try = get_font_object(font_path_str, size)
                line_height_try = _calc_line_height(draw, font_try, stroke_w=stroke_w)
                if line_height_try * len(lines) > box_h:
                    continue
                if _lines_fit_bands(draw, lines, font_try, box_w, box_h, stroke_w, bands):
                    actual_size = size
                    # Growth happened, so the whole-bubble regrow skip
                    # recorded at frame measurement no longer holds.
                    if "regrow:whole_bubble_measured" in stages_skipped:
                        stages_skipped.remove("regrow:whole_bubble_measured")
                    break
    if curved and centered and image is not None and fits_readably and lines:
        # The fit above validated sizes against bands measured on earlier
        # boxes; moves since then re-measure, so a validated size can still
        # cross the outline where it is actually drawn. Re-check the drawn
        # position — centred in the final box — and shrink until it holds,
        # or give the object back for review instead of lettering over art.
        final_bands = bands_for()
        if final_bands is not None:
            held = False
            # The greedy check runs first, unchanged: its verdict is what
            # the floor and the rectangle retry read, so it stands unless
            # it fails outright or lands below the source floor — those
            # are the cases where the old path gives the object back or
            # hands it to the rectangle retry, and a narrower wrap of the
            # same text at the fitted size is the better answer. A greedy
            # wrap can strand one long line on narrow rows while a narrower
            # wrap fits them, and shrinking to the greedy wrap strands the
            # block far below what the bubble holds.
            greedy_size = None
            greedy_lines = None
            for size in range(actual_size, readable - 1, -1):
                font_try = get_font_object(font_path_str, size)
                lines_try = _wrap_text(draw, text, font_try, box_w)
                if not lines_try or [w for line in lines_try for w in line.split()] != text.split():
                    continue
                if _calc_line_height(draw, font_try, stroke_w=stroke_w) * len(lines_try) > box_h:
                    continue
                if _lines_fit_bands(draw, lines_try, font_try, box_w, box_h, stroke_w, final_bands):
                    greedy_size, greedy_lines = size, lines_try
                    break
            floor_size = None
            if box_fit is not None:
                _floor_box = min(box_fit, size_ceiling_px) if size_ceiling_px else box_fit
                floor_size = int(SOURCE_FLOOR_RATIO * _floor_box)
            needs_rewrap = greedy_size is None or (floor_size is not None
                                                   and greedy_size < floor_size)
            if needs_rewrap:
                # The re-wrap is trusted only where the bands are real
                # interior: a fitted line sitting on a span narrower than
                # two fifths of the box is sitting on a fragment — letters
                # or a leak splitting the flood, not a lobe — so there the
                # greedy verdict stands, and a fit that needs the re-wrap
                # at every size fails here so the rectangle retry can
                # rescue it instead of lettering small on a sliver.
                font_try = get_font_object(font_path_str, actual_size)
                lh_try = _calc_line_height(draw, font_try, stroke_w=stroke_w)
                spans_try = _band_line_spans(box_w, box_h, len(lines), lh_try, final_bands)
                rewrap_trusted = (spans_try is not None
                                  and min(b - a for a, b in spans_try) >= 0.4 * box_w)
                if rewrap_trusted:
                    ok_try, lines_try = _fits(draw, text, box_w, box_h, font_path_str, actual_size,
                                              stroke_w, curved=True, bands=final_bands)
                    if ok_try:
                        actual_size, lines = actual_size, lines_try
                        held = True
            if not held and greedy_size is not None and not (
                    floor_size is not None and greedy_size < floor_size):
                actual_size, lines = greedy_size, greedy_lines
                held = True
            if not held:
                actual_size = readable
                lines = _wrap_text(draw, text, get_font_object(font_path_str, readable), box_w)
                fits_readably = False
    if centres_out is not None and curved and centered and image is not None and fits_readably and lines:
        # Each line is drawn on its own rows' interior centre, not the box
        # centre: the fit above already sized it against those intervals.
        bands = bands_for()
        if bands is not None:
            font_c = get_font_object(font_path_str, actual_size)
            line_height_c = _calc_line_height(draw, font_c, stroke_w=stroke_w)
            total_c = line_height_c * len(lines)
            if total_c <= box_h:
                centres = _band_line_centres(box_w, box_h, len(lines), line_height_c, bands)
                if centres is not None:
                    centres_out.extend(centres)
    if stats_out is not None:
        # The share rule is a fit constraint, judged here where the size
        # is known, not a post-hoc rejection: the caller retries at
        # share 0.0 on this verdict, it does not re-run the fit to learn it.
        share_ok = True
        if source_cap_px and not enlarge and min_source_share:
            from app.render.source_size import matching_font_px
            share_ok = actual_size >= min_source_share * matching_font_px(
                font_path_str, int(source_cap_px))
        # The floor share is a floor, not just a ceiling: a fit that
        # only holds far below what the box holds leaves the object for
        # review instead of lettering it small. The floor is judged
        # against the size the band allows, not the raw box fit: a
        # ceiling holding a short block below 60% of its roomy box is
        # the band working, not a fit failure.
        floor_box_fit = min(box_fit, size_ceiling_px) if size_ceiling_px else box_fit
        below_floor_ratio = bool(
            fits_readably and source_cap_px and not enlarge and box_fit is not None
            and actual_size < int(SOURCE_FLOOR_RATIO * floor_box_fit))
        stats_out.update({
            "frame": frame,
            "frame_source": frame_source,
            "below_readable_floor": not fits_readably,
            "readable_floor_px": readable,
            "bands_used": fit_bands is not None,
            "lobe": lobe_decision,
            "fallbacks": list(fallbacks),
            "stages_skipped": list(stages_skipped),
            "band_narrowing": dict(band_verdict),
            "cap_ceiling": cap_ceiling,
            "cap_relaxed": cap_relaxed,
            "size_ceiling_px": size_ceiling_px,
            "share_ok": share_ok,
            "below_floor_ratio": below_floor_ratio,
        })
    return actual_size, lines, fits_readably, (x1, y1, x2, y2)


ENLARGE_GROW_RATIO = 0.25  # each side of a region flagged enlarge grows by this share
ENLARGED_MIN_FONT_SIZE = 22
SOURCE_FLOOR_RATIO = 0.6  # source-matched text never drops below this share of the box-fit size
REFERENCE_PAGE_WIDTH = 800  # font size limits are for a page this wide
SOURCE_MATCH_MAX_FONT_SIZE = 200  # large source lettering may be matched past MAX_FONT_SIZE


def _ellipse_line_widths(box_w: int, box_h: int, line_count: int, line_height: int) -> list[float]:
    """Widest ink each centered line may hold inside the ellipse inscribed in the box.

    A bubble's usable width narrows toward its top and bottom, so a line's limit is the
    chord at the band edge farthest from the middle, not the full box width.
    """
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
    """Whether every line's ink plus stroke stays inside its chord of the inscribed ellipse."""
    if not lines:
        return True
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    limits = _ellipse_line_widths(box_w, box_h, len(lines), line_height)
    if len(limits) != len(lines):
        return False
    return all(
        draw.textbbox((0, 0), line, font=font)[2] + 2 * stroke_w <= limit
        for line, limit in zip(lines, limits))


BAND_COUNT = 24  # horizontal slices the measured bubble interior is split into
BAND_EDGE_PAD = 4  # ink keeps this margin inside a band, off the outline
BAND_FILL_TOLERANCE = 48  # grey levels a bubble interior may drift from its centre
BAND_MIN_FILL = 0.5  # a fill smaller than this is art, not a bubble interior
BUBBLE_BOX_EDGE_INSET = 8  # a bubble stays this far from the slice's left/right edges
BUBBLE_BOX_AREA_MIN = 0.35  # a bubble box may not shrink far below its region box
BUBBLE_BOX_AREA_MAX = 3.0  # nor grow far beyond it
BUBBLE_BOX_CENTRE_SHARE = 0.35  # a bubble box's centre stays this close to its region box
BUBBLE_BOX_PAPER_MIN = 0.65  # a bubble box is mostly paper, not art
BUBBLE_TRIM_DEVIATION_SHARE = 0.08  # a band whose interior sits this far off-centre ends the lobe
BUBBLE_TRIM_WIDTH_SHARE = 0.5  # a band narrower than this share of the widest ends the lobe
BUBBLE_TRIM_JUMP_ROWS = 12  # a thin stroke crossing the bubble is jumped, not a lobe end
BUBBLE_TRIM_HEIGHT_SHARE = 0.4  # a trim keeping less than this share of the height is no trim
RAY_EDGE_THRESHOLD = 80  # a Sobel step this strong is an outline, not paper noise
RAY_LOBE_SLACK = 16  # interior this far past a ray frame is a lost lobe, not edge width
BOX_RULE_EXPAND = 64  # how far past a region a narration frame's rules are searched for
DIVIDER_ROW_SHARE = 0.6  # a row this full of strong edge is a panel divider
DIVIDER_KEEP_SHARE = 0.4  # a cut keeping less than this share of the height is no cut


def _flood_seeds(w: int, h: int, letter_box: tuple[int, int, int, int] | None = None) -> list[tuple[int, int]]:
    """Seed points for a paper flood, in crop coordinates.

    Letters sit inside the bubble, so seeds sampled over the letter area
    flood the interior; a seed on paper outside the bubble floods the
    surroundings instead, and the union then calls that paper the bubble.
    Without a letter area the old nine-point grid over the whole crop
    stands, since a flood still needs a start.
    """
    if letter_box is not None:
        lx1, ly1, lx2, ly2 = (int(v) for v in letter_box)
        lx1, ly1 = max(0, lx1), max(0, ly1)
        lx2, ly2 = min(w, lx2), min(h, ly2)
        if lx2 - lx1 >= 4 and ly2 - ly1 >= 4:
            # Most points land on the paper between letters; one good
            # seed floods the whole interior, and fills grown from a
            # seed on a dark letter are too small and grey-wrong to join
            # the union.
            return [(lx1 + (2 * i + 1) * (lx2 - lx1) // 10,
                     ly1 + (2 * j + 1) * (ly2 - ly1) // 6)
                    for j in range(3) for i in range(5)]
    return [(w // 2, h // 2),
            (w // 4, h // 4), (3 * w // 4, h // 4), (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4),
            (w // 2, h // 4), (w // 2, 3 * h // 4), (w // 4, h // 2), (3 * w // 4, h // 2)]


def _paper_union(grey, seeds):
    """Union of the flood fills sharing the largest fill's paper, or None.

    A bubble split by a panel edge or a tail stroke floods in pieces; the
    pieces share the paper, so their union is the interior, not just the
    largest piece. The largest fill names the paper even when the centre
    seed lands outside the bubble, so art fills in another grey stay out.
    """
    try:
        import cv2
    except ImportError:
        return None
    h, w = grey.shape
    fills = []
    for sx, sy in seeds:
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        canvas = grey.copy()
        try:
            # Bits 8-15 of the flags are the value written into the mask; without
            # them the flood leaves the mask blank. FLOODFIXED_RANGE is 1<<16
            # (this cv2 build does not expose the name).
            cv2.floodFill(canvas, mask, (sx, sy), 0,
                          BAND_FILL_TOLERANCE, BAND_FILL_TOLERANCE,
                          4 | (1 << 16) | (255 << 8))
        except Exception:
            continue
        fills.append((mask[1:-1, 1:-1] > 0, int(grey[sy, sx])))
    if not fills:
        return None
    largest, paper_grey = max(fills, key=lambda item: float(item[0].mean()))
    filled = None
    for candidate, seed_grey in fills:
        if abs(seed_grey - paper_grey) <= BAND_FILL_TOLERANCE:
            filled = candidate if filled is None else (filled | candidate)
    return filled


def _row_runs(on):
    """Maximal runs of True in a row, hairline cracks merged."""
    cols = np.nonzero(on)[0]
    if len(cols) == 0:
        return []
    runs = []
    start = prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c - prev > 3:
            runs.append((start, prev + 1))
            start = c
        prev = c
    runs.append((start, prev + 1))
    return runs


def _outline_gap(mag_row, a: int, b: int):
    """Gap between strong outlines holding the middle of ``(a, b)``.

    A flood fill runs through a faint outline into same-grey surroundings,
    so its interval overstates the interior; the outline loop itself is
    still drawn, so the gap between its strokes is the interior that counts.
    Returns None when the row holds no outline, False when its middle sits
    on an outline stroke and the row says nothing about the interior.
    """
    cols = np.nonzero(mag_row[a:b] > RAY_EDGE_THRESHOLD)[0] + a
    if len(cols) == 0:
        return None
    spans = []
    start = prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c - prev > 2:
            spans.append((start, prev + 1))
            start = c
        prev = c
    spans.append((start, prev + 1))
    if len(spans) > 6:
        # Too many strong edges to be one outline loop bracketing the
        # interior: diagonal shading (or other interior art) fragments
        # the run into strips, and the gap holding the middle is one
        # strip, not the bubble. Cutting to it strands the text in a
        # sliver. A lobe waist or a leaked flood shows few spans — the
        # outline loop itself — so those still cut below.
        return None
    mid = (a + b) // 2
    left = a
    for s, e in spans:
        if s > left and left <= mid < s:
            return (left, s)
        left = max(left, e)
    if left < b and left <= mid < b:
        return (left, b)
    return False


def _bands_split_by_waist(bands, box_w: int) -> bool:
    """Whether a near-empty waist splits the flood into two flooded parts.

    Centred lines land on the waist and size to it, while the lobes above
    and below hold far wider rows — the bands describe two regions, not
    one bubble's interior. A bubble that simply ends partway down the box
    leaves empty bands with flood on one side only, and is not split.
    The run must be at least 3 bands: a 2-band dip is the narrowing step
    misreading letter rows (slice 25's bands 1-2 collapse to 10/2 px
    while the rows either side hold ~360), and the flood either side must
    clear half the box. Calibrated on slice 81 (3-band waist, 0.57/0.99
    flood) vs slice 25 (2-band dip, 0.47/0.45 flood).
    """
    if not bands or box_w <= 0:
        return False
    return _coverage_waist([sum(b - a for a, b in band) / box_w for band in bands])


def _coverage_waist(cov) -> bool:
    """Whether a >=3-row near-empty run separates two flooded halves."""
    start = None
    for i, c in enumerate(cov):
        if c < 0.15:
            if start is None:
                start = i
            continue
        if start is not None and i - start >= 3:
            if start > 0 and max(cov[:start]) > 0.5 and max(cov[i:]) > 0.5:
                return True
        start = None
    return False


_BANDS_CACHE: dict = {}
_BANDS_CACHE_MAX = 128


def _bubble_bands(image, box, band_count: int = BAND_COUNT, verdict: dict | None = None,
                  letter_box: tuple[int, int, int, int] | None = None):
    """Per-band ink intervals of the bubble interior holding ``box``, or None.

    The box sits inside its bubble, so flooding outward from several seeds on the
    cleaned image stops at the bubble outline; each band's intervals are the widths
    a centred line may use on those rows. A box the flood cannot cover — free text
    on art, a box cut by a panel edge — falls back to the rectangle/ellipse fit.
    When ``verdict`` is given, narrowing the flood leaves a trace there, so a
    fit that shrank can be told apart from one that never narrowed.
    ``letter_box`` seeds the flood from the letter area (absolute coordinates);
    without it the seeds spread over the whole box and one on paper outside
    the bubble can define the flood.
    Bands are cached per (image, box): one fit measures them several times
    on the same box, and the flood plus per-row gap analysis is the slow
    part of the render. The cache holds the image so its id cannot be
    reused by another image while an entry stands.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    width, height = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    grey = np.asarray(image.convert("L"))[y1:y2, x1:x2]
    if grey.size == 0:
        return None
    letter_key = tuple(int(v) for v in letter_box) if letter_box is not None else None
    cache_key = (id(image), (x1, y1, x2, y2), band_count, letter_key)
    hit = _BANDS_CACHE.get(cache_key)
    if hit is not None:
        _image_ref, bands, delta = hit
        if verdict is not None:
            for key, value in delta.items():
                if isinstance(value, bool):
                    verdict[key] = verdict.get(key, False) or value
                else:
                    verdict[key] = verdict.get(key, 0) + value
        return [list(band) for band in bands] if bands is not None else None
    h, w = grey.shape
    crop_letter = None
    if letter_box is not None:
        lx1, ly1, lx2, ly2 = (int(v) for v in letter_box)
        crop_letter = (lx1 - x1, ly1 - y1, lx2 - x1, ly2 - y1)
    letter_seeded = (crop_letter is not None and crop_letter[2] - crop_letter[0] >= 4
                     and crop_letter[3] - crop_letter[1] >= 4)
    filled = _paper_union(grey, _flood_seeds(w, h, crop_letter))
    delta: dict = {}
    waist_rows = False
    # A letter-seeded flood is the bubble the letters sit in, however
    # small a share of the box that is; the half-box floor only judges
    # grid-seeded floods, where a small fill is art, not a bubble.
    min_fill = 0.02 if letter_seeded else BAND_MIN_FILL
    if filled is None or float(filled.mean()) < min_fill:
        delta["bands_none"] = True
        if verdict is not None:
            verdict["bands_none"] = True
        _BANDS_CACHE[cache_key] = (image, None, delta)
        if len(_BANDS_CACHE) > _BANDS_CACHE_MAX:
            _BANDS_CACHE.pop(next(iter(_BANDS_CACHE)))
        return None
    try:
        import cv2

        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        mag_crop = np.hypot(gx, gy)
    except Exception:
        mag_crop = None
    bands = []
    gap_cut_rows = 0
    vote_bands = 0
    rolled_back = 0
    kept_narrowed = 0
    for i in range(band_count):
        top = i * h // band_count
        bottom = (i + 1) * h // band_count
        rows = filled[top:bottom]
        if rows.size == 0:
            bands.append([])
            continue
        if mag_crop is not None:
            cut = np.zeros_like(rows)
            centre = w // 2
            for r in range(rows.shape[0]):
                runs = _row_runs(rows[r])
                centre_run = None
                for a, b in runs:
                    if a <= centre < b:
                        centre_run = (a, b)
                        break
                for a, b in runs:
                    # A run on paper outside the bubble: it reaches the
                    # crop edge, the run holding the centre is a bounded
                    # interior run, and a strong edge (the outline)
                    # stands between the two. The gap-cut cannot see it
                    # — a paper run holds no outline inside itself, so
                    # the shave rule keeps it whole — and the cross-row
                    # union then calls the paper the interior, so a line
                    # clamped to the band crosses the outline where the
                    # bubble pinches. Drop it; the centre run is the
                    # interior that counts. Letter-seeded only: the
                    # letter area anchors the flood to one bubble, so a
                    # run reaching the crop edge is paper outside it.
                    # Without that anchor a run reaching the edge can be
                    # the interior itself (a bubble wider than its box),
                    # or a letter fragment in an uncleaned image, and
                    # dropping it shrinks healthy renders. A single-run
                    # row has no other run to drop against, and a row
                    # whose centre is not a bounded interior run is left
                    # alone.
                    if (letter_seeded and centre_run is not None
                            and (a, b) != centre_run
                            and centre_run[0] > 0 and centre_run[1] < w
                            and (a == 0 or b == w)):
                        lo, hi = ((b, centre_run[0]) if b <= centre_run[0]
                                  else (centre_run[1], a))
                        if lo < hi and bool(
                                (mag_crop[top + r, lo:hi]
                                 > RAY_EDGE_THRESHOLD).any()):
                            continue
                    gap = _outline_gap(mag_crop[top + r], a, b)
                    if gap is False:
                        continue
                    if gap is not None and gap[1] - gap[0] >= 0.85 * (b - a):
                        # A shave this small is the outline's own width, not
                        # a leak; cutting it only re-wraps healthy renders.
                        gap = None
                    if gap is not None:
                        gap_cut_rows += 1
                    ga, gb = gap if gap is not None else (a, b)
                    cut[r, ga:gb] = True
            refined = cut if cut.any() else rows
            waist_rows = _coverage_waist((refined.sum(axis=1) / max(1, w)).tolist())
            counts = [len(_row_runs(refined[r])) for r in range(refined.shape[0])
                      if refined[r].any()]
            cols_any = refined.any(axis=0)
            if (len(set(counts)) > 1
                    or len(_row_runs(cols_any)) < (max(counts) if counts else 0)):
                # Rows disagreeing about how many pieces the interior has
                # are a leak arguing with the outline: one faint-outline
                # row's merged run must not widen the whole band, so only
                # columns most rows agree on count as interior. A union
                # that merges pieces the rows keep separate is the same
                # argument in a single run — the outline between the
                # pieces drifts a few pixels row to row, so the union
                # bridges it while no row does.
                vote_bands += 1
                votes = refined.sum(axis=0)
                cols = votes > len(counts) // 2
            else:
                cols = cols_any
        else:
            cols = rows.any(axis=0)
        merged = _row_runs(cols)
        if mag_crop is not None:
            # Narrowing is undone band by band where it removed no
            # outline: columns the narrowers cut that hold no strong
            # edge are true interior (a tapering lobe, not a leak), so
            # the wider measurement stands. Columns cut past a strong
            # edge stay cut — that edge is the leak's outline.
            raw = _row_runs(rows.any(axis=0))
            if raw != merged:
                narrowed_set = set()
                for a, b in merged:
                    narrowed_set.update(range(a, b))
                extra = [c for a, b in raw for c in range(a, b)
                         if c not in narrowed_set]
                if extra and not bool((mag_crop[top:bottom, extra] > RAY_EDGE_THRESHOLD).any()):
                    merged = raw
                    rolled_back += 1
                else:
                    kept_narrowed += 1
        # Ink drawn to a band's edge touches the outline where the bubble
        # narrows, so each band keeps a small margin inside its own width.
        bands.append([(a + BAND_EDGE_PAD, b - BAND_EDGE_PAD) if b - a > 2 * BAND_EDGE_PAD else (a, b)
                      for a, b in merged])
    delta["gap_cut_rows"] = gap_cut_rows
    delta["vote_bands"] = vote_bands
    delta["bands_rolled_back"] = rolled_back
    delta["bands_kept_narrowed"] = kept_narrowed
    if gap_cut_rows or vote_bands:
        delta["narrowed"] = True
    if waist_rows or _bands_split_by_waist(bands, x2 - x1):
        # A waist-split flood sizes centred lines to the waist while the
        # lobes hold far wider rows; the rectangle/ellipse fit is the
        # honest fallback, not a fit to the narrowest rows.
        delta["bands_none"] = True
        delta["bands_split_waist"] = True
        if verdict is not None:
            for key, value in delta.items():
                if isinstance(value, bool):
                    verdict[key] = verdict.get(key, False) or value
                else:
                    verdict[key] = verdict.get(key, 0) + value
        _BANDS_CACHE[cache_key] = (image, None, delta)
        if len(_BANDS_CACHE) > _BANDS_CACHE_MAX:
            _BANDS_CACHE.pop(next(iter(_BANDS_CACHE)))
        return None
    if verdict is not None:
        for key, value in delta.items():
            if isinstance(value, bool):
                verdict[key] = verdict.get(key, False) or value
            else:
                verdict[key] = verdict.get(key, 0) + value
    _BANDS_CACHE[cache_key] = (image, [list(band) for band in bands], delta)
    if len(_BANDS_CACHE) > _BANDS_CACHE_MAX:
        _BANDS_CACHE.pop(next(iter(_BANDS_CACHE)))
    return bands


def _box_frame(image: Image.Image, box):
    """Frame interior of a narration box, from its own rules, or None.

    A narration frame is drawn as long dark rules; on the cleaned image
    the interior is paper, so a column (or row) dark over most of the
    box can only be a rule. One dimension is measured only when its two
    rules both show; a missing dimension keeps the region edge, and no
    rules at all means the frame is unmeasured, not guessed.
    """
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
        # A rule can sit flush with the region edge: the detector
        # region usually stops just past the rule, so the run ends
        # at the crop edge and the 2 px guard drops a genuine rule,
        # leaving the dimension unmeasured. The strict pass runs
        # first, so a box that already measures keeps its frame;
        # only a short-handed dimension retries with edge runs
        # admitted. A run touching the edge still shows the end the
        # frame uses — the left rule's inner end, the right rule's
        # inner start — and the pairing geometry below (straddling
        # the centre, interior at least half the region) is what
        # keeps an edge artifact from becoming a frame.
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
    # A region can sit inset from its frame, so a rule just past the
    # region edge is still this box's rule — but only when the strip
    # between rule and region is paper, which a neighbour's rule across
    # a panel gap is not. Each dimension is searched on its own expanded
    # crop, with the share denominator kept at the region extent, so
    # expanding one dimension cannot unfind the other's rules.
    # A stepped frame straddles the region edge: one rule sits inside
    # the region while the other sits just outside it, so neither the
    # inside-only pass nor the outside-only pairing below can pair
    # them. A missing outside side falls back to the inside pass's
    # outermost group on that side, converted to expanded coordinates;
    # the paper strip is only judged where the rule lies outside the
    # region, since the inside pass already trusts an inside rule
    # with no strip. A rule flush with the region edge has no strip
    # to judge, so a missing strip is skipped, not failed. The
    # fallback runs only where the inside pass measured nothing: a
    # dimension it already measured keeps its frame, so an outside
    # neighbour's rule cannot widen it through the fallback.
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
        # A group belongs to a side when its bulk lies past the region
        # edge: a rule ending a pixel past the edge still belongs to
        # that side, while a rule inside the region belongs to neither
        # side here and is reached through the fallback below instead.
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
            # The column hugging the rule carries the rule's own
            # antialiased edge, not the strip: judge the strip
            # without it, and a strip too thin to judge after the
            # trim is no strip at all — a panel gap is wide, so a
            # sliver cannot be the evidence that rejects the rule.
            if trim_left:
                strip = strip[:, 1:]
            else:
                strip = strip[:, :-1]
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
    """Interior of a system window, from its bright border lines, per dimension.

    A window's border is a bright line with a glow halo, not a dark rule,
    so it is found by brightness, not darkness: a thin column (or row)
    bright over most of the window's middle is a border side. A dimension
    is measured only when both its sides show such a line reaching into
    the region; a saturated stretch of bright art, a border cut off by
    the region edge, and a window bigger than its region all fail that
    gate and keep the inset for that dimension instead of guessing.
    Returns ``(fx1, fx2, fy1, fy2, measured_x, measured_y)``.
    """
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
        """Inner edge of the border line nearest ``edge``, or None.

        ``share`` runs over image coordinates starting at ``lo``; the
        line must reach into the region (a line entirely outside it is
        art beside the window, or a window bigger than its region).
        """
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

    left = side((grey[my1:my2, max(0, x1 - 16):x1 + 64] > 180).mean(axis=0),
                max(0, x1 - 16), x1, True)
    right = side((grey[my1:my2, x2 - 64:min(width, x2 + 16)] > 180).mean(axis=0),
                 x2 - 64, x2, False)
    top = side((grey[max(0, y1 - 16):y1 + 64, mx1:mx2] > 180).mean(axis=1),
               max(0, y1 - 16), y1, True)
    bottom = side((grey[y2 - 64:min(height, y2 + 16), mx1:mx2] > 180).mean(axis=1),
                  y2 - 64, y2, False)
    fx1, fx2, measured_x = x1, x2, False
    if left is not None and right is not None and left < (x1 + x2) // 2 < right \
            and right - left >= 40:
        fx1, fx2, measured_x = left, right, True
    fy1, fy2, measured_y = y1, y2, False
    if top is not None and bottom is not None and top < (y1 + y2) // 2 < bottom \
            and bottom - top >= 40:
        fy1, fy2, measured_y = top, bottom, True
    return (fx1, fx2, fy1, fy2, measured_x, measured_y)


def _two_lobes(bands, box_w: int, frame_measured: bool = True) -> bool:
    """Whether the measured bands hold two wide lobes with drifting centres.

    A tail is narrow, so it never counts as a lobe; only intervals at
    least half the widest count, and only when such intervals sit in both
    halves of the box with centres drifted apart. Every interval in a
    band counts, not just its widest: lobes sitting side by side share
    rows, so a band holding two wide intervals is two lobes, not one.
    A single bent body drifts too, but its wide intervals share one
    centre line; two lobes do not.
    With no frame measured the bands themselves are unproven — a flood
    leaked through a faint outline reads as two lobes — so the verdict
    is withheld and the trim runs instead of being vetoed by it.
    """
    if not bands or box_w <= 0 or not frame_measured:
        return False
    widest = 0
    for band in bands:
        for a, b in band:
            widest = max(widest, b - a)
    if widest <= 0:
        return False
    wide = [(i, (a + b) / 2) for i, band in enumerate(bands)
            for a, b in band if b - a >= 0.5 * widest]
    if len(wide) < 2:
        return False
    centres = [c for _, c in wide]
    if max(centres) - min(centres) <= 0.15 * box_w:
        return False
    half = len(bands) // 2
    return any(i < half for i, _ in wide) and any(i >= half for i, _ in wide)


def _edge_magnitude(image):
    """Sobel strength per pixel, for outlines too grey to stop a flood fill."""
    import cv2

    grey = np.asarray(image.convert("L"))
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _divider_cut(mag, box):
    """The box cut at a panel divider, keeping the side holding its centre.

    A bubble straddling a panel edge is lettered on the side its text came
    from; the divider row itself is a strong edge spanning the box. A cut
    that would keep too little, or a centre sitting on the divider, is no cut.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return box
    band = mag[y1:y2, x1:x2]
    if band.size == 0:
        return box
    score = (band > RAY_EDGE_THRESHOLD).mean(axis=1)
    rows = [i for i, s in enumerate(score) if s >= DIVIDER_ROW_SHARE]
    if not rows:
        return box
    groups = [[rows[0]]]
    for row in rows[1:]:
        if row - groups[-1][-1] <= 2:
            groups[-1].append(row)
        else:
            groups.append([row])
    cy = (y1 + y2) // 2 - y1
    for group in groups:
        if abs(cy - group[0]) <= 10 or abs(cy - group[-1]) <= 10:
            return box
    top, bottom = 0, y2 - y1
    for group in groups:
        if group[-1] < cy:
            top = max(top, group[-1] + 1)
        elif group[0] > cy:
            bottom = min(bottom, group[0])
    if bottom - top < DIVIDER_KEEP_SHARE * (y2 - y1):
        return box
    return (x1, y1 + top, x2, y1 + bottom)


def _seed_points(mag, dist, box):
    """Flat, deep points spread over ``box``, for ray-casting its bubble.

    One seed can sit on a hatched outline or a tapering tail, where rays
    find the edge band instead of the bubble. Several seeds across the box
    cover the bubble body; a seed on art or an outline is not flat or deep
    enough to count.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    bx, by = x2 - x1, y2 - y1
    height, width = mag.shape
    points = []
    for sx, sy in [(x1 + bx // 2, y1 + by // 2),
                   (x1 + bx // 4, y1 + by // 4), (x1 + 3 * bx // 4, y1 + by // 4),
                   (x1 + bx // 4, y1 + 3 * by // 4), (x1 + 3 * bx // 4, y1 + 3 * by // 4),
                   (x1 + bx // 2, y1 + by // 4), (x1 + bx // 2, y1 + 3 * by // 4),
                   (x1 + bx // 4, y1 + by // 2), (x1 + 3 * bx // 4, y1 + by // 2)]:
        if not (0 <= sx < width and 0 <= sy < height):
            continue
        if float(mag[sy, sx]) > RAY_EDGE_THRESHOLD:
            continue
        if float(dist[sy, sx]) < 20:
            continue
        points.append((sx, sy))
    return points


def _ray_loses_lobe(bands, box, pad, ray):
    """True when a ray frame cuts off interior the cut side's flood proves.

    A ray from inside one lobe stops at that lobe's outline, so a ray
    frame can be one lobe of a double bubble while claiming the whole.
    Only a two-lobed interior can lose a lobe, and only interior sticking
    out of the ray frame counts, so a ray that genuinely recovers the
    whole bubble is never rejected for it.
    """
    if not bands:
        return False
    cx1, cy1, cx2, cy2 = (int(v) for v in box)
    if not _two_lobes(bands, cx2 - cx1 - 2 * pad):
        return False
    left = right = top = bottom = None
    h = cy2 - cy1 - 2 * pad
    for i, band in enumerate(bands):
        if not band:
            continue
        if top is None:
            top = cy1 + pad + i * h // len(bands)
        bottom = cy1 + pad + (i + 1) * h // len(bands)
        for a, b in band:
            left = a if left is None else min(left, a)
            right = b if right is None else max(right, b)
    if left is None:
        return False
    left += cx1 + pad
    right += cx1 + pad
    return (left < ray[0] - RAY_LOBE_SLACK or right > ray[2] + RAY_LOBE_SLACK
            or top < ray[1] - RAY_LOBE_SLACK or bottom > ray[3] + RAY_LOBE_SLACK)


def _ray_boxes(mag, grey, box, seeds):
    """Bounding box of the outline loop around ``box``, or None.

    The detection region is a guess at the bubble: shifted, a subset, or
    stretched past a panel edge. Casting rays from flat points inside it
    to the first strong edge in each direction recovers the bubble itself —
    or its lobe, since a ray from inside a lobe stops at that lobe's
    outline. Each seed's box must look like this object's bubble on its
    own; their union is the bubble's full extent. A box that is mostly art,
    touches a slice edge like a panel leak, or sits far from the region is
    not this object's bubble, so the caller keeps the region box.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    height, width = mag.shape
    if x2 - x1 < 8 or y2 - y1 < 8 or not seeds:
        return None
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    box_area = max(1, (x2 - x1) * (y2 - y1))

    def first_edge(fixed, start, step, axis):
        hits = []
        for off in (-20, -10, 0, 10, 20):
            pos = start + step * 2
            line = fixed + off
            if axis == 0:
                if not (0 <= line < height):
                    continue
                row = mag[line, :]
                bound = width
            else:
                if not (0 <= line < width):
                    continue
                row = mag[:, line]
                bound = height
            while 0 <= pos < bound:
                if row[pos] > RAY_EDGE_THRESHOLD:
                    hits.append(pos)
                    break
                pos += step
        if len(hits) < 3:
            return None
        hits.sort()
        return hits[len(hits) // 2]

    def guards(rx1, ry1, rx2, ry2):
        if rx1 < BUBBLE_BOX_EDGE_INSET or rx2 > width - BUBBLE_BOX_EDGE_INSET:
            return False
        fill_area = (rx2 - rx1) * (ry2 - ry1)
        if not (BUBBLE_BOX_AREA_MIN * box_area <= fill_area <= BUBBLE_BOX_AREA_MAX * box_area):
            return False
        centre_dist = max(abs((rx1 + rx2) // 2 - cx), abs((ry1 + ry2) // 2 - cy))
        if centre_dist > BUBBLE_BOX_CENTRE_SHARE * max(x2 - x1, y2 - y1):
            return False
        paper = float((grey[ry1:ry2, rx1:rx2] >= 180).mean()) if ry2 > ry1 and rx2 > rx1 else 0.0
        return paper >= BUBBLE_BOX_PAPER_MIN

    passing = []
    for sx, sy in seeds:
        rx1 = first_edge(sy, sx, -1, 0)
        rx2 = first_edge(sy, sx, 1, 0)
        ry1 = first_edge(sx, sy, -1, 1)
        ry2 = first_edge(sx, sy, 1, 1)
        if None in (rx1, rx2, ry1, ry2):
            continue
        if not guards(rx1, ry1, rx2, ry2):
            continue
        # A ray can run through a faded outline into the bubble above;
        # the region box is the same object's detection, so the bubble
        # cannot lie far beyond it in any direction. The guards run on
        # the unclipped box first, so clipping cannot drag a neighbour's
        # bubble into passing range.
        cx1 = max(rx1, x1 - (x2 - x1) // 4)
        cx2 = min(rx2, x2 + (x2 - x1) // 4)
        cy1 = max(ry1, y1 - (y2 - y1) // 4)
        cy2 = min(ry2, y2 + (y2 - y1) // 4)
        if guards(cx1, cy1, cx2, cy2):
            passing.append((cx1, cy1, cx2, cy2))
    if not passing:
        return None
    ux1 = min(b[0] for b in passing)
    uy1 = min(b[1] for b in passing)
    ux2 = max(b[2] for b in passing)
    uy2 = max(b[3] for b in passing)
    if guards(ux1, uy1, ux2, uy2):
        return (ux1, uy1, ux2, uy2)
    passing.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return passing[0]


def _trim_bubble_box(image, bbox, letter_box: tuple[int, int, int, int] | None = None):
    """The lobe of ``bbox`` holding its widest row, or ``bbox`` untrimmed.

    Two lobes share one outline with an open junction, and a tail narrows
    away from the body; lines centred in the whole balloon sit badly in
    either. Walking out from the widest row while rows stay wide and
    centred keeps the lobe the text belongs to. A thin stroke crossing the
    bubble breaks the fill for a few rows and then the lobe resumes, so a
    short bad run with a good row past it is jumped, not a lobe end; a
    divider band never resumes, and neither does a lobe end. A trim that
    would keep too little of the balloon is not a lobe, so the whole box
    stands.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    width, height = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return bbox
    grey = np.asarray(image.convert("L"))[y1:y2, x1:x2]
    if grey.size == 0:
        return bbox
    h, w = grey.shape
    crop_letter = None
    if letter_box is not None:
        lx1, ly1, lx2, ly2 = (int(v) for v in letter_box)
        crop_letter = (lx1 - x1, ly1 - y1, lx2 - x1, ly2 - y1)
    filled = _paper_union(grey, _flood_seeds(w, h, crop_letter))
    if filled is None or float(filled.mean()) < BAND_MIN_FILL:
        return bbox
    centre = w // 2

    def holding(row):
        cols = np.nonzero(row)[0]
        if cols.size == 0:
            return None
        # Intervals of this row, merged over hairline cracks.
        intervals = []
        start = int(cols[0])
        prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c - prev > 3:
                intervals.append((start, prev + 1))
                start = c
            prev = c
        intervals.append((start, prev + 1))
        for a, b in intervals:
            if a <= centre <= b:
                return (a, b)
        return None

    held = [holding(filled[i]) for i in range(h)]
    anchor = None
    anchor_width = 0
    for i, iv in enumerate(held):
        if iv is not None and iv[1] - iv[0] > anchor_width:
            anchor, anchor_width = i, iv[1] - iv[0]
    if anchor is None or anchor_width <= 0:
        return bbox

    def good(iv):
        if iv is None or iv[1] - iv[0] < BUBBLE_TRIM_WIDTH_SHARE * anchor_width:
            return False
        return abs((iv[0] + iv[1]) // 2 - centre) <= BUBBLE_TRIM_DEVIATION_SHARE * w

    mag = None

    def divider_like(rows):
        # A panel divider is a continuous dark line across the box; a wash
        # band crossing it is edge-dense but mid-grey, not dark, so edge
        # strength alone cannot tell them apart — darkness can.
        return all(float((grey[i] < 110).mean()) >= DIVIDER_ROW_SHARE for i in rows)

    def jump_up(from_row):
        for k in range(1, BUBBLE_TRIM_JUMP_ROWS + 1):
            if from_row - k < 0:
                return None
            if good(held[from_row - k]):
                return k
        return None

    def jump_down(from_row):
        for k in range(1, BUBBLE_TRIM_JUMP_ROWS + 1):
            if from_row + k >= h:
                return None
            if good(held[from_row + k]):
                return k
        return None

    top = anchor
    while top > 0:
        if good(held[top - 1]):
            top -= 1
        else:
            jump = jump_up(top)
            if jump is not None and not divider_like(range(top - jump + 1, top)):
                top -= jump
            else:
                break
    bottom = anchor
    while bottom < h - 1:
        if good(held[bottom + 1]):
            bottom += 1
        else:
            jump = jump_down(bottom)
            if jump is not None and not divider_like(range(bottom + 1, bottom + jump)):
                bottom += jump
            else:
                break
    if bottom + 1 - top < BUBBLE_TRIM_HEIGHT_SHARE * h:
        return bbox
    return (x1, y1 + top, x2, y1 + bottom + 1)


def _band_line_widths(box_w: int, box_h: int, line_count: int, line_height: int, bands, top: int | None = None) -> list[int] | None:
    """Widest ink each straight line may hold inside the measured bubble bands, or None.

    A line's ink spans every row it covers, so it must sit inside the
    intersection of its rows' interiors, not just inside each row on its
    own: a bent balloon's lobes drift sideways, and a line centred on the
    box crosses an edge long before it fills any one lobe. A line whose
    rows' interiors never overlap cannot stay inside the bubble at all
    and the caller falls back.
    """
    spans = _band_line_spans(box_w, box_h, line_count, line_height, bands, top=top)
    if spans is None:
        return None
    return [b - a for a, b in spans]


def _band_line_spans(box_w: int, box_h: int, line_count: int, line_height: int, bands, top: int | None = None) -> list[tuple[int, int]] | None:
    """Per-line interior interval a straight line may use, or None."""
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
    """Centre of each line's interior interval, for drawing on the lobe, or None."""
    spans = _band_line_spans(box_w, box_h, line_count, line_height, bands, top=top)
    if spans is None:
        return None
    return [(a + b) / 2 for a, b in spans]


def _best_band_top(draw, lines, font, box_w: int, box_h: int, stroke_w: int, bands) -> int | None:
    """Top offset where every line fits its bands, closest to centred, or None."""
    if not lines:
        return None
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)
    if total_h > box_h:
        return None
    widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    centred = max(0, (box_h - total_h) // 2)
    step = max(4, line_height // 4)
    candidates = sorted(range(0, box_h - total_h + 1, step),
                        key=lambda top: abs(top - centred))
    for top in candidates:
        limits = _band_line_widths(box_w, box_h, len(lines), line_height, bands, top=top)
        if limits is not None and len(limits) == len(lines) and all(
                w + 2 * stroke_w <= limit for w, limit in zip(widths, limits)):
            return top
    return None


def _lines_fit_bands(draw, lines, font, box_w: int, box_h: int, stroke_w: int, bands) -> bool:
    """Whether every line's ink plus stroke stays inside its measured bubble bands."""
    if not lines:
        return True
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    limits = _band_line_widths(box_w, box_h, len(lines), line_height, bands)
    if limits is None or len(limits) != len(lines):
        return False
    return all(
        draw.textbbox((0, 0), line, font=font)[2] + 2 * stroke_w <= limit
        for line, limit in zip(lines, limits))


def _fits(draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int,
          curved: bool = False, bands=None) -> tuple[bool, list[str]]:
    font = get_font_object(font_path_str, size)
    lines = _wrap_text(draw, text, font, box_w)
    if not lines:
        return False, []
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    total_h = line_height * len(lines)
    widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    # A word broken across lines ("Khôn/g...") does not fit; a smaller size is tried first.
    whole = [word for line in lines for word in line.split()] == text.split()
    fits = whole and total_h <= box_h and max(widths) + 2 * stroke_w <= box_w
    if fits:
        if bands is not None:
            # Greedy lines fill the box width, but a bent balloon's rows
            # hold less; re-wrap at the narrowest span until the lines sit
            # inside their rows, or no narrower wrap helps.
            for _ in range(3):
                spans = _band_line_spans(box_w, box_h, len(lines), line_height, bands)
                if spans is None:
                    fits = False
                    break
                widths_now = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
                if all(w + 2 * stroke_w <= b - a for w, (a, b) in zip(widths_now, spans)):
                    break
                narrow = min(b - a for a, b in spans) - 2 * stroke_w
                if narrow <= 0:
                    fits = False
                    break
                new_lines = _wrap_text(draw, text, font, narrow)
                if (not new_lines or new_lines == lines
                        or [w for line in new_lines for w in line.split()] != text.split()):
                    fits = False
                    break
                lines = new_lines
                if line_height * len(lines) > box_h:
                    fits = False
                    break
            else:
                fits = False
        elif curved:
            limits = _ellipse_line_widths(box_w, box_h, len(lines), line_height)
            fits = len(limits) == len(lines) and all(
                w + 2 * stroke_w <= limit for w, limit in zip(widths, limits))
    return fits, lines


def _scan_sizes(hi: int, lo: int, limit: int = 200):
    """Sizes from hi down to lo + 1, largest first.

    Exact when the range is short; a coarse grid when it is long, so a
    bogus cap cannot make the fit scan hundreds of sizes. The smallest
    size is always tried, since a fit there is the last resort.
    """
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

    if (curved or bands is not None) and best_lines:
        # The ellipse/band predicate is not monotonic in size: a word break can
        # push a long line into a narrow chord at one size and not the next, so
        # a binary search can land far below the largest fitting size. Scan down
        # from the rectangle fit, which bounds every curved fit from above.
        for size in _scan_sizes(best_size, minimum_size - 1):
            ok, lines = _fits(draw, text, box_w, box_h, font_path_str, size, stroke_w,
                              curved=curved, bands=bands)
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
    """Words filled into lines no wider than ``width``, kept as word lists."""
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
    """Words filled into lines no wider than ``width``."""
    return [" ".join(group) for group in _greedy_groups(draw, words, font, width)]


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
    """As few lines as fit ``box_w``, made as even as those lines allow, with a lone last word pulled down
    from the previous line when it still fits."""
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
    groups = _greedy_groups(draw, words, font, hi)
    if len(groups) >= 2 and len(groups[-1]) == 1 and len(groups[-2]) >= 2:
        # Even widths can still strand one word alone on the last line;
        # moving the previous line's last word down fixes it when it
        # still fits, and the group move never splits a joined word.
        moved = groups[-2][-1]
        new_last = " ".join([moved] + groups[-1])
        if draw.textbbox((0, 0), new_last, font=font)[2] <= box_w:
            groups[-2] = groups[-2][:-1]
            groups[-1] = [moved] + groups[-1]
    return [" ".join(group) for group in groups]


def _wrap_text(draw, text: str, font, box_w: int, *, balance: bool = False,
               join_lines: bool = False) -> list[str]:
    """Lines of ``text`` no wider than ``box_w``; ``balance`` evens each wrapped line at the size finally drawn."""
    if join_lines:
        # A stored break is a guess about the box, not part of the words:
        # joining first lets a wide line stay one line instead of two small ones.
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
