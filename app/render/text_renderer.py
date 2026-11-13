from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_mask import mask_inscribed_rect
from app.parameters import (
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    RENDER_AUTO_STROKE_WIDTH,
    RENDER_DEFAULT_PADDING,
    RENDER_MIN_READABLE_FONT_SIZE,
    RENDER_PADDING_RATIO_MAX,
    RENDER_SAFE_CAPTION_EXPANSION,
    RENDER_SAFE_CAPTION_EXPANSION_RATIO,
    RENDER_STROKE_WIDTH_MAX,
)
from app.render.font_metrics import (
    LOW_CONTRAST_RATIO,
    LOW_CONTRAST_STROKE_WIDTH,
    _calc_line_height,
    _contrast,
    _draws_char,
    _light_background_grey,
    _line_start_x,
    _luminance,
    _missing_glyph_mask,
    _plain_punctuation,
    auto_detect_text_color,
    drawable,
    font_draws_text,
    get_font_object,
    get_font_path,
    list_available_fonts,
    parse_color,
)
from app.render.text_fitting import (
    BOX_RULE_EXPAND,
    ENLARGED_MIN_FONT_SIZE,
    ENLARGE_GROW_RATIO,
    MASK_BAND_PAD,
    REFERENCE_PAGE_WIDTH,
    SOURCE_FLOOR_RATIO,
    SOURCE_MATCH_MAX_FONT_SIZE,
    _CURVED_CONTAINERS,
    _DERIVED_MASK_CACHE,
    _anchored_top,
    _balanced_lines,
    _band_line_centres,
    _band_line_spans,
    _band_line_widths,
    _box_frame,
    _derived_mask,
    _ellipse_line_widths,
    _fit_text,
    _fits,
    _greedy_groups,
    _greedy_lines,
    _lines_fit_ellipse,
    _lines_fit_spans,
    _mask_bands,
    _mask_centroid,
    _resolve_mask,
    _scan_sizes,
    _screen_frame,
    _trim_mask_to_letters,
    _vi_words,
    _word_units,
    _wrap_text,
    _wrap_to_spans,
)


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
