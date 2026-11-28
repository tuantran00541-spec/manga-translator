import math
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.detector.bubble_mask import mask_inscribed_rect
from app.render.composition import compose_text
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
    _light_background_grey,
    _line_start_x,
    _luminance,
    auto_detect_text_color,
    drawable,
    font_draws_text,
    get_font_object,
    get_font_path,
    list_available_fonts,
    measure_line_ink,
    parse_color,
)
from app.render.text_fitting import (
    DEFAULT_BODY_FONT_SIZE,
    ENLARGED_MIN_FONT_SIZE,
    ENLARGE_GROW_RATIO,
    REFERENCE_PAGE_WIDTH,
    SOURCE_FLOOR_RATIO,
    SOURCE_MATCH_MAX_FONT_SIZE,
    _CURVED_CONTAINERS,
    _anchored_top,
    _band_line_centres,
    _band_line_spans,
    _box_frame,
    _derived_mask,
    _fit_text,
    _fits,
    _lines_fit_ellipse,
    _lines_fit_spans,
    _mask_bands,
    _mask_centroid,
    _mask_layout_y,
    _resolve_mask,
    _screen_frame,
    _trim_mask_to_letters,
    _wrap_text,
)
__all__ = [
    "_derived_mask",
    "_mask_centroid",
    "_trim_mask_to_letters",
    "font_draws_text",
    "list_available_fonts",
]




def _composite_rgba(image: Image.Image, patch_rgba: Image.Image, ox: int, oy: int) -> None:
    iw, ih = image.size
    pw, ph = patch_rgba.size
    dx0 = max(0, ox)
    dy0 = max(0, oy)
    dx1 = min(iw, ox + pw)
    dy1 = min(ih, oy + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0 = dx0 - ox
    sy0 = dy0 - oy
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    dst = image.crop((dx0, dy0, dx1, dy1)).convert("RGBA")
    src = patch_rgba.crop((sx0, sy0, sx1, sy1))
    blended = Image.alpha_composite(dst, src)
    image.paste(blended if image.mode == "RGBA" else blended.convert(image.mode), (dx0, dy0))


def _gradient_rgb_array(
    height: int,
    width: int,
    direction: str,
    stops_rgb: list[tuple[int, int, int]],
    extent: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    h = max(1, int(height))
    w = max(1, int(width))
    xp = np.linspace(0.0, 1.0, len(stops_rgb))
    stops_arr = np.asarray(stops_rgb, dtype=np.float32)
    if direction == "horizontal":
        t = np.linspace(*extent, w)
        ramp = np.stack([np.interp(t, xp, stops_arr[:, ch]) for ch in range(3)], axis=1)
        rgb = np.broadcast_to(ramp[None, :, :], (h, w, 3))
    else:
        t = np.linspace(*extent, h)
        ramp = np.stack([np.interp(t, xp, stops_arr[:, ch]) for ch in range(3)], axis=1)
        rgb = np.broadcast_to(ramp[:, None, :], (h, w, 3))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _draw_outer_glow(
    image: Image.Image,
    line_layouts: list[tuple[int, int, str]],
    ink_box: tuple[int, int, int, int],
    font,
    offsets: list[tuple[int, int]],
    glow_rgb: tuple[int, int, int],
    glow_radius: int,
    stroke_w: int,
) -> tuple[int, int, int, int]:
    radius = max(2, min(12, int(glow_radius)))
    pad_g = max(4, int(math.ceil(radius * 3.0)) + stroke_w + 2)
    px0 = ink_box[0] - pad_g
    py0 = ink_box[1] - pad_g
    pw = max(1, (ink_box[2] + pad_g) - px0)
    ph = max(1, (ink_box[3] + pad_g) - py0)
    mask = Image.new("L", (pw, ph), 0)
    mdraw = ImageDraw.Draw(mask)
    for start_x, cur_y, line in line_layouts:
        for dx, dy in offsets:
            mdraw.text(
                (start_x - px0 + dx, cur_y - py0 + dy),
                line,
                font=font,
                fill=255,
                stroke_width=stroke_w,
                stroke_fill=255,
            )
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=radius))
    alpha = np.clip(np.asarray(blurred, dtype=np.float32) * 1.35, 0.0, 255.0).astype(np.uint8)
    rgba = np.empty((ph, pw, 4), dtype=np.uint8)
    rgba[:, :, 0] = glow_rgb[0]
    rgba[:, :, 1] = glow_rgb[1]
    rgba[:, :, 2] = glow_rgb[2]
    rgba[:, :, 3] = alpha
    _composite_rgba(image, Image.fromarray(rgba, mode="RGBA"), px0, py0)
    return px0, py0, px0 + pw, py0 + ph


def _draw_gradient_line(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    start_x: int,
    cur_y: int,
    line: str,
    font,
    offsets: list[tuple[int, int]],
    is_bold: bool,
    direction: str,
    stops_rgb: list[tuple[int, int, int]],
    gradient_box: tuple[int, int, int, int] | None = None,
) -> None:
    gx0, gy0, gx1, gy1 = draw.textbbox((start_x, cur_y), line, font=font)
    bold = 1 if is_bold else 0
    lx0 = gx0 - 2
    ly0 = gy0 - 2
    lw = max(1, (gx1 + bold + 2) - lx0)
    lh = max(1, (gy1 + bold + 2) - ly0)
    mask = Image.new("L", (lw, lh), 0)
    mdraw = ImageDraw.Draw(mask)
    for dx, dy in offsets:
        mdraw.text((start_x - lx0 + dx, cur_y - ly0 + dy), line, font=font, fill=255)
    alpha = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return
    ny0, ny1 = int(ys.min()), int(ys.max()) + 1
    nx0, nx1 = int(xs.min()), int(xs.max()) + 1
    sub_alpha = alpha[ny0:ny1, nx0:nx1]
    extent = (0.0, 1.0)
    if gradient_box is not None:
        if direction == "horizontal":
            start, end = lx0 + nx0, lx0 + nx1
            lo, hi = gradient_box[0], gradient_box[2]
        else:
            start, end = ly0 + ny0, ly0 + ny1
            lo, hi = gradient_box[1], gradient_box[3]
        length = max(1, hi - lo - 1)
        extent = ((start - lo) / length, (end - 1 - lo) / length)
    grad_rgb = _gradient_rgb_array(ny1 - ny0, nx1 - nx0, direction, stops_rgb, extent)
    rgba = np.dstack([grad_rgb, sub_alpha])
    _composite_rgba(image, Image.fromarray(rgba, mode="RGBA"), lx0 + nx0, ly0 + ny0)


def _parse_color_profile(color_profile: dict | None) -> tuple[
    tuple[int, int, int] | None,
    str | None,
    list[tuple[int, int, int]] | None,
    tuple[int, int, int] | None,
    tuple[int, int, int] | None,
    int,
]:
    if not isinstance(color_profile, dict):
        return None, None, None, None, None, 0
    prof_fill = parse_color(color_profile.get("fill"), default=None) if color_profile.get("fill") else None
    grad_dir: str | None = None
    grad_stops: list[tuple[int, int, int]] | None = None
    grad = color_profile.get("gradient")
    if isinstance(grad, dict):
        d = str(grad.get("direction") or "").strip().lower()
        raw_stops = grad.get("stops") or []
        if d in ("vertical", "horizontal") and isinstance(raw_stops, (list, tuple)) and len(raw_stops) >= 2:
            parsed = [parse_color(s, default=None) for s in raw_stops[:5]]
            if all(p is not None for p in parsed):
                grad_dir = d
                grad_stops = parsed
    prof_stroke = (
        parse_color(color_profile.get("stroke_color"), default=None)
        if color_profile.get("stroke_color")
        else None
    )
    prof_glow = (
        parse_color(color_profile.get("glow_color"), default=None)
        if color_profile.get("glow_color")
        else None
    )
    glow_radius = 0
    if prof_glow is not None:
        try:
            glow_radius = max(2, min(12, int(round(float(color_profile.get("glow_radius") or 5)))))
        except (TypeError, ValueError):
            glow_radius = 5
    return prof_fill, grad_dir, grad_stops, prof_stroke, prof_glow, glow_radius


_SRGB_LUMINANCE = np.array([_luminance((v, v, v)) for v in range(256)])


def _background_luminances(image, glyph_boxes, line_layouts=None, font=None, stroke=0):
    samples = []
    for i, (x1, y1, x2, y2) in enumerate(glyph_boxes):
        bounds = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        crop = image.crop(bounds).convert("RGB")
        if line_layouts is not None:
            x, y, line = line_layouts[i]
            footprint = Image.new("L", crop.size)
            ImageDraw.Draw(footprint).text((x - bounds[0], y - bounds[1]), line, font=font,
                                           fill=255, stroke_width=stroke, stroke_fill=255)
            rgb = np.asarray(crop)[np.asarray(footprint) >= 128]
            if not rgb.size:
                continue
            luma = (_SRGB_LUMINANCE[rgb] * (0.2126, 0.7152, 0.0722)).sum(axis=-1)
            samples.extend(np.percentile(luma, (5, 20, 50, 80, 95)))
            continue
        rgb = np.asarray(crop)[::4, ::4]
        luma = (_SRGB_LUMINANCE[rgb] * (0.2126, 0.7152, 0.0722)).sum(axis=2)
        samples.extend(np.percentile(luma, (20, 50, 80)))
    return np.asarray(samples)


def _local_contrast(color, backgrounds):
    luma = _luminance(color)
    return (np.maximum(luma, backgrounds) + 0.05) / (np.minimum(luma, backgrounds) + 0.05)


def _adapt_palette(palette, backgrounds, threshold, outline, *, prefer_black_or_white=False):
    if not backgrounds.size:
        return palette

    def contrast(color):
        return _local_contrast(color, backgrounds)

    outlines = contrast(outline) >= threshold if outline is not None else False
    result = []
    for color in palette:
        readable = contrast(color) >= threshold
        if outline is not None and _contrast(color, outline) >= 2.0:
            readable = readable | outlines
        if readable.all():
            result.append(color)
            continue
        target = max(((0, 0, 0), (255, 255, 255)), key=lambda rgb: contrast(rgb).min())
        if prefer_black_or_white or contrast(target).min() < threshold:
            result.append(target)
            continue
        lo, hi = 0.0, 1.0
        for _ in range(8):
            blend = (lo + hi) / 2
            candidate = tuple(round(value + blend * (end - value)) for value, end in zip(color, target))
            if contrast(candidate).min() >= threshold:
                hi = blend
            else:
                lo = blend
        result.append(tuple(round(value + hi * (end - value)) for value, end in zip(color, target)))
    return result


def _art_ink(palette, backgrounds, preferred_outline):
    neutral = ((0, 0, 0), (255, 255, 255))
    threshold = 4.5

    def contrast(colours, outline=None):
        body = np.min([_local_contrast(color, backgrounds) for color in colours], axis=0)
        if outline is not None and all(_contrast(color, outline) >= 4.5 for color in colours):
            body = np.maximum(body, _local_contrast(outline, backgrounds))
        return float(body.min())

    if not backgrounds.size or contrast(palette, preferred_outline) >= threshold:
        return palette, preferred_outline
    adjusted = _adapt_palette(palette, backgrounds, threshold, None)
    if contrast(adjusted) >= threshold:
        return adjusted, preferred_outline if preferred_outline is not None and all(
            _contrast(color, preferred_outline) >= threshold for color in adjusted) else None
    for outline in neutral:
        if contrast(palette, outline) >= threshold:
            return palette, outline
    fill = min(neutral, key=lambda color: abs(_luminance(color) - _luminance(palette[len(palette) // 2])))
    outline = neutral[1] if fill == neutral[0] else neutral[0]
    return [fill] * len(palette), outline


def _mask_frame_coherent(
    mask: dict | None,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    verified_balloon: bool = False,
) -> tuple[int, int, int, int] | None:
    if mask is None or mask.get("kind") != "bubble":
        return None
    mx1, my1, mx2, my2 = (int(v) for v in mask["bbox"])
    mx1, my1 = max(0, mx1), max(0, my1)
    mx2, my2 = min(image_size[0], mx2), min(image_size[1], my2)
    if mx2 - mx1 < 16 or my2 - my1 < 16:
        return None
    bx1, by1, bx2, by2 = box
    # Reject only if the box and mask are completely disjoint (stale mask).
    if min(mx2, bx2) <= max(mx1, bx1) or min(my2, by2) <= max(my1, by1):
        return None
    return (mx1, my1, mx2, my2)


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
    color_profile: dict | None = None,
    adapt_contrast: bool | None = None,
    keep_compounds: bool = False,
    compose_layout: bool = False,
    optical_size_hint: int | None = None,
    reserved_boxes: tuple = (),
    typography_role: str | None = None,
    verified_balloon: bool = False,
    lock_frame: bool = False,
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
    text = unicodedata.normalize("NFC", str(text).strip())

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

    (
        profile_fill,
        profile_grad_dir,
        profile_grad_stops,
        profile_stroke_rgb,
        profile_glow_rgb,
        profile_glow_radius,
    ) = _parse_color_profile(color_profile)
    from app.ocr.visual_metadata import has_manual_color_profile, has_raw_color_profile

    measured_palette = has_raw_color_profile(color_profile)
    saved_palette = (
        (measured_palette or has_manual_color_profile(color_profile))
        and (fill in (None, "auto", "") or parse_color(fill, default=None) == profile_fill)
    )

    if fill is None or fill == "auto" or fill == "":
        if profile_fill is not None:
            text_color = profile_fill
            explicit_fill = True
        else:
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

    if stroke_w_auto and saved_palette and stroke_color in (None, "auto", ""):
        stroke_w = max(stroke_w, 2) if profile_stroke_rgb is not None else 0
    elif stroke_w_auto:
        if bg_color and bg_color not in ("transparent", "none", ""):
            bg_ref = parse_color(bg_color, default=(255, 255, 255))
        else:
            bg_grey = _light_background_grey(image, (x1, y1, x2, y2))
            bg_ref = (bg_grey, bg_grey, bg_grey)
        if _contrast(text_color, bg_ref) < LOW_CONTRAST_RATIO:
            stroke_w = max(stroke_w, LOW_CONTRAST_STROKE_WIDTH)
        if profile_stroke_rgb is not None:
            stroke_w = max(stroke_w, 2)

    profile_stroke_applied = False
    if stroke_color is None or stroke_color == "auto" or stroke_color == "":
        stroke_auto = True
        if profile_stroke_rgb is not None:
            stroke_c = profile_stroke_rgb
            profile_stroke_applied = True
        else:
            luminance = (text_color[0] * 299 + text_color[1] * 587 + text_color[2] * 114) / 1000
            stroke_c = (0, 0, 0) if luminance > 128 else (255, 255, 255)
    else:
        stroke_auto = False
        stroke_c = parse_color(stroke_color, default=(0, 0, 0))

    art_auto_ink = (not saved_palette and str(container or "").lower() == "free" and adapt_contrast is True
                    and stroke_w_auto and stroke_auto
                    and str(bg_color or "").lower() in ("", "none", "transparent"))
    if art_auto_ink:
        share = round(0.09 * min(raw_w, raw_h))
        stroke_w = max(stroke_w, min(share, 5, max(3, round(image.width * 0.004))))

    draw = ImageDraw.Draw(image)

    target_size = None
    if isinstance(font_size, (int, float)) and font_size > 0:
        target_size = int(font_size)
    elif isinstance(font_size, str) and font_size.isdigit() and int(font_size) > 0:
        target_size = int(font_size)

    font_path_str = str(font_path)

    if compose_layout and h_align == "center" and v_align == "middle" and source_letter_box is None:
        source_letter_box = (x1, y1, x2, y2)
    _mask, _mask_source = _resolve_mask(None if lock_frame else image, (x1, y1, x2, y2), container,
                                        source_letter_box, bubble_mask,
                                        letters, letters_box, verified_balloon=verified_balloon)
    curved = str(container or "").lower() in _CURVED_CONTAINERS or str(shape or "").lower() == "ellipse" \
        or (_mask is not None and _mask.get("kind") == "bubble")
    centred = h_align == "center" and v_align == "middle"
    if compose_layout and centred:
        optical_size_hint = optical_size_hint or target_size
        target_size = None
    if keep_compounds and _mask is not None and _mask.get("kind") == "bubble":
        left, top, right, bottom = _mask["bbox"]
        padding = max(padding, min(36, round(0.06 * min(right - left, bottom - top))))

    _fixed_height_fallback = False
    fixed_spans = None
    block_top = None
    fixed_frame = None
    positioned = None
    if target_size is not None:
        _readable_fixed = int(RENDER_MIN_READABLE_FONT_SIZE
                              * max(1.0, image.size[0] / REFERENCE_PAGE_WIDTH))
        if target_size < _readable_fixed:
            target_size = _readable_fixed

    if target_size is not None and curved and centred:
        measured = _mask_frame_coherent(_mask, (x1, y1, x2, y2), image.size, verified_balloon=verified_balloon)
        if measured is not None:
            fixed_frame = measured
            x1, y1, x2, y2 = measured
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2
        _fixed_font = get_font_object(font_path_str, target_size)
        _fixed_lines = _wrap_text(
            draw, text, _fixed_font, box_w, balance=True, keep_compounds=keep_compounds,
            stroke_w=stroke_w, bold_extra=int(is_bold),
        )
        if fixed_frame is not None:
            _bands = _mask_bands(_mask, (x1 + pad, y1 + pad, x2 - pad, y2 - pad))
            if _bands is not None:
                _widest = max((b - a for band in _bands for a, b in band), default=0)
                if _widest >= 0.4 * box_w:
                    _lh = _calc_line_height(draw, _fixed_font, stroke_w=stroke_w)
                    _anchor = _mask_layout_y(_mask, inner_body=keep_compounds) - (y1 + pad)
                    _top = _anchored_top(box_h, _lh * len(_fixed_lines), _anchor)
                    fixed_spans = _band_line_spans(box_w, box_h, len(_fixed_lines), _lh, _bands, top=_top)
                    if _lines_fit_spans(draw, _fixed_lines, _fixed_font, fixed_spans, stroke_w + int(is_bold)):
                        block_top = y1 + pad + _top
                    else:
                        target_size = None
                        _fixed_height_fallback = True
        elif not _lines_fit_ellipse(draw, _fixed_lines, _fixed_font, box_w, box_h, stroke_w + int(is_bold)):
            target_size = None
            _fixed_height_fallback = True

    if target_size is not None:
        _fixed_font = get_font_object(font_path_str, target_size)
        _fixed_lines = _wrap_text(
            draw, text, _fixed_font, box_w, balance=True, keep_compounds=keep_compounds,
            stroke_w=stroke_w, bold_extra=int(is_bold),
        )
        _bold_extra = 1 if is_bold else 0
        _whole_words = [w for ln in _fixed_lines for w in ln.split()] == text.split()
        _fixed_lh = _calc_line_height(draw, _fixed_font, stroke_w=stroke_w)
        _max_line_w = max(
            (measure_line_ink(draw, line, _fixed_font, stroke_w=stroke_w, bold=is_bold)[4] for line in _fixed_lines),
            default=0,
        )
        _fixed_min_y = min(
            (i * _fixed_lh + measure_line_ink(draw, ln, _fixed_font, stroke_w=stroke_w, bold=is_bold)[1]
             for i, ln in enumerate(_fixed_lines)),
            default=0,
        )
        _fixed_max_y = max(
            (i * _fixed_lh + measure_line_ink(draw, ln, _fixed_font, stroke_w=stroke_w, bold=is_bold)[3]
             for i, ln in enumerate(_fixed_lines)),
            default=0,
        )
        _fixed_ink_h = max(_fixed_lh * len(_fixed_lines), _fixed_max_y - _fixed_min_y)
        if (
            not _fixed_lines
            or not _whole_words
            or _fixed_ink_h > box_h
            or _max_line_w > box_w
        ):
            target_size = None
            _fixed_height_fallback = True

    if target_size is not None:
        actual_size = target_size
        font = get_font_object(font_path_str, actual_size)
        lines = _wrap_text(
            draw, text, font, box_w, balance=True, keep_compounds=keep_compounds,
            stroke_w=stroke_w, bold_extra=int(is_bold),
        )
        line_centres = [(a + b) / 2 for a, b in fixed_spans] if fixed_spans else None
        lines_source = "fixed_size"
        draw_spans = fixed_spans
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
            letters=letters, letters_box=letters_box, keep_compounds=keep_compounds,
            compose_layout=compose_layout, optical_size_hint=optical_size_hint,
            bold=is_bold, glow=profile_glow_radius * 3, reserved_boxes=reserved_boxes,
            typography_role=typography_role, verified_balloon=verified_balloon, lock_frame=lock_frame)
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
        positioned = fit_stats.get("composition")
        if fit_stats.get("omit_glow"):
            profile_glow_radius = 0
        lines = fitted_lines if positioned else _wrap_text(
            draw, text, font, box_w, balance=True, keep_compounds=keep_compounds,
            stroke_w=stroke_w, bold_extra=int(is_bold))
        lines_source = "balanced"
        draw_spans = spans_holder if spans_holder else None
        _lh_check = _calc_line_height(draw, font, stroke_w=stroke_w)
        _bold_extra = 1 if is_bold else 0
        _ink_min_y = min(
            (i * _lh_check + measure_line_ink(draw, ln, font, stroke_w=stroke_w, bold=is_bold)[1]
             for i, ln in enumerate(lines)),
            default=0,
        )
        _ink_max_y = max(
            (i * _lh_check + measure_line_ink(draw, ln, font, stroke_w=stroke_w, bold=is_bold)[3]
             for i, ln in enumerate(lines)),
            default=0,
        )
        if fitted_lines and (
            _lh_check * len(lines) > box_h
            or (_ink_max_y - _ink_min_y) > box_h
            or max((measure_line_ink(draw, ln, font, stroke_w=stroke_w, bold=is_bold)[4] for ln in lines), default=0) > box_w
        ):
            lines = fitted_lines
            lines_source = "fitted"
        if positioned:
            lines_source = "composition"
        elif centred and curved:
            if draw_spans is not None:
                if len(lines) != len(draw_spans):
                    lines = fitted_lines
                    lines_source = "fitted"
                elif (not _lines_fit_spans(draw, lines, font, draw_spans, stroke_w + _bold_extra)
                        and len(fitted_lines) == len(draw_spans)
                        and _lines_fit_spans(draw, fitted_lines, font, draw_spans, stroke_w + _bold_extra)):
                    lines = fitted_lines
                    lines_source = "fitted"
            elif not _lines_fit_ellipse(draw, lines, font, box_w, box_h, stroke_w + _bold_extra):
                if _lines_fit_ellipse(draw, fitted_lines, font, box_w, box_h, stroke_w + _bold_extra):
                    lines = fitted_lines
                    lines_source = "fitted"
    if stats is not None:
        stats["lines"] = len(lines)
        stats["size"] = actual_size
        stats["fit_box"] = (x1, y1, x2, y2)
        stats["fixed_size_height_fallback"] = _fixed_height_fallback
        if h_align == "center":
            stats["line_centring"] = "optical" if positioned else ("per_line" if line_centres else "box_centre")
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
            if positioned:
                for key in ("layout_score", "edge_clearance", "preferred_size"):
                    stats[key] = fit_stats[key]
        elif fixed_frame is not None:
            stats.update(frame=fixed_frame, frame_source="mask", mask_source=_mask_source)

    shape = str(shape or "").lower()
    if shape not in ("rectangle", "ellipse"):
        shape = "rectangle"
    bg_bbox = None
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
    ink_metrics = [
        measure_line_ink(draw, line, font, stroke_w=stroke_w, bold=is_bold)
        for line in lines
    ]
    if lines:
        block_min_y = min(i * line_height + m[1] for i, m in enumerate(ink_metrics))
        block_max_y = max(i * line_height + m[3] for i, m in enumerate(ink_metrics))
        actual_block_h = max(0, block_max_y - block_min_y)
    else:
        block_min_y = block_max_y = actual_block_h = 0

    if v_align == "top":
        start_y = (y1 + pad) - block_min_y
    elif v_align == "bottom":
        start_y = (y2 - pad) - block_max_y
    elif block_top is not None:
        start_y = int(block_top)
    else:
        start_y = round((y1 + pad) + (box_h - actual_block_h) / 2.0 - block_min_y)

    if lines and actual_block_h <= box_h:
        min_sy = (y1 + pad) - block_min_y
        max_sy = (y2 - pad) - block_max_y
        start_y = max(min_sy, min(start_y, max_sy))
    elif lines:
        min_sy = y1 - block_min_y
        max_sy = y2 - block_max_y
        if max_sy >= min_sy:
            start_y = max(min_sy, min(start_y, max_sy))

    offsets = [(0, 0), (1, 0), (0, 1), (1, 1)] if is_bold else [(0, 0)]

    ink_box = None
    glyph_boxes = []
    glyph_layouts = []
    line_layouts: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        if positioned:
            start_x, cur_y, _ = positioned[i]
            line_w = ink_metrics[i][4]
        else:
            start_x, line_w = _line_start_x(draw, line, font, x1=x1, pad=pad, box_w=box_w,
                                            h_align=h_align, centred=centred, curved=curved,
                                            line_centres=line_centres, draw_spans=draw_spans,
                                            i=i, line_count=len(lines), stroke_w=stroke_w,
                                            bold_extra=int(is_bold))
            cur_y = start_y + i * line_height
        line_layouts.append((start_x, cur_y, line))
        if line_w > 0:
            bold = 1 if is_bold else 0
            gx0, gy0, gx1, gy1 = draw.textbbox(
                (start_x, cur_y), line, font=font, stroke_width=stroke_w)
            gbox = (gx0, gy0, gx1 + bold, gy1 + bold)
            glyph_boxes.append(gbox)
            glyph_layouts.append((start_x, cur_y, line))
            ink_box = gbox if ink_box is None else (
                min(ink_box[0], gbox[0]), min(ink_box[1], gbox[1]),
                max(ink_box[2], gbox[2]), max(ink_box[3], gbox[3]))

    # Hard post-layout containment guard: keep every raster glyph pixel strictly inside the frame.
    if ink_box is not None:
        ix1, iy1, ix2, iy2 = ink_box
        nudge_x = 0
        if ix2 - ix1 <= (x2 - x1):
            if ix1 < x1:
                nudge_x = x1 - ix1
            elif ix2 > x2:
                nudge_x = x2 - ix2
        else:
            nudge_x = round(((x1 + x2) - (ix1 + ix2)) / 2.0)
        nudge_y = 0
        if iy2 - iy1 <= (y2 - y1):
            if iy1 < y1:
                nudge_y = y1 - iy1
            elif iy2 > y2:
                nudge_y = y2 - iy2
        else:
            nudge_y = round(((y1 + y2) - (iy1 + iy2)) / 2.0)
        if nudge_x or nudge_y:
            line_layouts = [(sx + nudge_x, sy + nudge_y, ln) for sx, sy, ln in line_layouts]
            glyph_layouts = [(sx + nudge_x, sy + nudge_y, ln) for sx, sy, ln in glyph_layouts]
            glyph_boxes = [(gx0 + nudge_x, gy0 + nudge_y, gx1 + nudge_x, gy1 + nudge_y)
                           for gx0, gy0, gx1, gy1 in glyph_boxes]
            ink_box = (ix1 + nudge_x, iy1 + nudge_y, ix2 + nudge_x, iy2 + nudge_y)

    if not saved_palette and (adapt_contrast is True or (adapt_contrast is None and explicit_fill)):
        if art_auto_ink:
            backgrounds = _background_luminances(image, glyph_boxes, glyph_layouts, font, stroke_w)
        else:
            backgrounds = _background_luminances(image, glyph_boxes)
        framed = str(container or "").lower() in ("bubble", "spiky", "box", "screen")
        threshold = 4.5 if framed else LOW_CONTRAST_RATIO
        palette = profile_grad_stops or [text_color]
        outline = stroke_c if stroke_w >= 2 and (profile_stroke_applied or not stroke_auto) else None
        plain_neutral = framed and profile_grad_stops is None and max(text_color) - min(text_color) <= 10
        if art_auto_ink:
            adjusted, art_outline = _art_ink(palette, backgrounds, outline)
            if art_outline is not None:
                if art_outline != outline or adjusted != palette:
                    profile_stroke_applied = False
                    if stats is not None:
                        stats["fill_fallback"] = "art_outline_contrast"
                if profile_glow_rgb is not None and _contrast(art_outline, profile_glow_rgb) < 3:
                    profile_glow_radius = 0
                stroke_c = art_outline
                stroke_w = min(stroke_w, max(2, round(actual_size * .06)))
            else:
                stroke_w = 0
                profile_stroke_applied = False
                if adjusted != palette:
                    profile_glow_radius = 0
        else:
            adjusted = _adapt_palette(palette, backgrounds, threshold, outline, prefer_black_or_white=plain_neutral)
        if adjusted != palette:
            if profile_grad_stops is None:
                text_color = adjusted[0]
            else:
                profile_grad_stops = adjusted
            if stroke_auto and not profile_stroke_applied and not art_auto_ink:
                stroke_c = (0, 0, 0) if _luminance(adjusted[len(adjusted) // 2]) > 0.179 else (255, 255, 255)
            if stats is not None and "fill_fallback" not in stats:
                stats["fill_fallback"] = "local_contrast"
        if (framed and stroke_w_auto and stroke_auto and not profile_stroke_applied
                and backgrounds.size and np.ptp(backgrounds) <= 0.05
                and all((_local_contrast(rgb, backgrounds) >= threshold).all() for rgb in adjusted)):
            stroke_w = 0

    draw_box = ink_box
    if profile_glow_rgb is not None and profile_glow_radius > 0 and ink_box is not None:
        draw_box = _draw_outer_glow(
            image,
            line_layouts,
            ink_box,
            font,
            offsets,
            profile_glow_rgb,
            profile_glow_radius,
            stroke_w,
        )

    use_gradient = bool(profile_grad_dir and profile_grad_stops and len(profile_grad_stops) >= 2)
    gradient_box = None
    if measured_palette and use_gradient and glyph_boxes:
        gradient_box = (min(g[0] for g in glyph_boxes) + stroke_w, min(g[1] for g in glyph_boxes) + stroke_w,
                        max(g[2] for g in glyph_boxes) - stroke_w, max(g[3] for g in glyph_boxes) - stroke_w)
    for start_x, cur_y, line in line_layouts:
        if use_gradient:
            if stroke_w > 0:
                for dx, dy in offsets:
                    draw.text(
                        (start_x + dx, cur_y + dy),
                        line,
                        font=font,
                        fill=stroke_c,
                        stroke_width=stroke_w,
                        stroke_fill=stroke_c,
                    )
            _draw_gradient_line(
                image,
                draw,
                start_x,
                cur_y,
                line,
                font,
                offsets,
                is_bold,
                profile_grad_dir,
                profile_grad_stops,
                gradient_box,
            )
        else:
            for dx, dy in offsets:
                draw.text(
                    (start_x + dx, cur_y + dy),
                    line,
                    font=font,
                    fill=text_color,
                    stroke_width=stroke_w,
                    stroke_fill=stroke_c,
                )
    if stats is not None:
        if ink_box is not None:
            stats["ink_box"] = ink_box
        if bg_bbox is not None:
            background = (bg_bbox[0], bg_bbox[1], bg_bbox[2] + 1, bg_bbox[3] + 1)
            draw_box = background if draw_box is None else (
                min(draw_box[0], background[0]), min(draw_box[1], background[1]),
                max(draw_box[2], background[2]), max(draw_box[3], background[3]))
        if draw_box is not None:
            stats["draw_box"] = draw_box
        if use_gradient or profile_stroke_applied or (profile_glow_rgb is not None and profile_glow_radius > 0):
            stats["color_profile_applied"] = {
                "gradient": use_gradient,
                "stroke": profile_stroke_applied,
                "glow": bool(profile_glow_rgb is not None and profile_glow_radius > 0 and ink_box is not None),
            }

    return image


def auto_fit(draw, text: str, box, image_size, font_path_str: str, *, padding: int = RENDER_DEFAULT_PADDING,
             stroke_w: int = RENDER_AUTO_STROKE_WIDTH, source_cap_px: int | None = None, enlarge: bool = False,
             opaque_caption: bool = False, curved: bool = False, image=None, centered: bool = True,
             container: str | None = None, source_letter_box: tuple[int, int, int, int] | None = None,
             centres_out: list | None = None, stats_out: dict | None = None,
             min_source_share: float = 0.0, size_ceiling_px: int | None = None,
             bubble_mask: dict | None = None, spans_out: list | None = None,
             letters=None,
             letters_box: tuple[int, int, int, int] | None = None,
             keep_compounds: bool = False, compose_layout: bool = False,
             optical_size_hint: int | None = None, bold: bool = False, glow: int = 0,
             reserved_boxes: tuple = (), typography_role: str | None = None, verified_balloon: bool = False,
             lock_frame: bool = False):
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

    mask, mask_source = _resolve_mask(None if lock_frame else image, (x1, y1, x2, y2), container,
                                      source_letter_box, bubble_mask,
                                      letters, letters_box, verified_balloon=verified_balloon)
    frame = (x1, y1, x2, y2)
    frame_source = "region_no_frame"
    fallbacks: list[str] = []
    use_mask = False
    if centered:
        measured = _mask_frame_coherent(mask, (x1, y1, x2, y2), image_size, verified_balloon=verified_balloon)
        if measured is not None:
            frame = measured
            frame_source = "mask"
            x1, y1, x2, y2 = frame
            use_mask = True
            raw_w, raw_h = x2 - x1, y2 - y1
            pad = max(2, min(padding, int(min(raw_w, raw_h) * RENDER_PADDING_RATIO_MAX)))
            box_w, box_h = raw_w - pad * 2, raw_h - pad * 2

    if not lock_frame and not use_mask and str(container or "").lower() == "screen":
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

    if not lock_frame and not use_mask and str(container or "").lower() == "box" and not curved and image is not None:
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
        if lock_frame or curved or str(container or "").lower() == "box":
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

    if compose_layout and centered and image is not None:
        size_hint = optical_size_hint if isinstance(optical_size_hint, (int, float)) and math.isfinite(optical_size_hint) else None
        preferred = int(source_match or size_hint or (DEFAULT_BODY_FONT_SIZE * scale))
        if container == "box" and typography_role == "title" and frame_source in ("mask", "box_rules"):
            preferred = max(preferred, min(round(32 * scale), round(0.12 * min(raw_w, raw_h))))
        if enlarge:
            preferred = round(preferred * 1.25)
        image_w, image_h = image_size
        inner_frame = (max(0, x1), max(0, y1), min(image_w, x2), min(image_h, y2))
        options = dict(preferred=min(preferred, maximum_size * 2), floor=floor, stroke=stroke_w,
                       bold=bold, mask=mask if use_mask else None, ellipse=curved and not use_mask,
                       anchor=source_letter_box, reserved_boxes=reserved_boxes,
                       preserve_paragraph_shape=verified_balloon and container == "box" and typography_role == "title",
                       verified_balloon=verified_balloon and use_mask)
        plan = compose_text(image, text, inner_frame, font_path_str, glow=glow, **options)
        omit_glow = plan is None and glow > 0
        if omit_glow:
            plan = compose_text(image, text, inner_frame, font_path_str, **options)
        if plan is not None:
            if stats_out is not None:
                stats_out.update(frame=inner_frame, frame_source=frame_source, floor_px=floor,
                                 composition=plan.lines, layout_score=plan.score,
                                 edge_clearance=plan.clearance, preferred_size=plan.target, omit_glow=omit_glow)
            return plan.size, [line for _, _, line in plan.lines], True, inner_frame
        fallbacks.append("compose_text_fallback")

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
            anchor_y = _mask_layout_y(mask, inner_body=keep_compounds) - padded[1]
            if bands is not None:
                widest = max((b - a for band in bands for a, b in band), default=0)
                if widest < 0.4 * fbw:
                    bands = None
                    fallbacks.append("mask_degenerate_rectangle")
        bold_extra = int(bool(bold))
        probe = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                          minimum_size=readable, maximum_size=maximum_size,
                          curved=bool(bands is not None or curved), bands=bands, anchor_y=anchor_y,
                          keep_compounds=keep_compounds, bold_extra=bold_extra)[0]
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
                                          curved=True, bands=bands, anchor_y=anchor_y,
                                          keep_compounds=keep_compounds, bold_extra=bold_extra)
            if not fits and fit_floor > readable:
                size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                              minimum_size=readable, maximum_size=fit_floor,
                                              curved=True, bands=bands, anchor_y=anchor_y,
                                              keep_compounds=keep_compounds, bold_extra=bold_extra)
        elif curved:
            size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                          minimum_size=fit_floor, maximum_size=ceiling,
                                          curved=True, bands=None, keep_compounds=keep_compounds,
                                          bold_extra=bold_extra)
            if not fits and fit_floor > readable:
                size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                              minimum_size=readable, maximum_size=fit_floor,
                                              curved=True, bands=None, keep_compounds=keep_compounds,
                                              bold_extra=bold_extra)
        else:
            size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                          minimum_size=fit_floor, maximum_size=ceiling,
                                          curved=False, bands=None, keep_compounds=keep_compounds,
                                          bold_extra=bold_extra)
            if not fits and fit_floor > readable:
                size, lines, fits = _fit_text(draw, text, fbw, fbh, font_path_str, stroke_w=stroke_w,
                                              minimum_size=readable, maximum_size=fit_floor,
                                              curved=False, bands=None, keep_compounds=keep_compounds,
                                              bold_extra=bold_extra)
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
            bold_extra = int(bool(bold))
            ok, flines = _fits(draw, text, rbw, rbh, font_path_str, floor, stroke_w,
                               curved=False, bands=None, keep_compounds=keep_compounds, bold_extra=bold_extra)
            rsize = floor
            if not ok and floor > readable:
                rsize, flines, ok = _fit_text(
                    draw, text, rbw, rbh, font_path_str, stroke_w=stroke_w,
                    minimum_size=readable, maximum_size=floor, curved=False, bands=None,
                    keep_compounds=keep_compounds, bold_extra=bold_extra)
            shrunk_below_readable = False
            if not ok and readable > MIN_FONT_SIZE:
                rsize, flines, ok = _fit_text(
                    draw, text, rbw, rbh, font_path_str, stroke_w=stroke_w,
                    minimum_size=MIN_FONT_SIZE, maximum_size=readable - 1, curved=False, bands=None,
                    keep_compounds=keep_compounds, bold_extra=bold_extra)
                shrunk_below_readable = ok
            if ok:
                size, lines = rsize, flines
                fits_readably = not shrunk_below_readable
                x1, y1, x2, y2 = rx1, ry1, rx2, ry2
                bands, anchor_y = None, None
                padded = (rx1 + rpad, ry1 + rpad, rx2 - rpad, ry2 - rpad)
                raw_w, raw_h = x2 - x1, y2 - y1
                pad = rpad
                box_w, box_h = rbw, rbh
                fallbacks.append("overflow_shrink_below_readable" if shrunk_below_readable else "outline_fit_failed_floor_rect")
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
