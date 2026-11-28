from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.render.font_metrics import font_draws_text, get_font_object, get_font_path
from app.render.text_renderer import _parse_color_profile


_DISPLAY_FONTS = ("emphasis.bungee", "emphasis.anton", "emphasis.bangers", "sfx.baloo-2")


@dataclass(frozen=True)
class _SourceFrame:
    quad: np.ndarray
    size: tuple[int, int]
    ink: np.ndarray
    angle: float
    tapered: bool


def _stem_slope(ink: np.ndarray, height: float) -> float:
    contour = cv2.morphologyEx(ink.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(contour, 1, np.pi / 180, max(12, round(.18 * height)),
                            minLineLength=max(12, round(.25 * height)), maxLineGap=3)
    if lines is None:
        return 0.0
    dx = lines[:, 0, 2] - lines[:, 0, 0]
    dy = lines[:, 0, 3] - lines[:, 0, 1]
    stems = (np.abs(dy) >= .25 * height) & (np.abs(dx) <= .38 * np.abs(dy))
    slopes = dx[stems] / dy[stems]
    if slopes.size < 3:
        return 0.0
    middle = float(np.median(slopes))
    return middle if np.mean(np.abs(slopes - middle) <= .14) >= .65 else 0.0


def _source_frame(letters: np.ndarray) -> _SourceFrame | None:
    """Estimate a single lettering band from connected source glyphs."""
    h, w = letters.shape
    _, labels, stats, centres = cv2.connectedComponentsWithStats((letters > 0).astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size < 3:
        return None
    ids = np.flatnonzero(areas >= max(24, .08 * areas.max())) + 1
    if len(ids) < 3:
        return None
    xs, ys = centres[ids].T
    glyph_bounds = stats[ids]
    glyph_lefts = glyph_bounds[:, cv2.CC_STAT_LEFT]
    span = (glyph_lefts + glyph_bounds[:, cv2.CC_STAT_WIDTH]).max() - glyph_lefts.min()
    if np.ptp(xs) < .50 * span:
        return None
    height = float(np.median(glyph_bounds[:, cv2.CC_STAT_HEIGHT]))
    slope, intercept = np.polyfit(xs, ys, 1)
    if abs(slope) > 1 or np.sqrt(np.mean((ys - slope * xs - intercept) ** 2)) > .16 * height:
        return None
    ink = np.isin(labels, ids)
    if np.count_nonzero(ink) < .70 * np.count_nonzero(letters):
        return None
    basis = np.array([[1., _stem_slope(ink, height)], [slope, 1.]])
    inverse = np.linalg.inv(basis)
    py, px = np.nonzero(ink)
    points = np.column_stack((px, py)) @ inverse.T
    left, top = np.percentile(points, 1, axis=0)
    right, bottom = np.percentile(points, 99, axis=0)
    width, thickness = right - left, bottom - top
    if width < 1.25 * thickness or thickness < 24:
        return None

    taper = 0.
    if len(ids) >= 4:
        along = centres[ids] @ inverse.T
        thicknesses = np.array([np.ptp(np.percentile(points[labels[py, px] == i, 1], (5, 95))) for i in ids])
        change, offset = np.polyfit(along[:, 0], thicknesses, 1)
        drift = float(change * width)
        error = float(np.sqrt(np.mean((thicknesses - change * along[:, 0] - offset) ** 2)))
        if .12 * thickness <= abs(drift) <= .48 * thickness and error <= .18 * abs(drift):
            taper = drift / 4
    quad = np.array([[left, top + taper], [right, top - taper],
                     [right, bottom + taper], [left, bottom - taper]]) @ basis.T
    centre = quad.mean(axis=0)
    offsets = quad - centre
    margins = np.where(offsets > 0, [w - 4, h - 4] - centre, centre - 3)
    scale = min(1., float(np.min(margins / np.maximum(1., np.abs(offsets)))))
    if scale < .72:
        return None
    return _SourceFrame((centre + scale * offsets).astype(np.float32),
                        (round(width * scale), round(thickness * scale)), ink,
                        float(np.degrees(np.arctan(slope))), bool(taper))


def _text_alpha(path, text, size, bold=False) -> Image.Image:
    font = get_font_object(str(path), size)
    left, top, right, bottom = font.getbbox(text)
    canvas = Image.new("L", (max(1, right - left + int(bold)), max(1, bottom - top + int(bold))))
    draw = ImageDraw.Draw(canvas)
    for dy in range(1 + int(bold)):
        for dx in range(1 + int(bold)):
            draw.text((dx - left, dy - top), text, font=font, fill=255)
    bounds = canvas.getbbox()
    return canvas.crop(bounds) if bounds else canvas


def _corners(width, height):
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)


def _match_font(frame: _SourceFrame, text: str, source: str, preferred: str) -> tuple[str, Path] | None:
    names = dict.fromkeys((preferred, *_DISPLAY_FONTS))
    probes = [(name, get_font_path(name)) for name in names]
    available = [(name, path) for name, path in probes if font_draws_text(path, text)]
    if not available:
        return None
    source = source.strip()
    if not (2 <= len(source) <= 24 and "\n" not in source and any(c.isalpha() for c in source)):
        return available[0]
    width, height = 144, 56
    warp = cv2.getPerspectiveTransform(frame.quad, _corners(width, height))
    shape = cv2.warpPerspective(frame.ink.astype(np.uint8), warp, (width, height)) != 0
    shape = cv2.dilate(shape.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
    best = None
    for name, path in available:
        if not font_draws_text(path, source):
            continue
        probe = _text_alpha(path, source, 72).resize((width, height), Image.Resampling.LANCZOS)
        mask = np.asarray(probe) >= 128
        overlap = np.count_nonzero(mask & shape) / max(1, np.count_nonzero(mask | shape))
        if best is None or overlap > best[0] + .025:
            best = (overlap, name, path)
    return available[0] if best is None else (best[1], best[2])


def _stroke_radius(raw, cleared, ink, color) -> int:
    if color is None:
        return 0
    distance = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    nearby = (distance > 0) & (distance <= 10) & cleared
    samples = nearby & (cv2.absdiff(raw, tuple(float(c) for c in color) + (0.0,)).max(axis=2) <= 32)
    if np.count_nonzero(samples) < max(24, .08 * np.count_nonzero(nearby)):
        return 0
    return min(8, max(1, round(float(np.percentile(distance[samples], 75)))))


def render_source_sfx(raw: Image.Image, clean: Image.Image, letters: np.ndarray, text: str, *,
                      source: str, font: str, color_profile: dict | None, blocked: np.ndarray,
                      bold: bool = False) -> tuple[Image.Image, dict] | None:
    """Fit new effect lettering to the verified planar footprint of the original glyphs."""
    w, h = clean.size
    if (raw.size != clean.size or letters.shape != (h, w) or blocked.shape != (h, w)
            or min(w, h) < 36 or w * h > 1_250_000 or not text or len(text) > 40 or "\n" in text):
        return None
    fill, gradient, _, outline, glow, _ = _parse_color_profile(color_profile)
    if gradient or glow is not None:
        return None
    frame = _source_frame(letters)
    if frame is None:
        return None
    original, background = np.asarray(raw.convert("RGB")), np.asarray(clean.convert("RGB"))
    cleared = cv2.absdiff(original, background).max(axis=2) >= 30
    if np.mean(cleared[frame.ink]) < .85:
        return None
    colors = original[frame.ink & cleared]
    if fill is not None:
        matching = cv2.absdiff(colors.reshape(-1, 1, 3), tuple(float(c) for c in fill) + (0.0,)).max(axis=2).ravel() <= 48
        if np.mean(matching) >= .35:
            colors = colors[matching]
    low, middle, high = np.percentile(colors, (15, 50, 85), axis=0)
    if np.max(high - low) > 96:
        return None
    fill = tuple(int(round(c)) for c in middle)
    fw, fh = frame.size
    halo = min(12, max(3, round(.05 * fh)))
    near = cv2.dilate(frame.ink.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * halo + 1,) * 2)) != 0
    same_ink = cv2.absdiff(original, (*map(float, fill), 0.0)).max(axis=2) <= 32
    if np.count_nonzero(near & same_ink & ~cleared) > max(30, .02 * np.count_nonzero(frame.ink)):
        return None
    selected = _match_font(frame, text, source, font)
    if selected is None:
        return None
    chosen, path = selected
    if min(fw, fh) < 24:
        return None
    raster = _text_alpha(path, text, max(48, round(fh * 1.4)), bold)
    if not raster.getbbox():
        return None
    aspect = raster.width / raster.height
    target_h = min(round(.96 * fh), round(.96 * fw / max(.1, .72 * aspect)))
    target_w = min(round(.96 * fw), round(1.35 * target_h * aspect))
    if min(target_w, target_h) < 12:
        return None
    slab = Image.new("L", (fw, fh))
    slab.paste(raster.resize((target_w, target_h), Image.Resampling.LANCZOS),
               ((fw - target_w) // 2, (fh - target_h) // 2))
    matrix = cv2.getPerspectiveTransform(_corners(fw, fh), frame.quad)
    alpha = cv2.warpPerspective(np.asarray(slab), matrix, (w, h), flags=cv2.INTER_LINEAR)
    radius = _stroke_radius(original, cleared, frame.ink, outline)
    stroke = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) if radius else alpha
    active = stroke > 0
    if (not active.any() or active[0].any() or active[-1].any() or active[:, 0].any() or active[:, -1].any()
            or np.any(active & blocked)):
        return None
    solid = alpha >= 128
    if not solid.any():
        return None
    contrast = cv2.absdiff(background, tuple(float(c) for c in fill) + (0.0,)).max(axis=2)
    if np.mean(contrast[solid] >= 80) < .80 and not radius:
        return None
    output = clean.copy()
    if radius:
        output.paste(outline, (0, 0, w, h), Image.fromarray(stroke))
    output.paste(fill, (0, 0, w, h), Image.fromarray(alpha))
    bounds = Image.fromarray(stroke).getbbox()
    return output, {"size": target_h, "lines": 1, "ink_box": bounds, "draw_box": bounds,
                    "sfx_angle": round(frame.angle, 1), "sfx_tapered": frame.tapered, "sfx_font": chosen}
