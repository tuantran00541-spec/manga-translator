from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app.detector.bubble_mask import (
    derive_bubble_mask,
    find_waist,
    mask_inscribed_rect,
    mask_row_spans,
    split_bubble_mask,
)
from app.parameters import (
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    RENDER_AUTO_STROKE_WIDTH,
    RENDER_MIN_READABLE_FONT_SIZE,
)
from app.render.font_metrics import _calc_line_height, get_font_object

MASK_BAND_PAD = 4
ENLARGE_GROW_RATIO = 0.25
ENLARGED_MIN_FONT_SIZE = 22
SOURCE_FLOOR_RATIO = 0.6
REFERENCE_PAGE_WIDTH = 800
DEFAULT_BODY_FONT_SIZE = 34
SOURCE_MATCH_MAX_FONT_SIZE = 200
BOX_RULE_EXPAND = 64

_CURVED_CONTAINERS = frozenset({"bubble", "spiky"})
_DERIVED_MASK_CACHE: OrderedDict = OrderedDict()
_DERIVED_MASK_CACHE_LOCK = threading.Lock()
_DERIVED_MASK_CACHE_MAX = 64


def _mask_bands(mask, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    bands = []
    for i in range(24):
        mid = y1 + ((2 * i + 1) * (y2 - y1)) // 48
        spans = mask_row_spans(mask, mid)
        padded = [(max(0, a - x1 + MASK_BAND_PAD),
                   min(x2 - x1, b - x1 - MASK_BAND_PAD)) for a, b in spans]
        bands.append([(a, b) for a, b in padded if b > a])
    return bands


def _lines_fit_spans(draw, lines, font, spans, stroke_w: int) -> bool:
    if not lines:
        return True
    if spans is None or len(spans) != len(lines):
        return False
    for line, (a, b) in zip(lines, spans):
        left, _, right, _ = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
        if right - left > b - a:
            return False
    return True


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


def _mask_has_narrow_waist(mask) -> bool:
    waist = find_waist(mask)
    if waist is None:
        return False
    if waist["orientation"] == "vertical":
        return True
    lobe_widths = waist.get("lobe_widths")
    if not lobe_widths:
        return False
    narrowest = max((b - a for a, b in mask_row_spans(mask, waist["cut_y"])), default=0)
    return 2 * narrowest < min(lobe_widths)


def _mask_layout_y(mask, *, inner_body: bool = False) -> int:
    centroid = _mask_centroid(mask)
    if centroid is None:
        bbox = mask.get("bbox") if isinstance(mask, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return (int(bbox[1]) + int(bbox[3])) // 2
        return 0
    _, centre = centroid
    if _mask_has_narrow_waist(mask):
        return int(mask.get("cy", centre))
    if inner_body:
        x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
        if x2 > x1 and y2 > y1:
            left, top, right, bottom = mask_inscribed_rect(mask)
            if (right - left) * (bottom - top) >= 0.3 * (x2 - x1) * (y2 - y1):
                return (top + bottom) // 2
    return centre


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


def _shift_derived_mask(mask: dict, dx: int, dy: int) -> dict:
    if not dx and not dy:
        return mask
    x1, y1, x2, y2 = mask["bbox"]
    shifted = dict(mask)
    shifted["bbox"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    if "cx" in shifted:
        shifted["cx"] = shifted["cx"] + dx
    if "cy" in shifted:
        shifted["cy"] = shifted["cy"] + dy
    return shifted


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, :, :3][:, :, ::-1].copy()
    return arr


def _derived_mask(image, box, letter_box):
    box_t = tuple(int(v) for v in box)
    letter_t = tuple(int(v) for v in letter_box) if letter_box is not None else None
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        width, height = image.size
        mode_sig = image.mode
    else:
        arr_full = np.asarray(image)
        height, width = arr_full.shape[:2]
        mode_sig = (arr_full.shape[2:] if arr_full.ndim > 2 else (), str(arr_full.dtype))

    x1, y1, x2, y2 = box_t
    margin = max(120, max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
    cx2, cy2 = min(width, x2 + margin), min(height, y2 + margin)
    has_crop = cx2 > cx1 and cy2 > cy1
    crop_arr = None
    if has_crop:
        if is_pil:
            crop_pil = image.crop((cx1, cy1, cx2, cy2))
            try:
                crop_arr = np.ascontiguousarray(crop_pil)
            finally:
                crop_pil.close()
        else:
            crop_arr = np.ascontiguousarray(arr_full[cy1:cy2, cx1:cx2])
        digest = hashlib.md5(memoryview(crop_arr)).hexdigest()
    else:
        digest = ""

    key = (box_t, letter_t, (width, height), mode_sig, digest)
    with _DERIVED_MASK_CACHE_LOCK:
        hit = _DERIVED_MASK_CACHE.get(key)
        if hit is not None:
            _DERIVED_MASK_CACHE.move_to_end(key)
            return copy.deepcopy(hit)

    seed = None
    if letter_t is not None:
        seed = (
            (letter_t[0] + letter_t[2]) // 2,
            (letter_t[1] + letter_t[3]) // 2,
        )

    mask = None
    full_crop = (cx1 == 0 and cy1 == 0 and cx2 == width and cy2 == height)
    if has_crop and crop_arr is not None:
        crop_bgr = _to_bgr(crop_arr)
        local_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        local_seed = (seed[0] - cx1, seed[1] - cy1) if seed is not None else None
        local_mask = derive_bubble_mask(crop_bgr, local_box, seed=local_seed)
        if local_mask.get("kind") == "bubble" or full_crop:
            mask = _shift_derived_mask(local_mask, cx1, cy1)

    if mask is None:
        arr = np.asarray(image) if is_pil else arr_full
        mask = derive_bubble_mask(_to_bgr(arr), box_t, seed=seed)

    with _DERIVED_MASK_CACHE_LOCK:
        _DERIVED_MASK_CACHE[key] = copy.deepcopy(mask)
        _DERIVED_MASK_CACHE.move_to_end(key)
        while len(_DERIVED_MASK_CACHE) > _DERIVED_MASK_CACHE_MAX:
            _DERIVED_MASK_CACHE.popitem(last=False)
    return mask


def _resolve_mask(
    image, box, container, letter_box, bubble_mask, letters=None, letters_box=None, *, verified_balloon=False,
):
    mask = bubble_mask if isinstance(bubble_mask, dict) else None
    if mask is not None:
        if verified_balloon and mask.get("kind") == "bubble":
            return mask, "verified"
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


def _ellipse_line_widths(box_w: int, box_h: int, line_count: int, line_height: int) -> list[float]:
    if line_count <= 0 or box_w <= 0 or box_h <= 0 or line_height <= 0:
        return []
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


def _lines_fit_ellipse(draw, lines, font, box_w: int, box_h: int, stroke_w: int) -> bool:
    if not lines:
        return True
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    limits = _ellipse_line_widths(box_w, box_h, len(lines), line_height)
    if len(limits) != len(lines):
        return False
    return all(
        draw.textbbox((0, 0), line, font=font)[2] + 2 * stroke_w <= limit
        for line, limit in zip(lines, limits)
    )


def _crop_gray_roi(image: Image.Image, rx1: int, ry1: int, rx2: int, ry2: int) -> np.ndarray:
    cropped = image.crop((rx1, ry1, rx2, ry2))
    try:
        if cropped.mode == "L":
            return np.asarray(cropped).copy()
        gray_img = cropped.convert("L")
        try:
            return np.asarray(gray_img).copy()
        finally:
            gray_img.close()
    finally:
        cropped.close()


def _box_frame(image: Image.Image, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    width, height = image.size
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if cx2 - cx1 < 40 or cy2 - cy1 < 40:
        return None
    ex1, ex2 = max(0, x1 - BOX_RULE_EXPAND), min(width, x2 + BOX_RULE_EXPAND)
    ey1, ey2 = max(0, y1 - BOX_RULE_EXPAND), min(height, y2 + BOX_RULE_EXPAND)
    roi_grey = _crop_gray_roi(image, ex1, ey1, ex2, ey2)
    grey = roi_grey[cy1 - ey1:cy2 - ey1, cx1 - ex1:cx2 - ex1]
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
    for axis in ("x", "y"):
        if axis == "x":
            crop = roi_grey[cy1 - ey1:cy2 - ey1, :] < 110
            share = crop.mean(axis=0) if crop.size else np.array([])
            lo, hi, extent = cx1 - ex1, cx2 - ex1, box_w
            strips = lambda a, b: roi_grey[cy1 - ey1:cy2 - ey1, a:b]
            first_groups = col_groups
            unmeasured = first_fx == (cx1, cx2)
        else:
            crop = roi_grey[:, cx1 - ex1:cx2 - ex1] < 110
            share = crop.mean(axis=1) if crop.size else np.array([])
            lo, hi, extent = cy1 - ey1, cy2 - ey1, box_h
            strips = lambda a, b: roi_grey[a:b, cx1 - ex1:cx2 - ex1]
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
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 40 or box_h < 40:
        return (x1, x2, y1, y2, False, False)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if cx2 - cx1 < 40 or cy2 - cy1 < 40:
        return (x1, x2, y1, y2, False, False)
    rx1, ry1 = max(0, cx1 - 16), max(0, cy1 - 16)
    rx2, ry2 = min(width, cx2 + 16), min(height, cy2 + 16)
    grey = _crop_gray_roi(image, rx1, ry1, rx2, ry2)
    my1 = max(ry1, min(ry2, y1 + box_h // 5)) - ry1
    my2 = max(ry1, min(ry2, y2 - box_h // 5)) - ry1
    mx1 = max(rx1, min(rx2, x1 + box_w // 5)) - rx1
    mx2 = max(rx1, min(rx2, x2 - box_w // 5)) - rx1

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

    def side(strip, axis: int, lo: int, edge: int, from_start: bool):
        if strip.size == 0:
            return None
        share = (strip > 180).mean(axis=axis)
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

    lx0, lx1 = max(rx1, x1 - 16), min(rx2, x1 + 64)
    left = side(grey[my1:my2, lx0 - rx1:lx1 - rx1], 0, lx0, x1, True)
    rx0, rx1_end = max(rx1, x2 - 64), min(rx2, x2 + 16)
    right = side(grey[my1:my2, rx0 - rx1:rx1_end - rx1], 0, rx0, x2, False)
    ty0, ty1 = max(ry1, y1 - 16), min(ry2, y1 + 64)
    top = side(grey[ty0 - ry1:ty1 - ry1, mx1:mx2], 1, ty0, y1, True)
    by0, by1 = max(ry1, y2 - 64), min(ry2, y2 + 16)
    bottom = side(grey[by0 - ry1:by1 - ry1, mx1:mx2], 1, by0, y2, False)
    fx1, fx2, measured_x = x1, x2, False
    if left is not None and right is not None and left < (x1 + x2) // 2 < right and right - left >= 40:
        fx1, fx2, measured_x = left, right, True
    fy1, fy2, measured_y = y1, y2, False
    if top is not None and bottom is not None and top < (y1 + y2) // 2 < bottom and bottom - top >= 40:
        fy1, fy2, measured_y = top, bottom, True
    return (fx1, fx2, fy1, fy2, measured_x, measured_y)


def _band_line_spans(box_w: int, box_h: int, line_count: int, line_height: int, bands, top: int | None = None) -> list[tuple[int, int]] | None:
    if line_count <= 0 or box_w <= 0 or box_h <= 0 or line_height <= 0 or not bands:
        return None
    total_h = line_height * line_count
    if top is None:
        top = max(0, (box_h - total_h) // 2)
    if top < 0 or top + total_h > box_h:
        return None
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
                return None
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


def _wrap_to_spans(draw, text: str, font, spans: list[tuple[int, int]], stroke_w: int = 0,
                   *, keep_compounds: bool = False) -> list[str] | None:
    raw_lines = [raw for raw in text.splitlines() if raw.strip()]
    if not raw_lines:
        return None
    out: list[str] = []
    span_idx = 0
    for raw_line in raw_lines:
        words = raw_line.split()
        if keep_compounds:
            words = _word_units(tuple(words))
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


def _fits(
    draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int,
    curved: bool = False, bands=None, anchor_y: int | None = None,
    keep_compounds: bool = False,
) -> tuple[bool, list[str]]:
    if box_w <= 0 or box_h <= 0 or not str(text or "").strip():
        return False, []
    font = get_font_object(font_path_str, size)
    lines = _wrap_text(draw, text, font, box_w, keep_compounds=keep_compounds)
    if not lines:
        return False, []
    line_height = _calc_line_height(draw, font, stroke_w=stroke_w)
    if line_height <= 0:
        return False, lines
    total_h = line_height * len(lines)
    widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    whole = [word for line in lines for word in line.split()] == text.split()
    fits = whole and total_h <= box_h and max(widths) + 2 * stroke_w <= box_w
    if fits:
        if bands is not None:
            top0 = _anchored_top(box_h, line_height * len(lines), anchor_y)
            spans0 = _band_line_spans(box_w, box_h, len(lines), line_height, bands, top=top0)
            if _lines_fit_spans(draw, lines, font, spans0, stroke_w):
                pass
            else:
                fitted_to_bands = None
                max_lines = box_h // line_height
                for k in range(len(lines), min(max_lines, len(lines) + 3) + 1):
                    top = _anchored_top(box_h, line_height * k, anchor_y)
                    spans = _band_line_spans(box_w, box_h, k, line_height, bands, top=top)
                    if spans is None:
                        continue
                    cand = _wrap_to_spans(draw, text, font, spans, stroke_w, keep_compounds=keep_compounds)
                    if cand is None or line_height * len(cand) > box_h:
                        continue
                    if len(cand) == k:
                        fitted_to_bands = cand
                        break
                    ctop = _anchored_top(box_h, line_height * len(cand), anchor_y)
                    cspans = _band_line_spans(box_w, box_h, len(cand), line_height, bands, top=ctop)
                    if _lines_fit_spans(draw, cand, font, cspans, stroke_w):
                        fitted_to_bands = cand
                        break
                if fitted_to_bands is not None:
                    lines = fitted_to_bands
                else:
                    fits = False
        elif curved:
            limits = _ellipse_line_widths(box_w, box_h, len(lines), line_height)
            fits = len(limits) == len(lines) and all(
                w + 2 * stroke_w <= limit for w, limit in zip(widths, limits)
            )
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
    keep_compounds: bool = False,
) -> tuple[int, list[str], bool]:
    minimum_size = max(MIN_FONT_SIZE, int(minimum_size))
    hi_max = max(minimum_size, min(SOURCE_MATCH_MAX_FONT_SIZE, int(maximum_size)))
    lo, hi = minimum_size, hi_max
    best_size = minimum_size
    best_lines: list[str] = []
    found_fit = False

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, lines = _fits(draw, text, box_w, box_h, font_path_str, mid, stroke_w,
                          keep_compounds=keep_compounds)
        if ok:
            best_size = mid
            best_lines = lines
            found_fit = True
            lo = mid + 1
        else:
            hi = mid - 1

    # Probe a small upward window past binary search in case font hinting is
    # slightly non-monotonic, stopping after 4 consecutive misses.
    misses = 0
    for size in range(best_size + 1, hi_max + 1):
        ok, lines = _fits(draw, text, box_w, box_h, font_path_str, size, stroke_w,
                          keep_compounds=keep_compounds)
        if ok:
            best_size = size
            best_lines = lines
            found_fit = True
            misses = 0
        else:
            misses += 1
            if misses >= 4:
                break

    if (curved or bands is not None) and best_lines:
        for size in _scan_sizes(best_size, minimum_size - 1):
            ok, lines = _fits(
                draw, text, box_w, box_h, font_path_str, size, stroke_w,
                curved=curved, bands=bands, anchor_y=anchor_y, keep_compounds=keep_compounds,
            )
            if ok:
                return size, lines, True
        min_font = get_font_object(font_path_str, minimum_size)
        return minimum_size, _wrap_text(
            draw, text, min_font, box_w, keep_compounds=keep_compounds
        ), False

    if not best_lines:
        min_font = get_font_object(font_path_str, minimum_size)
        best_lines = _wrap_text(draw, text, min_font, box_w, keep_compounds=keep_compounds)
    return best_size, best_lines, found_fit


def _greedy_groups(draw, words: Sequence[str], font, width: int) -> list[list[str]]:
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


def _greedy_lines(draw, words: Sequence[str], font, width: int) -> list[str]:
    return [" ".join(group) for group in _greedy_groups(draw, words, font, width)]


_VI_WORDS = Path(__file__).with_name("vi_words.txt")
_SYLLABLE = re.compile(r"(\W*)(\w+)(\W*)")


@lru_cache(maxsize=1)
def _vi_words() -> frozenset[str]:
    try:
        return frozenset(line for line in _VI_WORDS.read_text(encoding="utf-8").splitlines() if line)
    except OSError:
        return frozenset()


@lru_cache(maxsize=1024)
def _word_units(words: tuple[str, ...]) -> tuple[str, ...]:
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
    return tuple(units)


def _balanced_lines(draw, words: Sequence[str], font, box_w: int, *, keep_compounds: bool = False) -> list[str]:
    lines = _greedy_lines(draw, words, font, box_w)
    if len(lines) < 2:
        return lines
    if not keep_compounds:
        units = _word_units(tuple(words))
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


def _wrap_text(
    draw, text: str, font, box_w: int, *, balance: bool = False, join_lines: bool = False,
    keep_compounds: bool = False,
) -> list[str]:
    if join_lines:
        text = " ".join(text.split())
    raw_lines = text.splitlines()
    all_wrapped = []
    for raw_line in raw_lines:
        if not raw_line.strip():
            continue
        words = raw_line.split()
        if keep_compounds:
            units = _word_units(tuple(words))
            if all(draw.textbbox((0, 0), unit, font=font)[2] <= box_w for unit in units):
                lines = (_balanced_lines(draw, units, font, box_w, keep_compounds=True)
                         if balance else _greedy_lines(draw, units, font, box_w))
                all_wrapped.extend(lines)
                continue
        if balance and all(draw.textbbox((0, 0), word, font=font)[2] <= box_w for word in words):
            all_wrapped.extend(_balanced_lines(draw, words, font, box_w, keep_compounds=keep_compounds))
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
