from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app.detector.bubble_mask import (
    derive_bubble_mask,
    find_waist,
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
SOURCE_MATCH_MAX_FONT_SIZE = 200
BOX_RULE_EXPAND = 64

_CURVED_CONTAINERS = frozenset({"bubble", "spiky"})
_DERIVED_MASK_CACHE: dict = {}
_DERIVED_MASK_CACHE_MAX = 64


def _mask_bands(mask, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    bands = []
    for i in range(24):
        mid = y1 + ((2 * i + 1) * (y2 - y1)) // 48
        spans = mask_row_spans(mask, mid)
        bands.append([
            (a - x1 + MASK_BAND_PAD, b - x1 - MASK_BAND_PAD)
            for a, b in spans if b - a > 2 * MASK_BAND_PAD
        ])
    return bands


def _lines_fit_spans(draw, lines, font, spans, stroke_w: int) -> bool:
    if not lines:
        return True
    if spans is None or len(spans) != len(lines):
        return False
    return all(
        draw.textbbox((0, 0), line, font=font)[2] + 2 * stroke_w <= b - a
        for line, (a, b) in zip(lines, spans)
    )


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


def _derived_mask(image, box, letter_box):
    arr = np.asarray(image)
    key = (
        tuple(int(v) for v in box),
        tuple(int(v) for v in letter_box) if letter_box is not None else None,
        arr.shape,
        hashlib.md5(arr.tobytes()).hexdigest(),
    )
    hit = _DERIVED_MASK_CACHE.get(key)
    if hit is not None:
        return hit
    bgr = arr[:, :, ::-1].copy() if arr.ndim == 3 else arr
    seed = None
    if letter_box is not None:
        seed = (
            (int(letter_box[0]) + int(letter_box[2])) // 2,
            (int(letter_box[1]) + int(letter_box[3])) // 2,
        )
    mask = derive_bubble_mask(bgr, tuple(int(v) for v in box), seed=seed)
    if len(_DERIVED_MASK_CACHE) >= _DERIVED_MASK_CACHE_MAX:
        _DERIVED_MASK_CACHE.pop(next(iter(_DERIVED_MASK_CACHE), None), None)
    _DERIVED_MASK_CACHE[key] = mask
    return mask


def _resolve_mask(
    image, box, container, letter_box, bubble_mask, letters=None, letters_box=None,
):
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

    left = side(
        (grey[my1:my2, max(0, x1 - 16):min(width, x1 + 64)] > 180).mean(axis=0),
        max(0, x1 - 16), x1, True,
    )
    rx0 = max(0, x2 - 64)
    right = side(
        (grey[my1:my2, rx0:min(width, x2 + 16)] > 180).mean(axis=0),
        rx0, x2, False,
    )
    top = side(
        (grey[max(0, y1 - 16):min(height, y1 + 64), mx1:mx2] > 180).mean(axis=1),
        max(0, y1 - 16), y1, True,
    )
    by0 = max(0, y2 - 64)
    bottom = side(
        (grey[by0:min(height, y2 + 16), mx1:mx2] > 180).mean(axis=1),
        by0, y2, False,
    )
    fx1, fx2, measured_x = x1, x2, False
    if left is not None and right is not None and left < (x1 + x2) // 2 < right and right - left >= 40:
        fx1, fx2, measured_x = left, right, True
    fy1, fy2, measured_y = y1, y2, False
    if top is not None and bottom is not None and top < (y1 + y2) // 2 < bottom and bottom - top >= 40:
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


def _wrap_to_spans(draw, text: str, font, spans: list[tuple[int, int]], stroke_w: int = 0) -> list[str] | None:
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


def _fits(
    draw, text: str, box_w: int, box_h: int, font_path_str: str, size: int, stroke_w: int,
    curved: bool = False, bands=None, anchor_y: int | None = None,
) -> tuple[bool, list[str]]:
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
            ok, lines = _fits(
                draw, text, box_w, box_h, font_path_str, size, stroke_w,
                curved=curved, bands=bands, anchor_y=anchor_y,
            )
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


def _wrap_text(
    draw, text: str, font, box_w: int, *, balance: bool = False, join_lines: bool = False,
) -> list[str]:
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
