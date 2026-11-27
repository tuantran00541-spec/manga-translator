from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.detector.bubble_mask import mask_row_spans
from app.render.font_metrics import get_font_object
from app.render.text_fitting import DEFAULT_BODY_FONT_SIZE, REFERENCE_PAGE_WIDTH, _word_units


_UNSPACED = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af]")
_BREAK_BEFORE = frozenset("、。，；：！？）》」』】〕）…")
_VI_FORWARD_WORDS = frozenset({"bị", "bởi", "của", "đã", "đang", "để", "được", "ở", "sẽ", "từ", "về", "vì", "với"})


@dataclass(frozen=True)
class Composition:
    size: int
    lines: tuple[tuple[int, int, str], ...]
    score: float
    clearance: int
    target: int


def _word_spans(text: str) -> tuple[str, list[tuple[int, int]], list[int]]:
    spans, ends, paragraphs = [], [], []
    offset = 0
    for paragraph in text.splitlines():
        words = " ".join(paragraph.split())
        if not words:
            continue
        if _UNSPACED.search(words):
            starts = [match.start() for match in re.finditer(
                r"[\u2e80-\u9fff\uac00-\ud7af]|[^\s\u2e80-\u9fff\uac00-\ud7af]+", words
            ) if words[match.start()] not in _BREAK_BEFORE]
            if not starts or starts[0] != 0:
                starts.insert(0, 0)
            for first, last in zip(starts, starts[1:] + [len(words)]):
                spans.append((offset + first, offset + len(words[:last].rstrip())))
        else:
            spans.extend((offset + word.start(), offset + word.end())
                         for word in re.finditer(r"\S+", words))
        ends.extend([len(spans)] * (len(spans) - len(ends)))
        offset += len(words) + 1
        paragraphs.append(words)
    return "\n".join(paragraphs), spans, ends


def _break_costs(text, spans, ends, *, balloon=False):
    costs = {}
    start = 0
    while start < len(spans):
        end = ends[start]
        words = tuple(text[a:b] for a, b in spans[start:end])
        if not any(_UNSPACED.search(word) for word in words):
            offset = start
            for unit in _word_units(words):
                following = offset + len(unit.split())
                costs.update((i, .5) for i in range(offset + 1, following))
                offset = following
            for i, (left, right) in enumerate(zip(words, words[1:]), start + 1):
                right = right.rstrip(".,;:!?…")
                if left.istitle() and left.isalpha() and right.istitle() and right.isalpha():
                    costs[i] = .65
                if balloon and left.lstrip('"\'“‘(«—-').casefold() in _VI_FORWARD_WORDS:
                    costs[i] = .85
        start = end
    return costs


def _interior(frame, mask, ellipse):
    x1, y1, x2, y2 = frame
    w, h = x2 - x1, y2 - y1
    inside = np.ones((h, w), dtype=np.uint8)
    if mask is not None:
        inside.fill(0)
        for row in range(h):
            for left, right in mask_row_spans(mask, y1 + row):
                a, b = max(0, left - x1), min(w, right - x1)
                if a < b:
                    inside[row, a:b] = 1
    elif ellipse:
        inside.fill(0)
        cv2.ellipse(inside, (w // 2, h // 2), (max(1, w // 2 - 1), max(1, h // 2 - 1)),
                    0, 0, 360, 1, -1)
    return inside


def _edge_distances(inside):
    return cv2.distanceTransform(np.pad(inside, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def _exclude_art(safe, inside, image, frame, gap) -> bool:
    surface = image.crop(frame)
    gray = np.asarray(surface.convert("L"))
    occupied = inside != 0
    values = gray[occupied]
    if not values.size:
        return False
    if np.count_nonzero(values > 220) > 0.80 * values.size:
        foreign_ink = gray < 135
    elif np.count_nonzero(values < 42) > 0.80 * values.size:
        foreign_ink = gray > 122
    else:
        lab = cv2.cvtColor(np.asarray(surface.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
        deviation = np.linalg.norm(lab - np.median(lab[occupied], axis=0), axis=2)
        if np.mean(deviation[occupied] <= 16) < .8:
            return False
        foreign_ink = deviation > 32
    foreign_ink &= occupied
    if foreign_ink.any():
        count, flabels, fstats, _ = cv2.connectedComponentsWithStats(foreign_ink.astype(np.uint8), connectivity=8)
        min_art_area = max(16, int(0.002 * int(np.count_nonzero(occupied))))
        keep_ids = np.flatnonzero(fstats[1:, cv2.CC_STAT_AREA] >= min_art_area) + 1
        if keep_ids.size == 0:
            return False
        significant = np.isin(flabels, keep_ids)
        inside[significant] = 0
        safe &= _edge_distances(inside) > gap
        return True
    return False


def _line_spans(safe, top, n, height, step, anchor_x):
    if top < 0 or top + (n - 1) * step + height > safe.shape[0]:
        return ()
    spans = []
    for y in range(top, top + n * step, step):
        columns = safe[y:y + height].all(axis=0)
        edges = np.flatnonzero(np.diff(columns, prepend=False, append=False))
        if not edges.size:
            return ()
        spans.append(sorted(zip(edges[::2], edges[1::2]),
                            key=lambda ab: (max(ab[0] - anchor_x, anchor_x - ab[1], 0), ab[0] - ab[1])))
    return spans


def _ink_rows(line, font, bearing, width, top, height, edge, bold):
    """Measure the occupied span of each raster row, including the stroke and bold offsets."""
    bitmap = Image.new("L", (width, height))
    draw = ImageDraw.Draw(bitmap)
    for dy in range(1 + int(bold)):
        for dx in range(1 + int(bold)):
            draw.text((dx - bearing, dy - top), line, fill=255, font=font, stroke_width=edge)
    ink = np.asarray(bitmap) > 0
    rows = np.flatnonzero(ink.any(axis=1))
    active = ink[rows]
    return rows, active.argmax(axis=1), width - active[:, ::-1].argmax(axis=1)


def _place_ink_rows(runs, profile, top, width, anchor_x):
    """Intersect the allowed horizontal positions of all occupied rows at this baseline."""
    rows, lefts, rights = profile
    rows = rows + top
    if not rows.size or rows[0] < 0 or rows[-1] >= runs.shape[0]:
        return None
    origins = np.arange(runs.shape[1] - width + 1)
    ends = runs[rows[:, None], origins[None, :] + lefts[:, None]]
    fits = origins[(ends >= origins[None, :] + rights[:, None]).all(axis=0)]
    if fits.size:
        return int(fits[np.abs(fits + width / 2 - anchor_x).argmin()])
    return None


def _source_chamber(safe, frame, anchor, seed):
    x1, y1, x2, y2 = frame
    count, labels, stats, _ = cv2.connectedComponentsWithStats(safe.astype(np.uint8), connectivity=8)
    if count <= 2:
        return safe
    if anchor is not None:
        ax1, ay1, ax2, ay2 = anchor
        ink_area = labels[max(0, ay1 - y1):min(y2, ay2) - y1,
                          max(0, ax1 - x1):min(x2, ax2) - x1]
        if ink_area.size and ax2 > x1 and ay2 > y1 and ax1 < x2 and ay1 < y2:
            hits = np.bincount(ink_area.ravel(), minlength=count)[1:]
            if hits.any():
                return labels == int(hits.argmax()) + 1
    if seed is not None:
        sx, sy = seed
        if x1 <= sx < x2 and y1 <= sy < y2 and labels[sy - y1, sx - x1]:
            return labels == labels[sy - y1, sx - x1]
    return labels == int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1


def _track_tops(reference, minimum, maximum, height, step, lines, window_widths, shift, *, has_blockers):
    centred = [min(max(reference + offset, minimum), maximum)
               for offset in (0, -shift, shift, -2 * shift, 2 * shift)]
    yield from dict.fromkeys(centred)
    if not has_blockers:
        minimum, maximum = min(centred), max(centred)
    stride = max(2, height // 3, (maximum - minimum) // 48)
    starts = np.array([top for top in dict.fromkeys((*range(minimum, maximum + 1, stride), maximum))
                       if top not in centred], dtype=np.intp)
    if starts.size:
        widths = window_widths[starts[:, None] + step * np.arange(lines)].min(axis=1)
        best = np.lexsort((np.abs(starts - reference), -widths))
        yield from starts[best[:5]].tolist()


def compose_text(image, text, frame, font_path, *, preferred, floor, stroke, bold,
                 glow=0, mask=None, ellipse=False, anchor=None, reserved_boxes=(),
                 verified_balloon=False, preserve_paragraph_shape=False) -> Composition | None:
    x1, y1, x2, y2 = frame
    w, h = x2 - x1, y2 - y1
    if min(w, h) < 8:
        return None
    text = unicodedata.normalize("NFC", str(text))
    text, words, ends = _word_spans(text)
    if not words:
        return None
    break_cost = _break_costs(text, words, ends, balloon=verified_balloon)
    inside = _interior(frame, mask, ellipse)
    distance = _edge_distances(inside)
    gap = max(2, round(0.065 * min(w, h)))
    safe = distance > gap
    if mask is not None:
        seed = (int(mask.get("cx", x1)), int(mask.get("cy", y1)))
        safe = _source_chamber(safe, frame, anchor, seed)
    rows, cols = np.nonzero(safe)
    if not rows.size:
        return None
    weight = (distance[rows, cols] - gap) ** 2
    anchor_x = float(np.average(cols, weights=weight))
    anchor_y = float(np.average(rows, weights=weight))
    if anchor is not None and not verified_balloon:
        ax1, ay1, ax2, ay2 = anchor
        anchor_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        safe_area = int(rows.size)
        if safe_area > 1.8 * anchor_area:
            acx = max(0.0, min(float(w - 1), (ax1 + ax2) / 2.0 - x1))
            acy = max(0.0, min(float(h - 1), (ay1 + ay2) / 2.0 - y1))
            blend = min(0.75, 1.0 - (1.8 * anchor_area / safe_area))
            cand_x = anchor_x * (1.0 - blend) + acx * blend
            cand_y = anchor_y * (1.0 - blend) + acy * blend
            best_idx = int(np.argmin((cols - cand_x) ** 2 + (rows - cand_y) ** 2))
            anchor_x = float(cols[best_idx])
            anchor_y = float(rows[best_idx])
    art_excluded = False
    if mask is not None:
        art_excluded = _exclude_art(safe, inside, image, frame, gap)
    has_reserved = False
    for ox1, oy1, ox2, oy2 in reserved_boxes:
        left, right = max(0, ox1 - x1), min(w, ox2 - x1)
        top, bottom = max(0, oy1 - y1), min(h, oy2 - y1)
        if left < right and top < bottom:
            safe[top:bottom, left:right] = False
            has_reserved = True
    rows, cols = np.nonzero(safe)
    if not rows.size:
        return None
    row_widths = safe.sum(axis=1)
    runs = None
    if verified_balloon:
        breaks = np.where(safe, w, np.arange(w, dtype=np.int32))
        runs = np.minimum.accumulate(breaks[:, ::-1], axis=1)[:, ::-1]
    maximum_width = int(cols.max() - cols.min() + 1)
    min_y, max_y = int(rows.min()), int(rows.max()) + 1
    maximum_height = max_y - min_y
    paragraph_count = len(set(ends))
    compact = (mask is not None or ellipse) and len(words) >= 4 and paragraph_count == 1
    x_scale, y_scale = max(1, w), max(1, h)
    # Measure the stroke and glow inside the same clearance as the ink.
    edge = stroke + glow
    draw = ImageDraw.Draw(image)
    preferred = max(floor, preferred)
    ceiling = min(preferred, int(maximum_height * 1.45))
    sizes = {max(floor, round(ceiling * f)) for f in
             (1, .97, .94, .90, .85, .79, .72, .64, .56, .48, .40, .33, .26, .20, .14, 0)}
    one_line = False
    if verified_balloon and w >= h and paragraph_count == 1 and len(words) <= 5:
        body_size = DEFAULT_BODY_FONT_SIZE * max(1.0, image.width / REFERENCE_PAGE_WIDTH)
        probe_size = max(floor, round(min(0.76 * preferred, body_size)))
        font = get_font_object(font_path, probe_size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=edge)
        line_height = max(1, bottom + int(bold) - top)
        centre = round(anchor_y - line_height / 2)
        span = _line_spans(safe, centre, 1, line_height, 1, anchor_x)
        width = right - left + int(bold)
        one_line = bool(span and any(b - a >= width for a, b in span[0]))
        if not one_line and 0 < width <= maximum_width:
            profile = _ink_rows(text, font, left, width, top, line_height, edge, bold)
            one_line = _place_ink_rows(runs, profile, centre, width, anchor_x) is not None
        if one_line:
            sizes.add(probe_size)
    best = None

    for size in sorted(sizes, reverse=True):
        size_cost = 1.15 * math.log(preferred / size) ** 2
        if best is not None and size_cost >= best.score:
            break
        font = get_font_object(font_path, size)
        _, asc, _, desc = draw.textbbox((0, 0), text.replace("\n", " "), font=font, stroke_width=edge)
        ink_height = max(1, desc - asc + int(bold))
        step = max(round(size * 1.02), ink_height + max(2, round(size * 0.12)))
        max_lines = min(len(words), 1 + (maximum_height - ink_height) // step)
        if one_line:
            max_lines = min(max_lines, 1)
        if paragraph_count > max_lines:
            continue
        window_widths = np.lib.stride_tricks.sliding_window_view(row_widths, ink_height).min(axis=1)

        @lru_cache(maxsize=None)
        def measure(start, end):
            line = text[words[start][0]:words[end - 1][1]]
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font, stroke_width=edge)
            return line, left, right + int(bold) - left, top, bottom + int(bold) - top

        @lru_cache(maxsize=None)
        def ink_spans(y, height):
            spans = _line_spans(safe, y, 1, max(1, height), 1, anchor_x)
            return spans[0] if spans else ()

        @lru_cache(maxsize=256)
        def ink_profile(start, end):
            line, bearing, width, top, height = measure(start, end)
            return _ink_rows(line, font, bearing, width, top, height, edge, bold)

        @lru_cache(maxsize=1024)
        def place_profile(start, end, y, width):
            return _place_ink_rows(runs, ink_profile(start, end), y, width, anchor_x)

        paragraph_widths = dict(zip(dict.fromkeys(ends),
                                    (draw.textlength(line, font=font) for line in text.splitlines())))
        natural_width = sum(paragraph_widths.values())
        if preserve_paragraph_shape and paragraph_count > 1:
            paragraph_targets = {end: width / max(1, math.ceil(width / (.9 * maximum_width)))
                                 for end, width in paragraph_widths.items()}
        else:
            paragraph_targets = {}
        min_lines = max(paragraph_count, math.ceil(natural_width / (maximum_width + size * .5)))
        for n in range(min_lines, min(max_lines, min_lines + 7) + 1):
            if n > min_lines + 3 and best is not None:
                break
            mean_width = natural_width / n
            block_height = (n - 1) * step + ink_height
            reference = round(anchor_y - block_height / 2)
            shift = max(2, round(min(w, h) * .095))
            tops = _track_tops(reference, min_y, max_y - block_height, ink_height, step, n,
                               window_widths, shift, has_blockers=has_reserved or art_excluded)
            for top in tops:
                spans = [()] * n if verified_balloon else _line_spans(safe, top, n, ink_height, step, anchor_x)
                if not spans:
                    continue
                vertical_cost = 3 * ((top + block_height / 2 - anchor_y) / y_scale) ** 2
                base_score = size_cost + vertical_cost + .015 * (n - 1)
                if best is not None and base_score >= best.score:
                    continue
                # Retain the best partial layout for each word boundary.
                states = {0: (0., ())}
                for i, intervals in enumerate(spans):
                    next_states = {}
                    remaining = n - i - 1
                    limit = maximum_width if verified_balloon else max(b - a for a, b in intervals)
                    baseline = top + i * step - asc
                    target = mean_width * (1 + .12 * math.cos((i - (n - 1) / 2) * math.pi / max(1, n - 1)))
                    for start, (cost, placed) in states.items():
                        if start == len(words):
                            continue
                        stop = min(ends[start], len(words) - remaining)
                        for end in range(start + 1 if remaining else len(words), stop + 1):
                            line, bearing, width, ink_top, ink_h = measure(start, end)
                            if width > limit:
                                break
                            choices = ink_spans(baseline + ink_top, ink_h) if verified_balloon else intervals
                            span = min(((a, b) for a, b in choices if b - a >= width),
                                       key=lambda ab: max(ab[0] + width / 2 - anchor_x,
                                                          anchor_x - ab[1] + width / 2, 0), default=None)
                            left = None
                            if span is not None:
                                a, b = span
                                left = round(min(max(anchor_x - width / 2, a), b - width))
                            if (runs is not None and width > 0 and ink_h > 0
                                    and (left is None or abs(left + width / 2 - anchor_x) > gap)):
                                shaped = place_profile(start, end, baseline + ink_top, width)
                                if shaped is not None:
                                    left = shaped
                            if left is None:
                                continue
                            drift = (left + width / 2 - anchor_x) / x_scale
                            line_target = paragraph_targets.get(ends[start], target)
                            line_scale = paragraph_targets.get(ends[start], mean_width)
                            ragged = (width - line_target) / max(size, line_scale)
                            widow_limit = .8 if mask is not None and len(words) <= 6 else .58
                            widow = remaining == 0 and n > 1 and end == start + 1 and width < widow_limit * mean_width
                            wide = 2 * max(0, width / maximum_width - .76) ** 2 if compact else 0
                            value = cost + (.60 * ragged**2 + 7 * drift**2 + .24 * widow + wide
                                            + break_cost.get(end, 0)) / n
                            if end not in next_states or value < next_states[end][0]:
                                next_states[end] = (value, placed + ((x1 + left - bearing, y1 + baseline, line),))
                    states = next_states
                    if not states:
                        break
                if len(words) in states:
                    cost, lines = states[len(words)]
                    total = base_score + cost
                    if best is None or total < best.score:
                        best = Composition(size, lines, total, gap, preferred)
    return best
