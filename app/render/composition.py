from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.detector.bubble_mask import mask_row_spans
from app.render.font_metrics import get_font_object
from app.render.text_fitting import _word_units


_UNSPACED = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af]")
_BREAK_BEFORE = frozenset("、。，；：！？）》」』】〕）…")


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
            start = offset
            for unit in _word_units(tuple(words.split())):
                spans.append((start, start + len(unit)))
                start += len(unit) + 1
        ends.extend([len(spans)] * (len(spans) - len(ends)))
        offset += len(words) + 1
        paragraphs.append(words)
    return "\n".join(paragraphs), spans, ends


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


def _exclude_art(safe, inside, image, frame, gap):
    gray = np.asarray(image.crop(frame).convert("L"))
    values = gray[inside != 0]
    if values.size and np.count_nonzero(values > 220) > 0.80 * values.size:
        foreign_ink = (gray < 135) & inside.astype(bool)
    elif values.size and np.count_nonzero(values < 42) > 0.80 * values.size:
        foreign_ink = (gray > 122) & inside.astype(bool)
    else:
        return
    if foreign_ink.any():
        inside[foreign_ink] = 0
        safe &= _edge_distances(inside) > gap


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
                 glow=0, mask=None, ellipse=False, anchor=None, reserved_boxes=()) -> Composition | None:
    x1, y1, x2, y2 = frame
    w, h = x2 - x1, y2 - y1
    if min(w, h) < 8:
        return None
    text, words, ends = _word_spans(text)
    if not words:
        return None
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
    if mask is not None:
        _exclude_art(safe, inside, image, frame, gap)
    for ox1, oy1, ox2, oy2 in reserved_boxes:
        left, right = max(0, ox1 - x1), min(w, ox2 - x1)
        top, bottom = max(0, oy1 - y1), min(h, oy2 - y1)
        if left < right and top < bottom:
            safe[top:bottom, left:right] = False
    row_widths = safe.sum(axis=1)
    if not row_widths.any():
        return None
    maximum_width = int(cols.max() - cols.min() + 1)
    min_y, max_y = int(rows.min()), int(rows.max()) + 1
    maximum_height = max_y - min_y
    paragraph_count = len(set(ends))
    x_scale, y_scale = max(1, w), max(1, h)
    # Measure the stroke and glow inside the same clearance as the ink.
    edge = stroke + glow
    draw = ImageDraw.Draw(image)
    preferred = max(floor, preferred)
    ceiling = min(preferred, int(maximum_height * 1.45))
    sizes = sorted({max(floor, round(ceiling * f)) for f in
                    (1, .94, .87, .79, .70, .61, .52, .44, .36, .29, .22, .16, 0)}, reverse=True)
    best = None

    for size in sizes:
        size_cost = 1.15 * math.log(preferred / size) ** 2
        if best is not None and size_cost >= best.score:
            break
        font = get_font_object(font_path, size)
        _, asc, _, desc = draw.textbbox((0, 0), text.replace("\n", " "), font=font, stroke_width=edge)
        ink_height = max(1, desc - asc + int(bold))
        step = max(round(size * 1.02), ink_height + max(2, round(size * 0.12)))
        max_lines = min(len(words), 1 + (maximum_height - ink_height) // step)
        if paragraph_count > max_lines:
            continue
        window_widths = np.lib.stride_tricks.sliding_window_view(row_widths, ink_height).min(axis=1)

        @lru_cache(maxsize=None)
        def measure(start, end):
            line = text[words[start][0]:words[end - 1][1]]
            left, _, right, _ = draw.textbbox((0, 0), line, font=font, stroke_width=edge)
            return line, left, right + int(bold) - left

        natural_width = sum(draw.textlength(line, font=font) for line in text.splitlines())
        min_lines = max(paragraph_count, math.ceil(natural_width / (maximum_width + size * .5)))
        for n in range(min_lines, min(max_lines, min_lines + 7) + 1):
            if n > min_lines + 3 and best is not None:
                break
            mean_width = natural_width / n
            block_height = (n - 1) * step + ink_height
            reference = round(anchor_y - block_height / 2)
            shift = max(2, round(min(w, h) * .095))
            tops = _track_tops(reference, min_y, max_y - block_height, ink_height, step, n,
                               window_widths, shift, has_blockers=bool(reserved_boxes))
            for top in tops:
                spans = _line_spans(safe, top, n, ink_height, step, anchor_x)
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
                    limit = max(b - a for a, b in intervals)
                    target = mean_width * (1 + .12 * math.cos((i - (n - 1) / 2) * math.pi / max(1, n - 1)))
                    for start, (cost, placed) in states.items():
                        if start == len(words):
                            continue
                        stop = min(ends[start], len(words) - remaining)
                        for end in range(start + 1 if remaining else len(words), stop + 1):
                            line, bearing, width = measure(start, end)
                            if width > limit:
                                break
                            a, b = min(((a, b) for a, b in intervals if b - a >= width),
                                       key=lambda ab: max(ab[0] + width / 2 - anchor_x,
                                                          anchor_x - ab[1] + width / 2, 0))
                            left = round(min(max(anchor_x - width / 2, a), b - width))
                            drift = (left + width / 2 - anchor_x) / x_scale
                            ragged = (width - target) / max(size, mean_width)
                            widow = remaining == 0 and n > 1 and end == start + 1 and width < .58 * mean_width
                            value = cost + (.60 * ragged**2 + 7 * drift**2 + .24 * widow) / n
                            if end not in next_states or value < next_states[end][0]:
                                next_states[end] = (value, placed + ((x1 + left - bearing,
                                                                   y1 + top + i * step - asc, line),))
                    states = next_states
                    if not states:
                        break
                if len(words) in states:
                    cost, lines = states[len(words)]
                    total = base_score + cost
                    if best is None or total < best.score:
                        best = Composition(size, lines, total, gap, preferred)
    return best
