from __future__ import annotations

import copy

import cv2
import numpy as np

MASK_ROWS = 24
MASK_INSET_PX = 4

_CANNY_LOW = 70
_CANNY_HIGH = 140
_CROP_MARGIN_SHARE = 1.0
_CROP_MIN_MARGIN = 12
_FLOOD_MIN_BOX_SHARE = 0.15
_FLOOD_MAX_BOX_SHARE = 16.0
_MIN_NONEMPTY_ROWS = 20
_MIN_FILL_RATIO = 0.25

WAIST_MAX_RATIO = 0.85
WAIST_MIN_HEIGHT_SHARE = 0.06
WAIST_MIN_HEIGHT_PX = 6
LOBE_MIN_HEIGHT_SHARE = 0.25
LOBE_MIN_WIDTH_SHARE = 0.40
SPLIT_MIN_ROWS_SHARE = 0.30
SPLIT_MIN_INTERVAL_RATIO = 0.35
SPLIT_MIN_GAP_RATIO = 0.15
SPLIT_MAX_GAP_RATIO = 0.8
GAP_MIN_PX = 8
GAP_MIN_SIDE_SHARE = 0.10
GAP_MIN_PIECE_SHARE = 0.15
GAP_MIN_CONTAINMENT = 0.70
GAP_MAX_EXTENSION_RATIO = 0.5
GAP_MIN_STEP_RATIO = 0.15
GAP_MIN_RISE_RATIO = 0.05
GAP_MIN_OFFSET_RATIO = 0.15


def _empty_rows() -> list[list[list[int]]]:
    return [[] for _ in range(MASK_ROWS)]


def _none_mask(box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = (int(v) for v in box)
    return {
        "bbox": [x1, y1, x2, y2],
        "cx": (x1 + x2) // 2,
        "cy": (y1 + y2) // 2,
        "rows": _empty_rows(),
        "kind": "none",
    }


def _rect_mask(box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = (int(v) for v in box)
    w = max(0, x2 - x1)
    a = min(MASK_INSET_PX, w // 2)
    b = max(a, w - MASK_INSET_PX)
    return {
        "bbox": [x1, y1, x2, y2],
        "cx": (x1 + x2) // 2,
        "cy": (y1 + y2) // 2,
        "rows": [[[a, b]]] * MASK_ROWS if w > 0 else _empty_rows(),
        "kind": "rect",
    }


def _rows_from_spans(spans: list[tuple[int, int, list[tuple[int, int]]]],
                     bbox: tuple[int, int, int, int]) -> list[list[list[int]]]:
    bx1, by1, bx2, by2 = bbox
    height = max(1, by2 - by1)
    rows: list[list[list[int]]] = []
    for i in range(MASK_ROWS):
        band_y1 = by1 + (i * height) // MASK_ROWS
        band_y2 = by1 + ((i + 1) * height) // MASK_ROWS
        intervals: list[tuple[int, int]] = []
        for sy1, sy2, xs in spans:
            if sy2 <= band_y1 or sy1 >= band_y2:
                continue
            intervals.extend(xs)
        intervals.sort()
        merged: list[list[int]] = []
        for a, b in intervals:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        kept = []
        for a, b in merged:
            a2 = a + MASK_INSET_PX
            b2 = b - MASK_INSET_PX
            if b2 - a2 <= 0:
                a2, b2 = a, b
            if b2 > a2:
                kept.append([a2 - bx1, b2 - bx1])
        rows.append(kept)
    return rows


def _mask_spans(mask: dict) -> list[tuple[int, int, list[tuple[int, int]]]]:
    x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
    height = max(1, y2 - y1)
    spans = []
    for i, intervals in enumerate(mask["rows"]):
        if not intervals:
            continue
        band_y1 = y1 + (i * height) // MASK_ROWS
        band_y2 = y1 + ((i + 1) * height) // MASK_ROWS
        spans.append((band_y1, band_y2,
                      [(x1 + int(a), x1 + int(b)) for a, b in intervals]))
    return spans


def _seed_for(box: tuple[int, int, int, int], seed, letters) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    if seed is not None:
        return int(seed[0]), int(seed[1])
    if letters is not None:
        letters = np.asarray(letters)
        if letters.size and letters.any():
            ys, xs = np.nonzero(letters)
            return int(x1 + xs.mean()), int(y1 + ys.mean())
    return (x1 + x2) // 2, (y1 + y2) // 2


def _flood_seed(mask: np.ndarray, seed: tuple[int, int]) -> tuple[int, int] | None:
    h, w = mask.shape[:2]
    sx, sy = seed
    if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] == 0:
        return sx, sy
    for radius in (1, 2, 3, 5, 8, 12):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                x, y = sx + dx, sy + dy
                if 0 <= x < w and 0 <= y < h and mask[y, x] == 0:
                    return x, y
    return None


def derive_bubble_mask(image, box, *, seed=None, letters=None) -> dict:
    if image is None or getattr(image, "ndim", 0) < 2:
        return _none_mask(tuple(box))
    x1, y1, x2, y2 = (int(v) for v in box)
    height, width = image.shape[:2]
    if x2 <= x1 or y2 <= y1 or x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
        return _none_mask((x1, y1, x2, y2))
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return _none_mask((x1, y1, x2, y2))

    sx, sy = _seed_for((x1, y1, x2, y2), seed, letters)
    for share in (0.5, 1.0):
        mask = _derive_from_crop(image, (x1, y1, x2, y2), (sx, sy), letters,
                                 margin_share=share)
        if mask is not None:
            return mask
    full = _derive_from_crop(image, (x1, y1, x2, y2), (sx, sy), letters,
                             crop=(0, 0, width, height))
    if full is not None:
        return full
    return _rect_mask((cx1, cy1, cx2, cy2))


def _seed_candidates(box: tuple[int, int, int, int],
                     seed: tuple[int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = box
    base = [seed, ((x1 + x2) // 2, (y1 + y2) // 2)]
    for fy in (0.25, 0.5, 0.75):
        for fx in (0.25, 0.5, 0.75):
            base.append((int(x1 + fx * (x2 - x1)), int(y1 + fy * (y2 - y1))))
    seen: list[tuple[int, int]] = []
    for point in base:
        for offset in ((0, 0), (12, 0), (-12, 0), (0, 12), (0, -12)):
            candidate = (point[0] + offset[0], point[1] + offset[1])
            if candidate not in seen:
                seen.append(candidate)
    return seen


def _derive_from_crop(image, box, seed, letters, crop=None, margin_share=None) -> dict | None:
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    if crop is None:
        share = _CROP_MARGIN_SHARE if margin_share is None else margin_share
        margin = max(_CROP_MIN_MARGIN, int(share * max(x2 - x1, y2 - y1)))
        crop = (max(0, x1 - margin), max(0, y1 - margin),
                min(width, x2 + margin), min(height, y2 + margin))
    ox1, oy1, ox2, oy2 = crop
    crop_img = image[oy1:oy2, ox1:ox2]
    if crop_img.size == 0:
        return None
    ch, cw = crop_img.shape[:2]
    box_area = max(1, (x2 - x1) * (y2 - y1))

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    letter_points = None
    if letters is not None:
        letters = np.asarray(letters)
        if letters.size and letters.any() and letters.shape == (y2 - y1, x2 - x1):
            ys, xs = np.nonzero(letters)
            letter_points = (xs + x1 - ox1, ys + y1 - oy1)

    best = None
    for sx, sy in _seed_candidates(box, seed):
        start = _flood_seed(edges, (sx - ox1, sy - oy1))
        if start is None:
            continue
        flooded = edges.copy()
        retval, _, _, rect = cv2.floodFill(
            flooded, None, start, 127, loDiff=0, upDiff=0, flags=4)
        if retval < _FLOOD_MIN_BOX_SHARE * box_area:
            continue
        if retval > _FLOOD_MAX_BOX_SHARE * box_area:
            continue
        interior = flooded == 127
        if float((gray[interior] < 100).mean()) >= 0.5:
            continue
        filled = np.zeros(interior.shape, np.uint8)
        contours, _ = cv2.findContours(
            interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, contours, -1, 255, -1)
        interior = filled > 0
        coverage = 1.0
        if letter_points is not None:
            lx, ly = letter_points
            inside = interior[ly, lx]
            coverage = float(inside.mean())
            if coverage < 0.7:
                continue
        fx, fy, fw, fh = rect
        if (ox1 > 0 and fx <= 0) or (oy1 > 0 and fy <= 0):
            continue
        if (ox2 < width and fx + fw >= cw) or (oy2 < height and fy + fh >= ch):
            continue
        image_borders = (1 if ox1 == 0 and fx <= 0 else 0) \
            + (1 if oy1 == 0 and fy <= 0 else 0) \
            + (1 if ox2 == width and fx + fw >= cw else 0) \
            + (1 if oy2 == height and fy + fh >= ch else 0)
        if image_borders >= 3:
            continue
        key = (coverage, -retval)
        if best is None or key > best[0]:
            best = (key, interior, retval)
            if coverage >= 0.9:
                break
    if best is None:
        return None
    _, interior, _ = best

    ys, xs = np.nonzero(interior)
    if len(xs) == 0:
        return None
    bx1, by1 = int(xs.min()) + ox1, int(ys.min()) + oy1
    bx2, by2 = int(xs.max()) + 1 + ox1, int(ys.max()) + 1 + oy1
    if (bx2 - bx1) * (by2 - by1) <= 0 or (
            interior[by1 - oy1:by2 - oy1, bx1 - ox1:bx2 - ox1].sum()
            < _MIN_FILL_RATIO * (bx2 - bx1) * (by2 - by1)):
        return None
    spans: list[tuple[int, int, list[tuple[int, int]]]] = []
    for y in range(by1 - oy1, by2 - oy1):
        row = np.flatnonzero(interior[y])
        if len(row) == 0:
            continue
        runs: list[tuple[int, int]] = []
        start = int(row[0])
        prev = start
        for v in row[1:]:
            v = int(v)
            if v != prev + 1:
                runs.append((start + ox1, prev + 1 + ox1))
                start = v
            prev = v
        runs.append((start + ox1, prev + 1 + ox1))
        spans.append((y + oy1, y + oy1 + 1, runs))
    rows = _rows_from_spans(spans, (bx1, by1, bx2, by2))
    if sum(1 for row in rows if row) < _MIN_NONEMPTY_ROWS:
        return None
    return {
        "bbox": [bx1, by1, bx2, by2],
        "cx": seed[0],
        "cy": seed[1],
        "rows": rows,
        "kind": "bubble",
    }


def _band_index(y1: int, height: int, y: int) -> int:
    index = min(MASK_ROWS - 1, (int(y) - y1) * MASK_ROWS // height)
    while index + 1 < MASK_ROWS and int(y) >= y1 + ((index + 1) * height) // MASK_ROWS:
        index += 1
    return index


def mask_row_spans(mask: dict | None, y: int) -> list[list[int]]:
    if not isinstance(mask, dict) or mask.get("kind") == "none":
        return []
    try:
        x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
        rows = mask["rows"]
    except (KeyError, TypeError, ValueError):
        return []
    if y2 <= y1 or not (y1 <= int(y) < y2) or len(rows) != MASK_ROWS:
        return []
    height = y2 - y1
    index = _band_index(y1, height, y)
    return [[x1 + int(a), x1 + int(b)] for a, b in rows[index]]


def _widest_interval(row: list[list[int]]) -> tuple[int, int] | None:
    if not row:
        return None
    return max(((int(a), int(b)) for a, b in row), key=lambda ab: ab[1] - ab[0])


def mask_inscribed_rect(mask: dict | None) -> tuple[int, int, int, int]:
    if not isinstance(mask, dict):
        return (0, 0, 0, 0)
    try:
        x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
        rows = mask["rows"]
    except (KeyError, TypeError, ValueError):
        return (0, 0, 0, 0)
    if y2 <= y1 or len(rows) != MASK_ROWS:
        return (x1, y1, x2, y2)
    widest = [_widest_interval(row) for row in rows]
    if all(w is None for w in widest):
        return (x1, y1, x2, y2)
    row_h = (y2 - y1) / MASK_ROWS
    best = None
    for top in range(MASK_ROWS):
        if widest[top] is None:
            continue
        left, right = widest[top]
        for bottom in range(top, MASK_ROWS):
            if widest[bottom] is None:
                break
            left = max(left, widest[bottom][0])
            right = min(right, widest[bottom][1])
            if right - left <= 0:
                break
            area = (right - left) * (bottom - top + 1)
            key = (area, bottom - top, right - left)
            if best is None or key > best[0]:
                best = (key, (left, top, right, bottom))
    if best is None:
        return (x1, y1, x2, y2)
    _, (left, top, right, bottom) = best
    height = y2 - y1
    return (x1 + left, y1 + (top * height) // MASK_ROWS,
            x1 + right, y1 + ((bottom + 1) * height) // MASK_ROWS)


def find_waist(mask: dict | None, letters=None, box=None) -> dict | None:
    if not isinstance(mask, dict) or mask.get("kind") != "bubble":
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
        rows = mask["rows"]
    except (KeyError, TypeError, ValueError):
        return None
    if y2 <= y1 or x2 <= x1 or len(rows) != MASK_ROWS:
        return None
    height = y2 - y1
    width = x2 - x1
    row_h = height / MASK_ROWS

    waist = _interval_waist(rows, height, width, row_h, x1, y1)
    if waist is not None:
        return waist

    if letters is not None and box is not None:
        waist = _letter_gap_waist(mask, letters, box)
        if waist is not None:
            return waist

    if letters is not None and box is not None:
        waist = _row_gap_waist(mask, letters, box)
        if waist is not None:
            return waist

    widths = []
    for row in rows:
        if not row:
            widths.append(0)
        else:
            widths.append(max(int(b) for _, b in row) - min(int(a) for a, _ in row))
    return _profile_waist(widths, height, width, row_h, y1)


def _profile_waist(widths: list[int], height: int, width: int,
                   row_h: float, y1: int) -> dict | None:
    min_waist_rows = max(1, int(WAIST_MIN_HEIGHT_SHARE * MASK_ROWS))
    min_lobe_rows = max(1, int(LOBE_MIN_HEIGHT_SHARE * MASK_ROWS))
    best = None
    for start in range(min_lobe_rows, MASK_ROWS - min_lobe_rows - min_waist_rows + 1):
        for end in range(start + min_waist_rows,
                         MASK_ROWS - min_lobe_rows + 1):
            above = widths[:start]
            below = widths[end:]
            waist_rows = widths[start:end]
            if len(above) < min_lobe_rows or len(below) < min_lobe_rows:
                continue
            lobe_w = min(max(above), max(below))
            waist_w = max(waist_rows)
            if lobe_w < LOBE_MIN_WIDTH_SHARE * width:
                continue
            if waist_w > WAIST_MAX_RATIO * lobe_w:
                continue
            if (end - start) * row_h < WAIST_MIN_HEIGHT_PX:
                continue
            if max(above[-1], below[0]) < 0.8 * lobe_w:
                continue
            score = (lobe_w - waist_w, end - start)
            if best is None or score > best[0]:
                best = (score, start, end, waist_w, lobe_w)
    if best is None:
        return None
    _, start, end, waist_w, lobe_w = best
    return {
        "orientation": "horizontal",
        "cut_y": int(y1 + (start + end) / 2 * row_h),
        "waist_rows": (start, end),
        "waist_width": int(waist_w),
        "lobe_widths": (int(max(widths[:start])), int(max(widths[end:]))),
    }


def _interval_waist(rows: list, height: int, width: int, row_h: float,
                    x1: int, y1: int) -> dict | None:
    split_rows = []
    for i, row in enumerate(rows):
        if len(row) < 2:
            continue
        ordered = sorted((int(a), int(b)) for a, b in row)
        widest = max(b - a for a, b in ordered)
        second = sorted((b - a for a, b in ordered), reverse=True)[1]
        if second < SPLIT_MIN_INTERVAL_RATIO * widest:
            continue
        split_rows.append(i)
    if len(split_rows) < SPLIT_MIN_ROWS_SHARE * MASK_ROWS:
        return None
    if min(split_rows) >= MASK_ROWS // 2 or max(split_rows) < MASK_ROWS // 2:
        return None
    gaps = []
    for i in split_rows:
        ordered = sorted((int(a), int(b)) for a, b in rows[i])
        by_width = sorted(ordered, key=lambda ab: ab[1] - ab[0], reverse=True)[:2]
        by_width.sort()
        gaps.append((by_width[0][1], by_width[1][0],
                     (by_width[0][1] + by_width[1][0]) // 2))
    mids = sorted(g[2] for g in gaps)
    median_mid = mids[len(mids) // 2]
    if max(mids) - min(mids) > 0.25 * width:
        return None
    gap_width = float(np.median([g[1] - g[0] for g in gaps]))
    left_widths = []
    right_widths = []
    for i in split_rows:
        ordered = sorted((int(a), int(b)) for a, b in rows[i])
        by_width = sorted(ordered, key=lambda ab: ab[1] - ab[0], reverse=True)[:2]
        by_width.sort()
        left_widths.append(by_width[0][1] - by_width[0][0])
        right_widths.append(by_width[1][1] - by_width[1][0])
    lobe_min = min(float(np.median(left_widths)), float(np.median(right_widths)))
    if gap_width < SPLIT_MIN_GAP_RATIO * lobe_min:
        return None
    if gap_width > SPLIT_MAX_GAP_RATIO * lobe_min:
        return None
    return {
        "orientation": "vertical",
        "cut_x": int(x1 + median_mid),
        "gap": (int(float(np.median([g[0] for g in gaps]))),
                int(float(np.median([g[1] for g in gaps])))),
        "lobe_widths": (int(float(np.median(left_widths))),
                        int(float(np.median(right_widths)))),
        "split_rows": (min(split_rows), max(split_rows)),
    }


def _letters_inside_mask(mask: dict, xs, ys) -> float:
    if len(xs) == 0:
        return 0.0
    x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
    rows = mask["rows"]
    if y2 <= y1 or len(rows) != MASK_ROWS:
        return 0.0
    height = y2 - y1
    inside = 0
    for x, y in zip(xs, ys):
        if not (x1 <= x < x2 and y1 <= y < y2):
            continue
        index = _band_index(y1, height, int(y))
        for a, b in rows[index]:
            if x1 + int(a) <= x < x1 + int(b):
                inside += 1
                break
    return inside / len(xs)


def _mask_row_width_sum(mask: dict) -> int:
    return sum(int(b) - int(a) for row in mask["rows"] for a, b in row)


def _rows_split_in_both_caps(rows: list[list[list[int]]]) -> bool:
    def splits(band):
        if len(band) < 2:
            return False
        ordered = sorted((int(b) - int(a) for a, b in band), reverse=True)
        return ordered[1] >= SPLIT_MIN_INTERVAL_RATIO * ordered[0]

    return any(splits(row) for row in rows[: MASK_ROWS // 2]) \
        and any(splits(row) for row in rows[MASK_ROWS // 2:])


def _profile_corroborates(widths: list[int], rows: list[list[list[int]]] | None = None) -> bool:
    middle = widths[3:20]
    if len(middle) >= 3:
        for i in range(len(middle) - 2):
            if middle[i] > 0 and middle[i + 1] >= (1 + GAP_MIN_STEP_RATIO) * middle[i] \
                    and middle[i + 2] >= middle[i + 1]:
                return True
        for i in range(1, len(middle) - 1):
            if middle[i] < middle[i - 1] and middle[i] < middle[i + 1]:
                peak = max(middle[i + 1:])
                if peak >= (1 + GAP_MIN_RISE_RATIO) * middle[i]:
                    return True
    if rows is not None:
        return _rows_split_in_both_caps(rows)
    return False


def _letter_gap_waist(mask: dict, letters, box) -> dict | None:
    letters = np.asarray(letters)
    if letters.size == 0 or not letters.any():
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    if letters.shape != (y2 - y1, x2 - x1):
        return None
    ys, xs = np.nonzero(letters)
    xs = xs + x1
    ys = ys + y1
    total = len(xs)
    if total == 0:
        return None
    present = np.zeros(xs.max() + 1, dtype=bool)
    present[xs] = True
    gaps: list[tuple[int, int]] = []
    in_gap = False
    for x in range(int(xs.min()), int(xs.max()) + 1):
        if not present[x] and not in_gap:
            start = x
            in_gap = True
        elif present[x] and in_gap:
            gaps.append((start, x - 1))
            in_gap = False
    mx1, my1, mx2, my2 = (int(v) for v in mask["bbox"])
    union_area = _mask_row_width_sum(mask)
    if union_area <= 0:
        return None
    widths = []
    for row in mask["rows"]:
        if not row:
            widths.append(0)
        else:
            widths.append(max(int(b) for _, b in row) - min(int(a) for a, _ in row))
    if not _profile_corroborates(widths, mask["rows"]):
        return None
    best = None
    for gap_start, gap_end in gaps:
        if gap_end - gap_start + 1 < GAP_MIN_PX:
            continue
        cut_x = (gap_start + gap_end) // 2
        if not (mx1 < cut_x < mx2):
            continue
        left = xs < cut_x
        right = ~left
        left_share = float(left.sum()) / total
        right_share = float(right.sum()) / total
        if min(left_share, right_share) < GAP_MIN_SIDE_SHARE:
            continue
        pieces = split_bubble_mask(mask, {"orientation": "vertical", "cut_x": cut_x})
        if pieces is None:
            continue
        left_mask, right_mask = pieces
        if min(_mask_row_width_sum(left_mask), _mask_row_width_sum(right_mask)) < GAP_MIN_PIECE_SHARE * union_area:
            continue
        if _letters_inside_mask(left_mask, xs[left], ys[left]) < GAP_MIN_CONTAINMENT:
            continue
        if _letters_inside_mask(right_mask, xs[right], ys[right]) < GAP_MIN_CONTAINMENT:
            continue
        for piece, side_xs, side_ys in (
                (left_mask, xs[left], ys[left]),
                (right_mask, xs[right], ys[right])):
            letter_x1, letter_x2 = int(side_xs.min()), int(side_xs.max())
            letter_w = max(1, letter_x2 - letter_x1)
            ly1, ly2 = int(side_ys.min()), int(side_ys.max())
            span_x1, span_x2 = None, None
            for a, b, intervals in _mask_spans(piece):
                if b <= ly1 or a >= ly2:
                    continue
                for p, q in intervals:
                    span_x1 = p if span_x1 is None else min(span_x1, p)
                    span_x2 = q if span_x2 is None else max(span_x2, q)
            if span_x1 is None:
                break
            if (span_x1 < letter_x1 - GAP_MAX_EXTENSION_RATIO * letter_w
                    or span_x2 > letter_x2 + GAP_MAX_EXTENSION_RATIO * letter_w):
                break
        else:
            key = min(left_share, right_share)
            if best is None or key > best[0]:
                best = (key, cut_x, (gap_start, gap_end),
                        (left_share, right_share))
    if best is None:
        return None
    _, cut_x, gap, shares = best
    return {
        "orientation": "vertical",
        "cut_x": int(cut_x),
        "gap": (int(gap[0]), int(gap[1])),
        "lobe_widths": (int(cut_x - mx1), int(mx2 - cut_x)),
        "letter_shares": (round(float(shares[0]), 3), round(float(shares[1]), 3)),
    }


def _row_gap_waist(mask: dict, letters, box) -> dict | None:
    letters = np.asarray(letters)
    if letters.size == 0 or not letters.any():
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    if letters.shape != (y2 - y1, x2 - x1):
        return None
    ys, xs = np.nonzero(letters)
    xs = xs + x1
    ys = ys + y1
    total = len(xs)
    if total == 0:
        return None
    present = np.zeros(ys.max() + 1, dtype=bool)
    present[ys] = True
    gaps: list[tuple[int, int]] = []
    in_gap = False
    for y in range(int(ys.min()), int(ys.max()) + 1):
        if not present[y] and not in_gap:
            start = y
            in_gap = True
        elif present[y] and in_gap:
            gaps.append((start, y - 1))
            in_gap = False
    mx1, my1, mx2, my2 = (int(v) for v in mask["bbox"])
    union_area = _mask_row_width_sum(mask)
    if union_area <= 0:
        return None
    widths = []
    for row in mask["rows"]:
        if not row:
            widths.append(0)
        else:
            widths.append(max(int(b) for _, b in row) - min(int(a) for a, _ in row))
    if not _profile_corroborates(widths, mask["rows"]):
        return None
    mask_width = max(1, mx2 - mx1)
    best = None
    gap_lens = [g[1] - g[0] + 1 for g in gaps]
    for gap_start, gap_end in gaps:
        if gap_end - gap_start + 1 < GAP_MIN_PX:
            continue
        if sum(1 for length in gap_lens if length >= gap_end - gap_start + 1) > 1:
            continue
        cut_y = (gap_start + gap_end) // 2
        if not (my1 < cut_y < my2):
            continue
        above = ys < cut_y
        below = ~above
        above_share = float(above.sum()) / total
        below_share = float(below.sum()) / total
        if min(above_share, below_share) < GAP_MIN_SIDE_SHARE:
            continue
        pieces = split_bubble_mask(mask, {"orientation": "horizontal", "cut_y": cut_y})
        if pieces is None:
            continue
        upper_mask, lower_mask = pieces
        if min(_mask_row_width_sum(upper_mask), _mask_row_width_sum(lower_mask)) < GAP_MIN_PIECE_SHARE * union_area:
            continue
        if _letters_inside_mask(upper_mask, xs[above], ys[above]) < GAP_MIN_CONTAINMENT:
            continue
        if _letters_inside_mask(lower_mask, xs[below], ys[below]) < GAP_MIN_CONTAINMENT:
            continue
        above_cx = (int(xs[above].min()) + int(xs[above].max())) / 2
        below_cx = (int(xs[below].min()) + int(xs[below].max())) / 2
        if abs(above_cx - below_cx) < GAP_MIN_OFFSET_RATIO * mask_width:
            continue
        overlap = min(int(xs[above].max()), int(xs[below].max())) - max(int(xs[above].min()), int(xs[below].min()))
        if overlap < GAP_MIN_PX:
            continue
        key = min(above_share, below_share)
        if best is None or key > best[0]:
            best = (key, cut_y, (gap_start, gap_end),
                    (above_share, below_share))
    if best is None:
        return None
    _, cut_y, gap, shares = best
    return {
        "orientation": "horizontal",
        "cut_y": int(cut_y),
        "gap": (int(gap[0]), int(gap[1])),
        "lobe_heights": (int(cut_y - my1), int(my2 - cut_y)),
        "letter_shares": (round(float(shares[0]), 3), round(float(shares[1]), 3)),
    }


def split_bubble_mask(mask: dict, waist: dict) -> tuple[dict, dict] | None:
    if not isinstance(mask, dict) or mask.get("kind") != "bubble":
        return None
    spans = _mask_spans(mask)
    x1, y1, x2, y2 = (int(v) for v in mask["bbox"])
    if waist["orientation"] == "horizontal":
        cut = int(waist["cut_y"])
        above = [(a, b, xs) for a, b, xs in spans if b <= cut]
        below = [(a, b, xs) for a, b, xs in spans if a >= cut]
        if not above or not below:
            return None
        return (_mask_from_spans(above, mask), _mask_from_spans(below, mask))
    cut = int(waist["cut_x"])
    left = [(a, b, [(p, min(q, cut)) for p, q in xs if p < cut])
            for a, b, xs in spans]
    right = [(a, b, [(max(p, cut), q) for p, q in xs if q > cut])
             for a, b, xs in spans]
    left = [(a, b, xs) for a, b, xs in left if any(q > p for p, q in xs)]
    right = [(a, b, xs) for a, b, xs in right if any(q > p for p, q in xs)]
    if not left or not right:
        return None
    return (_mask_from_spans(left, mask), _mask_from_spans(right, mask))


def _mask_from_spans(spans: list[tuple[int, int, list[tuple[int, int]]]],
                     source: dict) -> dict:
    xs = [p for _, _, intervals in spans for p, q in intervals for p in (p, q)]
    ys = [v for a, b, _ in spans for v in (a, b)]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    rows = _rows_from_spans(spans, bbox)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    try:
        scx, scy = int(source["cx"]), int(source["cy"])
        if bbox[0] <= scx < bbox[2] and bbox[1] <= scy < bbox[3]:
            cx, cy = scx, scy
    except (KeyError, TypeError, ValueError):
        pass
    return {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "cx": cx, "cy": cy, "rows": rows, "kind": "bubble"}


def shift_bubble_mask(mask: dict | None, dx: int = 0, dy: int = 0) -> dict | None:
    if not isinstance(mask, dict):
        return None
    shifted = copy.deepcopy(mask)
    try:
        x1, y1, x2, y2 = (int(v) for v in shifted["bbox"])
        shifted["bbox"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        shifted["cx"] = int(shifted["cx"]) + dx
        shifted["cy"] = int(shifted["cy"]) + dy
    except (KeyError, TypeError, ValueError):
        return None
    return shifted


def clip_bubble_mask(mask: dict | None, box: tuple[int, int, int, int]) -> dict | None:
    if not isinstance(mask, dict) or mask.get("kind") != "bubble":
        return None
    try:
        bx1, by1, bx2, by2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None
    if bx2 <= bx1 or by2 <= by1:
        return None
    spans: list[tuple[int, int, list[tuple[int, int]]]] = []
    for a, b, intervals in _mask_spans(mask):
        top, bottom = max(a, by1), min(b, by2)
        if bottom <= top:
            continue
        clipped = [(max(p, bx1), min(q, bx2)) for p, q in intervals
                   if min(q, bx2) > max(p, bx1)]
        if clipped:
            spans.append((top, bottom, clipped))
    if not spans:
        return None
    clipped_mask = _mask_from_spans(spans, mask)
    width = clipped_mask["bbox"][2] - clipped_mask["bbox"][0]
    height = clipped_mask["bbox"][3] - clipped_mask["bbox"][1]
    if width < 16 or height < 16:
        return None
    return clipped_mask


def merge_bubble_masks(masks: list[dict | None]) -> dict | None:
    usable = [m for m in masks
              if isinstance(m, dict) and m.get("kind") in ("bubble", "rect")
              and isinstance(m.get("rows"), list) and len(m["rows"]) == MASK_ROWS]
    if not usable:
        return None
    if len(usable) == 1:
        return copy.deepcopy(usable[0])
    spans: list[tuple[int, int, list[tuple[int, int]]]] = []
    for mask in usable:
        spans.extend(_mask_spans(mask))
    xs = [p for _, _, intervals in spans for p, q in intervals for p in (p, q)]
    ys = [v for a, b, _ in spans for v in (a, b)]
    if not xs or not ys:
        return None
    bbox = (min(xs), min(ys), max(xs), max(ys))
    rows = _rows_from_spans(spans, bbox)
    kind = "bubble" if any(m.get("kind") == "bubble" for m in usable) else "rect"
    seeds = [(int(m["cx"]), int(m["cy"])) for m in usable
             if bbox[0] <= int(m.get("cx", 0)) < bbox[2]
             and bbox[1] <= int(m.get("cy", 0)) < bbox[3]]
    cx, cy = seeds[0] if seeds else ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "cx": cx, "cy": cy, "rows": rows, "kind": kind}
