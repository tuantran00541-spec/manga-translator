from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.detector.boxes import BubbleBox
from app.detector.bubble_mask import (
    GAP_MIN_CONTAINMENT,
    GAP_MIN_SIDE_SHARE,
    _letters_inside_mask,
    derive_bubble_mask,
    find_waist,
    split_bubble_mask,
)
from app.detector.ctd_mask import letter_mask, still_reads
from app.knobs import knob
from app.ort_utils import make_session

LETTERBOX_VALUE = 114
STRIDE = 32
NMS_IOU = knob("detect.nms_iou")
BOX_PAD = knob("detect.box_pad")
COLUMN_GAP = 16
HALVES_MIN_OVERLAP = 256
PAGE_WIDTH = 400
PAGE_OVERLAP = 0.15
PAGE_CONFIDENCE = 0.45
MASKLESS_MIN_SCORE = 0.5
MASKLESS_MIN_PIECES = 3
MASKLESS_PIECE_PX = 20
MASKLESS_PIECE_SHARE = 0.25
MASKLESS_DARK_DROP = 25
MASKLESS_SMALL_BOX_AREA = 15000
ROTATED_MIN_SCORE = PAGE_CONFIDENCE
RESCUE_MIN_SCORE = 0.10
RESCUE_WORD_MAX_W = 80
RESCUE_WORD_MAX_H = 35
RESCUE_WORD_MIN_AREA = 100
RESCUE_WORD_MIN_FILL = 0.35
RESCUE_WORD_PAD = 8
UNION_SHARE = knob("detect.union_share")
SAME_LINE = knob("detect.same_line")
SAME_LINE_X = knob("detect.same_line_x")


def _waist_letter_shares(box: BubbleBox, waist: dict, letters=None) -> tuple[float, float]:
    if letters is None:
        letters = getattr(box, "letters", None)
    if letters is None:
        return (0.0, 0.0)
    letters = np.asarray(letters)
    if letters.size == 0 or not letters.any():
        return (0.0, 0.0)
    ys, xs = np.nonzero(letters)
    if waist["orientation"] == "horizontal":
        first = ys + box.y1 < int(waist["cut_y"])
    else:
        first = xs + box.x1 < int(waist["cut_x"])
    total = len(xs)
    return (float(first.sum()) / total, float((~first).sum()) / total)


def _core_letters(box: BubbleBox):
    letters = getattr(box, "letters", None)
    if letters is None:
        return None
    letters = np.asarray(letters)
    core = getattr(box, "core", None)
    if core is None:
        return letters
    r0 = max(0, core[1] - box.y1)
    r1 = min(letters.shape[0], core[3] - box.y1)
    c0 = max(0, core[0] - box.x1)
    c1 = min(letters.shape[1], core[2] - box.x1)
    if r1 <= r0 or c1 <= c0:
        return letters
    clipped = np.zeros_like(letters)
    clipped[r0:r1, c0:c1] = letters[r0:r1, c0:c1]
    return clipped


def _clip_letters_to_cut(letters, box: BubbleBox, waist: dict, keep_first: bool):
    clipped = np.array(letters, copy=True)
    if waist["orientation"] == "horizontal":
        cut_local = min(max(int(waist["cut_y"]) - box.y1, 0), clipped.shape[0])
        if keep_first:
            clipped[cut_local:] = 0
        else:
            clipped[:cut_local] = 0
    else:
        cut_local = min(max(int(waist["cut_x"]) - box.x1, 0), clipped.shape[1])
        if keep_first:
            clipped[:, cut_local:] = 0
        else:
            clipped[:, :cut_local] = 0
    return clipped


def _trim_bubble_mask(box: BubbleBox) -> None:
    mask = getattr(box, "bubble_mask", None)
    letters = getattr(box, "letters", None)
    if (not isinstance(mask, dict) or mask.get("kind") != "bubble"
            or letters is None or not np.asarray(letters).any()):
        return
    waist = find_waist(mask, letters=letters, box=(box.x1, box.y1, box.x2, box.y2))
    if waist is None:
        return
    own = _core_letters(box)
    first_share, second_share = _waist_letter_shares(box, waist, own)
    if min(first_share, second_share) >= GAP_MIN_SIDE_SHARE:
        return
    pieces = split_bubble_mask(mask, waist)
    if pieces is None:
        return
    keep_first = first_share >= second_share
    keep = pieces[0] if keep_first else pieces[1]
    if own is None:
        return
    ys, xs = np.nonzero(np.asarray(own))
    if _letters_inside_mask(keep, xs + box.x1, ys + box.y1) < GAP_MIN_CONTAINMENT:
        return
    box.bubble_mask = keep
    box.letters = _clip_letters_to_cut(letters, box, waist, keep_first)


def _split_shared_union_masks(boxes: list[BubbleBox]) -> None:
    groups: dict[tuple, list[BubbleBox]] = {}
    for box in boxes:
        mask = getattr(box, "bubble_mask", None)
        if not isinstance(mask, dict) or mask.get("kind") != "bubble":
            continue
        groups.setdefault(tuple(int(v) for v in mask["bbox"]), []).append(box)
    for members in groups.values():
        if len(members) < 2:
            continue
        mask = members[0].bubble_mask
        mx1, my1, mx2, my2 = (int(v) for v in mask["bbox"])
        pooled = np.zeros((my2 - my1, mx2 - mx1), dtype=np.uint8)
        for box in members:
            own = _core_letters(box)
            if own is None:
                continue
            own = np.asarray(own)
            r0, r1 = max(0, box.y1 - my1), min(my2 - my1, box.y2 - my1)
            c0, c1 = max(0, box.x1 - mx1), min(mx2 - mx1, box.x2 - mx1)
            ar0, ac0 = r0 - (box.y1 - my1), c0 - (box.x1 - mx1)
            pooled[r0:r1, c0:c1] |= own[ar0:ar0 + (r1 - r0), ac0:ac0 + (c1 - c0)].astype(np.uint8)
        if not pooled.any():
            continue
        waist = find_waist(mask, letters=pooled, box=(mx1, my1, mx2, my2))
        if waist is None:
            continue
        pieces = split_bubble_mask(mask, waist)
        if pieces is None:
            continue
        for box in members:
            own = _core_letters(box)
            if own is None:
                continue
            letters = getattr(box, "letters", None)
            if letters is None or not np.asarray(letters).any():
                continue
            first_share, second_share = _waist_letter_shares(box, waist, own)
            if min(first_share, second_share) >= GAP_MIN_SIDE_SHARE:
                continue
            keep_first = first_share >= second_share
            keep = pieces[0] if keep_first else pieces[1]
            ys, xs = np.nonzero(np.asarray(own))
            if _letters_inside_mask(keep, xs + box.x1, ys + box.y1) < GAP_MIN_CONTAINMENT:
                continue
            box.bubble_mask = keep
            box.letters = _clip_letters_to_cut(letters, box, waist, keep_first)


def _attach_bubble_masks(image: np.ndarray, boxes: list[BubbleBox]) -> None:
    for box in boxes:
        box.bubble_mask = derive_bubble_mask(
            image, (box.x1, box.y1, box.x2, box.y2),
            letters=getattr(box, "letters", None))
        _trim_bubble_mask(box)
    _split_shared_union_masks(boxes)


def _split_waist_box(box: BubbleBox) -> list[BubbleBox]:
    from dataclasses import replace

    mask = getattr(box, "bubble_mask", None)
    letters = getattr(box, "letters", None)
    if (not isinstance(mask, dict) or mask.get("kind") != "bubble"
            or letters is None or not np.asarray(letters).any()):
        return [box]
    letters = np.asarray(letters)
    waist = find_waist(mask, letters=letters, box=(box.x1, box.y1, box.x2, box.y2))
    if waist is None:
        return [box]
    own = _core_letters(box)
    if own is None:
        return [box]
    own = np.asarray(own)
    first_share, second_share = _waist_letter_shares(box, waist, own)
    if min(first_share, second_share) < GAP_MIN_SIDE_SHARE:
        return [box]
    letters = own
    pieces = split_bubble_mask(mask, waist)
    if pieces is None:
        return [box]
    runs = _line_runs(letters.any(axis=1))
    run_heights = [e - s for s, e in runs]
    if len(run_heights) >= 2:
        height = float(np.median(run_heights))
    elif run_heights:
        height = float(run_heights[0]) / 3.0
    else:
        height = 20.0
    height = max(8.0, min(height, 60.0))
    reach = int(height)
    height_px, width_px = letters.shape
    horizontal = waist["orientation"] == "horizontal"
    cut_local = (int(waist["cut_y"]) if horizontal else int(waist["cut_x"])) \
        - (box.y1 if horizontal else box.x1)
    if cut_local <= 0 or cut_local >= (height_px if horizontal else width_px):
        return [box]
    parts = []
    for first, piece_mask in ((True, pieces[0]), (False, pieces[1])):
        if horizontal:
            part_letters = letters[:cut_local] if first else letters[cut_local:]
            row_offset = 0 if first else cut_local
            col_offset = 0
        else:
            part_letters = letters[:, :cut_local] if first else letters[:, cut_local:]
            row_offset = 0
            col_offset = 0 if first else cut_local
        if not part_letters.any():
            return [box]
        part_rows = np.flatnonzero(part_letters.any(axis=1)) + row_offset
        part_cols = np.flatnonzero(part_letters.any(axis=0)) + col_offset
        top, bottom = int(part_rows[0]), int(part_rows[-1]) + 1
        if bottom - top < reach:
            return [box]
        if horizontal:
            y1 = max(0 if first else cut_local, int(top) - reach)
            y2 = min(cut_local if first else height_px, int(bottom) + reach)
            x1 = max(0, int(part_cols[0]) - reach)
            x2 = min(width_px, int(part_cols[-1]) + 1 + reach)
        else:
            x1 = max(0 if first else cut_local, int(part_cols[0]) - reach)
            x2 = min(cut_local if first else width_px, int(part_cols[-1]) + 1 + reach)
            y1 = max(0, int(top) - reach)
            y2 = min(height_px, int(bottom) + reach)
        if x2 <= x1 or y2 <= y1:
            return [box]
        rect = (box.x1 + x1, box.y1 + y1, box.x1 + x2, box.y1 + y2)
        sliced_mask = None
        if box.mask is not None and np.asarray(box.mask).shape == letters.shape:
            sliced_mask = np.ascontiguousarray(np.asarray(box.mask)[y1:y2, x1:x2])
        parts.append(replace(
            box, x1=rect[0], y1=rect[1], x2=rect[2], y2=rect[3],
            mask=sliced_mask,
            letters=np.ascontiguousarray(letters[y1:y2, x1:x2]),
            core=_clip_core(box.core, rect),
            bubble_mask=piece_mask,
        ))
    if len(parts) != 2:
        return [box]
    return parts


def split_waist_boxes(boxes: list[BubbleBox]) -> list[BubbleBox]:
    out: list[BubbleBox] = []
    for box in boxes:
        out.extend(_split_waist_box(box))
    return out


def _text_box(x1, y1, x2, y2, score, mask, source_model, letters=None, core=None) -> BubbleBox:
    box = BubbleBox(
        x1, y1, x2, y2, score, mask,
        source_model=source_model, class_name="text", semantic_type="free_text",
        mask_source="text_segmenter", safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
        letters=letters,
        core=core,
    )
    return box


def _overlaps(a: list, b: list) -> bool:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    same_line = (iy >= SAME_LINE * min(a[3] - a[1], b[3] - b[1])
                 and ix >= SAME_LINE_X * min(a[2] - a[0], b[2] - b[0]))
    return ix > 0 and iy > 0 and (same_line or ix * iy >= UNION_SHARE * max(1, smaller))


SAME_DETECTION_IOU = 0.85


def _same_detection(a, b) -> bool:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter >= SAME_DETECTION_IOU * max(1, union)


def _unlettered_duplicate(a, b) -> bool:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy >= 0.9 * max(1, smaller)


JOIN_STACK_GAP = 0.2
JOIN_STACK_CENTRE = 0.15


def _letter_geometry(box: BubbleBox, core: tuple[int, int, int, int] | None = None):
    letters = getattr(box, "letters", None)
    if letters is None or not letters.any():
        return None
    y0, x0 = box.y1, box.x1
    if core is not None:
        r0, r1 = max(0, core[1] - box.y1), min(letters.shape[0], core[3] - box.y1)
        c0, c1 = max(0, core[0] - box.x1), min(letters.shape[1], core[2] - box.x1)
        if r1 <= r0 or c1 <= c0:
            return None
        letters = letters[r0:r1, c0:c1]
        if not letters.any():
            return None
        y0, x0 = box.y1 + r0, box.x1 + c0
    rows = np.flatnonzero(letters.any(axis=1)) + y0
    cols = np.flatnonzero(letters.any(axis=0)) + x0
    return (set(int(r) for r in rows),
            (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def _merge_geometry(a, b):
    if a is None:
        return b
    if b is None:
        return a
    rows_a, box_a = a
    rows_b, box_b = b
    return (rows_a | rows_b,
            (min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
             max(box_a[2], box_b[2]), max(box_a[3], box_b[3])))


def _union_core(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _letters_coherent(a, b) -> bool:
    if a is None or b is None:
        return False
    rows_a, box_a = a
    rows_b, box_b = b
    if rows_a & rows_b:
        return True
    (ux1, uy1, ux2, uy2), (lx1, ly1, lx2, ly2) = sorted((box_a, box_b), key=lambda box: box[1])
    gap = ly1 - uy2
    shorter = min(uy2 - uy1, ly2 - ly1)
    wider = max(ux2 - ux1, lx2 - lx1)
    offset = abs((ux1 + ux2) - (lx1 + lx2)) / 2
    return gap <= JOIN_STACK_GAP * shorter and offset <= JOIN_STACK_CENTRE * wider


def _join_grown(boxes: list[BubbleBox], cores: list[tuple[int, int, int, int]] | None = None) -> list[BubbleBox]:
    groups = [[[b.x1, b.y1, b.x2, b.y2, b.confidence],
               _letter_geometry(b, cores[i] if cores is not None else None),
               cores[i] if cores is not None else getattr(b, "core", None)]
              for i, b in enumerate(boxes)]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _overlaps(groups[i][0], groups[j][0]) and (
                        _letters_coherent(groups[i][1], groups[j][1])
                        or _same_detection(groups[i][0], groups[j][0])
                        or (groups[i][1] is None and groups[j][1] is None
                            and _unlettered_duplicate(groups[i][0], groups[j][0]))):
                    a, b = groups[i], groups[j]
                    a[0] = [min(a[0][0], b[0][0]), min(a[0][1], b[0][1]),
                            max(a[0][2], b[0][2]), max(a[0][3], b[0][3]), max(a[0][4], b[0][4])]
                    a[1] = _merge_geometry(a[1], b[1])
                    a[2] = _union_core(a[2], b[2])
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    rects = [g[0] for g in groups]
    if len(rects) == len(boxes):
        return boxes
    joined = []
    for (x1, y1, x2, y2, score), _geometry, core in groups:
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        letters = np.zeros((y2 - y1, x2 - x1), bool)
        for b in boxes:
            if b.x1 >= x1 and b.y1 >= y1 and b.x2 <= x2 and b.y2 <= y2:
                view = mask[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1]
                np.maximum(view, b.mask, out=view)
                if getattr(b, "letters", None) is not None:
                    letters[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1] |= b.letters
        joined.append(_text_box(x1, y1, x2, y2, score, mask, boxes[0].source_model,
                                letters if letters.any() else None, core=core))
    return joined


BLOCK_GAP = knob("detect.block_gap")
BLOCK_OFFSET = knob("detect.block_offset")
BLOCK_SPACING = knob("detect.block_spacing")
BLOCK_MIN_WIDTH = knob("detect.block_min_width")
BLOCK_LINE_SPREAD = knob("detect.block_line_spread")


def _separate(mask: np.ndarray, upper: list, lower: list) -> bool:
    def extent(group):
        cols = np.flatnonzero(mask[group[0][0]:group[-1][1]].any(axis=0))
        return float(cols[0]), float(cols[-1] + 1)

    if len(upper) < 2 or len(lower) < 2:
        return False
    (a1, a2), (b1, b2) = extent(upper), extent(lower)
    if min(a2 - a1, b2 - b1) < BLOCK_MIN_WIDTH * max(a2 - a1, b2 - b1):
        return False
    if abs((a1 + a2) - (b1 + b2)) / 2 > BLOCK_OFFSET * max(a2 - a1, b2 - b1):
        return True
    inner = [nxt[0] - line[1] for group in (upper, lower) for line, nxt in zip(group, group[1:])]
    return lower[0][0] - upper[-1][1] >= BLOCK_SPACING * max(inner)


def _clip_core(core: tuple[int, int, int, int] | None, rect: tuple[int, int, int, int]):
    if core is None:
        return None
    x1, y1 = max(core[0], rect[0]), max(core[1], rect[1])
    x2, y2 = min(core[2], rect[2]), min(core[3], rect[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _split_blocks(box: BubbleBox) -> list[BubbleBox]:
    letters = getattr(box, "letters", None)
    if letters is None or box.mask is None or not letters.any():
        return [box]
    rows = letters.any(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
    lines = [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]
    if len(lines) < 2:
        return [box]
    height = float(np.median([end - start for start, end in lines]))
    lines = [(start, end) for start, end in lines if end - start <= BLOCK_LINE_SPREAD * height]
    if len(lines) < 2:
        return [box]
    groups = [[lines[0]]]
    for line in lines[1:]:
        if line[0] - groups[-1][-1][1] >= BLOCK_GAP * height:
            groups.append([line])
        else:
            groups[-1].append(line)
    merged = [groups[0]]
    for group in groups[1:]:
        if _separate(letters, merged[-1], group):
            merged.append(group)
        else:
            merged[-1] = merged[-1] + group
    if len(merged) < 2:
        return [box]
    blocks = [(group[0][0], group[-1][1]) for group in merged]
    cuts = [0] + [(upper[1] + lower[0]) // 2 for upper, lower in zip(blocks, blocks[1:])] + [letters.shape[0]]
    if rows[cuts[1:-1]].any():
        return [box]
    reach = int(height)
    pieces = []
    for (top, bottom), band_top, band_bottom in zip(blocks, cuts, cuts[1:]):
        cols = np.flatnonzero(letters[top:bottom].any(axis=0))
        y1, y2 = max(int(band_top), int(top) - reach), min(int(band_bottom), int(bottom) + reach)
        x1, x2 = max(0, int(cols[0]) - reach), min(letters.shape[1], int(cols[-1]) + 1 + reach)
        rect = (box.x1 + x1, box.y1 + y1, box.x1 + x2, box.y1 + y2)
        pieces.append(_text_box(*rect, box.confidence,
                                np.ascontiguousarray(box.mask[y1:y2, x1:x2]), box.source_model, letters[y1:y2, x1:x2],
                                core=_clip_core(box.core, rect)))
    return pieces


def _line_runs(profile: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], profile.astype(np.int8), [0]))))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def _split_sideways(box: BubbleBox) -> list[BubbleBox]:
    letters = getattr(box, "letters", None)
    if letters is None or box.mask is None or not letters.any():
        return [box]
    lines = _line_runs(letters.any(axis=1))
    if len(lines) < 2:
        return [box]
    height = float(np.median([end - start for start, end in lines]))
    lines = [(start, end) for start, end in lines if end - start <= BLOCK_LINE_SPREAD * height]
    if len(lines) < 2:
        return [box]
    runs = _line_runs(letters.any(axis=0))
    if len(runs) < 2:
        return [box]
    width, left_end, right_start = max((nxt[0] - run[1], run[1], nxt[0]) for run, nxt in zip(runs, runs[1:]))
    if width < BLOCK_GAP * height:
        return [box]
    cut = (left_end + right_start) // 2
    sides = []
    for offset, part in ((0, letters[:, :cut]), (cut, letters[:, cut:])):
        part_lines = [(s, e) for s, e in _line_runs(part.any(axis=1)) if e - s <= BLOCK_LINE_SPREAD * height]
        part_cols = np.flatnonzero(part.any(axis=0))
        sides.append((part_lines, offset + int(part_cols[0]), offset + int(part_cols[-1]) + 1))
    (left_lines, lx1, lx2), (right_lines, rx1, rx2) = sides
    if len(left_lines) < 2 or len(right_lines) < 2:
        return [box]
    if min(lx2 - lx1, rx2 - rx1) < BLOCK_MIN_WIDTH * max(lx2 - lx1, rx2 - rx1):
        return [box]
    reach = int(height)
    pieces = []
    for part_lines, sx1, sx2, band_left, band_right in (
        (left_lines, lx1, lx2, 0, cut), (right_lines, rx1, rx2, cut, letters.shape[1]),
    ):
        top, bottom = part_lines[0][0], part_lines[-1][1]
        x1, x2 = max(int(band_left), int(sx1) - reach), min(int(band_right), int(sx2) + reach)
        y1, y2 = max(0, int(top) - reach), min(letters.shape[0], int(bottom) + reach)
        rect = (box.x1 + x1, box.y1 + y1, box.x1 + x2, box.y1 + y2)
        pieces.append(_text_box(*rect, box.confidence,
                                np.ascontiguousarray(box.mask[y1:y2, x1:x2]), box.source_model, letters[y1:y2, x1:x2],
                                core=_clip_core(box.core, rect)))
    return [part for piece in pieces for part in _split_sideways(piece)]


def _rows(output: np.ndarray, conf_threshold: float) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[-1] == 6:
        rows = output.reshape(-1, 6)
        return rows[rows[:, 4] >= conf_threshold][:, :5]
    preds = output.reshape(output.shape[-2], output.shape[-1]).T
    scores = preds[:, 4:].max(axis=1)
    keep = scores >= conf_threshold
    preds, scores = preds[keep], scores[keep]
    if not len(preds):
        return np.zeros((0, 5), np.float32)
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    picked = cv2.dnn.NMSBoxes(np.stack([boxes[:, 0], boxes[:, 1], w, h], axis=1).tolist(),
                              scores.tolist(), conf_threshold, NMS_IOU)
    picked = np.asarray(picked, dtype=int).reshape(-1)
    return np.concatenate([boxes[picked], scores[picked, None]], axis=1)


def _dark_pieces(image: np.ndarray, box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = (gray < max(20, int(np.median(gray)) - MASKLESS_DARK_DROP)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    cap = MASKLESS_PIECE_SHARE * (x2 - x1) * (y2 - y1)
    return int(((areas >= MASKLESS_PIECE_PX) & (areas <= cap)).sum())


def _has_letter_piece(image: np.ndarray, box: tuple[int, int, int, int], *, allow_wide: bool = False) -> bool:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size / 2:
        ink = 255 - ink
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width = ink.shape[:2]
    for x, y, w, h, area in stats[1:]:
        if area >= 12 and h >= 5 and (allow_wide or w <= 3 * h) and x > 0 and y > 0 and x + w < width and y + h < height:
            return True
    return False


def _stretch(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)


def _mask_coverage(shape: tuple[int, int], boxes: list[BubbleBox]) -> np.ndarray:
    cov = np.zeros(shape, bool)
    for box in boxes:
        if box.mask is not None:
            cov[box.y1:box.y2, box.x1:box.x2] |= box.mask > 0
        elif getattr(box, "letters", None) is not None:
            cov[box.y1:box.y2, box.x1:box.x2] |= box.letters
    return cov


def _word_blobs(binary: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    dil = cv2.dilate(binary.astype(np.uint8), np.ones((5, 17), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if 10 <= h <= 45 and 15 <= w <= 220 and area >= 60:
            out.append(((x, y, x + w, y + h), labels == i))
    return out


def _welded_words(image: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bg = cv2.medianBlur(gray, 51)
    ink = ((bg.astype(int) - gray.astype(int)) > MASKLESS_DARK_DROP).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    out = []
    for i in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[i])
        if (RESCUE_WORD_MAX_W >= w >= 20 and RESCUE_WORD_MAX_H >= h >= 12
                and area >= RESCUE_WORD_MIN_AREA and area / max(1, w * h) >= RESCUE_WORD_MIN_FILL):
            out.append(((x1 + x, y1 + y, x1 + x + w, y1 + y + h), (labels == i)[y:y + h, x:x + w]))
    return out


def _union_coherent(image: np.ndarray, boxes: list[tuple[int, int, int, int, float]]) -> list[tuple[int, int, int, int, float]]:
    groups = []
    for x1, y1, x2, y2, score in boxes:
        rect = (x1, y1, x2, y2)
        (gx1, gy1, gx2, gy2), _grown, letters = letter_mask(image, rect, with_letters=True)
        probe = _text_box(gx1, gy1, gx2, gy2, score, None, "kiuyha", letters, core=rect)
        groups.append([[x1, y1, x2, y2, score], _letter_geometry(probe, rect)])
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _overlaps(groups[i][0], groups[j][0]) and (
                        _letters_coherent(groups[i][1], groups[j][1])
                        or _same_detection(groups[i][0], groups[j][0])
                        or (groups[i][1] is None and groups[j][1] is None
                            and _unlettered_duplicate(groups[i][0], groups[j][0]))):
                    a, b = groups[i], groups[j]
                    a[0] = [min(a[0][0], b[0][0]), min(a[0][1], b[0][1]),
                            max(a[0][2], b[0][2]), max(a[0][3], b[0][3]), max(a[0][4], b[0][4])]
                    a[1] = _merge_geometry(a[1], b[1])
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return [(int(a), int(b), int(c), int(d), float(e)) for a, b, c, d, e in (g[0] for g in groups)]


class KiuyhaTextDetector:
    def __init__(self, model_path, conf_threshold: float = knob("detect.confidence"), session=None):
        self.session = session if session is not None else make_session(model_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        height, width = inp.shape[2], inp.shape[3]
        self.fixed = (int(height), int(width)) if isinstance(height, int) and isinstance(width, int) else None
        self.conf_threshold = float(conf_threshold)
        self._metrics = threading.local()
        self.source_model = Path(str(model_path)).name

    def _input_size(self, height: int, width: int) -> tuple[int, int]:
        if self.fixed is not None:
            return self.fixed
        return int(math.ceil(height / STRIDE) * STRIDE), int(math.ceil(width / STRIDE) * STRIDE)

    def raw_boxes(self, image: np.ndarray, conf_threshold: float | None = None) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        in_h, in_w = self._input_size(h, w)
        scale = min(in_w / w, in_h / h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        pad_x, pad_y = (in_w - rw) // 2, (in_h - rh) // 2
        canvas = np.full((in_h, in_w, 3), LETTERBOX_VALUE, np.uint8)
        resized = image if (rw, rh) == (w, h) else cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        rows = _rows(self.session.run(None, {self.input_name: blob})[0],
                     self.conf_threshold if conf_threshold is None else conf_threshold)
        boxes = []
        for x1, y1, x2, y2, score in rows:
            bx1 = int(max(0.0, (x1 - pad_x) / scale))
            by1 = int(max(0.0, (y1 - pad_y) / scale))
            bx2 = int(min(float(w), math.ceil((x2 - pad_x) / scale)))
            by2 = int(min(float(h), math.ceil((y2 - pad_y) / scale)))
            if bx2 > bx1 and by2 > by1:
                boxes.append((bx1, by1, bx2, by2, float(score)))
        return boxes

    def halves_plan(self, height: int, width: int) -> tuple[float, list[tuple[int, int]]] | None:
        if self.fixed is None or self.fixed[0] != self.fixed[1]:
            return None
        size = self.fixed[0]
        scale = min(1.0, (size - COLUMN_GAP) / (2.0 * width))
        overlap = int(2 * size / scale) - height
        if overlap < HALVES_MIN_OVERLAP:
            overlap = min(height, HALVES_MIN_OVERLAP)
            scale = min(scale, 2.0 * size / (height + overlap))
        if scale < 1.2 * min(size / width, size / height):
            return None
        half = (height + min(overlap, height) + 1) // 2
        return scale, [(0, half), (height - half, height)]

    def _raw_slice_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        found = self._coarse_boxes(image) + self.page_boxes(image)
        if not found:
            found = self._rotated_boxes(image)
        return found

    def detect_slice(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        found = _union_coherent(image, self._raw_slice_boxes(image))
        return [(max(0, x1 - BOX_PAD), max(0, y1 - BOX_PAD), min(w, x2 + BOX_PAD), min(h, y2 + BOX_PAD), score)
                for x1, y1, x2, y2, score in found]

    def _rotated_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        found = []
        for code, clockwise in ((cv2.ROTATE_90_CLOCKWISE, True), (cv2.ROTATE_90_COUNTERCLOCKWISE, False)):
            turned = cv2.rotate(image, code)
            for x1, y1, x2, y2, score in self.raw_boxes(turned):
                if score < ROTATED_MIN_SCORE:
                    continue
                if clockwise:
                    box = (y1, h - 1 - x2, y2, h - 1 - x1)
                else:
                    box = (w - 1 - y2, x1, w - 1 - y1, x2)
                bx1, by1, bx2, by2 = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
                if bx2 > bx1 and by2 > by1:
                    found.append((bx1, by1, bx2, by2, score))
        return found

    def _coarse_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        plan = self.halves_plan(h, w)
        if plan is None:
            found = self.raw_boxes(image)
        else:
            scale, halves = plan
            size = self.fixed[0]
            canvas = np.full((size, size, 3), LETTERBOX_VALUE, np.uint8)
            columns, x = [], 0
            for y0, y1 in halves:
                rw, rh = min(size - x, int(w * scale)), min(size, int((y1 - y0) * scale))
                canvas[:rh, x:x + rw] = cv2.resize(image[y0:y1], (rw, rh), interpolation=cv2.INTER_LINEAR)
                columns.append((x, rw, rh, y0, y1))
                x += rw + COLUMN_GAP
            found = []
            for cx1, cy1, cx2, cy2, score in self.raw_boxes(canvas):
                px, rw, rh, y0, y1 = columns[0] if (cx1 + cx2) / 2 < columns[1][0] else columns[1]
                sx, sy = rw / w, rh / (y1 - y0)
                bx1, bx2 = int((max(cx1, px) - px) / sx), int(math.ceil((min(cx2, px + rw) - px) / sx))
                by1, by2 = int(min(cy1, rh) / sy) + y0, int(math.ceil(min(cy2, rh) / sy)) + y0
                if bx2 - bx1 > 4 and by2 - by1 > 4:
                    found.append((bx1, by1, bx2, by2, score))
        return found

    def page_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        if self.fixed is None:
            return self.raw_boxes(image)
        size_h, size_w = self.fixed
        scale = min(1.0, PAGE_WIDTH / w)
        sw = max(1, int(round(w * scale)))
        window = int(size_h / scale)
        step = int(window * (1 - PAGE_OVERLAP))
        starts = [0] if window >= h else list(range(0, h - window, step)) + [h - window]
        per_canvas = max(1, (size_w + COLUMN_GAP) // (sw + COLUMN_GAP))
        found = []
        for first in range(0, len(starts), per_canvas):
            canvas = np.full((size_h, size_w, 3), LETTERBOX_VALUE, np.uint8)
            columns = []
            for k, y0 in enumerate(starts[first:first + per_canvas]):
                part = image[y0:y0 + window]
                rh = min(size_h, int(round(part.shape[0] * scale)))
                x = k * (sw + COLUMN_GAP)
                canvas[:rh, x:x + sw] = cv2.resize(part, (sw, rh), interpolation=cv2.INTER_AREA)
                columns.append((x, rh, y0, part.shape[0]))
            for cx1, cy1, cx2, cy2, score in self.raw_boxes(canvas):
                if score < PAGE_CONFIDENCE:
                    continue
                centre = (cx1 + cx2) / 2
                column = next((c for c in columns if c[0] <= centre < c[0] + sw), None)
                if column is None:
                    continue
                x, rh, y0, part_h = column
                bx1, bx2 = int(max(0, cx1 - x) / scale), int(math.ceil(min(sw, cx2 - x) / scale))
                by1, by2 = int(max(0, cy1) / scale), int(math.ceil(min(rh, cy2) / scale))
                if bx2 - bx1 > 4 and by2 - by1 > 4:
                    found.append((min(w, bx1), by1 + y0, min(w, bx2), min(h, by2 + y0), score))
        return found

    def _rescue_boxes(self, image: np.ndarray, kept: list[BubbleBox]) -> list[BubbleBox]:
        h, w = image.shape[:2]
        cov = _mask_coverage(image.shape[:2], kept)
        dilated = cv2.dilate(cov.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        low = [box for box in self.raw_boxes(image, RESCUE_MIN_SCORE)
               if box[4] < self.conf_threshold]
        stretched = None
        found: list[list] = []

        def consider(rect, pixels, score, emit=None, welded=False):
            x1, y1, x2, y2 = rect
            if x2 - x1 < 15 or y2 - y1 < 10 or not pixels.any():
                return
            if float((pixels & dilated[y1:y2, x1:x2]).sum()) / float(pixels.sum()) >= 0.5:
                return
            found.append([*(emit or rect), score, welded])

        for x1, y1, x2, y2, score in sorted(low, key=lambda b: (b[1], b[0])):
            if _dark_pieces(image, (x1, y1, x2, y2)) < MASKLESS_MIN_PIECES:
                continue
            if stretched is None:
                stretched = _stretch(image)
            (sx1, sy1, sx2, sy2), _, letters = letter_mask(stretched, (x1, y1, x2, y2), with_letters=True)
            if letters is not None and letters.any():
                uncov = letters & ~cov[sy1:sy2, sx1:sx2]
                for (wx1, wy1, wx2, wy2), blob in _word_blobs(uncov):
                    consider((sx1 + wx1, sy1 + wy1, sx1 + wx2, sy1 + wy2),
                             blob[wy1:wy2, wx1:wx2], score)
            for (wx1, wy1, wx2, wy2), pixels in _welded_words(image, (x1, y1, x2, y2)):
                consider((wx1, wy1, wx2, wy2), pixels, score,
                         emit=(max(0, wx1 - RESCUE_WORD_PAD), max(0, wy1 - RESCUE_WORD_PAD),
                               min(w, wx2 + RESCUE_WORD_PAD), min(h, wy2 + RESCUE_WORD_PAD)),
                         welded=True)
        merged: list[list] = []
        for rect in sorted(found, key=lambda r: (r[1], r[0])):
            for other in merged:
                ix = max(0, min(rect[2], other[2]) - max(rect[0], other[0]))
                iy = max(0, min(rect[3], other[3]) - max(rect[1], other[1]))
                if ix * iy >= 0.5 * min((rect[2] - rect[0]) * (rect[3] - rect[1]),
                                       (other[2] - other[0]) * (other[3] - other[1])):
                    other[0], other[1] = min(other[0], rect[0]), min(other[1], rect[1])
                    other[2], other[3] = max(other[2], rect[2]), max(other[3], rect[3])
                    other[4] = max(other[4], rect[4])
                    other[5] = other[5] or rect[5]
                    break
            else:
                merged.append(list(rect))
        boxes = []
        for x1, y1, x2, y2, score, welded in merged:
            if not _has_letter_piece(image, (x1, y1, x2, y2), allow_wide=welded):
                continue
            full = np.full((y2 - y1, x2 - x1), 255, np.uint8)
            boxes.append(_text_box(x1, y1, x2, y2, score, full, self.source_model, None))
        return boxes

    def text_boxes(self, image: np.ndarray) -> list[BubbleBox]:
        boxes = []
        cores = []
        for x1, y1, x2, y2, score in self._raw_slice_boxes(image):
            core = (x1, y1, x2, y2)
            (x1, y1, x2, y2), mask, letters = letter_mask(image, core, with_letters=True)
            if mask.any():
                boxes.append(_text_box(x1, y1, x2, y2, score, mask.astype(np.uint8) * 255, self.source_model, letters, core=core))
                cores.append(core)
            elif (score >= MASKLESS_MIN_SCORE and y2 > y1 and x2 > x1
                  and (_dark_pieces(image, (x1, y1, x2, y2)) >= MASKLESS_MIN_PIECES
                       or (x2 - x1) * (y2 - y1) <= MASKLESS_SMALL_BOX_AREA)):
                full = np.full((y2 - y1, x2 - x1), 255, np.uint8)
                boxes.append(_text_box(x1, y1, x2, y2, score, full, self.source_model, None, core=core))
                cores.append(core)
        kept = [part for box in _join_grown(boxes, cores) for piece in _split_blocks(box) for part in _split_sideways(piece)]
        final = kept + self._rescue_boxes(image, kept)
        _attach_bubble_masks(image, final)
        return final

    def leftover_boxes(self, clean: np.ndarray, targets) -> list[BubbleBox]:
        boxes = []
        for t in targets:
            if still_reads(clean, (t.x1, t.y1, t.x2, t.y2)):
                (x1, y1, x2, y2), mask = letter_mask(clean, (t.x1, t.y1, t.x2, t.y2))
                if mask.any():
                    boxes.append(_text_box(x1, y1, x2, y2, t.confidence, mask.astype(np.uint8) * 255, self.source_model))
        return boxes

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        started = time.perf_counter()
        boxes = self.text_boxes(image)
        self._metrics.value = {"text_model_ms": (time.perf_counter() - started) * 1000.0, "result_boxes": len(boxes)}
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._metrics, "value", {}) or {})
