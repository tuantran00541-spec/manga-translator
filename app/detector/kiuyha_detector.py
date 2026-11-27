from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.detector.boxes import BubbleBox, _merge_mask_evidence, apply_final_nms
from app.detector.bubble_mask import (
    GAP_MIN_CONTAINMENT,
    GAP_MIN_SIDE_SHARE,
    _letters_inside_mask,
    derive_bubble_mask,
    find_waist,
    split_bubble_mask,
)
from app.detector.ctd_mask import chromatic_stroke_mask, letter_mask, probability, still_reads
from app.detector.ink_mask import (
    CONTRAST_ARTWORK_MASK,
    LOCAL_INK_MASK,
    LARGE_TEXT_MASK_AREA,
    PAPER_INK_MASK,
    PRECISE_INK_MASK_SOURCES,
    expand_local_ink,
    fills_rectangle,
    flat_background_ink,
    isolated_paper_ink,
    protect_free_text_art,
)
from app.knobs import knob
from app.model_contracts import validate_kiuyha_session
from app.ort_utils import make_session
from app.parameters import DETECTOR_MAX_BOX_AREA_RATIO

LETTERBOX_VALUE = 114
STRIDE = 32
NMS_IOU = knob("detect.nms_iou")
BOX_PAD = knob("detect.box_pad")
COLUMN_GAP = 16
HALVES_MIN_OVERLAP = 256
PAGE_WIDTH = 400
PAGE_OVERLAP = 0.15
PAGE_CONFIDENCE = 0.45
OVERVIEW_WIDTHS = (440, 315, 225)
OVERVIEW_GAP = 24
OVERVIEW_PAD_Y = 64
OVERVIEW_CONFIDENCE = 0.55
OVERVIEW_MEDIUM_CONFIDENCE = 0.65
OVERVIEW_WHITE_BUBBLE_CONFIDENCE = 0.30
OVERVIEW_MIN_LONG = 350
OVERVIEW_MIN_SHORT = 160
OVERVIEW_MEDIUM_MIN_LONG = 64
OVERVIEW_MEDIUM_MIN_SHORT = 20
OVERVIEW_WHITE_BUBBLE_MIN_WIDTH_SHARE = 0.60
OVERVIEW_WHITE_BUBBLE_BRIGHT_L = 215
OVERVIEW_WHITE_BUBBLE_MIN_SHARE = 0.35
OVERVIEW_WHITE_BUBBLE_DARK_L = 50
OVERVIEW_WHITE_BUBBLE_DARK_SHARE = 0.15
OVERVIEW_EXISTING_OVERLAP = 0.35
OVERVIEW_SELF_OVERLAP = 0.40
PRE_CTD_DEDUPE_IOU = 0.72
AUX_MODEL_NAME = "kiuyha-tuned-transient-1024.onnx"
AUX_MODEL_ENV = "MANGA_KIUYHA_AUX_MODEL"
MASKLESS_MIN_SCORE = 0.5
MASKLESS_MIN_PIECES = 3
MASKLESS_PIECE_PX = 20
MASKLESS_PIECE_SHARE = 0.25
MASKLESS_DARK_DROP = 25
MASKLESS_SMALL_BOX_AREA = 15000
MASKLESS_MIN_COVERAGE = 0.02
ROTATED_MIN_SCORE = PAGE_CONFIDENCE
ROTATED_AUX_MIN_SCORE = 0.50
NATIVE_AUX_CONFIDENCE = 0.25
NATIVE_AUX_STRICT_CONF = 0.32
ENHANCED_CTD_MIN_SCORE = 0.35
ENHANCED_CTD_AUX_MIN_SCORE = 0.50
RESCUE_MIN_SCORE = 0.10
RESCUE_WORD_MAX_W = 80
RESCUE_WORD_MAX_H = 35
RESCUE_WORD_MIN_AREA = 100
RESCUE_WORD_MIN_FILL = 0.35
RESCUE_WORD_MIN_ASPECT = 2.0
RESCUE_MIN_CONTRAST = 40.0
RESCUE_UNSTRETCHED_CTD_MIN = 0.35
RESCUE_WORD_PAD = 8
TIGHT_BOX_PAD = 4
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
        if bottom - top < max(6, int(0.6 * reach)):
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


def split_waist_boxes(boxes: list[BubbleBox], _depth: int = 0) -> list[BubbleBox]:
    if _depth >= 3:
        return list(boxes)
    out: list[BubbleBox] = []
    changed = False
    for box in boxes:
        parts = _split_waist_box(box)
        if len(parts) > 1:
            changed = True
        out.extend(parts)
    return split_waist_boxes(out, _depth + 1) if changed else out


def _text_box(
    x1,
    y1,
    x2,
    y2,
    score,
    mask,
    source_model,
    letters=None,
    core=None,
    mask_source: str = "text_segmenter",
) -> BubbleBox:
    box = BubbleBox(
        x1, y1, x2, y2, score, mask,
        source_model=source_model, class_name="text", semantic_type="free_text",
        mask_source=mask_source, safe_to_inpaint=True, ocr_eligible=True,
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


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter) / float(max(1, union))


def _dedupe_raw_boxes(
    boxes: list[tuple[int, int, int, int, float]],
    iou_threshold: float = PRE_CTD_DEDUPE_IOU,
) -> list[tuple[int, int, int, int, float]]:
    if len(boxes) <= 1:
        return list(boxes)
    ordered = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []
    for box in ordered:
        matched_idx = None
        for idx, prev in enumerate(kept):
            if _box_iou(box, prev) >= iou_threshold:
                matched_idx = idx
                break
        if matched_idx is not None:
            prev = kept[matched_idx]
            if box[4] >= 0.65 * prev[4]:
                ux1 = min(prev[0], box[0])
                uy1 = min(prev[1], box[1])
                ux2 = max(prev[2], box[2])
                uy2 = max(prev[3], box[3])
                prev_area = max(1, (prev[2] - prev[0]) * (prev[3] - prev[1]))
                if (ux2 - ux1) * (uy2 - uy1) <= 1.25 * prev_area:
                    kept[matched_idx] = (ux1, uy1, ux2, uy2, max(prev[4], box[4]))
            continue
        kept.append(box)
    return kept


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


def _has_border_wall(
    image: np.ndarray | None,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    *,
    vertical: bool,
) -> bool:
    if image is None or x2 <= x1 or y2 <= y1:
        return False
    h, w = image.shape[:2]
    ry1, ry2 = max(0, int(y1)), min(h, int(y2))
    rx1, rx2 = max(0, int(x1)), min(w, int(x2))
    if ry2 - ry1 < 4 or rx2 - rx1 < 4:
        return False
    patch = image[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    dark = gray < 95
    coverage = dark.any(axis=1) if vertical else dark.any(axis=0)
    return float(coverage.mean()) >= 0.65


def _letters_coherent(a, b, image: np.ndarray | None = None) -> bool:
    if a is None or b is None:
        return False
    rows_a, box_a = a
    rows_b, box_b = b
    if rows_a & rows_b:
        horiz_gap = max(box_b[0] - box_a[2], box_a[0] - box_b[2])
        min_line_h = min(box_a[3] - box_a[1], box_b[3] - box_b[1])
        if horiz_gap > max(14, int(1.2 * max(8, min_line_h))):
            return False
        if horiz_gap >= 2:
            oy1 = max(box_a[1], box_b[1])
            oy2 = min(box_a[3], box_b[3])
            gx1 = min(box_a[2], box_b[2])
            gx2 = max(box_a[0], box_b[0])
            if _has_border_wall(image, oy1, oy2, gx1, gx2, vertical=True):
                return False
        return True
    (ux1, uy1, ux2, uy2), (lx1, ly1, lx2, ly2) = sorted((box_a, box_b), key=lambda box: box[1])
    gap = ly1 - uy2
    shorter = min(uy2 - uy1, ly2 - ly1)
    wider = max(ux2 - ux1, lx2 - lx1)
    offset = abs((ux1 + ux2) - (lx1 + lx2)) / 2
    if image is not None and gap >= 2:
        ox1, ox2 = max(ux1, lx1), min(ux2, lx2)
        if _has_border_wall(image, uy2, ly1, ox1 if ox2 > ox1 else min(ux1, lx1), ox2 if ox2 > ox1 else max(ux2, lx2), vertical=False):
            return False
        return gap <= 0.55 * shorter and offset <= 0.25 * wider
    return gap <= JOIN_STACK_GAP * shorter and offset <= JOIN_STACK_CENTRE * wider


def _join_grown(
    boxes: list[BubbleBox],
    cores: list[tuple[int, int, int, int]] | None = None,
    image: np.ndarray | None = None,
) -> list[BubbleBox]:
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
                        _letters_coherent(groups[i][1], groups[j][1], image=image)
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
        sources = set()
        for b in boxes:
            if b.x1 >= x1 and b.y1 >= y1 and b.x2 <= x2 and b.y2 <= y2:
                view = mask[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1]
                np.maximum(view, b.mask, out=view)
                if getattr(b, "letters", None) is not None:
                    letters[b.y1 - y1:b.y2 - y1, b.x1 - x1:b.x2 - x1] |= b.letters
                sources.add(b.mask_source)
        src_kind = next(iter(sources)) if len(sources) == 1 else "text_segmenter"
        if len(sources) > 1 and sources <= PRECISE_INK_MASK_SOURCES:
            src_kind = CONTRAST_ARTWORK_MASK
        if "border_protected_display" in sources:
            src_kind = "border_protected_display"
        joined.append(_text_box(x1, y1, x2, y2, score, mask, boxes[0].source_model,
                                letters if letters.any() else None, core=core, mask_source=src_kind))
    return joined


BLOCK_GAP = knob("detect.block_gap")
BLOCK_OFFSET = knob("detect.block_offset")
BLOCK_SPACING = knob("detect.block_spacing")
BLOCK_MIN_WIDTH = knob("detect.block_min_width")
BLOCK_LINE_SPREAD = knob("detect.block_line_spread")


def _separate(mask: np.ndarray, upper: list, lower: list,
              image: np.ndarray | None = None, origin: tuple[int, int] = (0, 0),
              height: float = 0.0) -> bool:
    def extent(group):
        cols = np.flatnonzero(mask[group[0][0]:group[-1][1]].any(axis=0))
        return float(cols[0]), float(cols[-1] + 1)

    gap = lower[0][0] - upper[-1][1]
    if len(upper) < 2 or len(lower) < 2:
        if image is not None and height > 0 and gap >= 1.25 * height:
            (a1, a2), (b1, b2) = extent(upper), extent(lower)
            if min(a2 - a1, b2 - b1) >= BLOCK_MIN_WIDTH * max(a2 - a1, b2 - b1):
                ox, oy = origin
                gx1, gx2 = int(min(a1, b1)) + ox, int(max(a2, b2)) + ox
                if _has_border_wall(image, oy + upper[-1][1], oy + lower[0][0], gx1, gx2, vertical=False):
                    return True
        return False
    (a1, a2), (b1, b2) = extent(upper), extent(lower)
    if min(a2 - a1, b2 - b1) < BLOCK_MIN_WIDTH * max(a2 - a1, b2 - b1):
        return False
    if abs((a1 + a2) - (b1 + b2)) / 2 > BLOCK_OFFSET * max(a2 - a1, b2 - b1):
        return True
    inner = [nxt[0] - line[1] for group in (upper, lower) for line, nxt in zip(group, group[1:])]
    return gap >= BLOCK_SPACING * max(inner)


def _clip_core(core: tuple[int, int, int, int] | None, rect: tuple[int, int, int, int]):
    if core is None:
        return None
    x1, y1 = max(core[0], rect[0]), max(core[1], rect[1])
    x2, y2 = min(core[2], rect[2]), min(core[3], rect[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _split_blocks(box: BubbleBox, image: np.ndarray | None = None) -> list[BubbleBox]:
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
        if _separate(letters, merged[-1], group, image=image, origin=(box.x1, box.y1), height=height):
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
                                core=_clip_core(box.core, rect), mask_source=box.mask_source))
    return pieces


def _line_runs(profile: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], profile.astype(np.int8), [0]))))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def _split_sideways(box: BubbleBox, image: np.ndarray | None = None) -> list[BubbleBox]:
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
        if not part_lines or not len(part_cols):
            return [box]
        sides.append((part_lines, offset + int(part_cols[0]), offset + int(part_cols[-1]) + 1))
    (left_lines, lx1, lx2), (right_lines, rx1, rx2) = sides
    if len(left_lines) < 2 or len(right_lines) < 2:
        if (
            image is None
            or not left_lines
            or not right_lines
            or width < 1.25 * height
            or not _has_border_wall(
                image,
                box.y1 + min(left_lines[0][0], right_lines[0][0]),
                box.y1 + max(left_lines[-1][1], right_lines[-1][1]),
                box.x1 + left_end,
                box.x1 + right_start,
                vertical=True,
            )
        ):
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
                                core=_clip_core(box.core, rect), mask_source=box.mask_source))
    return [part for piece in pieces for part in _split_sideways(piece, image=image)]


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


def _bright_pieces(image: np.ndarray, box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = (gray > min(235, int(np.median(gray)) + MASKLESS_DARK_DROP)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    cap = MASKLESS_PIECE_SHARE * (x2 - x1) * (y2 - y1)
    return int(((areas >= MASKLESS_PIECE_PX) & (areas <= cap)).sum())


def _is_white_bubble_display(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    if x2 - x1 < OVERVIEW_WHITE_BUBBLE_MIN_WIDTH_SHARE * w:
        return False
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bright = float((gray >= OVERVIEW_WHITE_BUBBLE_BRIGHT_L).mean())
    dark = float((gray <= OVERVIEW_WHITE_BUBBLE_DARK_L).mean())
    return bright >= OVERVIEW_WHITE_BUBBLE_MIN_SHARE and dark >= OVERVIEW_WHITE_BUBBLE_DARK_SHARE


def _restore_outline_glyphs(gray, letters, protected, labels, stats, candidates, core):
    """Accept small outlined glyphs when CTD, a text row and surrounding white paper agree."""
    if not np.any(letters):
        return
    heights = stats[np.unique(labels[letters]), cv2.CC_STAT_HEIGHT]
    if heights.size == 0:
        return
    cap = float(np.median(heights))
    if not np.isfinite(cap) or cap < 8:
        return
    left, top, right, bottom = core
    pad = round(cap)
    collar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    margin = max(8, round(.26 * cap))
    for cid in candidates:
        x, y, w, h, _ = (int(v) for v in stats[cid])
        if not (.48 * cap <= h <= 1.25 * cap and 3 <= w <= .9 * cap
                and left + 6 <= x and x + w <= right - 6 and top + 6 <= y and y + h <= bottom - 6):
            continue
        row = letters[y:y + h, max(0, x - pad):min(gray.shape[1], x + w + pad)]
        if np.mean(row.any(axis=1)) < .55:
            continue
        bounds = np.s_[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
        comp = labels[bounds] == cid
        values = gray[bounds]
        _, faint = cv2.connectedComponents(((values < 228) & ~letters[bounds]).astype(np.uint8))
        faint_ids = np.unique(faint[comp])
        faint_ids = faint_ids[faint_ids > 0]
        if faint_ids.size == 0:
            continue
        outline = np.isin(faint, faint_ids)
        if outline[0].any() or outline[-1].any() or outline[:, 0].any() or outline[:, -1].any():
            continue
        paper = (cv2.dilate(outline.astype(np.uint8), collar) > 0) & ~outline & ~letters[bounds]
        if np.count_nonzero(paper) < 20 or np.mean(values[paper] >= 228) < .85:
            continue
        letters[bounds] |= comp
        protected[bounds] &= ~comp


def _white_bubble_display_mask(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Build a border-protected glyph mask for oversized bold brush text inside white spiky bubbles.

    Separates interior bold letter components from thin spiky bubble borders and adjacent panel art
    so jagged speech-bubble outlines and neighboring panel artwork stay intact.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box
    pad = 40
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
    crop = image[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < 32 or cw < 32 or crop.ndim != 3:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Require a predominantly bright outer page margin so large bubbles inside dark/flaming
    # panels fall back to CTD letter_mask instead of losing edge words to dark panel bleed.
    border_px = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float((border_px >= 195).mean()) < 0.65:
        return None
    prob = probability(crop)

    dark = (gray < 140).astype(np.uint8)
    # Morphologically close narrow white letter-outline gaps so outer page margin never leaks inside.
    wall = cv2.morphologyEx(
        (gray < 185).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    free = (wall == 0).astype(np.uint8)
    _, flabels = cv2.connectedComponents(free, connectivity=4)
    edge_ids = set(
        np.unique(
            np.concatenate([flabels[0, :], flabels[-1, :], flabels[:, 0], flabels[:, -1]])
        ).tolist()
    )
    edge_ids.discard(0)
    outer_margin = np.zeros((ch, cw), bool)
    for eid in edge_ids:
        region = flabels == eid
        if region.sum() >= 0.015 * ch * cw:
            outer_margin |= region
    if not outer_margin.any():
        return None
    near_outer = cv2.dilate(
        outer_margin.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    ) > 0

    n_c, clabels, cstats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    letter_mask_local = np.zeros((ch, cw), bool)
    protect_local = np.zeros((ch, cw), bool)
    unassigned: list[tuple[int, int]] = []
    outline_candidates = []

    for cid in range(1, n_c):
        cx, cy, bw, bh, area = (int(v) for v in cstats[cid])
        if area < 6:
            continue
        roi = np.s_[cy:cy + bh, cx:cx + bw]
        comp_roi = clabels[roi] == cid
        touches_edge = (cx <= 1) or (cy <= 1) or (cx + bw >= cw - 1) or (cy + bh >= ch - 1)
        ctd_mean = float(prob[roi][comp_roi].mean())
        ctd_high = float((prob[roi][comp_roi] >= 0.45).mean())
        outer_touch = float((comp_roi & near_outer[roi]).sum()) / max(1, area)
        fill = area / max(1.0, bw * bh)

        if touches_edge and area > 2000:
            protect_local[roi] |= comp_roi
            continue
        if fill < 0.28 and (outer_touch >= 0.40 or (outer_touch >= 0.22 and ctd_high < 0.70)):
            protect_local[roi] |= comp_roi
            if ctd_high >= .70:
                outline_candidates.append(cid)
            continue
        in_core = (
            cx + bw > (x1 - cx1) + 4
            and cx < (x2 - cx1) - 4
            and cy + bh > (y1 - cy1) + 4
            and cy < (y2 - cy1) - 4
        )
        if not in_core:
            protect_local[roi] |= comp_roi
            continue
        if ctd_high >= 0.10 or (ctd_mean >= 0.14 and area >= 40):
            letter_mask_local[roi] |= comp_roi
        elif outer_touch >= 0.25 and fill < 0.28:
            protect_local[roi] |= comp_roi
        else:
            unassigned.append((cid, area))

    if not letter_mask_local.any() or not protect_local.any():
        return None

    near_core_letters = cv2.dilate(
        letter_mask_local.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65)),
    ) > 0
    for cid, area in unassigned:
        cx, cy, bw, bh, _ = (int(v) for v in cstats[cid])
        roi = np.s_[cy:cy + bh, cx:cx + bw]
        comp_roi = clabels[roi] == cid
        if area < 4000 and (comp_roi & near_core_letters[roi]).any():
            letter_mask_local[roi] |= comp_roi
    if outline_candidates:
        _restore_outline_glyphs(gray, letter_mask_local, protect_local, clabels, cstats,
                                outline_candidates, (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1))

    protect_wall = cv2.dilate(
        protect_local.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    # Keep the erase authority close to the glyphs.  A wide halo here used to
    # absorb the faint texture and parts of the outline of a large white
    # speech bubble, which then made the whole interior look airbrushed.
    near_letters = cv2.dilate(
        letter_mask_local.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ) > 0
    chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
    letter_halo = near_letters & ((gray < 248) | (chroma > 5)) & ~protect_wall
    final_local = (
        cv2.dilate(
            (letter_mask_local | letter_halo).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        > 0
    ) & ~protect_wall
    if not final_local.any():
        return None
    return (cx1, cy1, cx2, cy2), final_local.astype(np.uint8) * 255, letter_mask_local


def _resolve_aux_model_path(model_path) -> Path | None:
    override = os.getenv(AUX_MODEL_ENV, "").strip()
    if override.lower() in {"0", "false", "no", "off", "none"}:
        return None
    base = Path(str(model_path))
    if override:
        candidate = Path(override) if Path(override).is_absolute() else base.parent / override
        return candidate if candidate.is_file() else None
    if base.name == AUX_MODEL_NAME:
        return None
    sibling = base.parent / AUX_MODEL_NAME
    return sibling if sibling.is_file() else None


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


def _word_blobs(
    binary: np.ndarray,
    *,
    max_h: int = 64,
    max_w: int = 280,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    dil = cv2.dilate(binary.astype(np.uint8), np.ones((5, 17), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if 10 <= h <= max_h and 15 <= w <= max_w and area >= 60:
            out.append(((x, y, x + w, y + h), binary[y:y + h, x:x + w] & (labels[y:y + h, x:x + w] == i)))
    return out


def _has_strong_aux_letter_evidence(letters: np.ndarray | None, mask_size: int) -> bool:
    """Require genuine multi-glyph or dense CTD letter seeds on low-confidence aux detections."""
    if letters is None or not np.any(letters) or mask_size <= 0:
        return False
    letter_px = int(np.count_nonzero(letters))
    if letter_px < 48 or letter_px / float(mask_size) < 0.08:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(letters.astype(np.uint8), connectivity=8)
    valid = [row for row in stats[1:] if int(row[cv2.CC_STAT_AREA]) >= 6]
    if len(valid) >= 3 and letter_px >= 120:
        return True
    if len(valid) >= 2 and letter_px >= 64 and letter_px / float(mask_size) >= 0.12:
        return True
    if letter_px >= 2500:
        return True
    return False


def _is_periodic_screentone_letters(letters: np.ndarray) -> bool:
    """Detect uniform periodic dot grids / screentones misread by CTD on low-score boxes."""
    if letters is None or not np.any(letters):
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(letters.astype(np.uint8), connectivity=8)
    if n < 15:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    small = (areas >= 2) & (areas <= 48)
    if int(np.count_nonzero(small)) < 14:
        return False
    sel_areas = areas[small].astype(np.float32)
    med_area = float(np.median(sel_areas))
    if med_area >= 24.0:
        return False
    iqr = float(np.percentile(sel_areas, 75) - np.percentile(sel_areas, 25))
    h_std = float(np.std(heights[small].astype(np.float32)))
    return iqr / max(1.0, med_area) < 0.35 and h_std < 1.8


def _passes_glyph_plausibility(
    image_shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    mask: np.ndarray,
    letters: np.ndarray | None,
    score: float,
    *,
    is_aux: bool = False,
) -> bool:
    """Reject full-panel artwork FPs, sub-glyph noise specks, and periodic halftone grids."""
    img_h, img_w = image_shape[:2]
    img_area = max(1, img_h * img_w)
    rw = max(1, rect[2] - rect[0])
    rh = max(1, rect[3] - rect[1])
    box_area = rw * rh
    if box_area > DETECTOR_MAX_BOX_AREA_RATIO * img_area and score < 0.75:
        return False
    if letters is None or not np.any(letters):
        return False
    letter_px = int(np.count_nonzero(letters))
    if score < 0.65 or is_aux:
        if letter_px < 12:
            return False
    if score < 0.45 and _is_periodic_screentone_letters(letters):
        return False
    return True


def _chromatic_contrast_probe(crop: np.ndarray) -> np.ndarray | None:
    """Convert isoluminant colored text into dark-on-white BGR using border-relative Lab chroma."""
    if crop.ndim != 3 or crop.shape[2] < 3 or min(crop.shape[:2]) < 12:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[..., 1:3]
    chroma = np.hypot(ab[..., 0] - 128.0, ab[..., 1] - 128.0)
    if float(chroma.mean()) <= 18.0:
        return None
    h, w = crop.shape[:2]
    band = max(2, min(8, min(h, w) // 8))
    border_ab = np.concatenate([
        ab[:band].reshape(-1, 2),
        ab[-band:].reshape(-1, 2),
        ab[:, :band].reshape(-1, 2),
        ab[:, -band:].reshape(-1, 2),
    ])
    bg_ab = np.median(border_ab, axis=0)
    dist = np.linalg.norm(ab - bg_ab, axis=-1)
    if float(np.percentile(dist, 96)) < 16.0:
        return None
    inv_gray = np.clip(255.0 - dist * 4.5, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR)


def _enhanced_letter_mask(
    image: np.ndarray,
    core: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Retry CTD letter segmentation on locally contrast-enhanced, inverted, or chroma crops."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = core
    pad = max(64, int(0.45 * max(x2 - x1, y2 - y1)))
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = image[py1:py2, px1:px2]
    if crop.size == 0 or min(crop.shape[:2]) < 8 or crop.ndim != 3:
        return None
    local_core = (x1 - px1, y1 - py1, x2 - px1, y2 - py1)
    variants = [_stretch(crop)]
    gray = cv2.cvtColor(crop[local_core[1]:local_core[3], local_core[0]:local_core[2]], cv2.COLOR_BGR2GRAY)
    if gray.size and float(np.median(gray)) < 155.0:
        variants.append(255 - crop)
    chroma_probe = _chromatic_contrast_probe(crop)
    if chroma_probe is not None:
        variants.append(chroma_probe)
    for variant in variants:
        (lx1, ly1, lx2, ly2), mask, letters = letter_mask(variant, local_core, with_letters=True)
        if not mask.size or fills_rectangle(mask):
            continue
        if np.count_nonzero(mask) / mask.size < MASKLESS_MIN_COVERAGE:
            continue
        if letters is None or np.count_nonzero(letters) < 16:
            continue
        rect = (px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2)
        if _has_letter_piece(image, rect) or np.count_nonzero(letters) >= 48:
            return rect, mask, letters
    return None


def _tighten_box(box: BubbleBox, *, pad: int = TIGHT_BOX_PAD) -> BubbleBox:
    """Trim empty zero-padding around a box tightly to its non-zero mask/letters pixels."""
    from dataclasses import replace

    h = box.y2 - box.y1
    w = box.x2 - box.x1
    if h <= 0 or w <= 0:
        return box
    support = None
    if box.mask is not None and box.mask.shape == (h, w) and np.any(box.mask > 0):
        support = box.mask > 0
    elif box.letters is not None and np.asarray(box.letters).shape == (h, w) and np.any(box.letters):
        support = np.asarray(box.letters, dtype=bool)
    if support is None:
        return box
    ys, xs = np.nonzero(support)
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(h, int(ys.max()) + 1 + pad)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(w, int(xs.max()) + 1 + pad)
    if r0 == 0 and r1 == h and c0 == 0 and c1 == w:
        return box
    rect = (box.x1 + c0, box.y1 + r0, box.x1 + c1, box.y1 + r1)
    sliced_mask = (
        np.ascontiguousarray(box.mask[r0:r1, c0:c1])
        if box.mask is not None and box.mask.shape == (h, w)
        else box.mask
    )
    sliced_letters = (
        np.ascontiguousarray(np.asarray(box.letters)[r0:r1, c0:c1])
        if box.letters is not None and np.asarray(box.letters).shape == (h, w)
        else box.letters
    )
    return replace(
        box,
        x1=rect[0],
        y1=rect[1],
        x2=rect[2],
        y2=rect[3],
        mask=sliced_mask,
        letters=sliced_letters,
        core=_clip_core(box.core, rect),
    )


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
        fill = area / max(1, w * h)
        if (
            RESCUE_WORD_MAX_W >= w >= 20
            and RESCUE_WORD_MAX_H >= h >= 12
            and w >= RESCUE_WORD_MIN_ASPECT * h
            and area >= RESCUE_WORD_MIN_AREA
            and fill >= RESCUE_WORD_MIN_FILL
        ):
            out.append(((x1 + x, y1 + y, x1 + x + w, y1 + y + h), labels[y:y + h, x:x + w] == i))
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
    def __init__(
        self,
        model_path,
        conf_threshold: float = knob("detect.confidence"),
        session=None,
        aux_session=None,
    ):
        if session is not None:
            self.session = session
        else:
            self.session = make_session(model_path)
            validate_kiuyha_session(self.session)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        height, width = inp.shape[2], inp.shape[3]
        self.fixed = (int(height), int(width)) if isinstance(height, int) and isinstance(width, int) else None
        self.conf_threshold = float(conf_threshold)
        self._metrics = threading.local()
        self._low_cache = threading.local()
        self._medium_aux_cache = threading.local()
        self.source_model = Path(str(model_path)).name
        self._aux_session = None
        self._aux_input_name = None
        self._aux_fixed = None
        if aux_session is not None:
            self._attach_aux_session(aux_session)
        elif session is None:
            aux_path = _resolve_aux_model_path(model_path)
            if aux_path is not None:
                try:
                    loaded_aux = make_session(aux_path)
                    validate_kiuyha_session(loaded_aux)
                    self._attach_aux_session(loaded_aux)
                except Exception:
                    self._aux_session = None

    def _attach_aux_session(self, aux_session) -> None:
        aux_inp = aux_session.get_inputs()[0]
        ah, aw = aux_inp.shape[2], aux_inp.shape[3]
        if isinstance(ah, int) and isinstance(aw, int):
            self._aux_session = aux_session
            self._aux_input_name = aux_inp.name
            self._aux_fixed = (int(ah), int(aw))

    def _run_raw_boxes(
        self,
        session,
        input_name: str,
        fixed: tuple[int, int] | None,
        image: np.ndarray,
        conf_threshold: float,
    ) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        in_h, in_w = fixed if fixed is not None else (
            int(math.ceil(h / STRIDE) * STRIDE),
            int(math.ceil(w / STRIDE) * STRIDE),
        )
        scale = min(in_w / w, in_h / h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        pad_x, pad_y = (in_w - rw) // 2, (in_h - rh) // 2
        canvas = np.full((in_h, in_w, 3), LETTERBOX_VALUE, np.uint8)
        resized = (
            image
            if (rw, rh) == (w, h)
            else cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        )
        canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        rows = _rows(session.run(None, {input_name: blob})[0], conf_threshold)
        boxes = []
        for x1, y1, x2, y2, score in rows:
            bx1 = int(max(0.0, (x1 - pad_x) / scale))
            by1 = int(max(0.0, (y1 - pad_y) / scale))
            bx2 = int(min(float(w), math.ceil((x2 - pad_x) / scale)))
            by2 = int(min(float(h), math.ceil((y2 - pad_y) / scale)))
            if bx2 > bx1 and by2 > by1:
                boxes.append((bx1, by1, bx2, by2, float(score)))
        return boxes

    def raw_boxes(self, image: np.ndarray, conf_threshold: float | None = None) -> list[tuple[int, int, int, int, float]]:
        return self._run_raw_boxes(
            self.session,
            self.input_name,
            self.fixed,
            image,
            self.conf_threshold if conf_threshold is None else conf_threshold,
        )

    def _call_raw_boxes(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        if conf_threshold is None or "raw_boxes" in self.__dict__:
            return self.raw_boxes(image)
        return self.raw_boxes(image, conf_threshold)

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

    def _overview_boxes(
        self,
        image: np.ndarray,
        existing: list[tuple[int, int, int, int, float]],
        medium_out: set[tuple[int, int, int, int]] | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        if self._aux_session is None or self._aux_fixed is None:
            return []
        h, w = image.shape[:2]
        if max(h, w) < OVERVIEW_MEDIUM_MIN_LONG or min(h, w) < OVERVIEW_MEDIUM_MIN_SHORT:
            return []
        size_h, size_w = self._aux_fixed
        canvas = np.full((size_h, size_w, 3), LETTERBOX_VALUE, np.uint8)
        scaled = []
        max_h = max(32, size_h - OVERVIEW_PAD_Y)
        for target_w in OVERVIEW_WIDTHS:
            scale = min(1.0, target_w / w, max_h / h)
            sw = max(1, int(round(w * scale)))
            sh = max(1, int(round(h * scale)))
            scaled.append((sw, sh, scale))
        total_w = sum(d[0] for d in scaled) + OVERVIEW_GAP * max(0, len(scaled) - 1)
        x = max(OVERVIEW_GAP, (size_w - total_w) // 2)
        columns = []
        for sw, sh, scale in scaled:
            if x + sw > size_w:
                break
            oy = (size_h - sh) // 2
            canvas[oy:oy + sh, x:x + sw] = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            columns.append((x, oy, sw, sh, scale))
            x += sw + OVERVIEW_GAP
        if not columns:
            return []
        raw = self._run_raw_boxes(
            self._aux_session,
            self._aux_input_name,
            self._aux_fixed,
            canvas,
            OVERVIEW_WHITE_BUBBLE_CONFIDENCE,
        )
        candidates: list[tuple[int, int, int, int, float, bool]] = []
        for cx1, cy1, cx2, cy2, score in raw:
            centre = (cx1 + cx2) / 2
            col = next((c for c in columns if c[0] <= centre < c[0] + c[2]), None)
            if col is None:
                continue
            ox, oy, sw, sh, scale = col
            bx1 = int(max(0, cx1 - ox) / scale)
            bx2 = int(min(w, math.ceil(min(sw, cx2 - ox) / scale)))
            by1 = int(max(0, cy1 - oy) / scale)
            by2 = int(min(h, math.ceil(min(sh, cy2 - oy) / scale)))
            bw, bh = bx2 - bx1, by2 - by1
            if bw <= 0 or bh <= 0:
                continue
            rect = (bx1, by1, bx2, by2)
            is_large = max(bw, bh) >= OVERVIEW_MIN_LONG and min(bw, bh) >= OVERVIEW_MIN_SHORT
            if is_large:
                if score < OVERVIEW_CONFIDENCE and not _is_white_bubble_display(image, rect):
                    continue
                candidates.append((bx1, by1, bx2, by2, float(score), False))
            elif (
                score >= OVERVIEW_MEDIUM_CONFIDENCE
                and max(bw, bh) >= OVERVIEW_MEDIUM_MIN_LONG
                and min(bw, bh) >= OVERVIEW_MEDIUM_MIN_SHORT
                and (
                    _dark_pieces(image, rect) >= MASKLESS_MIN_PIECES
                    or _bright_pieces(image, rect) >= MASKLESS_MIN_PIECES
                )
            ):
                candidates.append((bx1, by1, bx2, by2, float(score), True))
        candidates.sort(key=lambda b: b[4], reverse=True)
        kept: list[tuple[int, int, int, int, float]] = []
        for bx1, by1, bx2, by2, score, is_medium in candidates:
            box = (bx1, by1, bx2, by2, score)
            bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
            area = max(1, bw * bh)
            if any(
                max(0, min(bx2, k[2]) - max(bx1, k[0])) * max(0, min(by2, k[3]) - max(by1, k[1]))
                >= OVERVIEW_SELF_OVERLAP * min(area, max(1, (k[2] - k[0]) * (k[3] - k[1])))
                for k in kept
            ):
                continue
            overlapping = [
                ex for ex in existing
                if min(bx2, ex[2]) > max(bx1, ex[0]) and min(by2, ex[3]) > max(by1, ex[1])
            ]
            if any(
                max(0, min(bx2, ex[2]) - max(bx1, ex[0])) * max(0, min(by2, ex[3]) - max(by1, ex[1]))
                >= OVERVIEW_EXISTING_OVERLAP * area
                for ex in overlapping
            ):
                can_supersede = (
                    not is_medium
                    and score >= max(OVERVIEW_CONFIDENCE, max(ex[4] for ex in overlapping) + 0.10)
                    and area >= 1.25 * max(max(1, (ex[2] - ex[0]) * (ex[3] - ex[1])) for ex in overlapping)
                    and (
                        len(overlapping) == 1
                        or all(
                            _box_iou(overlapping[i], overlapping[j]) >= 0.15
                            for i in range(len(overlapping))
                            for j in range(i + 1, len(overlapping))
                        )
                    )
                )
                if can_supersede:
                    gw, gh = min(bw, 128), min(bh, 128)
                    sx, sy = gw / bw, gh / bh
                    cov_grid = np.zeros((gh, gw), dtype=bool)
                    for ex in overlapping:
                        cx1 = max(0, int(math.floor((max(bx1, ex[0]) - bx1) * sx)))
                        cy1 = max(0, int(math.floor((max(by1, ex[1]) - by1) * sy)))
                        cx2 = min(gw, int(math.ceil((min(bx2, ex[2]) - bx1) * sx)))
                        cy2 = min(gh, int(math.ceil((min(by2, ex[3]) - by1) * sy)))
                        if cx2 > cx1 and cy2 > cy1:
                            cov_grid[cy1:cy2, cx1:cx2] = True
                    can_supersede = float(cov_grid.mean()) < 0.78
                if not can_supersede:
                    continue
            kept.append(box)
            if is_medium and medium_out is not None:
                medium_out.add((bx1, by1, bx2, by2))
        return kept

    def _native_aux_boxes(
        self,
        image: np.ndarray,
        existing: list[tuple[int, int, int, int, float]],
        medium_out: set[tuple[int, int, int, int]] | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        """Tile the slice at native 1.0x resolution with the aux detector to catch small/stylized text."""
        if self._aux_session is None or self._aux_fixed is None:
            return []
        h, w = image.shape[:2]
        if min(h, w) < 32:
            return []
        size_h, size_w = self._aux_fixed
        tile_w = min(w, size_w)
        tile_h = min(h, size_h)
        step_x = max(1, int(tile_w * 0.80))
        step_y = max(1, int(tile_h * 0.80))
        xs = [0] if w <= tile_w else list(range(0, w - tile_w, step_x)) + [w - tile_w]
        ys = [0] if h <= tile_h else list(range(0, h - tile_h, step_y)) + [h - tile_h]
        tiles = [(x0, y0, tile_w, tile_h) for y0 in ys for x0 in xs]

        raw_found: list[tuple[int, int, int, int, float]] = []
        # Pack multiple narrow tiles side-by-side when 2+ tiles fit inside one canvas width.
        per_canvas = max(1, (size_w + COLUMN_GAP) // (tile_w + COLUMN_GAP)) if tile_w * 2 + COLUMN_GAP <= size_w else 1
        for first in range(0, len(tiles), per_canvas):
            batch = tiles[first:first + per_canvas]
            if len(batch) == 1:
                x0, y0, tw, th = batch[0]
                crop = image[y0:y0 + th, x0:x0 + tw]
                for bx1, by1, bx2, by2, score in self._run_raw_boxes(
                    self._aux_session, self._aux_input_name, self._aux_fixed, crop, NATIVE_AUX_CONFIDENCE
                ):
                    if bx2 - bx1 > 4 and by2 - by1 > 4:
                        raw_found.append((x0 + bx1, y0 + by1, x0 + bx2, y0 + by2, score))
            else:
                canvas = np.full((size_h, size_w, 3), LETTERBOX_VALUE, np.uint8)
                slots = []
                cx = 0
                for x0, y0, tw, th in batch:
                    canvas[:th, cx:cx + tw] = image[y0:y0 + th, x0:x0 + tw]
                    slots.append((cx, x0, y0, tw, th))
                    cx += tw + COLUMN_GAP
                for rx1, ry1, rx2, ry2, score in self._run_raw_boxes(
                    self._aux_session, self._aux_input_name, self._aux_fixed, canvas, NATIVE_AUX_CONFIDENCE
                ):
                    mid_x = (rx1 + rx2) / 2.0
                    slot = next((s for s in slots if s[0] <= mid_x < s[0] + s[3]), None)
                    if slot is None:
                        continue
                    sx, x0, y0, tw, th = slot
                    bx1 = x0 + max(0, rx1 - sx)
                    bx2 = x0 + min(tw, rx2 - sx)
                    by1 = y0 + max(0, ry1)
                    by2 = y0 + min(th, ry2)
                    if bx2 - bx1 > 4 and by2 - by1 > 4:
                        raw_found.append((bx1, by1, bx2, by2, score))

        kept: list[tuple[int, int, int, int, float]] = []
        for bx1, by1, bx2, by2, score in _dedupe_raw_boxes(raw_found):
            if any(_box_iou((bx1, by1, bx2, by2), ex) >= 0.60 for ex in existing):
                continue
            area = max(1, (bx2 - bx1) * (by2 - by1))
            if any(
                max(0, min(bx2, ex[2]) - max(bx1, ex[0])) * max(0, min(by2, ex[3]) - max(by1, ex[1]))
                >= 0.72 * area
                for ex in existing
            ):
                continue
            kept.append((bx1, by1, bx2, by2, score))
            if medium_out is not None:
                medium_out.add((bx1, by1, bx2, by2))
        return kept

    def _packed_rotated_aux_boxes(
        self,
        image: np.ndarray,
        existing: list[tuple[int, int, int, int, float]],
        medium_out: set[tuple[int, int, int, int]] | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        """Check 90-degree CW & CCW orientations in a single packed aux canvas even when upright boxes exist."""
        if self._aux_session is None or self._aux_fixed is None:
            return []
        h, w = image.shape[:2]
        if min(h, w) < 64 or max(h, w) > 5000:
            return []
        size_h, size_w = self._aux_fixed
        slot_h = (size_h - COLUMN_GAP) // 2
        rot_w, rot_h = h, w
        scale = min(1.0, size_w / rot_w, slot_h / rot_h)
        rw, rh = max(1, int(round(rot_w * scale))), max(1, int(round(rot_h * scale)))
        if min(rw, rh) < 32:
            return []
        canvas = np.full((size_h, size_w, 3), LETTERBOX_VALUE, np.uint8)
        ox = (size_w - rw) // 2
        slots = []
        for idx, (code, clockwise) in enumerate(((cv2.ROTATE_90_CLOCKWISE, True), (cv2.ROTATE_90_COUNTERCLOCKWISE, False))):
            turned = cv2.rotate(image, code)
            oy = idx * (rh + COLUMN_GAP)
            canvas[oy:oy + rh, ox:ox + rw] = cv2.resize(turned, (rw, rh), interpolation=cv2.INTER_AREA)
            slots.append((ox, oy, rw, rh, scale, clockwise))

        raw = self._run_raw_boxes(
            self._aux_session, self._aux_input_name, self._aux_fixed, canvas, ROTATED_AUX_MIN_SCORE
        )
        kept: list[tuple[int, int, int, int, float]] = []
        for cx1, cy1, cx2, cy2, score in raw:
            mid_y = (cy1 + cy2) / 2.0
            slot = next((s for s in slots if s[1] <= mid_y < s[1] + s[3]), None)
            if slot is None:
                continue
            sx, sy, sw, sh, sc, clockwise = slot
            tx1 = int(max(0, cx1 - sx) / sc)
            tx2 = int(min(rot_w, math.ceil(min(sw, cx2 - sx) / sc)))
            ty1 = int(max(0, cy1 - sy) / sc)
            ty2 = int(min(rot_h, math.ceil(min(sh, cy2 - sy) / sc)))
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            if clockwise:
                bx1, by1, bx2, by2 = ty1, h - tx2, ty2, h - tx1
            else:
                bx1, by1, bx2, by2 = w - ty2, tx1, w - ty1, tx2
            bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(w, bx2), min(h, by2)
            if bx2 - bx1 <= 8 or by2 - by1 <= 8:
                continue
            area = max(1, (bx2 - bx1) * (by2 - by1))
            if any(
                max(0, min(bx2, ex[2]) - max(bx1, ex[0])) * max(0, min(by2, ex[3]) - max(by1, ex[1]))
                >= 0.35 * area
                for ex in existing
            ):
                continue
            kept.append((bx1, by1, bx2, by2, score))
            if medium_out is not None:
                medium_out.add((bx1, by1, bx2, by2))
        return kept

    def _raw_slice_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        low: list[tuple[int, int, int, int, float]] = []
        medium_aux: set[tuple[int, int, int, int]] = set()
        found = self._coarse_boxes(image, low_out=low)
        if self.fixed is not None:
            found = found + self.page_boxes(image, low_out=low)
        found = found + self._overview_boxes(image, found, medium_out=medium_aux)
        found = found + self._native_aux_boxes(image, found, medium_out=medium_aux)
        found = found + self._packed_rotated_aux_boxes(image, found, medium_out=medium_aux)
        if not found:
            found = self._rotated_boxes(image)
        self._low_cache.value = low
        self._medium_aux_cache.value = medium_aux
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
                    box = (y1, h - x2, y2, h - x1)
                else:
                    box = (w - y2, x1, w - y1, x2)
                bx1, by1, bx2, by2 = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
                if bx2 > bx1 and by2 > by1:
                    found.append((bx1, by1, bx2, by2, score))
        return found

    def _coarse_boxes(
        self,
        image: np.ndarray,
        low_out: list[tuple[int, int, int, int, float]] | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        if self.fixed is not None and self.fixed[0] == self.fixed[1]:
            size = self.fixed[0]
            base_scale = min(1.0, (size - COLUMN_GAP) / max(1.0, 2.0 * w))
            max_chunk_h = max(size * 2, int(2.0 * size / max(base_scale, 0.25)) - HALVES_MIN_OVERLAP)
            if h > max_chunk_h:
                step = max(1, max_chunk_h - HALVES_MIN_OVERLAP)
                starts = list(range(0, h - max_chunk_h, step)) + [h - max_chunk_h]
                found: list[tuple[int, int, int, int, float]] = []
                for y_off in starts:
                    chunk_low: list[tuple[int, int, int, int, float]] | None = [] if low_out is not None else None
                    for bx1, by1, bx2, by2, score in self._coarse_boxes(
                        image[y_off:y_off + max_chunk_h], low_out=chunk_low
                    ):
                        found.append((bx1, by1 + y_off, bx2, by2 + y_off, score))
                    if low_out is not None and chunk_low:
                        low_out.extend(
                            (bx1, by1 + y_off, bx2, by2 + y_off, score)
                            for bx1, by1, bx2, by2, score in chunk_low
                        )
                return found
        plan = self.halves_plan(h, w)
        if (
            plan is None
            and self.fixed is not None
            and self.fixed[0] == self.fixed[1]
            and w > int(1.3 * h)
            and w > self.fixed[0]
        ):
            size = self.fixed[0]
            half_w = min(w, max(size, (w + HALVES_MIN_OVERLAP + 1) // 2))
            spread_found: list[tuple[int, int, int, int, float]] = []
            for x_off in (0, w - half_w):
                chunk_low: list[tuple[int, int, int, int, float]] | None = [] if low_out is not None else None
                for bx1, by1, bx2, by2, score in self._coarse_boxes(
                    image[:, x_off:x_off + half_w], low_out=chunk_low
                ):
                    spread_found.append((bx1 + x_off, by1, bx2 + x_off, by2, score))
                if low_out is not None and chunk_low:
                    low_out.extend(
                        (bx1 + x_off, by1, bx2 + x_off, by2, score)
                        for bx1, by1, bx2, by2, score in chunk_low
                    )
            return spread_found
        min_conf = min(self.conf_threshold, RESCUE_MIN_SCORE) if low_out is not None else None
        if plan is None:
            raw = self._call_raw_boxes(image, min_conf)
            if low_out is None:
                return raw
            found = []
            for box in raw:
                if box[4] >= self.conf_threshold:
                    found.append(box)
                elif box[4] >= RESCUE_MIN_SCORE:
                    low_out.append(box)
            return found
        scale, halves = plan
        size = self.fixed[0]
        canvas = np.full((size, size, 3), LETTERBOX_VALUE, np.uint8)
        columns, x = [], 0
        for y0, y1 in halves:
            rw, rh = min(size - x, int(w * scale)), min(size, int((y1 - y0) * scale))
            canvas[:rh, x:x + rw] = cv2.resize(
                image[y0:y1], (rw, rh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            )
            columns.append((x, rw, rh, y0, y1))
            x += rw + COLUMN_GAP
        found = []
        for cx1, cy1, cx2, cy2, score in self._call_raw_boxes(canvas, min_conf):
            px, rw, rh, y0, y1 = columns[0] if (cx1 + cx2) / 2 < columns[1][0] else columns[1]
            sx, sy = rw / w, rh / (y1 - y0)
            bx1, bx2 = int((max(cx1, px) - px) / sx), int(math.ceil((min(cx2, px + rw) - px) / sx))
            by1, by2 = int(min(cy1, rh) / sy) + y0, int(math.ceil(min(cy2, rh) / sy)) + y0
            if bx2 - bx1 > 4 and by2 - by1 > 4:
                if score >= self.conf_threshold:
                    found.append((bx1, by1, bx2, by2, score))
                elif low_out is not None and score >= RESCUE_MIN_SCORE:
                    low_out.append((bx1, by1, bx2, by2, score))
        return found

    def page_boxes(
        self,
        image: np.ndarray,
        low_out: list[tuple[int, int, int, int, float]] | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        h, w = image.shape[:2]
        min_conf = min(self.conf_threshold, RESCUE_MIN_SCORE) if low_out is not None else None
        if self.fixed is None:
            raw = self._call_raw_boxes(image, min_conf)
            if low_out is None:
                return raw
            found = []
            for box in raw:
                if box[4] >= self.conf_threshold:
                    found.append(box)
                elif box[4] >= RESCUE_MIN_SCORE:
                    low_out.append(box)
            return found
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
            batch = starts[first:first + per_canvas]
            columns = []
            for k, y0 in enumerate(batch):
                part = image[y0:y0 + window]
                rh = min(size_h, int(round(part.shape[0] * scale)))
                x = k * (sw + COLUMN_GAP)
                canvas[:rh, x:x + sw] = cv2.resize(part, (sw, rh), interpolation=cv2.INTER_AREA)
                columns.append((x, rh, y0, part.shape[0]))
            overview_col = None
            if (
                first + per_canvas >= len(starts)
                and self._aux_session is None
                and h > size_h
                and max(h, w) >= OVERVIEW_MIN_LONG
                and min(h, w) >= OVERVIEW_MIN_SHORT
            ):
                used_w = len(batch) * sw + max(0, len(batch) - 1) * COLUMN_GAP
                ov_scale = min(1.0, 225 / w, max(32, size_h - OVERVIEW_PAD_Y) / h)
                ov_sw = max(1, int(round(w * ov_scale)))
                ov_sh = max(1, int(round(h * ov_scale)))
                if used_w + COLUMN_GAP + ov_sw <= size_w:
                    ov_x = used_w + COLUMN_GAP
                    ov_y = (size_h - ov_sh) // 2
                    canvas[ov_y:ov_y + ov_sh, ov_x:ov_x + ov_sw] = cv2.resize(
                        image, (ov_sw, ov_sh), interpolation=cv2.INTER_AREA
                    )
                    overview_col = (ov_x, ov_y, ov_sw, ov_sh, ov_scale)
            for cx1, cy1, cx2, cy2, score in self._call_raw_boxes(canvas, min_conf):
                centre = (cx1 + cx2) / 2
                column = next((c for c in columns if c[0] <= centre < c[0] + sw), None)
                if column is not None:
                    x, rh, y0, part_h = column
                    bx1, bx2 = int(max(0, cx1 - x) / scale), int(math.ceil(min(sw, cx2 - x) / scale))
                    by1, by2 = int(max(0, cy1) / scale), int(math.ceil(min(rh, cy2) / scale))
                    if bx2 - bx1 > 4 and by2 - by1 > 4:
                        mapped = (min(w, bx1), by1 + y0, min(w, bx2), min(h, by2 + y0), score)
                        if score >= PAGE_CONFIDENCE:
                            found.append(mapped)
                        elif low_out is not None and score >= RESCUE_MIN_SCORE:
                            low_out.append(mapped)
                elif overview_col is not None:
                    ov_x, ov_y, ov_sw, ov_sh, ov_scale = overview_col
                    if ov_x <= centre < ov_x + ov_sw:
                        bx1 = int(max(0, cx1 - ov_x) / ov_scale)
                        bx2 = int(min(w, math.ceil(min(ov_sw, cx2 - ov_x) / ov_scale)))
                        by1 = int(max(0, cy1 - ov_y) / ov_scale)
                        by2 = int(min(h, math.ceil(min(ov_sh, cy2 - ov_y) / ov_scale)))
                        bw, bh = bx2 - bx1, by2 - by1
                        if (
                            max(bw, bh) >= OVERVIEW_MIN_LONG
                            and min(bw, bh) >= OVERVIEW_MIN_SHORT
                            and (
                                score >= PAGE_CONFIDENCE
                                or (
                                    score >= OVERVIEW_WHITE_BUBBLE_CONFIDENCE
                                    and _is_white_bubble_display(image, (bx1, by1, bx2, by2))
                                )
                            )
                        ):
                            found.append((bx1, by1, bx2, by2, float(score)))
        return found

    def _rescue_boxes(self, image: np.ndarray, kept: list[BubbleBox]) -> list[BubbleBox]:
        h, w = image.shape[:2]
        cov = _mask_coverage(image.shape[:2], kept)
        dilated = cv2.dilate(cov.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        cached_low = getattr(self._low_cache, "value", None)
        self._low_cache.value = None
        if cached_low is not None:
            low = _dedupe_raw_boxes(cached_low)
        else:
            low = _dedupe_raw_boxes(
                [box for box in self.raw_boxes(image, RESCUE_MIN_SCORE) if box[4] < self.conf_threshold]
            )
        found: list[tuple[BubbleBox, bool]] = []

        kept_rects = [(b.x1, b.y1, b.x2, b.y2) for b in kept]

        def consider(rect, pixels, score, emit=None, welded=False):
            x1, y1, x2, y2 = rect
            rw, rh = x2 - x1, y2 - y1
            if rw < 15 or rh < 10 or not pixels.any():
                return
            rect_area = max(1, rw * rh)
            if any(
                max(0, min(x2, kx2) - max(x1, kx1)) * max(0, min(y2, ky2) - max(y1, ky1)) >= 0.60 * rect_area
                for kx1, ky1, kx2, ky2 in kept_rects
            ):
                return
            if np.count_nonzero(pixels & dilated[y1:y2, x1:x2]) >= 0.5 * np.count_nonzero(pixels):
                return
            cx1, cy1 = max(0, x1 - 8), max(0, y1 - 8)
            cx2, cy2 = min(w, x2 + 8), min(h, y2 + 8)
            raw_crop = image[cy1:cy2, cx1:cx2]
            raw_gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY) if raw_crop.ndim == 3 else raw_crop
            if raw_gray.size and float(np.percentile(raw_gray, 95) - np.percentile(raw_gray, 5)) < RESCUE_MIN_CONTRAST:
                return
            if not welded:
                _, _, pstats, _ = cv2.connectedComponentsWithStats(pixels.astype(np.uint8), connectivity=8)
                n_parts = int(np.count_nonzero(pstats[1:, cv2.CC_STAT_AREA] >= 6))
                if n_parts < 2 and not (rw >= RESCUE_WORD_MIN_ASPECT * rh and score >= self.conf_threshold):
                    return
                if score < self.conf_threshold:
                    ux1, uy1 = max(0, x1 - 8), max(0, y1 - 8)
                    ux2, uy2 = min(w, x2 + 8), min(h, y2 + 8)
                    unstretched_peak = float(probability(image[uy1:uy2, ux1:ux2]).max())
                    if unstretched_peak < RESCUE_UNSTRETCHED_CTD_MIN:
                        if not (unstretched_peak >= 0.18 and n_parts >= 2 and np.count_nonzero(pixels) >= 36):
                            return
            ex1, ey1, ex2, ey2 = emit or rect
            letters = np.zeros((ey2 - ey1, ex2 - ex1), bool)
            letters[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1] = pixels
            mask = expand_local_ink(letters)
            if not fills_rectangle(mask):
                box = _tighten_box(_text_box(
                    ex1, ey1, ex2, ey2, score, mask, self.source_model, letters, mask_source=LOCAL_INK_MASK,
                ))
                found.append((box, welded))

        for x1, y1, x2, y2, score in sorted(low, key=lambda b: (b[1], b[0])):
            if float(dilated[y1:y2, x1:x2].mean()) >= 0.85:
                continue
            dark_evidence = _dark_pieces(image, (x1, y1, x2, y2)) >= MASKLESS_MIN_PIECES
            bright_evidence = False
            if not dark_evidence and score >= 0.18:
                crop_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image[y1:y2, x1:x2]
                if crop_gray.size and float(np.median(crop_gray)) < 110.0:
                    bright_evidence = _bright_pieces(image, (x1, y1, x2, y2)) >= MASKLESS_MIN_PIECES
            piece_evidence = dark_evidence or bright_evidence
            if score < self.conf_threshold and not piece_evidence:
                continue
            pad = max(64, int(0.5 * max(x2 - x1, y2 - y1)))
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
            raw_patch = image[py1:py2, px1:px2]
            probe_crops = [_stretch(raw_patch)]
            if bright_evidence and raw_patch.ndim == 3:
                probe_crops.append(255 - raw_patch)
            best_letters = None
            best_coords = (0, 0, 0, 0)
            for probe in probe_crops:
                (lsx1, lsy1, lsx2, lsy2), _, cand_letters = letter_mask(
                    probe,
                    (x1 - px1, y1 - py1, x2 - px1, y2 - py1),
                    with_letters=True,
                )
                if cand_letters is not None and cand_letters.any():
                    best_letters = cand_letters
                    best_coords = (px1 + lsx1, py1 + lsy1, px1 + lsx2, py1 + lsy2)
                    break
            sx1, sy1, sx2, sy2 = best_coords
            letters = best_letters
            if letters is not None and letters.any():
                uncov = letters & ~cov[sy1:sy2, sx1:sx2]
                max_h = 120 if score >= self.conf_threshold else 64
                max_w = 480 if score >= self.conf_threshold else 280
                for (wx1, wy1, wx2, wy2), pixels in _word_blobs(uncov, max_h=max_h, max_w=max_w):
                    consider((sx1 + wx1, sy1 + wy1, sx1 + wx2, sy1 + wy2), pixels, score)
            if not dark_evidence:
                continue
            for (wx1, wy1, wx2, wy2), pixels in _welded_words(image, (x1, y1, x2, y2)):
                consider((wx1, wy1, wx2, wy2), pixels, score,
                         emit=(max(0, wx1 - RESCUE_WORD_PAD), max(0, wy1 - RESCUE_WORD_PAD),
                               min(w, wx2 + RESCUE_WORD_PAD), min(h, wy2 + RESCUE_WORD_PAD)),
                         welded=True)
        merged: list[tuple[BubbleBox, bool]] = []
        for box, welded in sorted(found, key=lambda item: (item[0].y1, item[0].x1)):
            for i, (other, other_welded) in enumerate(merged):
                ix = max(0, min(box.x2, other.x2) - max(box.x1, other.x1))
                iy = max(0, min(box.y2, other.y2) - max(box.y1, other.y1))
                if ix * iy >= 0.5 * min(box.mask.size, other.mask.size):
                    merged[i] = (_merge_mask_evidence([other, box]), welded or other_welded)
                    break
                oy = max(0, min(box.y2, other.y2) - max(box.y1, other.y1))
                gap = max(other.x1 - box.x2, box.x1 - other.x2)
                min_h = min(box.y2 - box.y1, other.y2 - other.y1)
                if oy >= 0.6 * min_h and 0 <= gap <= 0.5 * min_h:
                    merged[i] = (_merge_mask_evidence([other, box]), welded or other_welded)
                    break
            else:
                merged.append((box, welded))
        return [
            box for box, welded in merged
            if not fills_rectangle(box.mask)
            and _has_letter_piece(
                image,
                (max(0, box.x1 - 8), max(0, box.y1 - 8), min(w, box.x2 + 8), min(h, box.y2 + 8)),
                allow_wide=welded,
            )
        ]

    def text_boxes(self, image: np.ndarray) -> list[BubbleBox]:
        self._low_cache.value = None
        self._medium_aux_cache.value = set()
        raw_boxes = self._raw_slice_boxes(image)
        medium_aux = getattr(self._medium_aux_cache, "value", None) or set()
        self._medium_aux_cache.value = set()
        boxes = []
        cores = []
        proposals = []
        paper_boxes = []

        def add_box(box: BubbleBox) -> None:
            tight = _tighten_box(box)
            if tight.mask_source == PAPER_INK_MASK:
                paper_boxes.append(tight)
            else:
                boxes.append(tight)
                cores.append(tight.core)

        def defer_box(core, score, reason):
            proposals.append(BubbleBox(
                *core, score, source_model=self.source_model, class_name="text",
                source_role="text_proposal", needs_review=True, deferred_reason=reason,
            ))

        def add_segmented(rect, score, mask, letters, core):
            if letters is None or not letters.any():
                defer_box(core, score, "missing_glyph_mask")
                return
            mask, letters = mask > 0, letters > 0
            source = "text_segmenter"
            if mask.size >= LARGE_TEXT_MASK_AREA:
                x1, y1, x2, y2 = rect
                gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                paper = isolated_paper_ink(gray, mask, letters)
                if paper is not None:
                    mask, letters = paper
                    source = PAPER_INK_MASK
                else:
                    bounded = protect_free_text_art(gray, mask, letters)
                    if bounded is not None:
                        mask, letters = bounded
                        source = CONTRAST_ARTWORK_MASK
                if not mask.any():
                    defer_box(core, score, "possible_artwork")
                    return
            add_box(_text_box(
                *rect, score, mask.astype(np.uint8) * 255, self.source_model, letters,
                core=core, mask_source=source,
            ))

        for x1, y1, x2, y2, score in _dedupe_raw_boxes(raw_boxes):
            core = (x1, y1, x2, y2)
            is_medium_aux = core in medium_aux
            bw, bh = x2 - x1, y2 - y1
            if (
                max(bw, bh) >= OVERVIEW_MIN_LONG
                and min(bw, bh) >= OVERVIEW_MIN_SHORT
                and _is_white_bubble_display(image, core)
            ):
                protected = _white_bubble_display_mask(image, core)
                if protected is not None:
                    (px1, py1, px2, py2), pmask, pletters = protected
                    add_box(_text_box(
                        px1, py1, px2, py2, score, pmask, self.source_model, pletters,
                        core=core, mask_source="border_protected_display",
                    ))
                    continue
            (x1, y1, x2, y2), mask, letters = letter_mask(image, core, with_letters=True)
            if not mask.size:
                continue
            coverage = np.count_nonzero(mask) / mask.size
            dense = fills_rectangle(mask)
            if coverage >= MASKLESS_MIN_COVERAGE and not dense:
                if is_medium_aux and score < NATIVE_AUX_STRICT_CONF and not _has_strong_aux_letter_evidence(letters, mask.size):
                    continue
                if not _passes_glyph_plausibility(image.shape[:2], (x1, y1, x2, y2), mask, letters, score, is_aux=is_medium_aux):
                    continue
                add_segmented((x1, y1, x2, y2), score, mask, letters, core)
                continue
            if (
                (not is_medium_aux and score >= ENHANCED_CTD_MIN_SCORE)
                or (is_medium_aux and score >= ENHANCED_CTD_AUX_MIN_SCORE)
            ):
                enhanced = _enhanced_letter_mask(image, core)
                if enhanced is not None:
                    rect, enhanced_mask, enhanced_letters = enhanced
                    if _passes_glyph_plausibility(image.shape[:2], rect, enhanced_mask, enhanced_letters, score, is_aux=is_medium_aux):
                        add_segmented(rect, score, enhanced_mask, enhanced_letters, core)
                    continue
            if is_medium_aux or score < MASKLESS_MIN_SCORE:
                continue

            crop = image[y1:y2, x1:x2]
            if not dense or letters is None or not 8 <= np.count_nonzero(letters) <= 0.45 * mask.size:
                letters = flat_background_ink(crop)
            local_mask = expand_local_ink(letters) if letters is not None else None
            if local_mask is not None and not fills_rectangle(local_mask):
                add_box(_text_box(
                    x1, y1, x2, y2, score, local_mask, self.source_model, letters, core=core, mask_source=LOCAL_INK_MASK,
                ))
                continue

            rect = (x1, y1, x2, y2)
            small = mask.size <= MASKLESS_SMALL_BOX_AREA
            if not small and not (
                _dark_pieces(image, rect) >= MASKLESS_MIN_PIECES
                or _bright_pieces(image, rect) >= MASKLESS_MIN_PIECES
            ):
                continue
            artwork = None if small else chromatic_stroke_mask(crop)
            if artwork is not None:
                artwork_mask, letters = artwork
                add_box(_text_box(
                    x1, y1, x2, y2, score, artwork_mask, self.source_model, letters,
                    core=core, mask_source=CONTRAST_ARTWORK_MASK,
                ))
                continue
            defer_box(core, score, "missing_glyph_mask")
        kept = [
            _tighten_box(part)
            for box in _join_grown(boxes, cores, image=image)
            for piece in _split_blocks(box, image=image)
            for part in _split_sideways(piece, image=image)
        ]
        kept.extend(paper_boxes)
        final = kept + self._rescue_boxes(image, kept)
        _attach_bubble_masks(image, final)
        return final + apply_final_nms(proposals)

    def leftover_boxes(self, clean: np.ndarray, targets) -> list[BubbleBox]:
        boxes = []
        for t in targets:
            source = getattr(t, "mask_source", "")
            if source in PRECISE_INK_MASK_SOURCES or source == "border_protected_display":
                continue
            if still_reads(clean, (t.x1, t.y1, t.x2, t.y2)):
                (x1, y1, x2, y2), mask = letter_mask(clean, (t.x1, t.y1, t.x2, t.y2))
                if not mask.any():
                    continue
                target_mask = getattr(t, "mask", None)
                target_h, target_w = t.y2 - t.y1, t.x2 - t.x1
                if (
                    target_mask is not None
                    and target_mask.shape == (target_h, target_w)
                    and np.any(target_mask > 0)
                ):
                    support = cv2.dilate(
                        (target_mask > 0).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    ) > 0
                    limited = np.zeros_like(mask, dtype=bool)
                    ix1, iy1 = max(x1, t.x1), max(y1, t.y1)
                    ix2, iy2 = min(x2, t.x2), min(y2, t.y2)
                    if ix2 > ix1 and iy2 > iy1:
                        limited[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = (
                            mask[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]
                            & support[iy1 - t.y1:iy2 - t.y1, ix1 - t.x1:ix2 - t.x1]
                        )
                    mask = limited
                if mask.any():
                    rows, cols = np.nonzero(mask)
                    mx1, my1 = int(cols.min()), int(rows.min())
                    mx2, my2 = int(cols.max()) + 1, int(rows.max()) + 1
                    boxes.append(_text_box(
                        x1 + mx1,
                        y1 + my1,
                        x1 + mx2,
                        y1 + my2,
                        t.confidence,
                        mask[my1:my2, mx1:mx2].astype(np.uint8) * 255,
                        self.source_model,
                    ))
        return boxes

    def detect(self, image: np.ndarray) -> list[BubbleBox]:
        started = time.perf_counter()
        boxes = self.text_boxes(image)
        self._metrics.value = {"text_model_ms": (time.perf_counter() - started) * 1000.0, "result_boxes": len(boxes)}
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._metrics, "value", {}) or {})
