"""Detected text boxes and the NMS that merges overlapping ones."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.detector.bubble_mask import MASK_ROWS, merge_bubble_masks
from app.parameters import DETECTOR_FINAL_NMS_IOU, DETECTOR_NMS_SCORE_FLOOR

SEPARATE_LOBE_MAX_SHARED = 0.10


def _mask_spans(mask: dict) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """A stored mask as (y1, y2, [(x1, x2), ...]) spans in slice coords.

    Rows are bbox-local half-open x-intervals, 24 bands over the bbox
    height with integer band edges (see app/detector/bubble_mask).
    """
    x1, y1, _x2, y2 = (int(v) for v in mask["bbox"])
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


def _spans_area(spans: list[tuple[int, int, list[tuple[int, int]]]]) -> int:
    return sum((y2 - y1) * sum(b - a for a, b in xs) for y1, y2, xs in spans)


def _intervals_overlap(xs: list[tuple[int, int]], ys: list[tuple[int, int]]) -> int:
    """Shared width of two sorted, non-overlapping interval lists."""
    shared = 0
    j = 0
    for a, b in xs:
        while j < len(ys) and ys[j][1] <= a:
            j += 1
        k = j
        while k < len(ys) and ys[k][0] < b:
            shared += max(0, min(b, ys[k][1]) - max(a, ys[k][0]))
            k += 1
    return shared


def _spans_overlap_area(
    a: list[tuple[int, int, list[tuple[int, int]]]],
    b: list[tuple[int, int, list[tuple[int, int]]]],
) -> int:
    shared = 0
    for ay1, ay2, axs in a:
        for by1, by2, bxs in b:
            overlap_y = min(ay2, by2) - max(ay1, by1)
            if overlap_y > 0:
                shared += overlap_y * _intervals_overlap(axs, bxs)
    return shared


def bubble_masks_are_separate_lobes(a: dict | None, b: dict | None) -> bool:
    """True when two masks are different lobes of one waist, not one bubble.

    Two pieces of a waist split hold disjoint interiors; two boxes in the
    same bubble hold the same interior twice. Only a closed outline counts:
    a rect mask is the box itself, so two fragments of one free-text block
    would read as disjoint and must never veto.
    """
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    if a.get("kind") != "bubble" or b.get("kind") != "bubble":
        return False
    spans_a, spans_b = _mask_spans(a), _mask_spans(b)
    if not spans_a or not spans_b:
        return False
    smaller = min(_spans_area(spans_a), _spans_area(spans_b))
    if smaller <= 0:
        return False
    return _spans_overlap_area(spans_a, spans_b) <= SEPARATE_LOBE_MAX_SHARED * smaller


@dataclass
class BubbleBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    mask: np.ndarray | None = None
    source_model: str = "unknown"
    class_id: int = 0
    class_name: str = "unknown"
    semantic_type: str = "unknown"
    mask_source: str = "none"
    safe_to_inpaint: bool = False
    ocr_eligible: bool = False
    needs_review: bool = False
    source_role: str = "unknown"
    deferred_reason: str | None = None
    verify_region_only: bool = False
    letters: np.ndarray | None = field(default=None, compare=False)
    bubble_mask: dict | None = field(default=None, compare=False)
    core: tuple[int, int, int, int] | None = None

    @property
    def verified_mask(self) -> bool:
        h = self.y2 - self.y1
        w = self.x2 - self.x1
        return (
            self.mask is not None
            and self.mask.ndim == 2
            and self.mask.shape == (h, w)
            and bool(np.any(self.mask > 0))
        )


def box_iou(a: BubbleBox, b: BubbleBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    return intersection / float(max(1, area_a + area_b - intersection))


def _merge_mask_evidence(boxes: list[BubbleBox]) -> BubbleBox:
    """One box whose mask is the union of every overlapping box's mask."""
    base = max(boxes, key=lambda box: float(box.confidence))
    evidence = [box for box in boxes if box.verified_mask]
    if not evidence:
        x1, y1 = min(b.x1 for b in boxes), min(b.y1 for b in boxes)
        x2, y2 = max(b.x2 for b in boxes), max(b.y2 for b in boxes)
        if (x1, y1, x2, y2) == (base.x1, base.y1, base.x2, base.y2):
            return replace(base)
        cored = [box.core for box in boxes if getattr(box, "core", None) is not None]
        merged_core = None
        if cored:
            merged_core = (min(c[0] for c in cored), min(c[1] for c in cored),
                           max(c[2] for c in cored), max(c[3] for c in cored))
        return replace(base, x1=x1, y1=y1, x2=x2, y2=y2, letters=None, core=merged_core,
                       bubble_mask=merge_bubble_masks(
                           [getattr(b, "bubble_mask", None) for b in boxes]))
    x1, y1 = min([base.x1] + [b.x1 for b in evidence]), min([base.y1] + [b.y1 for b in evidence])
    x2, y2 = max([base.x2] + [b.x2 for b in evidence]), max([base.y2] + [b.y2 for b in evidence])
    merged = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for box in evidence:
        view = merged[box.y1 - y1:box.y2 - y1, box.x1 - x1:box.x2 - x1]
        np.maximum(view, box.mask, out=view)
    merged_letters = None
    lettered = [box for box in boxes if getattr(box, "letters", None) is not None]
    if lettered:
        merged_letters = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        for box in lettered:
            letters = np.asarray(box.letters)
            if letters.shape != (box.y2 - box.y1, box.x2 - box.x1):
                continue
            ly1, lx1 = box.y1, box.x1
            core = getattr(box, "core", None)
            if core is not None:
                r0 = max(0, core[1] - box.y1)
                r1 = min(letters.shape[0], core[3] - box.y1)
                c0 = max(0, core[0] - box.x1)
                c1 = min(letters.shape[1], core[2] - box.x1)
                if r1 <= r0 or c1 <= c0:
                    continue
                letters = letters[r0:r1, c0:c1]
                ly1, lx1 = box.y1 + r0, box.x1 + c0
            ix1, iy1 = max(lx1, x1), max(ly1, y1)
            ix2, iy2 = min(lx1 + letters.shape[1], x2), min(ly1 + letters.shape[0], y2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            view = merged_letters[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]
            view |= letters[iy1 - ly1:iy2 - ly1, ix1 - lx1:ix2 - lx1].astype(bool)
        if not merged_letters.any():
            merged_letters = None
    merged_core = None
    cored = [box.core for box in boxes if getattr(box, "core", None) is not None]
    if cored:
        merged_core = (min(c[0] for c in cored), min(c[1] for c in cored),
                       max(c[2] for c in cored), max(c[3] for c in cored))
    mask_source = base.mask_source if base.verified_mask else max(evidence, key=lambda b: float(b.confidence)).mask_source
    merged_bubble_mask = merge_bubble_masks([getattr(b, "bubble_mask", None) for b in boxes])
    return replace(
        base, x1=x1, y1=y1, x2=x2, y2=y2, mask=merged, mask_source=mask_source,
        safe_to_inpaint=bool(np.any(merged > 0) and any(b.safe_to_inpaint for b in evidence)),
        ocr_eligible=any(b.ocr_eligible for b in boxes),
        needs_review=any(b.needs_review for b in boxes),
        letters=merged_letters, core=merged_core, bubble_mask=merged_bubble_mask,
    )


def _nms_group(members: list[BubbleBox], iou_threshold: float) -> list[BubbleBox]:
    rects = [[b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1] for b in members]
    scores = [max(DETECTOR_NMS_SCORE_FLOOR, float(b.confidence)) for b in members]
    kept = [int(i) for i in np.array(cv2.dnn.NMSBoxes(rects, scores, DETECTOR_NMS_SCORE_FLOOR, iou_threshold)).flatten()]
    if not kept or members[kept[0]].source_role != "text_segmenter":
        return [members[i] for i in kept]
    buckets = {index: [members[index]] for index in kept}
    singles: list[BubbleBox] = []
    for index, box in enumerate(members):
        if index in buckets or not box.verified_mask:
            continue
        target = max(kept, key=lambda k: box_iou(box, members[k]))
        if box_iou(box, members[target]) > iou_threshold:
            if bubble_masks_are_separate_lobes(box.bubble_mask, members[target].bubble_mask):
                singles.append(box)
            else:
                buckets[target].append(box)
    return [_merge_mask_evidence(buckets[index]) for index in kept] + singles


def apply_final_nms(boxes: list[BubbleBox], iou_threshold: float = DETECTOR_FINAL_NMS_IOU) -> list[BubbleBox]:
    """Drop duplicate boxes per model and type, keeping the masks of the ones folded in."""
    groups: dict[tuple[str, str], list[BubbleBox]] = {}
    for b in boxes:
        groups.setdefault((b.source_model, b.semantic_type), []).append(b)
    result = [kept for members in groups.values() for kept in _nms_group(members, iou_threshold)]
    return sorted(result, key=lambda b: b.confidence, reverse=True)
