"""Detected text boxes and the NMS that merges overlapping ones."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.detector.bubble_mask import merge_bubble_masks
from app.parameters import DETECTOR_FINAL_NMS_IOU, DETECTOR_NMS_SCORE_FLOOR


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
    # The letter rows alone, without outline or glow; they tell one text
    # from another where the rectangles overlap. A plain attribute would
    # not survive dataclasses.replace, so this is a field: the seam and
    # shift paths rebuild boxes with replace and the fold below reads it.
    letters: np.ndarray | None = field(default=None, compare=False)
    # The bubble interior the text is lettered into, as per-row
    # x-intervals over the outline bbox (see app/detector/bubble_mask).
    # A plain dict, so it survives dataclasses.replace and JSON.
    bubble_mask: dict | None = field(default=None, compare=False)
    # The raw detector rect the box grew from, in the box's own frame.
    # Letters the growth pulled in past it belong to the neighbour the
    # growth reached, not this box, so coherence is judged inside it.
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
        # No verified mask anywhere: the union still spans every grouped
        # box, or the merge silently drops the rects it was meant to fold
        # in. The letters are dropped when the rect grows — a base-only
        # letters array would be read against the wrong rows in the wider
        # rect — and the mask stays the base's, unverified, so the shape
        # check in verified_mask now fails honestly.
        x1, y1 = min(b.x1 for b in boxes), min(b.y1 for b in boxes)
        x2, y2 = max(b.x2 for b in boxes), max(b.y2 for b in boxes)
        if (x1, y1, x2, y2) == (base.x1, base.y1, base.x2, base.y2):
            return replace(base)
        cored = [box.core for box in boxes if getattr(box, "core", None) is not None]
        merged_core = None
        if cored:
            merged_core = (min(c[0] for c in cored), min(c[1] for c in cored),
                           max(c[2] for c in cored), max(c[3] for c in cored))
        # The rect grew, so the base's bubble alone no longer
        # describes it -- but the union of the members' bubbles does,
        # and the waist split needs it. (The erase masks are a
        # different story: with none verified there is nothing to union.)
        return replace(base, x1=x1, y1=y1, x2=x2, y2=y2, letters=None, core=merged_core,
                       bubble_mask=merge_bubble_masks(
                           [getattr(b, "bubble_mask", None) for b in boxes]))
    # The base's own rect joins the union even when the base has no
    # verified mask: the highest-confidence box is the merge's anchor,
    # and leaving its rect out shrinks the merged box away from it.
    x1, y1 = min([base.x1] + [b.x1 for b in evidence]), min([base.y1] + [b.y1 for b in evidence])
    x2, y2 = max([base.x2] + [b.x2 for b in evidence]), max([base.y2] + [b.y2 for b in evidence])
    merged = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for box in evidence:
        view = merged[box.y1 - y1:box.y2 - y1, box.x1 - x1:box.x2 - x1]
        np.maximum(view, box.mask, out=view)
    # The letters merge like the mask, into the same frame, so the merged
    # box's letter geometry still describes its own rect; a base-only
    # letters array would be read against the wrong rows downstream.
    # Each member's letters are clipped to its own core first: the growth
    # pulls a neighbour's letters in past the core, and a seam box in
    # particular carries a fragment of the text above the cut. Carried
    # into the union, that fragment splits the letter rows and the topmost
    # group alone is read as the lettering.
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
    # The bubble masks union like the erase masks: the merged box's
    # interior is the union of its members' interiors (a double bubble
    # folded from two lobe boxes keeps both lobes, for the waist split).
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
    for index, box in enumerate(members):
        if index in buckets or not box.verified_mask:
            continue
        target = max(kept, key=lambda k: box_iou(box, members[k]))
        if box_iou(box, members[target]) > iou_threshold:
            buckets[target].append(box)
    return [_merge_mask_evidence(buckets[index]) for index in kept]


def apply_final_nms(boxes: list[BubbleBox], iou_threshold: float = DETECTOR_FINAL_NMS_IOU) -> list[BubbleBox]:
    """Drop duplicate boxes per model and type, keeping the masks of the ones folded in."""
    groups: dict[tuple[str, str], list[BubbleBox]] = {}
    for b in boxes:
        groups.setdefault((b.source_model, b.semantic_type), []).append(b)
    result = [kept for members in groups.values() for kept in _nms_group(members, iou_threshold)]
    return sorted(result, key=lambda b: b.confidence, reverse=True)
