"""Detected text boxes and the NMS that merges overlapping ones."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

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
        return replace(base)
    x1, y1 = min(b.x1 for b in evidence), min(b.y1 for b in evidence)
    x2, y2 = max(b.x2 for b in evidence), max(b.y2 for b in evidence)
    merged = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for box in evidence:
        view = merged[box.y1 - y1:box.y2 - y1, box.x1 - x1:box.x2 - x1]
        np.maximum(view, box.mask, out=view)
    mask_source = base.mask_source if base.verified_mask else max(evidence, key=lambda b: float(b.confidence)).mask_source
    return replace(
        base, x1=x1, y1=y1, x2=x2, y2=y2, mask=merged, mask_source=mask_source,
        safe_to_inpaint=bool(np.any(merged > 0) and any(b.safe_to_inpaint for b in evidence)),
        ocr_eligible=any(b.ocr_eligible for b in boxes),
        needs_review=any(b.needs_review for b in boxes),
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
