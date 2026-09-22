from __future__ import annotations

from app.detector.bubble_detector import BubbleBox
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
import app.inpaint.fast_lama_inpainter as fast_lama
from app.inpaint.lama_inpainter import Inpainter


class OverlapAwareAdaptiveFastInpainter(AdaptiveFastInpainter):
    """ Union near-duplicate destructive authorities before size-based grouping. """

    @staticmethod
    def _clusters_strongly_overlap(
        left: list[BubbleBox],
        right: list[BubbleBox],
    ) -> bool:
        threshold = float(fast_lama._BUBBLE_FASTPATH_OVERLAP_RATIO)
        for a in left:
            if not bool(a.safe_to_inpaint):
                continue
            area_a = max(0, int(a.x2) - int(a.x1)) * max(
                0, int(a.y2) - int(a.y1)
            )
            if area_a <= 0:
                continue
            for b in right:
                if not bool(b.safe_to_inpaint):
                    continue
                area_b = max(0, int(b.x2) - int(b.x1)) * max(
                    0, int(b.y2) - int(b.y1)
                )
                if area_b <= 0:
                    continue
                ix1 = max(int(a.x1), int(b.x1))
                iy1 = max(int(a.y1), int(b.y1))
                ix2 = min(int(a.x2), int(b.x2))
                iy2 = min(int(a.y2), int(b.y2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if intersection / float(min(area_a, area_b)) >= threshold:
                    return True
        return False

    @classmethod
    def _cluster_boxes(cls, boxes: list[BubbleBox]) -> list[list[BubbleBox]]:
        clusters = [list(group) for group in Inpainter._cluster_boxes(boxes)]
        changed = True
        while changed:
            changed = False
            for left_index in range(len(clusters)):
                if changed:
                    break
                for right_index in range(left_index + 1, len(clusters)):
                    if not cls._clusters_strongly_overlap(
                        clusters[left_index], clusters[right_index]
                    ):
                        continue
                    clusters[left_index].extend(clusters.pop(right_index))
                    changed = True
                    break
        return clusters
