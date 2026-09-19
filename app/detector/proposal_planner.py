from __future__ import annotations

from dataclasses import dataclass

from app.detector.evidence import DetectionEvidence
from app.detector.page_context import PageContext


@dataclass(frozen=True)
class DetectionPlan:
    rois: tuple[tuple[int, int, int, int], ...]
    deferred: tuple[DetectionEvidence, ...]
    needs_full_page_fallback: bool
    reason: str | None = None


class ProposalPlanner:
    """Pure geometry/budget planner.

    It knows nothing about ONNX, LaMa, model authority, or destructive masks.
    """

    def __init__(
        self,
        *,
        pad_x: int,
        pad_y: int,
        max_rois: int,
        max_source_side: int,
        merge_overlap: float,
    ) -> None:
        if min(pad_x, pad_y) < 0:
            raise ValueError("ROI padding must be non-negative")
        if max_rois <= 0 or max_source_side <= 0:
            raise ValueError("ROI budget limits must be positive")
        if not 0.0 <= merge_overlap <= 1.0:
            raise ValueError("merge_overlap must be in [0,1]")
        self.pad_x = int(pad_x)
        self.pad_y = int(pad_y)
        self.max_rois = int(max_rois)
        self.max_source_side = int(max_source_side)
        self.merge_overlap = float(merge_overlap)

    @staticmethod
    def _overlap_fraction(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / float(min(area_a, area_b))

    def plan(
        self,
        *,
        page: PageContext,
        proposals: list[DetectionEvidence],
    ) -> DetectionPlan:
        rois: list[tuple[int, int, int, int]] = []
        deferred: list[DetectionEvidence] = []

        ranked = sorted(proposals, key=lambda item: item.confidence, reverse=True)
        for proposal in ranked:
            x1, y1, x2, y2 = proposal.bbox
            x1 = max(0, x1 - self.pad_x)
            y1 = max(0, y1 - self.pad_y)
            x2 = min(page.width, x2 + self.pad_x)
            y2 = min(page.height, y2 + self.pad_y)
            candidate = (x1, y1, x2, y2)
            if (
                x2 <= x1
                or y2 <= y1
                or (x2 - x1) > self.max_source_side
                or (y2 - y1) > self.max_source_side
            ):
                deferred.append(proposal)
                continue

            merged = False
            for index, current in enumerate(rois):
                if self._overlap_fraction(candidate, current) < self.merge_overlap:
                    continue
                union = (
                    min(candidate[0], current[0]),
                    min(candidate[1], current[1]),
                    max(candidate[2], current[2]),
                    max(candidate[3], current[3]),
                )
                if (
                    (union[2] - union[0]) <= self.max_source_side
                    and (union[3] - union[1]) <= self.max_source_side
                ):
                    rois[index] = union
                    merged = True
                    break
            if merged:
                continue

            if len(rois) >= self.max_rois:
                deferred.append(proposal)
                continue
            rois.append(candidate)

        needs_fallback = bool(deferred)
        return DetectionPlan(
            rois=tuple(rois),
            deferred=tuple(deferred),
            needs_full_page_fallback=needs_fallback,
            reason="proposal_geometry_or_budget" if needs_fallback else None,
        )
