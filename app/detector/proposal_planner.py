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

    def _bounded_axis(
        self,
        start: int,
        end: int,
        bound: int,
        pad: int,
    ) -> tuple[int, int] | None:
        start = max(0, min(int(start), bound))
        end = max(start, min(int(end), bound))
        if end <= start or end - start > self.max_source_side:
            return None
        padded_start = max(0, start - int(pad))
        padded_end = min(bound, end + int(pad))
        if padded_end - padded_start <= self.max_source_side:
            return padded_start, padded_end

        target = min(self.max_source_side, bound)
        center = (start + end) // 2
        window_start = max(0, min(bound - target, center - target // 2))
        if window_start > start:
            window_start = start
        if window_start + target < end:
            window_start = end - target
        window_start = max(0, min(bound - target, window_start))
        return window_start, window_start + target

    def plan_shape(
        self,
        *,
        image_shape: tuple[int, int],
        proposals: list[DetectionEvidence],
    ) -> DetectionPlan:
        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0 or not proposals:
            return DetectionPlan((), (), False, None)

        ranked = sorted(
            proposals,
            key=lambda item: (
                -float(item.confidence),
                max(1, int(item.width) * int(item.height)),
                int(item.bbox[1]),
                int(item.bbox[0]),
            ),
        )
        rois: list[tuple[int, int, int, int]] = []
        deferred: list[DetectionEvidence] = []

        for proposal in ranked:
            x1, y1, x2, y2 = proposal.bbox
            xs = self._bounded_axis(x1, x2, width, self.pad_x)
            ys = self._bounded_axis(y1, y2, height, self.pad_y)
            if xs is None or ys is None:
                deferred.append(proposal)
                continue
            candidate = (xs[0], ys[0], xs[1], ys[1])

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
                    union[2] - union[0] <= self.max_source_side
                    and union[3] - union[1] <= self.max_source_side
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

    def plan(
        self,
        *,
        page: PageContext,
        proposals: list[DetectionEvidence],
    ) -> DetectionPlan:
        return self.plan_shape(
            image_shape=(page.height, page.width),
            proposals=proposals,
        )
