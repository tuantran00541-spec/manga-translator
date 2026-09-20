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


@dataclass(frozen=True)
class ClusteredDetectionPlan:
    """Geometry-only ROI plan with explicit proposal coverage."""

    rois: tuple[tuple[int, int, int, int], ...]
    covered: tuple[DetectionEvidence, ...]
    deferred: tuple[DetectionEvidence, ...]
    candidate_roi_count: int
    deferred_roi_count: int
    group_count: int

    @property
    def needs_full_page_fallback(self) -> bool:
        return bool(self.deferred)


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
        merge_gap: int = 0,
    ) -> None:
        if min(pad_x, pad_y) < 0:
            raise ValueError("ROI padding must be non-negative")
        if max_rois <= 0 or max_source_side <= 0:
            raise ValueError("ROI budget limits must be positive")
        if not 0.0 <= merge_overlap <= 1.0:
            raise ValueError("merge_overlap must be in [0,1]")
        if merge_gap < 0:
            raise ValueError("merge_gap must be non-negative")
        self.pad_x = int(pad_x)
        self.pad_y = int(pad_y)
        self.max_rois = int(max_rois)
        self.max_source_side = int(max_source_side)
        self.merge_overlap = float(merge_overlap)
        self.merge_gap = int(merge_gap)

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

    @staticmethod
    def _rect_gap(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return gap_x, gap_y

    def _clusterable(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> bool:
        if self._overlap_fraction(a, b) >= self.merge_overlap:
            return True
        gap_x, gap_y = self._rect_gap(a, b)
        return gap_x <= self.merge_gap and gap_y <= self.merge_gap

    @staticmethod
    def _union_rects(
        rects: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        return (
            min(rect[0] for rect in rects),
            min(rect[1] for rect in rects),
            max(rect[2] for rect in rects),
            max(rect[3] for rect in rects),
        )

    @staticmethod
    def _tile_axis(
        start: int,
        end: int,
        *,
        bound: int,
        limit: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        start = max(0, min(int(start), int(bound)))
        end = max(start, min(int(end), int(bound)))
        if end <= start:
            return []
        limit = max(1, min(int(limit), int(bound)))
        if end - start <= limit:
            return [(start, end)]
        overlap = max(0, min(int(overlap), limit - 1))
        stride = max(1, limit - overlap)
        out: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            stop = min(end, cursor + limit)
            if stop - cursor < limit and end - limit >= start:
                cursor = end - limit
                stop = end
            item = (cursor, stop)
            if not out or item != out[-1]:
                out.append(item)
            if stop >= end:
                break
            cursor += stride
        return out

    @classmethod
    def _rect_covered_by_rois(
        cls,
        rect: tuple[int, int, int, int],
        rois: list[tuple[int, int, int, int]],
    ) -> bool:
        x1, y1, x2, y2 = rect
        if x2 <= x1 or y2 <= y1:
            return True
        intersections: list[tuple[int, int, int, int]] = []
        x_breaks = {x1, x2}
        for rx1, ry1, rx2, ry2 in rois:
            ix1, iy1 = max(x1, rx1), max(y1, ry1)
            ix2, iy2 = min(x2, rx2), min(y2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            intersections.append((ix1, iy1, ix2, iy2))
            x_breaks.add(ix1)
            x_breaks.add(ix2)
        if not intersections:
            return False

        xs = sorted(x_breaks)
        covered_area = 0
        for left, right in zip(xs, xs[1:]):
            if right <= left:
                continue
            intervals: list[tuple[int, int]] = []
            for ix1, iy1, ix2, iy2 in intersections:
                if ix1 <= left and ix2 >= right:
                    intervals.append((iy1, iy2))
            if not intervals:
                continue
            intervals.sort()
            merged_y = 0
            cy1, cy2 = intervals[0]
            for iy1, iy2 in intervals[1:]:
                if iy1 <= cy2:
                    cy2 = max(cy2, iy2)
                else:
                    merged_y += max(0, cy2 - cy1)
                    cy1, cy2 = iy1, iy2
            merged_y += max(0, cy2 - cy1)
            covered_area += (right - left) * merged_y
        return covered_area >= (x2 - x1) * (y2 - y1)

    def plan_clustered_shape(
        self,
        *,
        image_shape: tuple[int, int],
        proposals: list[DetectionEvidence],
        source_pixel_budget: int | None = None,
        tensor_pixel_budget: int | None = None,
        tensor_pixels_per_roi: int = 0,
        tile_overlap: int = 0,
        span_short_axis: bool = False,
    ) -> ClusteredDetectionPlan:
        """Merge nearby proposals before tiling and report real proposal coverage.

        Unlike the legacy retry planner, this method does not treat every
        deferred *tile* as a deferred proposal. A proposal is deferred only when
        its original bbox is not fully covered by the scheduled ROI union.
        """

        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0 or not proposals:
            return ClusteredDetectionPlan((), (), tuple(proposals), 0, 0, 0)

        entries: list[
            tuple[
                DetectionEvidence,
                tuple[int, int, int, int],
                tuple[int, int, int, int],
                float,
            ]
        ] = []
        for proposal in proposals:
            x1, y1, x2, y2 = map(int, proposal.bbox)
            x1 = max(0, min(x1, width))
            x2 = max(x1, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(y1, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            meta = dict(proposal.metadata or {})
            pad_x = max(0, int(meta.get("pad_x", self.pad_x)))
            pad_y = max(0, int(meta.get("pad_y", self.pad_y)))
            cluster_rect = (x1, y1, x2, y2)
            roi_rect = (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width, x2 + pad_x),
                min(height, y2 + pad_y),
            )
            priority = float(meta.get("priority", 1.0))
            entries.append((proposal, cluster_rect, roi_rect, priority))

        if not entries:
            return ClusteredDetectionPlan(
                (), (), tuple(proposals), 0, 0, 0
            )

        # Cluster on the original proposal geometry. Padding belongs to the
        # resulting authority crop, not to connectivity: clustering padded boxes
        # can create a transitive chain that turns an entire page into one ROI.
        pending = list(range(len(entries)))
        groups: list[list[int]] = []
        while pending:
            seed = pending.pop(0)
            group = [seed]
            changed = True
            while changed:
                changed = False
                group_rect = self._union_rects([entries[i][1] for i in group])
                keep: list[int] = []
                for idx in pending:
                    candidate = entries[idx][1]
                    union = self._union_rects([group_rect, candidate])
                    bounded_union = (
                        union[2] - union[0] <= self.max_source_side
                        and union[3] - union[1] <= self.max_source_side
                    )
                    if self._clusterable(group_rect, candidate) and bounded_union:
                        group.append(idx)
                        changed = True
                    else:
                        keep.append(idx)
                pending = keep
            groups.append(group)

        ranked_rois: list[
            tuple[tuple[float, int, int, int], tuple[int, int, int, int]]
        ] = []
        serial = 0
        for group in groups:
            group_rect = self._union_rects([entries[i][2] for i in group])
            gx1, gy1, gx2, gy2 = group_rect
            if span_short_axis and height >= width and width <= self.max_source_side:
                gx1, gx2 = 0, width
            elif span_short_axis and width > height and height <= self.max_source_side:
                gy1, gy2 = 0, height

            xs = self._tile_axis(
                gx1,
                gx2,
                bound=width,
                limit=self.max_source_side,
                overlap=tile_overlap,
            )
            ys = self._tile_axis(
                gy1,
                gy2,
                bound=height,
                limit=self.max_source_side,
                overlap=tile_overlap,
            )
            group_priority = min(entries[i][3] for i in group)
            group_area = max(1, (gx2 - gx1) * (gy2 - gy1))
            for ay, by in ys:
                for ax, bx in xs:
                    ranked_rois.append(
                        (
                            (group_priority, group_area, ay, serial),
                            (ax, ay, bx, by),
                        )
                    )
                    serial += 1

        dedup: dict[
            tuple[int, int, int, int],
            tuple[float, int, int, int],
        ] = {}
        for priority, roi in ranked_rois:
            previous = dedup.get(roi)
            if previous is None or priority < previous:
                dedup[roi] = priority
        ordered = [
            roi for roi, _priority in
            sorted(dedup.items(), key=lambda item: (item[1], item[0]))
        ]

        scheduled: list[tuple[int, int, int, int]] = []
        used_source = 0
        used_tensor = 0
        source_budget = (
            None if source_pixel_budget is None
            else max(0, int(source_pixel_budget))
        )
        tensor_budget = (
            None if tensor_pixel_budget is None
            else max(0, int(tensor_pixel_budget))
        )
        tensor_per_roi = max(0, int(tensor_pixels_per_roi))
        for roi in ordered:
            x1, y1, x2, y2 = roi
            pixels = max(0, (x2 - x1) * (y2 - y1))
            fits = len(scheduled) < self.max_rois
            if source_budget is not None:
                fits = fits and used_source + pixels <= source_budget
            if tensor_budget is not None and tensor_per_roi > 0:
                fits = fits and used_tensor + tensor_per_roi <= tensor_budget
            if not fits:
                continue
            scheduled.append(roi)
            used_source += pixels
            used_tensor += tensor_per_roi

        covered: list[DetectionEvidence] = []
        deferred: list[DetectionEvidence] = []
        for proposal in proposals:
            bbox = tuple(map(int, proposal.bbox))
            if self._rect_covered_by_rois(bbox, scheduled):
                covered.append(proposal)
            else:
                deferred.append(proposal)

        return ClusteredDetectionPlan(
            rois=tuple(scheduled),
            covered=tuple(covered),
            deferred=tuple(deferred),
            candidate_roi_count=len(ordered),
            deferred_roi_count=max(0, len(ordered) - len(scheduled)),
            group_count=len(groups),
        )

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
