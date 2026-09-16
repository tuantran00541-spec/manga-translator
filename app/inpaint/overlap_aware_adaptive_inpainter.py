from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
import app.inpaint.fast_lama_inpainter as fast_lama
from app.inpaint.lama_inpainter import Inpainter


class OverlapAwareAdaptiveFastInpainter(AdaptiveFastInpainter):
    """Union duplicate authority and compact dense outlined text before LaMa.

    The base clusterer intentionally limits cluster span for CPU cost. That limit
    should not split two authorities that almost contain one another: unioning
    their existing masks adds no destructive pixels, while processing them in
    separate LaMa jobs can repaint the same text region repeatedly.

    Some comic text segmenters emit a dense *text-region* mask for large outlined
    free text instead of glyph support. On artwork this asks LaMa to repaint a
    whole panel-sized block and can still leave a glyph-shaped ghost. A narrowly
    gated image-evidence pass converts only high-contrast black/white outlined
    groups into compact glyph authority. Any ambiguous case falls back to the
    original detector masks unchanged.
    """

    _OUTLINED_DENSE_FRACTION_MIN = 0.55
    _OUTLINED_DARK_MAX = 60
    _OUTLINED_LIGHT_MIN = 200
    _OUTLINED_PAIR_RADIUS_FRACTION = 0.008
    _OUTLINED_PAIR_RADIUS_MIN = 3
    _OUTLINED_PAIR_RADIUS_MAX = 12
    _OUTLINED_COMPONENT_AREA_FRACTION = 0.00025
    _OUTLINED_COMPONENT_FILL_MIN = 0.15
    _OUTLINED_COMPONENT_AUTHORITY_OVERLAP_MIN = 0.75
    _OUTLINED_COMPONENT_HEIGHT_MAX_FRACTION = 0.45
    _OUTLINED_COMPONENT_WIDTH_MAX_FRACTION = 0.60
    _OUTLINED_SCALE_LOW = 0.70
    _OUTLINED_SCALE_HIGH = 1.35
    _OUTLINED_MIN_COMPONENTS = 3
    _OUTLINED_LINE_CENTER_TOLERANCE = 0.55
    _OUTLINED_LINE_MIN_COMPONENTS = 2
    _OUTLINED_LINE_MIN_SPAN_SCALE = 1.50
    _OUTLINED_SINGLE_LINE_MIN_COMPONENTS = 3
    _OUTLINED_SINGLE_LINE_MIN_SPAN_SCALE = 3.0
    _OUTLINED_FILLED_AUTHORITY_OVERLAP_MIN = 0.90
    _OUTLINED_COMPACT_FRACTION_MAX = 0.70
    _OUTLINED_MARGIN_FRACTION = 0.05
    _OUTLINED_MARGIN_MIN = 2
    _OUTLINED_MARGIN_MAX = 12
    _OUTLINED_GUARD_EXTRA = 2
    _OUTLINED_FINAL_FRACTION_MAX = 0.75

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

    @staticmethod
    def _outlined_refinement_candidate(box: BubbleBox) -> bool:
        return bool(
            box.source_role == "text_segmenter"
            and box.safe_to_inpaint
            and not box.needs_review
            and box.verified_mask
            and box.semantic_type in {"speech_bubble", "free_text", "text"}
        )

    @classmethod
    def _outlined_overlap_groups(
        cls,
        boxes: list[BubbleBox],
    ) -> list[list[int]]:
        candidates = [
            index
            for index, box in enumerate(boxes)
            if cls._outlined_refinement_candidate(box)
        ]
        if not candidates:
            return []

        parent = {index: index for index in candidates}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for offset, left in enumerate(candidates):
            for right in candidates[offset + 1 :]:
                if cls._clusters_strongly_overlap(
                    [boxes[left]],
                    [boxes[right]],
                ):
                    union(left, right)

        groups: dict[int, list[int]] = {}
        for index in candidates:
            groups.setdefault(find(index), []).append(index)
        return sorted(groups.values(), key=lambda group: min(group))

    @staticmethod
    def _union_group_authority(
        group: list[BubbleBox],
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        if not group:
            return None
        x1 = min(int(box.x1) for box in group)
        y1 = min(int(box.y1) for box in group)
        x2 = max(int(box.x2) for box in group)
        y2 = max(int(box.y2) for box in group)
        if x2 <= x1 or y2 <= y1:
            return None

        authority = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for box in group:
            mask = box.mask
            box_w = int(box.x2) - int(box.x1)
            box_h = int(box.y2) - int(box.y1)
            if mask is None or box_w <= 0 or box_h <= 0:
                continue
            if mask.shape[:2] != (box_h, box_w):
                mask = cv2.resize(
                    mask,
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            dx1, dy1 = int(box.x1) - x1, int(box.y1) - y1
            target = authority[dy1 : dy1 + box_h, dx1 : dx1 + box_w]
            authority[dy1 : dy1 + box_h, dx1 : dx1 + box_w] = np.maximum(
                target,
                mask,
            )

        if not np.any(authority > 127):
            return None
        return (x1, y1, x2, y2), authority

    @classmethod
    def _dominant_height_components(
        cls,
        components: list[dict],
    ) -> tuple[list[dict], float]:
        if not components:
            return [], 0.0

        best: list[dict] = []
        best_area = -1
        for anchor in components:
            height = max(1.0, float(anchor["height"]))
            low = height * cls._OUTLINED_SCALE_LOW
            high = height * cls._OUTLINED_SCALE_HIGH
            cohort = [
                component
                for component in components
                if low <= float(component["height"]) <= high
            ]
            cohort_area = sum(int(component["area"]) for component in cohort)
            if len(cohort) > len(best) or (
                len(cohort) == len(best) and cohort_area > best_area
            ):
                best = cohort
                best_area = cohort_area

        if len(best) < cls._OUTLINED_MIN_COMPONENTS:
            return [], 0.0
        dominant = float(np.median([component["height"] for component in best]))
        low = dominant * cls._OUTLINED_SCALE_LOW
        high = dominant * cls._OUTLINED_SCALE_HIGH
        coherent = [
            component
            for component in best
            if low <= float(component["height"]) <= high
        ]
        if len(coherent) < cls._OUTLINED_MIN_COMPONENTS:
            return [], 0.0
        return coherent, dominant

    @classmethod
    def _coherent_text_line_components(
        cls,
        components: list[dict],
        dominant_height: float,
    ) -> list[dict]:
        if not components or dominant_height <= 0:
            return []

        tolerance = max(
            2.0,
            dominant_height * cls._OUTLINED_LINE_CENTER_TOLERANCE,
        )
        lines: list[dict] = []
        for component in sorted(components, key=lambda item: item["center_y"]):
            best_line = None
            best_distance = None
            for line in lines:
                distance = abs(float(component["center_y"]) - float(line["center_y"]))
                if distance <= tolerance and (
                    best_distance is None or distance < best_distance
                ):
                    best_line = line
                    best_distance = distance
            if best_line is None:
                lines.append(
                    {
                        "center_y": float(component["center_y"]),
                        "components": [component],
                    }
                )
            else:
                best_line["components"].append(component)
                best_line["center_y"] = float(
                    np.mean(
                        [
                            item["center_y"]
                            for item in best_line["components"]
                        ]
                    )
                )

        coherent_lines: list[list[dict]] = []
        for line in lines:
            members = line["components"]
            x1 = min(int(item["x"]) for item in members)
            x2 = max(int(item["x"]) + int(item["width"]) for item in members)
            span = x2 - x1
            if (
                len(members) >= cls._OUTLINED_LINE_MIN_COMPONENTS
                and span >= dominant_height * cls._OUTLINED_LINE_MIN_SPAN_SCALE
            ):
                coherent_lines.append(members)

        if not coherent_lines:
            return []
        if len(coherent_lines) == 1:
            members = coherent_lines[0]
            x1 = min(int(item["x"]) for item in members)
            x2 = max(int(item["x"]) + int(item["width"]) for item in members)
            if (
                len(members) < cls._OUTLINED_SINGLE_LINE_MIN_COMPONENTS
                or (x2 - x1)
                < dominant_height * cls._OUTLINED_SINGLE_LINE_MIN_SPAN_SCALE
            ):
                return []

        return [
            component
            for line in coherent_lines
            for component in line
        ]

    @classmethod
    def _outlined_dense_glyph_mask(
        cls,
        crop: np.ndarray,
        authority_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int | float]] | None:
        if (
            crop is None
            or crop.size == 0
            or authority_mask is None
            or authority_mask.size == 0
        ):
            return None
        height, width = authority_mask.shape[:2]
        if crop.shape[:2] != (height, width) or min(height, width) < 16:
            return None

        authority = authority_mask > 127
        authority_pixels = int(np.count_nonzero(authority))
        area = max(1, int(height * width))
        if authority_pixels / float(area) < cls._OUTLINED_DENSE_FRACTION_MIN:
            return None

        gray = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if crop.ndim == 3
            else crop
        )
        dark = gray <= cls._OUTLINED_DARK_MAX
        light = gray >= cls._OUTLINED_LIGHT_MIN

        pair_radius = int(
            np.clip(
                round(min(height, width) * cls._OUTLINED_PAIR_RADIUS_FRACTION),
                cls._OUTLINED_PAIR_RADIUS_MIN,
                cls._OUTLINED_PAIR_RADIUS_MAX,
            )
        )
        pair_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pair_radius * 2 + 1, pair_radius * 2 + 1),
        )
        near_light = cv2.dilate(light.astype(np.uint8), pair_kernel) > 0
        near_dark = cv2.dilate(dark.astype(np.uint8), pair_kernel) > 0
        paired = (dark & near_light) | (light & near_dark)
        paired = cv2.morphologyEx(
            paired.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (paired > 127).astype(np.uint8),
            connectivity=8,
        )
        min_component_area = max(
            24,
            int(round(area * cls._OUTLINED_COMPONENT_AREA_FRACTION)),
        )
        components: list[dict] = []
        for label in range(1, count):
            x, y, comp_w, comp_h, comp_area = (
                int(value) for value in stats[label]
            )
            if comp_area < min_component_area:
                continue
            if (
                comp_h > height * cls._OUTLINED_COMPONENT_HEIGHT_MAX_FRACTION
                or comp_w > width * cls._OUTLINED_COMPONENT_WIDTH_MAX_FRACTION
            ):
                continue
            bbox_area = max(1, comp_w * comp_h)
            if (
                comp_area / float(bbox_area)
                < cls._OUTLINED_COMPONENT_FILL_MIN
            ):
                continue

            component = labels == label
            overlap = int(np.count_nonzero(component & authority))
            if (
                overlap / float(max(1, comp_area))
                < cls._OUTLINED_COMPONENT_AUTHORITY_OVERLAP_MIN
            ):
                continue
            components.append(
                {
                    "label": label,
                    "x": x,
                    "y": y,
                    "width": comp_w,
                    "height": comp_h,
                    "area": comp_area,
                    "center_y": float(centroids[label][1]),
                }
            )

        coherent, dominant_height = cls._dominant_height_components(components)
        coherent = cls._coherent_text_line_components(
            coherent,
            dominant_height,
        )
        if len(coherent) < cls._OUTLINED_MIN_COMPONENTS:
            return None

        filled = np.zeros((height, width), dtype=np.uint8)
        for component in coherent:
            binary = (
                (labels == int(component["label"])).astype(np.uint8) * 255
            )
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                cv2.drawContours(filled, contours, -1, 255, thickness=-1)

        filled_pixels = int(np.count_nonzero(filled > 127))
        if filled_pixels <= 0:
            return None
        filled_inside = int(np.count_nonzero((filled > 127) & authority))
        if (
            filled_inside / float(filled_pixels)
            < cls._OUTLINED_FILLED_AUTHORITY_OVERLAP_MIN
        ):
            return None
        if (
            filled_pixels / float(authority_pixels)
            > cls._OUTLINED_COMPACT_FRACTION_MAX
        ):
            return None

        margin = int(
            np.clip(
                round(dominant_height * cls._OUTLINED_MARGIN_FRACTION),
                cls._OUTLINED_MARGIN_MIN,
                cls._OUTLINED_MARGIN_MAX,
            )
        )
        margin_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (margin * 2 + 1, margin * 2 + 1),
        )
        refined = cv2.dilate(filled, margin_kernel, iterations=1)

        guard_radius = margin + int(cls._OUTLINED_GUARD_EXTRA)
        guard_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (guard_radius * 2 + 1, guard_radius * 2 + 1),
        )
        guard = cv2.dilate(
            authority.astype(np.uint8) * 255,
            guard_kernel,
            iterations=1,
        )
        refined = np.where(guard > 127, refined, 0).astype(np.uint8)
        refined_pixels = int(np.count_nonzero(refined > 127))
        if refined_pixels <= 0:
            return None
        if (
            refined_pixels / float(authority_pixels)
            > cls._OUTLINED_FINAL_FRACTION_MAX
        ):
            return None

        return refined, {
            "authority_pixels": authority_pixels,
            "filled_pixels": filled_pixels,
            "refined_pixels": refined_pixels,
            "component_count": len(coherent),
            "dominant_height": round(dominant_height, 3),
            "margin": margin,
        }

    @classmethod
    def _prepare_outlined_dense_text_boxes(
        cls,
        image: np.ndarray,
        boxes: list[BubbleBox],
    ) -> tuple[list[BubbleBox], dict[str, int]]:
        groups = cls._outlined_overlap_groups(boxes)
        if not groups:
            return list(boxes), {
                "outlined_text_refined_groups": 0,
                "outlined_text_collapsed_boxes": 0,
                "outlined_text_authority_pixels": 0,
                "outlined_text_refined_pixels": 0,
                "outlined_text_saved_pixels": 0,
                "outlined_text_max_margin": 0,
            }

        replacements: dict[int, BubbleBox] = {}
        removed: set[int] = set()
        metrics = {
            "outlined_text_refined_groups": 0,
            "outlined_text_collapsed_boxes": 0,
            "outlined_text_authority_pixels": 0,
            "outlined_text_refined_pixels": 0,
            "outlined_text_saved_pixels": 0,
            "outlined_text_max_margin": 0,
        }
        image_h, image_w = image.shape[:2]

        for indexes in groups:
            members = [boxes[index] for index in indexes]
            merged = cls._union_group_authority(members)
            if merged is None:
                continue
            (x1, y1, x2, y2), authority = merged
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(image_w, x2), min(image_h, y2)
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                continue
            crop = image[y1:y2, x1:x2]
            refined = cls._outlined_dense_glyph_mask(crop, authority)
            if refined is None:
                continue
            mask, stats = refined

            seed = max(members, key=lambda box: float(box.confidence))
            semantic_type = (
                "free_text"
                if any(
                    member.semantic_type in {"free_text", "text"}
                    for member in members
                )
                else seed.semantic_type
            )
            canonical = replace(
                seed,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=max(float(member.confidence) for member in members),
                mask=mask,
                semantic_type=semantic_type,
                mask_source="text_segmenter",
                safe_to_inpaint=True,
                ocr_eligible=True,
                needs_review=False,
                source_role="text_segmenter",
                deferred_reason=None,
            )
            first = min(indexes)
            replacements[first] = canonical
            removed.update(index for index in indexes if index != first)

            source_pixels = int(stats["authority_pixels"])
            refined_pixels = int(stats["refined_pixels"])
            metrics["outlined_text_refined_groups"] += 1
            metrics["outlined_text_collapsed_boxes"] += max(0, len(indexes) - 1)
            metrics["outlined_text_authority_pixels"] += source_pixels
            metrics["outlined_text_refined_pixels"] += refined_pixels
            metrics["outlined_text_saved_pixels"] += max(
                0,
                source_pixels - refined_pixels,
            )
            metrics["outlined_text_max_margin"] = max(
                metrics["outlined_text_max_margin"],
                int(stats["margin"]),
            )

        prepared: list[BubbleBox] = []
        for index, box in enumerate(boxes):
            if index in removed:
                continue
            prepared.append(replacements.get(index, box))
        return prepared, metrics

    def inpaint(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
        *,
        protected_regions: list[dict] | None = None,
    ) -> np.ndarray:
        prepared, refinement_metrics = self._prepare_outlined_dense_text_boxes(
            image,
            boxes,
        )
        result = super().inpaint(
            image,
            prepared,
            protected_regions=protected_regions,
        )
        metrics = getattr(self._metrics_local, "value", None)
        if isinstance(metrics, dict):
            metrics.update(refinement_metrics)
        return result
