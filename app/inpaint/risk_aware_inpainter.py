from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.detector.mask_builder import build_mask
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
from app.inpaint.lama_inpainter import Inpainter
from app.parameters import (
    INPAINT_COALESCE_ENABLED,
    INPAINT_COALESCE_MAX_BOXES,
    INPAINT_COALESCE_MAX_BATCH_SIZE,
    INPAINT_COALESCE_MAX_DIM,
    INPAINT_COALESCE_MAX_GAP,
    INPAINT_COALESCE_MAX_PADDING_RATIO,
    INPAINT_COALESCE_UNION_SLACK_PIXELS,
    INPAINT_COALESCE_UNION_SLACK_RATIO,
    INPAINT_RISK_CHROMA_HIGH,
    INPAINT_RISK_CHROMA_MEDIUM,
    INPAINT_RISK_CONTEXT_RADIUS,
    INPAINT_RISK_EDGE_DENSITY_HIGH,
    INPAINT_RISK_EDGE_DENSITY_MEDIUM,
    INPAINT_RISK_GRADIENT_HIGH,
    INPAINT_RISK_GRADIENT_MEDIUM,
    INPAINT_RISK_HIGH_SCORE,
    INPAINT_RISK_LONG_ASPECT,
    INPAINT_RISK_MEDIUM_SCORE,
    INPAINT_RISK_MIN_CONTEXT_PIXELS,
    INPAINT_RISK_ROUTER_ENABLED,
    INPAINT_RISK_VARIATION_HIGH,
    INPAINT_RISK_VARIATION_MEDIUM,
    DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM,
    INPAINT_SIZE,
)


@dataclass(frozen=True, slots=True)
class InpaintRisk:
    """Measured context risk for a single authorized inpaint region.

    The measurements deliberately exclude pixels inside the destructive mask.
    The router can therefore use nearby artwork as a warning signal without
    turning that artwork into deletion authority.
    """

    level: str
    score: float
    variation_span: float
    context_std: float
    edge_density: float
    gradient_mean: float
    chroma_std: float
    aspect: float
    context_pixels: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "level": self.level,
            "score": round(float(self.score), 6),
            "variation_span": round(float(self.variation_span), 4),
            "context_std": round(float(self.context_std), 4),
            "edge_density": round(float(self.edge_density), 6),
            "gradient_mean": round(float(self.gradient_mean), 4),
            "chroma_std": round(float(self.chroma_std), 4),
            "aspect": round(float(self.aspect), 4),
            "context_pixels": int(self.context_pixels),
        }


def _binary_context(local_mask: np.ndarray) -> np.ndarray:
    """Return a bounded near-mask context, falling back to all non-mask pixels."""
    mask = np.asarray(local_mask) > 127
    if mask.ndim != 2:
        return np.zeros(mask.shape[:2] if mask.ndim > 1 else (0, 0), dtype=bool)

    radius = max(1, int(INPAINT_RISK_CONTEXT_RADIUS))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    near = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask
    if int(np.count_nonzero(near)) >= int(INPAINT_RISK_MIN_CONTEXT_PIXELS):
        return near

    all_non_mask = ~mask
    if int(np.count_nonzero(all_non_mask)) >= int(INPAINT_RISK_MIN_CONTEXT_PIXELS):
        return all_non_mask
    return all_non_mask


def _safe_percentile_span(values: np.ndarray) -> float:
    if values.size == 0:
        return 255.0
    values = values.astype(np.float32, copy=False)
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def measure_inpaint_risk(crop: np.ndarray, local_mask: np.ndarray) -> InpaintRisk:
    """Measure texture/gradient/chroma risk without changing the input mask.

    ``score`` is the maximum normalized signal.  Max-normalization makes the
    policy one-sided: a single strong gradient or edge signal is enough to
    leave the cheap path, while multiple weak signals can still produce a
    reviewable medium decision.  All normalization thresholds are centralized
    in :mod:`app.parameters`.
    """
    if crop is None or crop.size == 0 or local_mask is None:
        return InpaintRisk(
            "high", 1.0, 255.0, 255.0, 1.0, 255.0, 128.0, 1.0, 0
        )

    if local_mask.ndim != 2 or crop.shape[:2] != local_mask.shape[:2]:
        return InpaintRisk(
            "high", 1.0, 255.0, 255.0, 1.0, 255.0, 128.0, 1.0, 0
        )

    context = _binary_context(local_mask)
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else np.asarray(crop)
    if gray.ndim != 2:
        return InpaintRisk(
            "high", 1.0, 255.0, 255.0, 1.0, 255.0, 128.0, 1.0, 0
        )

    values = gray[context]
    context_pixels = int(values.size)
    if context_pixels == 0:
        return InpaintRisk(
            "high", 1.0, 255.0, 255.0, 1.0, 255.0, 128.0, 1.0, 0
        )

    variation_span = _safe_percentile_span(values)
    context_std = float(values.astype(np.float32, copy=False).std())

    edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
    edge_density = float(np.mean(edges[context])) if context_pixels else 1.0

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient_mean = float(np.mean(gradient[context])) if context_pixels else 255.0

    chroma_std = 0.0
    if crop.ndim == 3 and crop.shape[2] >= 3:
        lab = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2LAB)
        ab = lab[context, 1:3].astype(np.float32, copy=False)
        if ab.size:
            chroma_std = float(np.max(ab.std(axis=0)))

    crop_h, crop_w = crop.shape[:2]
    aspect = max(crop_h, crop_w) / float(max(1, min(crop_h, crop_w)))

    signals = [
        variation_span / max(1e-6, float(INPAINT_RISK_VARIATION_HIGH)),
        edge_density / max(1e-6, float(INPAINT_RISK_EDGE_DENSITY_HIGH)),
        gradient_mean / max(1e-6, float(INPAINT_RISK_GRADIENT_HIGH)),
        chroma_std / max(1e-6, float(INPAINT_RISK_CHROMA_HIGH)),
    ]
    if aspect >= float(INPAINT_RISK_LONG_ASPECT):
        signals.append(
            1.0
            + (aspect - float(INPAINT_RISK_LONG_ASPECT))
            / max(1.0, float(INPAINT_RISK_LONG_ASPECT))
        )
    else:
        signals.append(0.0)
    score = max(0.0, float(max(signals)))

    # Keep raw medium/high thresholds meaningful even when a caller sets a
    # custom score threshold. This also makes report interpretation stable.
    medium_signal = max(
        variation_span >= float(INPAINT_RISK_VARIATION_MEDIUM),
        edge_density >= float(INPAINT_RISK_EDGE_DENSITY_MEDIUM),
        gradient_mean >= float(INPAINT_RISK_GRADIENT_MEDIUM),
        chroma_std >= float(INPAINT_RISK_CHROMA_MEDIUM),
    )
    high_signal = max(
        variation_span >= float(INPAINT_RISK_VARIATION_HIGH),
        edge_density >= float(INPAINT_RISK_EDGE_DENSITY_HIGH),
        gradient_mean >= float(INPAINT_RISK_GRADIENT_HIGH),
        chroma_std >= float(INPAINT_RISK_CHROMA_HIGH),
        aspect >= float(INPAINT_RISK_LONG_ASPECT),
    )
    if high_signal or score >= float(INPAINT_RISK_HIGH_SCORE):
        level = "high"
    elif medium_signal or score >= float(INPAINT_RISK_MEDIUM_SCORE):
        level = "medium"
    else:
        level = "low"

    return InpaintRisk(
        level,
        score,
        variation_span,
        context_std,
        edge_density,
        gradient_mean,
        chroma_std,
        aspect,
        context_pixels,
    )


class RiskAwareFastInpainter(AdaptiveFastInpainter):
    """E5 candidate: cheap paths only for demonstrably low-risk context.

    Existing FastInpainter behavior is retained for low-risk flat regions.
    Medium/high-risk regions fall through to the normal LaMa implementation;
    no risk decision expands or changes the authorized destructive mask.
    """

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        super()._begin_metrics(boxes=boxes)
        self._metrics_local.value.update(
            {
                "risk_evaluations": 0,
                "risk_low_regions": 0,
                "risk_medium_regions": 0,
                "risk_high_regions": 0,
                "risk_forced_lama_regions": 0,
                "risk_fastpath_rejections": 0,
                "risk_score_sum_milli": 0,
                "risk_score_max_milli": 0,
            }
        )

    def _observe_risk(self, risk: InpaintRisk) -> None:
        self._metric_add("risk_evaluations")
        self._metric_add(f"risk_{risk.level}_regions")
        score_milli = max(0, int(round(float(risk.score) * 1000.0)))
        self._metric_add("risk_score_sum_milli", score_milli)
        metrics = self._metrics_local.value
        metrics["risk_score_max_milli"] = max(
            int(metrics.get("risk_score_max_milli", 0)), score_milli
        )

    @staticmethod
    def assess_risk(crop: np.ndarray, local_mask: np.ndarray) -> InpaintRisk:
        return measure_inpaint_risk(crop, local_mask)

    def _allow_bubble_fast_fill(
        self,
        crop: np.ndarray,
        local_mask: np.ndarray,
    ) -> bool:
        if not INPAINT_RISK_ROUTER_ENABLED:
            return True
        risk = self.assess_risk(crop, local_mask)
        self._observe_risk(risk)
        if risk.level == "low":
            return True
        self._metric_add("risk_fastpath_rejections")
        return False

    def _smart_paint_region(
        self,
        image: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple,
        feather: bool = False,
        force_lama: bool = False,
    ) -> np.ndarray:
        if force_lama or not INPAINT_RISK_ROUTER_ENABLED:
            return super()._smart_paint_region(
                image,
                local_mask,
                crop_box,
                feather=feather,
                force_lama=force_lama,
            )

        cx1, cy1, cx2, cy2 = (int(value) for value in crop_box)
        crop = image[cy1:cy2, cx1:cx2]
        risk = self.assess_risk(crop, local_mask)
        self._observe_risk(risk)
        if risk.level != "low":
            self._metric_add("risk_forced_lama_regions")
            return super()._smart_paint_region(
                image,
                local_mask,
                crop_box,
                feather=feather,
                force_lama=True,
            )
        return super()._smart_paint_region(
            image,
            local_mask,
            crop_box,
            feather=feather,
            force_lama=False,
        )


class CoalescingRiskAwareInpainter(RiskAwareFastInpainter):
    """E6 candidate: merge only small, compatible LaMa source groups.

    ``Inpainter.inpaint`` already consumes a list of clusters. Overriding that
    planning seam lets E6 reuse the exact existing mask construction and final
    authorized compositing. The class only changes the model context supplied
    to one call; the benchmark must therefore compare quality before promotion.
    """

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        super()._begin_metrics(boxes=boxes)
        self._metrics_local.value.update(
            {
                "coalesce_evaluations": 0,
                "coalesce_merges": 0,
                "coalesce_rejected_incompatible": 0,
                "coalesce_rejected_geometry": 0,
                "coalesce_saved_clusters": 0,
                "coalesce_union_overhead_pixels": 0,
                "coalesce_batch_groups": 0,
                "coalesce_batch_items": 0,
                "coalesce_batch_padding_pixels": 0,
                "coalesce_batch_fallbacks": 0,
            }
        )

    @staticmethod
    def _cluster_bounds(cluster: list) -> tuple[int, int, int, int]:
        return (
            min(int(box.x1) for box in cluster),
            min(int(box.y1) for box in cluster),
            max(int(box.x2) for box in cluster),
            max(int(box.y2) for box in cluster),
        )

    @staticmethod
    def _gap(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        return (
            max(0, max(first[0], second[0]) - min(first[2], second[2])),
            max(0, max(first[1], second[1]) - min(first[3], second[3])),
        )

    @classmethod
    def _compatible(cls, first: list, second: list) -> tuple[bool, str, int]:
        if not first or not second:
            return False, "empty", 0
        combined = first + second
        if len(combined) > int(INPAINT_COALESCE_MAX_BOXES):
            return False, "incompatible", 0
        if any(
            not bool(getattr(box, "safe_to_inpaint", False))
            or bool(getattr(box, "needs_review", False))
            or not bool(getattr(box, "verified_mask", False))
            for box in combined
        ):
            return False, "incompatible", 0

        semantic_types = {str(getattr(box, "semantic_type", "")) for box in combined}
        source_roles = {str(getattr(box, "source_role", "")) for box in combined}
        if len(semantic_types) > 1 or len(source_roles) > 1:
            return False, "incompatible", 0

        first_bounds = cls._cluster_bounds(first)
        second_bounds = cls._cluster_bounds(second)
        gap_x, gap_y = cls._gap(first_bounds, second_bounds)
        if max(gap_x, gap_y) > int(INPAINT_COALESCE_MAX_GAP):
            return False, "geometry", 0

        union = (
            min(first_bounds[0], second_bounds[0]),
            min(first_bounds[1], second_bounds[1]),
            max(first_bounds[2], second_bounds[2]),
            max(first_bounds[3], second_bounds[3]),
        )
        union_w = max(0, union[2] - union[0])
        union_h = max(0, union[3] - union[1])
        if max(union_w, union_h) > int(INPAINT_COALESCE_MAX_DIM):
            return False, "geometry", 0

        first_area = max(1, (first_bounds[2] - first_bounds[0]) * (first_bounds[3] - first_bounds[1]))
        second_area = max(1, (second_bounds[2] - second_bounds[0]) * (second_bounds[3] - second_bounds[1]))
        union_area = union_w * union_h
        allowed = first_area + second_area + max(
            int(INPAINT_COALESCE_UNION_SLACK_PIXELS),
            int(round((first_area + second_area) * INPAINT_COALESCE_UNION_SLACK_RATIO)),
        )
        if union_area > allowed:
            return False, "geometry", 0
        return True, "ok", union_area - first_area - second_area

    def _coalesce_cluster_groups(
        self,
        base_clusters: list[list],
    ) -> list[list[list]]:
        """Return merged clusters while retaining their original components."""
        if not INPAINT_COALESCE_ENABLED or len(base_clusters) < 2:
            return [[list(cluster)] for cluster in base_clusters]

        result: list[list[list]] = []
        for cluster in base_clusters:
            merged_index = None
            best_overhead = None
            for index, existing_components in enumerate(result):
                existing = [
                    box
                    for component in existing_components
                    for box in component
                ]
                self._metric_add("coalesce_evaluations")
                ok, reason, overhead = self._compatible(existing, cluster)
                if not ok:
                    self._metric_add(
                        "coalesce_rejected_incompatible"
                        if reason == "incompatible"
                        else "coalesce_rejected_geometry"
                    )
                    continue
                if best_overhead is None or overhead < best_overhead:
                    merged_index = index
                    best_overhead = overhead
            if merged_index is None:
                result.append([list(cluster)])
            else:
                result[merged_index].append(list(cluster))
                self._metric_add("coalesce_merges")

        result.sort(
            key=lambda components: (
                min(box.y1 for component in components for box in component),
                min(box.x1 for component in components for box in component),
            )
        )
        self._metrics_local.value["coalesce_saved_clusters"] = max(
            0, len(base_clusters) - len(result)
        )
        if len(base_clusters) != len(result):
            baseline_pixels = sum(
                max(0, self._cluster_bounds(group)[2] - self._cluster_bounds(group)[0])
                * max(0, self._cluster_bounds(group)[3] - self._cluster_bounds(group)[1])
                for group in base_clusters
            )
            merged_pixels = sum(
                max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
                for components in result
                for bounds in [
                    self._cluster_bounds(
                        [box for component in components for box in component]
                    )
                ]
            )
            self._metrics_local.value["coalesce_union_overhead_pixels"] = max(
                0, int(merged_pixels - baseline_pixels)
            )
        return result

    def _cluster_boxes(self, boxes: list) -> list[list]:
        base_clusters = Inpainter._cluster_boxes(boxes)
        groups = self._coalesce_cluster_groups(base_clusters)
        return [
            [box for component in components for box in component]
            for components in groups
        ]

    def _component_plan(
        self,
        image: np.ndarray,
        component: list,
        width: int,
        height: int,
        protected_regions: list[dict] | None,
    ) -> dict:
        bx1 = min(box.x1 for box in component)
        by1 = min(box.y1 for box in component)
        bx2 = max(box.x2 for box in component)
        by2 = max(box.y2 for box in component)
        crop_box = self._compute_crop_region(
            bx1, by1, bx2, by2, width, height
        )
        cx1, cy1, cx2, cy2 = crop_box
        local_boxes = [
            self._local_box(box, cx1, cy1) for box in component
        ]
        local_mask = build_mask(
            (cy2 - cy1, cx2 - cx1),
            local_boxes,
            image[cy1:cy2, cx1:cx2],
        )
        local_mask = self._subtract_protected_regions(
            local_mask, crop_box, protected_regions
        )
        return {
            "crop_box": crop_box,
            "mask": local_mask,
            "crop": image[cy1:cy2, cx1:cx2].copy(),
        }

    @staticmethod
    def _prepare_lama_canvas(
        crop: np.ndarray,
        local_mask: np.ndarray,
        *,
        dynamic: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        crop_h, crop_w = crop.shape[:2]
        if dynamic:
            scale = min(
                1.0,
                DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM / max(crop_h, crop_w),
            )
            new_h = max(1, int(round(crop_h * scale)))
            new_w = max(1, int(round(crop_w * scale)))
            if (new_h, new_w) != (crop_h, crop_w):
                resized = cv2.resize(
                    crop, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                resized_mask = Inpainter._resize_mask_preserve_support(
                    local_mask, new_w, new_h
                )
            else:
                resized = crop
                resized_mask = local_mask
        else:
            scale = INPAINT_SIZE / max(crop_h, crop_w)
            new_h = max(1, int(round(crop_h * scale)))
            new_w = max(1, int(round(crop_w * scale)))
            resized = cv2.resize(
                crop,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            resized_mask = Inpainter._resize_mask_preserve_support(
                local_mask, new_w, new_h
            )

        canvas_size = INPAINT_SIZE if not dynamic else None
        canvas_h = (
            int(canvas_size)
            if canvas_size is not None
            else max(8, ((new_h + 7) // 8) * 8)
        )
        canvas_w = (
            int(canvas_size)
            if canvas_size is not None
            else max(8, ((new_w + 7) // 8) * 8)
        )
        if new_h > canvas_h or new_w > canvas_w:
            raise ValueError(
                f"LaMa canvas exceeds configured bounds: {(new_h, new_w)}"
            )
        pad_y = (canvas_h - new_h) // 2
        pad_x = (canvas_w - new_w) // 2
        canvas = cv2.copyMakeBorder(
            resized,
            pad_y,
            canvas_h - new_h - pad_y,
            pad_x,
            canvas_w - new_w - pad_x,
            cv2.BORDER_REPLICATE,
        )
        mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_mask
        return canvas, mask_canvas, (pad_x, pad_y, new_w, new_h)

    @staticmethod
    def _restore_lama_canvas(
        painted: np.ndarray,
        crop_shape: tuple[int, int],
        content: tuple[int, int, int, int],
    ) -> np.ndarray:
        pad_x, pad_y, width, height = content
        restored = painted[pad_y:pad_y + height, pad_x:pad_x + width]
        crop_h, crop_w = crop_shape
        if restored.shape[:2] != (crop_h, crop_w):
            restored = cv2.resize(
                restored, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC
            )
        return restored

    def _prepare_batch_item(self, item: dict) -> np.ndarray:
        """Prepare one independent crop, retaining the E5 dynamic ROI."""
        model_crop = item["crop"]
        model_mask = item["mask"]
        model_box = item["crop_box"]
        region = self._dynamic_roi_region(
            model_crop, model_mask, model_box
        )
        if region is not None:
            (
                model_crop,
                model_mask,
                model_box,
                source_pixels,
                roi_pixels,
            ) = region
            self._metric_add("roi_lama_regions")
            self._metric_add("roi_lama_source_pixels", source_pixels)
            self._metric_add("roi_lama_input_pixels", roi_pixels)
            self._metric_add(
                "roi_lama_saved_pixels", source_pixels - roi_pixels
            )
        item["model_crop"] = model_crop
        item["model_mask"] = model_mask
        item["model_box"] = model_box
        canvas, mask_canvas, content = self._prepare_lama_canvas(
            model_crop, model_mask, dynamic=self.dynamic_lama
        )
        item["mask_canvas"] = mask_canvas
        item["content"] = content
        return canvas

    def _paint_lama_batch(
        self,
        image: np.ndarray,
        plans: list[dict],
    ) -> np.ndarray:
        """Paint high-risk plans in bounded batches, with safe fallback."""
        if not plans:
            return image
        self._ensure_session()
        pending: list[dict] = []

        def flush(items: list[dict]) -> None:
            nonlocal image
            if not items:
                return
            canvases = [item["canvas"] for item in items]
            masks = [item["mask_canvas"] for item in items]
            try:
                painted = self._run_lama_batch(canvases, masks)
            except Exception:
                self._metric_add("coalesce_batch_fallbacks")
                for item in items:
                    self._metric_add("lama_regions")
                    image = Inpainter._smart_paint_region(
                        self,
                        image,
                        item["mask"],
                        item["crop_box"],
                        force_lama=True,
                    )
                return

            self._metric_add("coalesce_batch_groups")
            self._metric_add("coalesce_batch_items", len(items))
            self._metric_add("lama_regions", len(items))
            padded_area = max(
                int(canvas.shape[0] * canvas.shape[1])
                for canvas in canvases
            ) * len(canvases)
            source_area = sum(
                int(canvas.shape[0] * canvas.shape[1]) for canvas in canvases
            )
            self._metric_add(
                "coalesce_batch_padding_pixels",
                max(0, padded_area - source_area),
            )
            for item, painted_canvas in zip(items, painted):
                restored = self._restore_lama_canvas(
                    painted_canvas,
                    item["model_crop"].shape[:2],
                    item["content"],
                )
                x1, y1, x2, y2 = item["model_box"]
                original = image[y1:y2, x1:x2]
                mask_3d = (item["model_mask"] > 127)[:, :, None]
                image[y1:y2, x1:x2] = np.where(
                    mask_3d, restored, original
                )

        for item in plans:
            canvas = self._prepare_batch_item(item)
            if pending:
                current_max_h = max(
                    int(pending_item["canvas"].shape[0])
                    for pending_item in pending
                )
                current_max_w = max(
                    int(pending_item["canvas"].shape[1])
                    for pending_item in pending
                )
                proposed_max_h = max(current_max_h, int(canvas.shape[0]))
                proposed_max_w = max(current_max_w, int(canvas.shape[1]))
                proposed_area = proposed_max_h * proposed_max_w * (len(pending) + 1)
                source_area = sum(
                    int(pending_item["canvas"].shape[0] * pending_item["canvas"].shape[1])
                    for pending_item in pending
                ) + int(canvas.shape[0] * canvas.shape[1])
                padding_ratio = (
                    (proposed_area - source_area) / float(max(1, source_area))
                )
                if (
                    len(pending) >= int(INPAINT_COALESCE_MAX_BATCH_SIZE)
                    or padding_ratio > float(INPAINT_COALESCE_MAX_PADDING_RATIO)
                ):
                    flush(pending)
                    pending = []
            item["canvas"] = canvas
            pending.append(item)
        flush(pending)
        return image

    def inpaint(
        self,
        image: np.ndarray,
        boxes: list,
        *,
        protected_regions: list[dict] | None = None,
    ) -> np.ndarray:
        """Coalesce model contexts while preserving per-cluster mask authority.

        ``build_mask`` can choose a larger adaptive dilation when its crop
        contains textured content.  Rebuilding one mask over a merged crop
        would therefore make the candidate's authority depend on coalescing.
        Each pre-coalesce component is built in its original crop and then
        copied into the shared context.  LaMa sees one context; ownership stays
        byte-for-byte tied to the control components.
        """
        self._begin_metrics(boxes=len(boxes))
        if not boxes:
            return image.copy()

        result = image.copy()
        remaining = []
        for box in boxes:
            if self._try_bubble_fast_fill(result, box, protected_regions):
                continue
            remaining.append(box)
        if not remaining:
            self._metrics_local.value["clusters"] = 0
            return result

        height, width = result.shape[:2]
        base_clusters = Inpainter._cluster_boxes(remaining)
        split_clusters: list[list] = []
        for cluster in base_clusters:
            parts = self._split_oversized_cluster_area(cluster, width, height)
            if len(parts) > 1:
                self._metric_add("split_clusters", len(parts) - 1)
            split_clusters.extend(parts)
        groups = self._coalesce_cluster_groups(split_clusters)
        self._metrics_local.value["clusters"] = len(groups)

        for components in groups:
            if len(components) == 1:
                plan = self._component_plan(
                    result,
                    components[0],
                    width,
                    height,
                    protected_regions,
                )
                result = super()._smart_paint_region(
                    result, plan["mask"], plan["crop_box"]
                )
                continue

            batch_plans: list[dict] = []
            for component in components:
                plan = self._component_plan(
                    result,
                    component,
                    width,
                    height,
                    protected_regions,
                )
                risk = self.assess_risk(plan["crop"], plan["mask"])
                self._observe_risk(risk)
                if risk.level == "low":
                    result = Inpainter._smart_paint_region(
                        self,
                        result,
                        plan["mask"],
                        plan["crop_box"],
                    )
                else:
                    self._metric_add("risk_forced_lama_regions")
                    batch_plans.append(plan)

            if len(batch_plans) == 1:
                self._metric_add("lama_regions")
                result = Inpainter._smart_paint_region(
                    self,
                    result,
                    batch_plans[0]["mask"],
                    batch_plans[0]["crop_box"],
                    force_lama=True,
                )
            elif batch_plans:
                result = self._paint_lama_batch(result, batch_plans)

        return result


__all__ = [
    "CoalescingRiskAwareInpainter",
    "InpaintRisk",
    "RiskAwareFastInpainter",
    "measure_inpaint_risk",
]
