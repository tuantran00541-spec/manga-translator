from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_CHIPS,
    FOCUS_SOURCE_PIXEL_BUDGET,
    FOCUS_TENSOR_PIXEL_BUDGET,
    AdaptiveFocusCombinedTextDetector,
    _adaptive_detect,
    _proposal_has_full_text,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.parameters import (
    DETECTOR_FOCUS_HARD_MAX_CHIPS,
    DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP,
    DETECTOR_INPUT_SIZE,
    DETECTOR_TALL_IMAGE_FACTOR,
)


def _finish_focus_from_full(
    detector: YoloDetector,
    image: np.ndarray,
    proposals: list[BubbleBox],
    full_boxes: list[BubbleBox],
) -> tuple[list[BubbleBox], dict[str, int], list[BubbleBox], float]:
    """Finish tall-page focus retries after a prefetched full text pass.

    This mirrors ``_focus_text_detect`` after its full-image call. Keeping it as
    a separate helper lets the bubble model and the independent full text pass
    overlap without changing how uncovered proposals, focus budgets, NMS or
    deferred review regions are decided.
    """
    h, w = image.shape[:2]
    uncovered = [
        proposal
        for proposal in proposals
        if not _proposal_has_full_text(proposal, full_boxes)
    ]
    fallback_image = image if not proposals and not full_boxes else None
    extra = max(0, len(uncovered) - 1) // DETECTOR_FOCUS_PROPOSALS_PER_EXTRA_CHIP
    adaptive_max_chips = min(
        DETECTOR_FOCUS_HARD_MAX_CHIPS,
        FOCUS_MAX_CHIPS + extra,
    )
    scale = adaptive_max_chips / float(max(1, FOCUS_MAX_CHIPS))
    chips, deferred = plan_focus_chips(
        h,
        w,
        uncovered,
        max_chips=adaptive_max_chips,
        source_pixel_budget=int(round(FOCUS_SOURCE_PIXEL_BUDGET * scale)),
        tensor_pixel_budget=int(round(FOCUS_TENSOR_PIXEL_BUDGET * scale)),
        fallback_image=fallback_image,
    )

    chip_started = time.perf_counter()
    all_boxes = list(full_boxes)
    for x1, y1, x2, y2 in chips:
        crop = image[y1:y2, x1:x2]
        if crop.size:
            all_boxes.extend(detector._detect_single_plain(crop, x1, y1))
    chip_ms = (time.perf_counter() - chip_started) * 1000.0

    boxes = detector._nms_boxes(all_boxes)
    result = [
        detector._with_semantics(box)
        for box in detector._filter_invalid(boxes, w, h)
    ]
    deferred_boxes = [
        BubbleBox(
            x1,
            y1,
            x2,
            y2,
            0.0,
            None,
            source_model="adaptive_scheduler",
            class_name="focus_deferred",
            semantic_type="review_region",
            mask_source="none",
            safe_to_inpaint=False,
            ocr_eligible=False,
            needs_review=True,
            source_role="scheduler",
            deferred_reason="focus_budget_exhausted",
        )
        for x1, y1, x2, y2 in deferred
    ]
    metrics = {
        "focus_proposals": len(proposals),
        "focus_uncovered_proposals": len(uncovered),
        "focus_chip_calls": len(chips),
        "focus_source_pixels": sum(
            (x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in chips
        ),
        "focus_tensor_pixels": len(chips)
        * DETECTOR_INPUT_SIZE
        * DETECTOR_INPUT_SIZE,
        "focus_deferred_regions": len(deferred_boxes),
        "focus_fallback_calls": int(bool(fallback_image is not None and chips)),
    }
    return result, metrics, deferred_boxes, chip_ms


class ParallelAdaptiveFocusCombinedTextDetector(AdaptiveFocusCombinedTextDetector):
    """ ``ChapterPipeline`` requests ``parallel=True`` only when there is a single page worker. In that case bubble detection and the text model's full-image pass are independent and may use separate CPU/OpenVINO streams. Recovery and focus retries remain sequential because focus geometry depends on bubble/MSER proposals. With ``parallel=False`` this class is byte-for-byte behaviorally delegated to the validated adaptive detector. """

    def detect(
        self,
        image: np.ndarray,
        *,
        parallel: bool = False,
    ) -> list[BubbleBox]:
        if not parallel:
            return super().detect(image, parallel=False)

        started_at = time.perf_counter()
        h, w = image.shape[:2]
        short_page = h <= DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR

        def bubble_prefetch():
            call_started = time.perf_counter()
            boxes = _adaptive_detect(self._bubble_model, image)
            return boxes, (time.perf_counter() - call_started) * 1000.0

        def text_prefetch():
            call_started = time.perf_counter()
            if short_page:
                raw = self._text_model._detect_single(image, 0, 0)
                boxes = [
                    self._text_model._with_semantics(box)
                    for box in self._text_model._filter_invalid(raw, w, h)
                ]
            else:
                # Keep this raw just like _focus_text_detect: geometry-only
                # completeness checks run before final NMS/semantic filtering.
                boxes = self._text_model._detect_single_plain(image, 0, 0)
            return boxes, (time.perf_counter() - call_started) * 1000.0

        prefetch_started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="detector-prefetch",
        ) as pool:
            bubble_future = pool.submit(bubble_prefetch)
            text_future = pool.submit(text_prefetch)
            bubble_boxes, bubble_ms = bubble_future.result()
            prefetched_text, text_prefetch_ms = text_future.result()
        prefetch_wall_ms = (time.perf_counter() - prefetch_started) * 1000.0

        proposal_started = time.perf_counter()
        recovery_boxes = self.recovery.detect(image, existing=bubble_boxes)
        proposal_ms = (time.perf_counter() - proposal_started) * 1000.0
        proposals = list(bubble_boxes) + list(recovery_boxes)

        if short_page:
            text_boxes = list(prefetched_text)
            focus_metrics = {
                "focus_proposals": len(proposals),
                "focus_uncovered_proposals": 0,
                "focus_chip_calls": 0,
                "focus_source_pixels": 0,
                "focus_tensor_pixels": 0,
                "focus_deferred_regions": 0,
                "focus_fallback_calls": 0,
            }
            deferred_boxes: list[BubbleBox] = []
            focus_chip_ms = 0.0
        else:
            (
                text_boxes,
                focus_metrics,
                deferred_boxes,
                focus_chip_ms,
            ) = _finish_focus_from_full(
                self._text_model,
                image,
                proposals,
                list(prefetched_text),
            )

        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = CombinedTextDetector.detect(self, image, parallel=False)

        if deferred_boxes:
            result.extend(deferred_boxes)

        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics["bubble_model_ms"] = round(bubble_ms, 3)
        metrics["text_model_ms"] = round(text_prefetch_ms + focus_chip_ms, 3)
        metrics["focus_prefetch_mser_ms"] = round(proposal_ms, 3)
        metrics["focus_prefetch_proposals"] = len(recovery_boxes)
        metrics["mser_ms"] = round(
            float(metrics.get("mser_ms", 0.0)) + proposal_ms,
            3,
        )
        metrics.update(focus_metrics)
        metrics["parallel_prefetch_enabled"] = 1
        metrics["parallel_prefetch_wall_ms"] = round(prefetch_wall_ms, 3)
        metrics["parallel_prefetch_overlap_ms"] = round(
            max(0.0, bubble_ms + text_prefetch_ms - prefetch_wall_ms),
            3,
        )
        metrics["total_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0,
            3,
        )
        self._metrics_local.value = metrics
        return result
