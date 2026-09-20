from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from app.config import TEXT_SEGMENTER_MODEL
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.mask_builder import build_mask
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
from app.parameters import TEXT_CONF_THRESHOLD


@dataclass
class OneShotCleanupResult:
    image: np.ndarray
    mask: np.ndarray
    boxes: list[BubbleBox]
    metrics: dict[str, float | int]


class OneShotTextMaskDetector:
    """One ONNX forward; decode normal first, then rescue from the same outputs.

    The expensive model forward always happens exactly once. We first postprocess
    the captured ONNX outputs at the normal text threshold. Only if that yields
    no verified cleanup authority do we re-run postprocess on the *same tensors*
    at the lower rescue threshold. This keeps the fast path identical to the
    original one-shot detector while rescuing known low-confidence text without
    another inference.
    """

    RESCUE_CONF_THRESHOLD = 0.12

    def __init__(self, detector: YoloDetector | None = None):
        self.detector = detector or YoloDetector(
            TEXT_SEGMENTER_MODEL,
            TEXT_CONF_THRESHOLD,
            use_tta=False,
            model_role="text_segmenter",
        )

    @staticmethod
    def _accept(box: BubbleBox) -> BubbleBox | None:
        if box.source_role != "text_segmenter" or not box.verified_mask:
            return None
        return replace(
            box,
            semantic_type="free_text",
            mask_source="text_segmenter",
            safe_to_inpaint=True,
            ocr_eligible=True,
            needs_review=False,
            deferred_reason=None,
        )

    def _accept_many(self, raw_boxes: list[BubbleBox]) -> list[BubbleBox]:
        accepted: list[BubbleBox] = []
        for raw in raw_boxes:
            box = self._accept(raw)
            if box is not None:
                accepted.append(box)
        return accepted

    def _single_forward_outputs(self, image: np.ndarray):
        """Run preprocess + ONNX once and return raw outputs + geometry."""
        blob, transform = self.detector._preprocess(image, offset_x=0, offset_y=0)
        if blob is None or transform is None:
            return None, None
        outputs = self.detector.session.run(
            None,
            {self.detector.input_name: blob},
        )
        return outputs, transform

    def _postprocess_at_threshold(
        self,
        outputs,
        transform,
        threshold: float,
    ) -> list[BubbleBox]:
        original_conf = self.detector.conf_threshold
        try:
            self.detector.conf_threshold = float(threshold)
            return self.detector._postprocess(outputs, transform)
        finally:
            self.detector.conf_threshold = original_conf

    def detect(self, image: np.ndarray) -> tuple[list[BubbleBox], dict[str, float | int]]:
        started = time.perf_counter()

        outputs, transform = self._single_forward_outputs(image)
        if outputs is None or transform is None:
            return [], {
                "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "detector_forward_calls": 0,
                "detector_boxes": 0,
                "accepted_mask_boxes": 0,
                "normal_conf_boxes": 0,
                "low_conf_rescue": 0,
                "low_conf_rescue_boxes": 0,
                "rescue_conf_threshold": float(self.RESCUE_CONF_THRESHOLD),
            }

        # Fast path: preserve the original one-shot behavior and cost.
        normal_raw = self._postprocess_at_threshold(
            outputs,
            transform,
            TEXT_CONF_THRESHOLD,
        )
        normal_boxes = self._accept_many(normal_raw)

        low_conf_rescue = False
        rescue_raw: list[BubbleBox] = []
        boxes = normal_boxes

        # Tail path: no second inference. Decode the already-captured tensors
        # again at a lower threshold only when normal authority is empty.
        if not normal_boxes:
            rescue_raw = self._postprocess_at_threshold(
                outputs,
                transform,
                min(TEXT_CONF_THRESHOLD, self.RESCUE_CONF_THRESHOLD),
            )
            rescue_boxes = self._accept_many(rescue_raw)
            if rescue_boxes:
                boxes = rescue_boxes
                low_conf_rescue = True

        raw_count = len(normal_raw) if normal_boxes else len(rescue_raw)
        return boxes, {
            "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "detector_forward_calls": 1,
            "detector_boxes": int(raw_count),
            "accepted_mask_boxes": int(len(boxes)),
            "normal_conf_boxes": int(len(normal_boxes)),
            "low_conf_rescue": int(low_conf_rescue),
            "low_conf_rescue_boxes": int(len(boxes) if low_conf_rescue else 0),
            "rescue_conf_threshold": float(self.RESCUE_CONF_THRESHOLD),
        }

    def detect_mask(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
        boxes, metrics = self.detect(image)
        mask = (
            build_mask(image.shape[:2], boxes, image)
            if boxes
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )
        return mask, {
            **metrics,
            "mask_pixels": int(np.count_nonzero(mask > 127)),
        }


class OneShotCleanupPipeline:
    """Single-forward detector with same-output rescue + production inpainter."""

    def __init__(
        self,
        *,
        detector: OneShotTextMaskDetector | None = None,
        inpainter: AdaptiveFastInpainter | None = None,
        padding: int = 0,
    ):
        self.detector = detector or OneShotTextMaskDetector()
        self.inpainter = inpainter or AdaptiveFastInpainter()
        self.padding = int(padding)

    def clean(self, image: np.ndarray) -> OneShotCleanupResult:
        total_started = time.perf_counter()
        boxes, detector_metrics = self.detector.detect(image)

        authority = (
            build_mask(image.shape[:2], boxes, image)
            if boxes
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )

        inpaint_started = time.perf_counter()
        cleaned = self.inpainter.inpaint(image, boxes)
        inpaint_ms = (time.perf_counter() - inpaint_started) * 1000.0

        inpaint_metrics = self.inpainter.last_metrics()
        metrics = {
            **detector_metrics,
            "mask_pixels": int(np.count_nonzero(authority > 127)),
            "inpaint_ms": round(inpaint_ms, 3),
            "lama_model_runs": int(inpaint_metrics.get("lama_model_runs", 0)),
            "smart_fill_regions": int(inpaint_metrics.get("smart_fill_regions", 0)),
            "bubble_fast_fill_regions": int(
                inpaint_metrics.get("bubble_fast_fill_regions", 0)
            ),
            "clusters": int(inpaint_metrics.get("clusters", 0)),
            "total_ms": round((time.perf_counter() - total_started) * 1000.0, 3),
        }
        return OneShotCleanupResult(cleaned, authority, boxes, metrics)
