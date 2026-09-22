from __future__ import annotations

from dataclasses import replace
import threading
import time

import numpy as np

from app.config import TEXT_SEGMENTER_MODEL
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.fast_residue_detector import FastResidueAdaptiveFocusCombinedTextDetector
from app.parameters import TEXT_CONF_THRESHOLD


class OneShotTextMaskDetector:
    """One ONNX forward; decode normal first, then rescue from the same outputs.

    The expensive model forward always happens exactly once. We first postprocess
    the captured ONNX outputs at the normal text threshold, then decode the same
    tensors at the lower rescue threshold and append only genuinely low-confidence
    verified masks. This recovers weak free text even on pages that already contain
    normal-confidence dialogue without paying for another model inference.
    """

    RESCUE_CONF_THRESHOLD = 0.12

    def __init__(self, detector: YoloDetector | None = None):
        self.detector = detector or YoloDetector(
            TEXT_SEGMENTER_MODEL,
            TEXT_CONF_THRESHOLD,
            use_tta=False,
            model_role="text_segmenter",
        )
        self._decode_lock = threading.Lock()

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

    def _run_session(self, blob):
        """Small seam so benchmarks/tests can count the real ONNX call."""
        return self.detector.session.run(
            None,
            {self.detector.input_name: blob},
        )

    def _single_forward_outputs(self, image: np.ndarray):
        """Run preprocess + ONNX once and return raw outputs + geometry."""
        blob, transform = self.detector._preprocess(image, offset_x=0, offset_y=0)
        if blob is None or transform is None:
            return None, None
        outputs = self._run_session(blob)
        return outputs, transform

    def _postprocess_at_threshold(
        self,
        outputs,
        transform,
        threshold: float,
    ) -> list[BubbleBox]:
        with self._decode_lock:
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

        # Tail decode: no second inference. Always inspect the already-captured
        # tensors at the rescue threshold. Keeping only boxes below the normal
        # threshold avoids duplicating the normal-confidence detections returned
        # by the same NMS pass while recovering weak text on mixed-confidence pages.
        rescue_raw = self._postprocess_at_threshold(
            outputs,
            transform,
            min(TEXT_CONF_THRESHOLD, self.RESCUE_CONF_THRESHOLD),
        )
        rescue_boxes = self._accept_many(rescue_raw)
        low_conf_boxes = [
            box
            for box in rescue_boxes
            if float(box.confidence) < float(TEXT_CONF_THRESHOLD)
        ]
        # A low detector score is useful recall evidence, but the real-chapter
        # visual audit found that directly granting it destructive authority can
        # damage artwork. Keep weak masks review-only; the post-inpaint verifier
        # gets a focused second look and may promote only a freshly verified
        # text-segmenter mask into the repair pass.
        low_conf_review = [
            replace(
                box,
                safe_to_inpaint=False,
                ocr_eligible=True,
                needs_review=True,
                deferred_reason="low_confidence_unconfirmed",
            )
            for box in low_conf_boxes
        ]
        boxes = [*normal_boxes, *low_conf_review]
        low_conf_rescue = bool(low_conf_boxes)

        raw_count = len(rescue_raw)
        return boxes, {
            "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "detector_forward_calls": 1,
            "detector_boxes": int(raw_count),
            "accepted_mask_boxes": int(len(boxes)),
            "normal_conf_boxes": int(len(normal_boxes)),
            "low_conf_rescue": int(low_conf_rescue),
            "low_conf_rescue_boxes": int(len(low_conf_boxes)),
            "rescue_conf_threshold": float(self.RESCUE_CONF_THRESHOLD),
        }

class OneShotProductionDetector(FastResidueAdaptiveFocusCombinedTextDetector):
    """Production adapter for the one-forward text-mask detector.

    Detection itself uses only ``text_segmenter.onnx``. The inherited fast
    residue verifier is retained because it only needs ``self.text_detector``;
    bubble YOLO, MSER recovery, adaptive focus and their model sessions are
    never constructed here.
    """

    def __init__(self):
        self.core = OneShotTextMaskDetector()
        self.text_detector = self.core.detector
        self._metrics_local = threading.local()
        self._residue_metrics_local = threading.local()
        self._residue_metrics_lock = threading.Lock()
        self._residue_totals: dict[str, int] = {}
        self._residue_flat_gate_enabled = True

    def detect(self, image: np.ndarray, *, parallel: bool = False) -> list[BubbleBox]:
        boxes, metrics = self.core.detect(image)
        detector_ms = float(metrics.get("detector_ms", 0.0))
        self._metrics_local.value = {
            "bubble_model_ms": 0.0,
            "text_model_ms": detector_ms,
            "mser_ms": 0.0,
            "text_grayscale_fallback_runs": 0,
            "text_grayscale_fallback_calls": 0,
            "text_grayscale_fallback_source_pixels": 0,
            "text_grayscale_fallback_deferred_regions": 0,
            "text_grayscale_fallback_ms": 0.0,
            "text_grayscale_fallback_proposals": 0,
            "mser_segmenter_promotion_calls": 0,
            "mser_segmenter_promotions": 0,
            "mser_segmenter_promotion_deferred_regions": 0,
            "result_boxes": int(len(boxes)),
            "review_boxes": int(sum(1 for box in boxes if box.needs_review)),
            "detector_forward_calls": int(metrics.get("detector_forward_calls", 1)),
            "normal_conf_boxes": int(metrics.get("normal_conf_boxes", 0)),
            "low_conf_rescue": int(metrics.get("low_conf_rescue", 0)),
            "low_conf_rescue_boxes": int(metrics.get("low_conf_rescue_boxes", 0)),
            "total_ms": detector_ms,
        }
        return boxes
