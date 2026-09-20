from __future__ import annotations

from dataclasses import dataclass, replace
import time

import cv2
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
    """One cheap full-slice pass with one bounded low-confidence retry.

    Normal slices cost exactly one text-segmenter forward. Only when that pass
    returns no verified mask do we retry the same full slice once at a lower
    detector confidence. Destructive authority still requires a verified
    text-segmenter mask.
    """

    FALLBACK_CONF_THRESHOLD = 0.12

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

    def detect(self, image: np.ndarray) -> tuple[list[BubbleBox], dict[str, float | int]]:
        started = time.perf_counter()

        # Fast path: exactly one whole-slice forward.
        raw_boxes = self.detector._detect_single(image, 0, 0)
        boxes = self._accept_many(raw_boxes)

        fallback_triggered = not boxes
        fallback_raw_boxes: list[BubbleBox] = []
        fallback_forward_calls = 0

        if fallback_triggered:
            original_conf = getattr(self.detector, "conf_threshold", None)
            try:
                if original_conf is not None:
                    self.detector.conf_threshold = min(
                        float(original_conf),
                        self.FALLBACK_CONF_THRESHOLD,
                    )
                fallback_forward_calls = 1
                fallback_raw_boxes = self.detector._detect_single(image, 0, 0)
            finally:
                if original_conf is not None:
                    self.detector.conf_threshold = original_conf

            boxes = self._accept_many(fallback_raw_boxes)

        total_forwards = 1 + fallback_forward_calls
        return boxes, {
            "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "detector_forward_calls": total_forwards,
            "detector_boxes": int(len(raw_boxes) + len(fallback_raw_boxes)),
            "accepted_mask_boxes": int(len(boxes)),
            "fallback_triggered": int(fallback_triggered),
            "fallback_forward_calls": int(fallback_forward_calls),
            "fallback_boxes": int(len(fallback_raw_boxes)),
            "fallback_conf_threshold": float(self.FALLBACK_CONF_THRESHOLD),
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
    """Cheap detector with zero-box fallback + production AdaptiveFastInpainter."""

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
