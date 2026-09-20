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
    """One cheap full-slice pass with a bounded zero-box fallback.

    Normal slices still cost exactly one text-segmenter forward. Only when the
    verified full-slice pass returns no usable text mask do we spend four more
    forwards on an overlapping 2x2 grid. The fallback uses the same segmenter,
    the same confidence threshold and the same verified-mask authority contract.
    """

    FALLBACK_TILE_RATIO = 0.60

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

    @classmethod
    def _fallback_windows(
        cls,
        image: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []

        tile_w = min(w, max(1, int(round(w * cls.FALLBACK_TILE_RATIO))))
        tile_h = min(h, max(1, int(round(h * cls.FALLBACK_TILE_RATIO))))

        x_starts = sorted({0, max(0, w - tile_w)})
        y_starts = sorted({0, max(0, h - tile_h)})
        return [
            (x, y, min(w, x + tile_w), min(h, y + tile_h))
            for y in y_starts
            for x in x_starts
        ]

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
            for x1, y1, x2, y2 in self._fallback_windows(image):
                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                fallback_forward_calls += 1
                fallback_raw_boxes.extend(
                    self.detector._detect_single(tile, x1, y1)
                )

            if fallback_raw_boxes:
                # The detector's existing NMS understands global page geometry
                # because every tile was decoded with page offsets.
                fallback_raw_boxes = self.detector._nms_boxes(fallback_raw_boxes)
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
