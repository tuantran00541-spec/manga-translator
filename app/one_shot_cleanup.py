from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from app.config import TEXT_SEGMENTER_MODEL
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.inpaint.lama_inpainter import Inpainter
from app.parameters import TEXT_CONF_THRESHOLD


@dataclass
class OneShotCleanupResult:
    image: np.ndarray
    mask: np.ndarray
    roi: tuple[int, int, int, int] | None
    metrics: dict[str, float | int | list[int]]


class OneShotTextMaskDetector:
    """Deliberately dumb detector path: one text-segmenter forward per image.

    There is no bubble detector, TTA, tall-image windowing, recovery, grayscale
    retry, residue verification, semantic promotion, or review path here.
    The text segmenter is asked one question only: which pixels look like manga
    text? That includes text inside speech bubbles and free text.
    """

    def __init__(self, detector: YoloDetector | None = None):
        self.detector = detector or YoloDetector(
            TEXT_SEGMENTER_MODEL,
            TEXT_CONF_THRESHOLD,
            use_tta=False,
            model_role="text_segmenter",
        )

    @staticmethod
    def _union_verified_masks(
        image_shape: tuple[int, int],
        boxes: list[BubbleBox],
    ) -> tuple[np.ndarray, int]:
        height, width = (int(v) for v in image_shape)
        page_mask = np.zeros((height, width), dtype=np.uint8)
        accepted = 0

        for box in boxes:
            if box.source_role != "text_segmenter" or not box.verified_mask:
                continue

            x1 = max(0, min(width, int(box.x1)))
            y1 = max(0, min(height, int(box.y1)))
            x2 = max(0, min(width, int(box.x2)))
            y2 = max(0, min(height, int(box.y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            box_w = max(1, int(box.x2) - int(box.x1))
            box_h = max(1, int(box.y2) - int(box.y1))
            local_mask = box.mask
            if local_mask is None:
                continue
            if local_mask.shape != (box_h, box_w):
                local_mask = cv2.resize(
                    local_mask,
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            src_x1 = x1 - int(box.x1)
            src_y1 = y1 - int(box.y1)
            src_x2 = src_x1 + (x2 - x1)
            src_y2 = src_y1 + (y2 - y1)
            clipped = local_mask[src_y1:src_y2, src_x1:src_x2]
            if clipped.shape != (y2 - y1, x2 - x1):
                continue
            if not np.any(clipped > 127):
                continue

            target = page_mask[y1:y2, x1:x2]
            page_mask[y1:y2, x1:x2] = np.maximum(
                target,
                (clipped > 127).astype(np.uint8) * 255,
            )
            accepted += 1

        return page_mask, accepted

    def detect_mask(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
        started = time.perf_counter()

        # Intentionally bypass YoloDetector.detect(). That production method may
        # split tall images, add a full-image retry and run TTA. The experiment
        # requires exactly one detector forward for the entire slice.
        boxes = self.detector._detect_single(image, 0, 0)
        mask, accepted = self._union_verified_masks(image.shape[:2], boxes)

        return mask, {
            "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "detector_forward_calls": 1,
            "detector_boxes": int(len(boxes)),
            "accepted_mask_boxes": int(accepted),
            "mask_pixels": int(np.count_nonzero(mask > 127)),
        }


class OneShotCleanupPipeline:
    """The dumbest cleanup path: detector mask -> whole image -> one LaMa call."""

    def __init__(
        self,
        *,
        detector: OneShotTextMaskDetector | None = None,
        inpainter: Inpainter | None = None,
        padding: int = 0,
    ):
        self.detector = detector or OneShotTextMaskDetector()
        self.inpainter = inpainter or Inpainter()
        # Kept only so the existing benchmark CLI stays compatible. Deliberately
        # unused: this experiment does not crop, cluster, pad, or plan regions.
        self.padding = int(padding)

    def clean(self, image: np.ndarray) -> OneShotCleanupResult:
        total_started = time.perf_counter()
        mask, detector_metrics = self.detector.detect_mask(image)

        if not np.any(mask > 127):
            metrics = {
                **detector_metrics,
                "inpaint_ms": 0.0,
                "lama_model_runs": 0,
                "roi_pixels": 0,
                "total_ms": round((time.perf_counter() - total_started) * 1000.0, 3),
            }
            return OneShotCleanupResult(image.copy(), mask, None, metrics)

        begin_metrics = getattr(self.inpainter, "_begin_metrics", None)
        if callable(begin_metrics):
            begin_metrics()

        inpaint_started = time.perf_counter()
        # No ROI, no components, no merge, no SmartFill, no tile planner.
        # Give the whole slice and the union text mask to LaMa once.
        painted = self.inpainter._lama_fill_single(image, mask)
        inpaint_ms = (time.perf_counter() - inpaint_started) * 1000.0

        if painted.shape != image.shape:
            raise ValueError(
                f"One-shot LaMa returned {painted.shape}; expected {image.shape}"
            )

        output = image.copy()
        authority = mask > 127
        output[authority] = painted[authority]

        last_metrics = getattr(self.inpainter, "last_metrics", None)
        lama_metrics = last_metrics() if callable(last_metrics) else {}
        lama_runs = int(lama_metrics.get("lama_model_runs", 1))

        height, width = image.shape[:2]
        full_frame = (0, 0, int(width), int(height))
        metrics = {
            **detector_metrics,
            "inpaint_ms": round(inpaint_ms, 3),
            "lama_model_runs": lama_runs,
            "roi_pixels": int(width * height),
            "roi_shape": [int(height), int(width)],
            "total_ms": round((time.perf_counter() - total_started) * 1000.0, 3),
        }
        return OneShotCleanupResult(output, mask, full_frame, metrics)
