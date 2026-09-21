from __future__ import annotations

from dataclasses import replace
import os
import threading
import time

import numpy as np

from app.config import YOLO26_SEG_MODEL
from app.detector.bubble_detector import BubbleBox, YoloDetector


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


class ExtremeYolo26TextDetector:
    """YOLO26-only experiment with frame/balloon-guided text focus.

    The first pass keeps Manga109 frame/text/balloon boxes, but only verified
    class=text masks can ever become destructive authority.  Frame and balloon
    are bbox-only context.  On a primary zero-text slice, up to a tiny bounded
    number of context crops are re-run through the same YOLO26 model so text is
    presented at a larger effective scale.  No legacy detector participates.
    """

    MODEL_ROLE = "manga109_yolo26_seg"

    def __init__(self, conf_threshold: float | None = None):
        self.conf_threshold = float(
            conf_threshold
            if conf_threshold is not None
            else os.getenv("MANGA_YOLO26_TEXT_CONF", "0.01")
        )
        self.focus_conf_threshold = _env_float(
            "MANGA_YOLO26_FOCUS_TEXT_CONF",
            0.006,
            minimum=0.0001,
            maximum=self.conf_threshold,
        )
        self.focus_max_regions = _env_int(
            "MANGA_YOLO26_CONTEXT_MAX_REGIONS",
            2,
            minimum=0,
            maximum=8,
        )
        self.focus_pad_ratio = _env_float(
            "MANGA_YOLO26_CONTEXT_PAD_RATIO",
            0.16,
            minimum=0.0,
            maximum=0.75,
        )
        self.focus_max_area_ratio = _env_float(
            "MANGA_YOLO26_CONTEXT_MAX_AREA_RATIO",
            0.72,
            minimum=0.05,
            maximum=1.0,
        )
        self._detector = YoloDetector(
            YOLO26_SEG_MODEL,
            self.conf_threshold,
            use_tta=False,
            model_role=self.MODEL_ROLE,
        )
        self._local = threading.local()

    @staticmethod
    def _authority_text_boxes(raw: list[BubbleBox]) -> list[BubbleBox]:
        boxes: list[BubbleBox] = []
        for box in raw:
            if box.class_name != "text" or not box.verified_mask:
                continue
            boxes.append(
                replace(
                    box,
                    semantic_type="text",
                    mask_source=ExtremeYolo26TextDetector.MODEL_ROLE,
                    safe_to_inpaint=True,
                    ocr_eligible=True,
                    needs_review=False,
                    source_role=ExtremeYolo26TextDetector.MODEL_ROLE,
                    deferred_reason=None,
                )
            )
        return boxes

    @staticmethod
    def _region_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aa = max(1, (ax2 - ax1) * (ay2 - ay1))
        bb = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(max(1, aa + bb - inter))

    def _context_regions(
        self,
        raw: list[BubbleBox],
        image_shape: tuple[int, ...],
    ) -> list[tuple[int, int, int, int]]:
        if self.focus_max_regions <= 0:
            return []

        h, w = image_shape[:2]
        page_area = max(1, h * w)
        contexts = [
            box
            for box in raw
            if box.class_name in {"balloon", "frame"}
        ]
        # Balloon is the strongest local text prior. Frame is a fallback when
        # a page has no useful balloon proposal.
        contexts.sort(
            key=lambda box: (
                0 if box.class_name == "balloon" else 1,
                -float(box.confidence),
            )
        )

        regions: list[tuple[int, int, int, int]] = []
        for box in contexts:
            bw = max(1, box.x2 - box.x1)
            bh = max(1, box.y2 - box.y1)
            pad = max(16, int(round(max(bw, bh) * self.focus_pad_ratio)))
            x1 = max(0, box.x1 - pad)
            y1 = max(0, box.y1 - pad)
            x2 = min(w, box.x2 + pad)
            y2 = min(h, box.y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(page_area)
            if area_ratio > self.focus_max_area_ratio:
                continue
            region = (x1, y1, x2, y2)
            if any(self._region_iou(region, prev) >= 0.70 for prev in regions):
                continue
            regions.append(region)
            if len(regions) >= self.focus_max_regions:
                break
        return regions

    def _focused_text_retry(
        self,
        image: np.ndarray,
        regions: list[tuple[int, int, int, int]],
    ) -> tuple[list[BubbleBox], float]:
        recovered: list[BubbleBox] = []
        started = time.perf_counter()
        old_threshold = float(self._detector.conf_threshold)
        try:
            # detect() is intentionally single-threaded in this experiment, so
            # a scoped threshold override does not race another inference.
            self._detector.conf_threshold = self.focus_conf_threshold
            for x1, y1, x2, y2 in regions:
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_raw = self._detector.detect(crop)
                for box in self._authority_text_boxes(crop_raw):
                    recovered.append(
                        replace(
                            box,
                            x1=box.x1 + x1,
                            y1=box.y1 + y1,
                            x2=box.x2 + x1,
                            y2=box.y2 + y1,
                        )
                    )
        finally:
            self._detector.conf_threshold = old_threshold
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return recovered, elapsed_ms

    def detect(
        self,
        image: np.ndarray,
        parallel: bool = False,
    ) -> list[BubbleBox]:
        del parallel
        started = time.perf_counter()

        primary_started = time.perf_counter()
        raw = self._detector.detect(image)
        primary_ms = (time.perf_counter() - primary_started) * 1000.0
        primary_text = self._authority_text_boxes(raw)

        frame_contexts = sum(box.class_name == "frame" for box in raw)
        balloon_contexts = sum(box.class_name == "balloon" for box in raw)

        focus_regions: list[tuple[int, int, int, int]] = []
        recovered: list[BubbleBox] = []
        focus_ms = 0.0
        if not primary_text:
            focus_regions = self._context_regions(raw, image.shape)
            if focus_regions:
                recovered, focus_ms = self._focused_text_retry(image, focus_regions)

        boxes = primary_text + recovered
        if len(boxes) > 1:
            boxes = self._detector._nms_boxes(boxes)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._local.metrics = {
            "bubble_model_ms": 0.0,
            "text_model_ms": round(elapsed_ms, 3),
            "primary_model_ms": round(primary_ms, 3),
            "context_focus_ms": round(focus_ms, 3),
            "mser_ms": 0.0,
            "text_grayscale_fallback_runs": 0,
            "text_grayscale_fallback_ms": 0.0,
            "total_ms": round(elapsed_ms, 3),
            "raw_instances": int(len(raw)),
            "frame_contexts": int(frame_contexts),
            "balloon_contexts": int(balloon_contexts),
            "primary_text_instances": int(len(primary_text)),
            "context_focus_calls": int(len(focus_regions)),
            "context_recovered_text": int(len(recovered)),
            "text_instances": int(len(boxes)),
            "mask_pixels": int(
                sum(
                    np.count_nonzero(box.mask > 0)
                    for box in boxes
                    if box.mask is not None
                )
            ),
            "confidence_threshold": float(self.conf_threshold),
            "focus_conf_threshold": float(self.focus_conf_threshold),
            "text_segmenter_calls": 0,
            "bubble_yolo_calls": 0,
            "recovery_calls": 0,
            "residue_verify_calls": 0,
        }
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._local, "metrics", {}))

    def verify_post_inpaint_residue(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
    ) -> list[BubbleBox]:
        del image, boxes
        self._local.residue_metrics = {
            "disabled": 1,
            "regions": 0,
        }
        return []

    def last_residue_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._local, "residue_metrics", {}))
