from __future__ import annotations

from dataclasses import replace
import os
import threading
import time

import numpy as np

from app.config import YOLO26_SEG_MODEL
from app.detector.bubble_detector import BubbleBox, YoloDetector


class ExtremeYolo26TextDetector:
    """Extreme experiment: YOLO26 segmentation text masks are the only detector path.

    The Manga109 model has three classes: frame, text, balloon. This adapter drops
    frame/balloon completely and returns only verified class=text instance masks.
    There is no bubble detector, text_segmenter, MSER, recovery, focus pass, rescue
    pass, or residue detector in this path.
    """

    MODEL_ROLE = "manga109_yolo26_seg"

    def __init__(self, conf_threshold: float | None = None):
        self.conf_threshold = float(
            conf_threshold
            if conf_threshold is not None
            else os.getenv("MANGA_YOLO26_TEXT_CONF", "0.01")
        )
        self._detector = YoloDetector(
            YOLO26_SEG_MODEL,
            self.conf_threshold,
            use_tta=False,
            model_role=self.MODEL_ROLE,
        )
        self._local = threading.local()

    def detect(
        self,
        image: np.ndarray,
        parallel: bool = False,
    ) -> list[BubbleBox]:
        # parallel is intentionally ignored: this experiment has exactly one model.
        started = time.perf_counter()
        raw = self._detector.detect(image)

        boxes: list[BubbleBox] = []
        for box in raw:
            if box.class_name != "text" or not box.verified_mask:
                continue
            boxes.append(
                replace(
                    box,
                    semantic_type="text",
                    mask_source=self.MODEL_ROLE,
                    safe_to_inpaint=True,
                    ocr_eligible=True,
                    needs_review=False,
                    source_role=self.MODEL_ROLE,
                    deferred_reason=None,
                )
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._local.metrics = {
            "bubble_model_ms": 0.0,
            "text_model_ms": round(elapsed_ms, 3),
            "mser_ms": 0.0,
            "text_grayscale_fallback_runs": 0,
            "text_grayscale_fallback_ms": 0.0,
            "total_ms": round(elapsed_ms, 3),
            "raw_instances": int(len(raw)),
            "text_instances": int(len(boxes)),
            "mask_pixels": int(
                sum(np.count_nonzero(box.mask > 0) for box in boxes if box.mask is not None)
            ),
            "confidence_threshold": float(self.conf_threshold),
        }
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._local, "metrics", {}))

    def verify_post_inpaint_residue(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
    ) -> list[BubbleBox]:
        # Extreme path: no second detector pass after inpaint.
        self._local.residue_metrics = {
            "disabled": 1,
            "input_boxes": int(len(boxes)),
        }
        return []

    def last_residue_metrics(self) -> dict[str, float | int]:
        return dict(getattr(self._local, "residue_metrics", {}))
