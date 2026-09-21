from __future__ import annotations

import os
import time

import numpy as np

from app.config import MANGA109_YOLO26_SEG_MODEL
from app.detector.bubble_detector import BubbleBox, YoloDetector


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


class Manga109Yolo26TextDetector:
    """Extreme branch detector: YOLO26 class=text mask is cleanup authority.

    No bubble detector, text_segmenter, MSER, adaptive focus, recovery, rescue,
    or residue model participates in automatic detection. The Manga109 YOLO26
    segmentation model runs directly and only its class=text instance masks are
    returned to the normal inpaint path.
    """

    def __init__(self, conf_threshold: float | None = None):
        self.conf_threshold = (
            _env_float("MANGA_YOLO26_TEXT_CONF", 0.01)
            if conf_threshold is None
            else float(conf_threshold)
        )
        self.model = YoloDetector(
            MANGA109_YOLO26_SEG_MODEL,
            self.conf_threshold,
            use_tta=False,
            model_role="manga109_yolo26_seg",
        )
        self._last_metrics: dict[str, float | int] = {}
        self._last_residue_metrics: dict[str, float | int] = {
            "disabled": 1,
            "regions": 0,
        }

    def detect(
        self,
        image: np.ndarray,
        parallel: bool = False,
    ) -> list[BubbleBox]:
        del parallel
        started = time.perf_counter()
        boxes = [
            box
            for box in self.model.detect(image)
            if (
                box.class_name == "text"
                and box.verified_mask
                and box.safe_to_inpaint
            )
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        mask_pixels = int(
            sum(
                int(np.count_nonzero(box.mask > 0))
                for box in boxes
                if box.mask is not None
            )
        )
        self._last_metrics = {
            "total_ms": round(elapsed_ms, 3),
            "detector_total_ms": round(elapsed_ms, 3),
            "yolo26_model_ms": round(elapsed_ms, 3),
            "detector_forward_calls": 1,
            "detector_boxes": int(len(boxes)),
            "authorized": int(len(boxes)),
            "mask_pixels": mask_pixels,
            "conf_threshold": float(self.conf_threshold),
            "extreme_yolo26_only": 1,
            "text_segmenter_calls": 0,
            "bubble_yolo_calls": 0,
            "recovery_calls": 0,
            "residue_verify_calls": 0,
        }
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        return dict(self._last_metrics)

    def verify_post_inpaint_residue(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
    ) -> list[BubbleBox]:
        del image, boxes
        self._last_residue_metrics = {
            "disabled": 1,
            "regions": 0,
        }
        return []

    def last_residue_metrics(self) -> dict[str, float | int]:
        return dict(self._last_residue_metrics)
