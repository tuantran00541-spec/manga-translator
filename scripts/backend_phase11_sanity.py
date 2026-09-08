#!/usr/bin/env python3
"""Deterministic safety/equivalence checks for Phase 11 ROI grayscale retry."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.parameters import (
    DETECTOR_GRAYSCALE_FALLBACK_MAX_ROIS,
    DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeTextDetector:
    def __init__(self, target: tuple[int, int, int, int]):
        self.target = target
        self.calls: list[tuple[int, int, int, int]] = []

    def _detect_single(self, image, offset_x, offset_y):
        h, w = image.shape[:2]
        self.calls.append((int(offset_x), int(offset_y), int(w), int(h)))
        tx1, ty1, tx2, ty2 = self.target
        if not (
            offset_x <= tx1 < tx2 <= offset_x + w
            and offset_y <= ty1 < ty2 <= offset_y + h
        ):
            return []
        mask = np.full((ty2 - ty1, tx2 - tx1), 255, dtype=np.uint8)
        return [
            BubbleBox(
                tx1,
                ty1,
                tx2,
                ty2,
                0.91,
                mask,
                source_model="text_segmenter.onnx",
                class_id=0,
                class_name="text_comic",
                semantic_type="text",
                mask_source="model",
                safe_to_inpaint=False,
                ocr_eligible=False,
                needs_review=True,
                source_role="text_segmenter",
            )
        ]

    @staticmethod
    def _with_semantics(box: BubbleBox) -> BubbleBox:
        safe = bool(box.verified_mask and box.source_role == "text_segmenter")
        return replace(
            box,
            mask_source="text_segmenter" if safe else "none",
            safe_to_inpaint=safe,
            ocr_eligible=safe,
            needs_review=not safe,
        )

    @staticmethod
    def _filter_invalid(boxes, _width, _height):
        return list(boxes)


def roi_retry_check():
    image = np.zeros((5000, 800, 3), dtype=np.uint8)
    image[2480:2600, 260:520] = (20, 120, 230)
    proposal = BubbleBox(
        120,
        2350,
        700,
        2750,
        0.88,
        None,
        source_model="bubble_yolo.onnx",
        class_name="text_free",
        semantic_type="free_text",
        source_role="bubble_detector",
        safe_to_inpaint=False,
        ocr_eligible=True,
        needs_review=True,
    )
    target = (300, 2490, 460, 2570)
    fake = FakeTextDetector(target)
    detector = CombinedTextDetector.__new__(CombinedTextDetector)
    detector.text_detector = fake

    recovered, metrics = detector._grayscale_text_retry(image, [proposal])
    check(metrics["calls"] == 1, "compact miss must use exactly one ROI retry")
    check(len(fake.calls) == 1, "retry unexpectedly ran more than one detector crop")
    ox, oy, cw, ch = fake.calls[0]
    check((ox, oy) != (0, 0) or (cw, ch) != (800, 5000), "fallback reran the full page")
    check(cw <= DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE, "ROI width exceeded source bound")
    check(ch <= DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE, "ROI height exceeded source bound")
    check(metrics["source_pixels"] < image.shape[0] * image.shape[1] * 0.5, "ROI did not materially bound source pixels")
    check(len(recovered) == 1, "known grayscale text target was not recovered")
    box = recovered[0]
    check((box.x1, box.y1, box.x2, box.y2) == target, "ROI coordinate remap changed geometry")
    check(box.verified_mask, "ROI retry lost the text-segmenter mask")
    check(box.safe_to_inpaint and not box.needs_review, "verified segmenter retry lost erase authority")
    print("phase11 bounded grayscale ROI recovery PASS")


def budget_and_fail_safe_check():
    image_shape = (7000, 800)
    proposals = [
        BubbleBox(80, 400 + i * 1800, 720, 700 + i * 1800, 0.90 - i * 0.05,
                  semantic_type="free_text", source_role="bubble_detector", needs_review=True)
        for i in range(3)
    ]
    rois, deferred = CombinedTextDetector._plan_grayscale_fallback_rois(
        image_shape,
        proposals,
    )
    check(len(rois) == DETECTOR_GRAYSCALE_FALLBACK_MAX_ROIS, "ROI call budget was not enforced")
    check(deferred >= 1, "budget exhaustion was not surfaced")

    giant = BubbleBox(
        10,
        100,
        790,
        100 + DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE + 100,
        0.99,
        semantic_type="free_text",
        source_role="bubble_detector",
        safe_to_inpaint=False,
        needs_review=True,
    )
    giant_rois, giant_deferred = CombinedTextDetector._plan_grayscale_fallback_rois(
        (4000, 800),
        [giant],
    )
    check(not giant_rois and giant_deferred == 1, "oversized proposal bypassed ROI bound")
    check(not giant.safe_to_inpaint and giant.needs_review, "deferred proposal gained destructive authority")
    print("phase11 ROI budget/fail-safe PASS")


roi_retry_check()
budget_and_fail_safe_check()
print("backend grayscale ROI sanity: phase 11 PASS")
