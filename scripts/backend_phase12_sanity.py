#!/usr/bin/env python3
"""Deterministic safety/behavior checks for Phase 12 focus refinement budget."""
from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.detector.adaptive_focus_detector import (
    FOCUS_MAX_CHIPS,
    FOCUS_MAX_SOURCE_SIDE,
    _focus_text_detect,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox
from app.parameters import DETECTOR_FOCUS_MAX_CHIPS


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeTextDetector:
    model_role = "text_segmenter"

    def __init__(self):
        self.calls: list[tuple[int, int, int, int]] = []

    def _detect_single_plain(self, image, offset_x, offset_y):
        h, w = image.shape[:2]
        self.calls.append((int(offset_x), int(offset_y), int(w), int(h)))
        # Full-page pass deliberately has no result. The first bounded focus crop
        # returns one text box so the contract also exercises global remapping.
        if offset_x == 0 and offset_y == 0 and h > FOCUS_MAX_SOURCE_SIDE:
            return []
        x1 = int(offset_x + min(40, max(0, w - 80)))
        y1 = int(offset_y + min(40, max(0, h - 80)))
        x2 = min(int(offset_x + w), x1 + 40)
        y2 = min(int(offset_y + h), y1 + 40)
        if x2 <= x1 or y2 <= y1:
            return []
        mask = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
        return [
            BubbleBox(
                x1, y1, x2, y2, 0.90, mask,
                source_model="text_segmenter.onnx",
                class_name="text_comic",
                semantic_type="text",
                mask_source="text_segmenter",
                safe_to_inpaint=True,
                ocr_eligible=True,
                needs_review=False,
                source_role="text_segmenter",
            )
        ]

    @staticmethod
    def _nms_boxes(boxes):
        return list(boxes)

    @staticmethod
    def _with_semantics(box):
        return box

    @staticmethod
    def _filter_invalid(boxes, _width, _height):
        return list(boxes)


def scheduler_budget_check():
    check(DETECTOR_FOCUS_MAX_CHIPS == 1, "Phase 12 parameter default must be one focus chip")
    check(FOCUS_MAX_CHIPS == 1, "adaptive detector did not consume the Phase 12 focus budget")

    proposal = BubbleBox(
        100, 200, 800, 3500, 0.91, None,
        source_model="bubble_yolo.onnx",
        class_name="text_free",
        semantic_type="free_text",
        source_role="bubble_detector",
        safe_to_inpaint=False,
        ocr_eligible=True,
        needs_review=True,
    )
    chips, deferred = plan_focus_chips(4096, 900, [proposal])
    check(len(chips) == 1, "focus scheduler exceeded the one-chip Phase 12 budget")
    check(bool(deferred), "budget exhaustion was not surfaced as deferred focus work")
    check(
        all(max(x2 - x1, y2 - y1) <= FOCUS_MAX_SOURCE_SIDE for x1, y1, x2, y2 in chips),
        "scheduled focus chip exceeded the source-side bound",
    )
    print("phase12 one-chip scheduler budget PASS")


def focus_detect_fail_safe_check():
    image = np.zeros((4096, 900, 3), dtype=np.uint8)
    proposal = BubbleBox(
        100, 200, 800, 3500, 0.91, None,
        source_model="bubble_yolo.onnx",
        class_name="text_free",
        semantic_type="free_text",
        source_role="bubble_detector",
        safe_to_inpaint=False,
        ocr_eligible=True,
        needs_review=True,
    )
    fake = FakeTextDetector()
    text, metrics, deferred = _focus_text_detect(fake, image, [proposal])

    check(len(fake.calls) == 2, "expected one full RGB pass plus exactly one focus refinement")
    full = fake.calls[0]
    focus = fake.calls[1]
    check(full == (0, 0, 900, 4096), "Phase 12 removed or changed the full RGB text pass")
    check(focus != full, "focus refinement unexpectedly reran the full page")
    check(focus[2] <= FOCUS_MAX_SOURCE_SIDE and focus[3] <= FOCUS_MAX_SOURCE_SIDE,
          "focus refinement exceeded the bounded source geometry")
    check(metrics["focus_chip_calls"] == 1, "focus metrics did not report one refinement call")
    check(metrics["focus_deferred_regions"] == len(deferred) and len(deferred) > 0,
          "deferred focus regions were not surfaced")
    check(bool(text), "bounded focus refinement failed to preserve recovered text evidence")
    check(all(not box.safe_to_inpaint for box in deferred),
          "budget-exhausted focus region gained destructive authority")
    check(all(box.needs_review for box in deferred),
          "budget-exhausted focus region was not review-only")
    check(all(box.deferred_reason == "focus_budget_exhausted" for box in deferred),
          "deferred focus provenance was lost")
    print("phase12 full-pass preservation + deferred fail-safe PASS")


scheduler_budget_check()
focus_detect_fail_safe_check()
print("backend focus budget sanity: phase 12 PASS")
