#!/usr/bin/env python3
"""Contract for model-specific detector resolution and focused retry routing."""
from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.detector.bubble_detector as yolo_mod
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class Meta:
    def __init__(self, name, shape, type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type


class Session:
    def get_inputs(self):
        return [Meta("images", [1, 3, 640, 640])]

    def get_outputs(self):
        return [
            Meta("output0", [1, 37, 8400]),
            Meta("output1", [1, 32, 160, 160]),
        ]


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def _detect_single(self, image, offset_x, offset_y):
        self.calls += 1
        return []

    def _detect_single_plain(self, image, offset_x, offset_y):
        self.calls += 1
        return []

    @staticmethod
    def _with_semantics(box):
        return box

    @staticmethod
    def _filter_invalid(boxes, width, height):
        return list(boxes)


def per_model_input_size_check():
    with mock.patch.object(yolo_mod, "make_session", return_value=Session()):
        detector = YoloDetector(
            "text_segmenter_640.onnx",
            0.20,
            model_role="text_segmenter",
            input_size=640,
        )
    blob, transform = detector._preprocess(
        np.zeros((240, 320, 3), dtype=np.uint8)
    )
    check(blob.shape == (1, 3, 640, 640), "detector ignored model-specific input size")
    check(
        (transform.input_h, transform.input_w) == (640, 640),
        "letterbox transform ignored model-specific input size",
    )


def focused_retry_routes_to_specialized_detector_check():
    detector = CombinedTextDetector.__new__(CombinedTextDetector)
    primary = FakeDetector()
    specialized = FakeDetector()
    detector.text_detector = primary
    detector._retry_text_detector = specialized
    detector._retry_text_detector_resolved = True

    proposal = BubbleBox(
        40, 50, 220, 140, 0.9, None,
        source_model="bubble_yolo.onnx",
        class_name="text_free",
        semantic_type="free_text",
        source_role="bubble_detector",
        needs_review=True,
    )
    detector._focused_text_retry(
        np.zeros((300, 400, 3), dtype=np.uint8),
        [proposal],
        grayscale=False,
    )
    check(specialized.calls == 1, "focused retry did not use specialized segmenter")
    check(primary.calls == 0, "focused retry unexpectedly used primary 1024 segmenter")


per_model_input_size_check()
focused_retry_routes_to_specialized_detector_check()
print("focused segmenter 640 sanity PASS")
