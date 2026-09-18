#!/usr/bin/env python3
"""Contract for model-specific detector resolution and focused retry routing."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.detector.bubble_detector as yolo_mod
import app.detector.combined_detector as combined_mod
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.detector.fast_residue_detector import FastResidueAdaptiveFocusCombinedTextDetector


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




def lazy_optional_specialized_detector_check():
    detector = CombinedTextDetector.__new__(CombinedTextDetector)
    primary = FakeDetector()
    specialized = FakeDetector()
    detector.text_detector = primary
    detector._residue_text_detector = None
    detector._residue_text_detector_resolved = False
    detector._residue_text_detector_lock = threading.Lock()

    calls = []

    def factory(model_path, conf_threshold, use_tta=None, *, model_role, input_size=None):
        calls.append({
            "model_path": str(model_path),
            "model_role": model_role,
            "input_size": input_size,
        })
        return specialized

    with tempfile.TemporaryDirectory() as temp_dir:
        candidate = Path(temp_dir) / "text_segmenter_640.onnx"
        candidate.write_bytes(b"placeholder")
        with (
            mock.patch.object(combined_mod, "YoloDetector", side_effect=factory),
            mock.patch.object(
                combined_mod,
                "TEXT_SEGMENTER_RETRY_MODEL",
                candidate,
                create=True,
            ),
            mock.patch.object(
                combined_mod,
                "DETECTOR_RETRY_SEGMENTER_ENABLED",
                True,
                create=True,
            ),
            mock.patch.object(
                combined_mod,
                "DETECTOR_RETRY_INPUT_SIZE",
                640,
                create=True,
            ),
        ):
            check(
                detector.residue_text_detector is specialized,
                "available specialized model was not lazy-loaded",
            )
            check(
                detector.residue_text_detector is specialized,
                "specialized detector was not cached",
            )

    check(len(calls) == 1, "specialized segmenter was loaded more than once")
    check(calls[0]["input_size"] == 640, "specialized detector did not use 640 input")
    check(calls[0]["model_role"] == "text_segmenter", "specialized detector role changed")


def residue_verify_routes_to_specialized_detector_check():
    detector = FastResidueAdaptiveFocusCombinedTextDetector.__new__(
        FastResidueAdaptiveFocusCombinedTextDetector
    )
    primary = FakeDetector()
    specialized = FakeDetector()
    detector.text_detector = primary
    detector._residue_text_detector = specialized
    detector._residue_text_detector_resolved = True
    detector._residue_text_detector_lock = threading.Lock()
    detector._residue_flat_gate_enabled = False
    detector._residue_metrics_local = threading.local()
    detector._residue_metrics_lock = threading.Lock()
    detector._residue_totals = {}

    mask = np.full((60, 120), 255, dtype=np.uint8)
    source = BubbleBox(
        80, 90, 200, 150, 0.95, mask,
        source_model="text_segmenter.onnx",
        class_name="text_comic",
        semantic_type="text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        source_role="text_segmenter",
    )
    detector.verify_post_inpaint_residue(
        np.zeros((300, 400, 3), dtype=np.uint8),
        [source],
    )
    check(specialized.calls == 1, "residue verifier did not use specialized segmenter")
    check(primary.calls == 0, "residue verifier unexpectedly used primary 1024 segmenter")

def focused_retry_stays_on_primary_detector_check():
    detector = CombinedTextDetector.__new__(CombinedTextDetector)
    primary = FakeDetector()
    specialized = FakeDetector()
    detector.text_detector = primary
    detector._residue_text_detector = specialized
    detector._residue_text_detector_resolved = True

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
    check(primary.calls == 1, "focused retry left the validated primary 1024 segmenter")
    check(specialized.calls == 0, "residue-only 640 segmenter leaked into destructive retry")


per_model_input_size_check()
focused_retry_stays_on_primary_detector_check()
lazy_optional_specialized_detector_check()
residue_verify_routes_to_specialized_detector_check()
print("focused segmenter 640 sanity PASS")
