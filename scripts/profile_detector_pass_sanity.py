#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.profile_processing import detector_phase_specs, summarize_detector_passes


def check(condition, message):
    if not condition:
        raise AssertionError(message)


events = [
    {
        "name": "yolo.forward",
        "model": "bubble_yolo.onnx",
        "phase": "primary",
        "ms": 12.5,
        "source_shape": [700, 1000],
        "input_shape": [1, 3, 1024, 1024],
    },
    {
        "name": "yolo.forward",
        "model": "text_segmenter.onnx",
        "phase": "primary",
        "ms": 20.0,
        "source_shape": [700, 1000],
        "input_shape": [1, 3, 1024, 1024],
    },
    {
        "name": "yolo.forward",
        "model": "text_segmenter.onnx",
        "phase": "grayscale_retry",
        "ms": 18.0,
        "source_shape": [240, 320],
        "input_shape": [1, 3, 1024, 1024],
    },
    {
        "name": "yolo.forward",
        "model": "text_segmenter.onnx",
        "phase": "mser_promotion",
        "ms": 17.0,
        "source_shape": [180, 260],
        "input_shape": [1, 3, 1024, 1024],
    },
    {
        "name": "yolo.forward",
        "model": "text_segmenter.onnx",
        "phase": "residue_verify",
        "ms": 16.0,
        "source_shape": [120, 220],
        "input_shape": [1, 3, 1024, 1024],
    },
]

phase_specs = detector_phase_specs()
check(
    ("OneShotProductionDetector", "verify_post_inpaint_residue", "residue_verify")
    in phase_specs,
    "production residue verification must be attributed as residue_verify",
)

summary = summarize_detector_passes(events)

check(summary["total_forward_calls"] == 5, "forward call total")
check(summary["by_phase"]["primary"]["calls"] == 2, "primary call count")
check(summary["by_phase"]["grayscale_retry"]["calls"] == 1, "grayscale call count")
check(summary["by_phase"]["mser_promotion"]["calls"] == 1, "MSER promotion call count")
check(summary["by_phase"]["residue_verify"]["calls"] == 1, "residue call count")
check(summary["by_model"]["text_segmenter.onnx"]["calls"] == 4, "segmenter call count")
check(
    summary["by_phase"]["grayscale_retry"]["source_pixels"] == 240 * 320,
    "grayscale source-pixel attribution",
)
check(
    summary["by_phase"]["grayscale_retry"]["model_pixels"] == 1024 * 1024,
    "grayscale model-pixel attribution",
)
check(
    summary["by_phase"]["grayscale_retry"]["model_to_source_pixel_ratio"] > 13.0,
    "small ROI should expose fixed-1024 amplification",
)

print("detector pass telemetry sanity PASS")
