from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "yolo26_segmentation_spike.py"
WORKFLOW = ROOT / ".github" / "workflows" / "yolo26-segmentation-spike.yml"
EXPECTED_PT_SHA256 = "0b4376e426fa96af3976afa6a2602421dacf2dec96ef87b4a44f5e8d4971cb6f"


def _load_spike():
    assert SCRIPT.is_file(), "YOLO26 segmentation spike script is missing"
    spec = importlib.util.spec_from_file_location("yolo26_segmentation_spike", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_metrics_measure_reference_recall_candidate_precision_and_iou():
    spike = _load_spike()
    reference = np.zeros((4, 4), dtype=np.uint8)
    reference[0:2, 0:2] = 255
    candidate = np.zeros((4, 4), dtype=np.uint8)
    candidate[0:2, 1:3] = 255

    metrics = spike.mask_metrics(reference, candidate)

    assert metrics["reference_pixels"] == 4
    assert metrics["candidate_pixels"] == 4
    assert metrics["intersection_pixels"] == 2
    assert metrics["false_positive_pixels"] == 2
    assert metrics["pixel_recall"] == 0.5
    assert metrics["pixel_precision"] == 0.5
    assert metrics["pixel_iou"] == (2 / 6)


def test_candidate_mask_union_clips_instance_masks_to_page():
    spike = _load_spike()

    class Box:
        x1, y1, x2, y2 = -1, -1, 2, 2
        mask = np.full((3, 3), 255, dtype=np.uint8)

    union = spike.candidate_mask_union((3, 3, 3), [Box()])

    assert union.shape == (3, 3)
    assert int(np.count_nonzero(union)) == 4


def test_shadow_yolo26s_workflow_is_pinned_and_benchmark_only():
    assert WORKFLOW.is_file(), "YOLO26 segmentation workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")

    assert EXPECTED_PT_SHA256 in text
    assert "ShadowB/Manga109-panel-balloon-text-yolov26-segmentation" in text
    assert "best.pt" in text
    assert "--text-class-id 1" in text
    assert "--sizes 640,768,1024" in text
    assert "models/text_segmenter.onnx" not in text
    assert "cp " not in text or "models/text_segmenter.onnx" not in text


def test_shadow_yolo26s_spike_reuses_production_mask_decoder_without_relaxing_contracts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "YoloDetector" in text
    assert "_detect_single_plain" in text
    assert "SimpleNamespace" in text
    assert "validate_detector_session" not in text
