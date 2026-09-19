from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "yolo26_proposal_spike.py"
WORKFLOW = ROOT / ".github" / "workflows" / "yolo26-proposal-spike.yml"
EXPECTED_PT_SHA256 = "73e0fb587ea3afe0d17aa9f0c3b1f5a8001b3ecbc3c77091e0730654b0da9146"


def _load_spike():
    assert SCRIPT.is_file(), "YOLO26 proposal spike script is missing"
    spec = importlib.util.spec_from_file_location("yolo26_proposal_spike", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_padded_proposal_union_measures_authority_pixel_coverage():
    spike = _load_spike()
    reference = np.zeros((10, 10), dtype=np.uint8)
    reference[2:4, 2:4] = 255
    proposals = [(2, 2, 3, 4)]

    tight = spike.build_proposal_union(reference.shape, proposals, pad=0)
    padded = spike.build_proposal_union(reference.shape, proposals, pad=1)

    assert spike.mask_coverage(reference, tight) == 0.5
    assert spike.mask_coverage(reference, padded) == 1.0


def test_center_recall_uses_padded_proposal_geometry():
    spike = _load_spike()
    references = [(2, 2, 4, 4), (8, 8, 10, 10)]
    proposals = [(1, 1, 3, 3), (7, 7, 8, 8)]

    assert spike.center_recall(references, proposals, pad=0) == 0.5
    assert spike.center_recall(references, proposals, pad=1) == 1.0


def test_yolo26_workflow_is_pinned_proposal_only_and_non_destructive():
    assert WORKFLOW.is_file(), "YOLO26 proposal workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")

    assert EXPECTED_PT_SHA256 in text
    assert "manga_panel_detector_fp32.pt" in text
    assert "yolo26_manga_proposal_640.onnx" in text
    assert "yolo26_manga_proposal_1024.onnx" in text
    assert "--text-class-id 1" in text
    assert "--max-pages 32" in text
    assert "authority_pixel_coverage" in text
    assert "center_recall" in text
    assert "cp models/yolo26_manga_proposal_640.onnx models/bubble_yolo.onnx" not in text
    assert "cp models/yolo26_manga_proposal_1024.onnx models/bubble_yolo.onnx" not in text
    assert "cp models/yolo26_manga_proposal_640.onnx models/text_segmenter.onnx" not in text
    assert "cp models/yolo26_manga_proposal_1024.onnx models/text_segmenter.onnx" not in text


def test_yolo26_export_uses_implicit_raw_one_to_many_default():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "nms=None" not in text
    assert 'format="onnx"' in text
