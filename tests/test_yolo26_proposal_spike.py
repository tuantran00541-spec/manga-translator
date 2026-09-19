from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

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


def test_merge_proposal_sets_deduplicates_exact_geometry():
    spike = _load_spike()

    merged = spike.merge_proposal_sets(
        [(1, 1, 3, 3), (5, 5, 7, 7)],
        [(1, 1, 3, 3), (9, 9, 10, 10)],
    )

    assert merged == [(1, 1, 3, 3), (5, 5, 7, 7), (9, 9, 10, 10)]


def test_direct_script_execution_bootstraps_repo_imports(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-dir",
            str(empty),
            "--candidate-640",
            str(tmp_path / "missing-640.onnx"),
            "--candidate-1024",
            str(tmp_path / "missing-1024.onnx"),
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = result.stdout + result.stderr

    assert result.returncode != 0
    assert "No module named 'app'" not in combined
    assert "No benchmark images" in combined


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


def test_yolo26_export_uses_explicit_non_end2end_raw_head_for_pinned_exporter():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "nms=None" not in text
    assert 'format="onnx"' in text
    assert "end2end=False" in text


def test_yolo26_spike_benchmarks_existing_cheap_proposal_stack():
    script = SCRIPT.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "BUBBLE_DETECTOR_MODEL" in script
    assert "SecondaryTextRecovery" in script
    assert '"bubble+yolo26"' in script
    assert '"bubble+yolo26+mser"' in script
    assert '"source_stack"' in script
    assert "source_stack" in workflow


def test_yolo26_spike_benchmarks_bubble_free_yolo26_mser_stack():
    script = SCRIPT.read_text(encoding="utf-8")

    assert '"yolo26+mser"' in script
