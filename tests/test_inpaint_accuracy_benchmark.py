from __future__ import annotations

import numpy as np

from scripts.benchmark_inpaint_accuracy import _metric_row, _synthetic_case


def test_accuracy_metrics_report_perfect_ground_truth_and_authority_safety():
    reference = np.full((12, 16, 3), 120, dtype=np.uint8)
    source = reference.copy()
    source[4:8, 5:11] = 10
    output = reference.copy()
    target = np.zeros((12, 16), dtype=np.uint8)
    target[4:8, 5:11] = 255
    authority = target.copy()

    metrics = _metric_row(source, reference, output, target, authority)

    assert metrics["target_mae"] == 0.0
    assert metrics["target_residual_fraction_gt24"] == 0.0
    assert metrics["outside_changed_pixels_exact"] == 0
    assert metrics["outside_changed_pixels_gt8"] == 0


def test_accuracy_metrics_detect_changes_outside_authority_mask():
    reference = np.full((10, 14, 3), 120, dtype=np.uint8)
    source = reference.copy()
    output = reference.copy()
    target = np.zeros((10, 14), dtype=np.uint8)
    target[3:7, 4:10] = 255
    authority = target.copy()
    output[0, 0] = 121

    metrics = _metric_row(source, reference, output, target, authority)

    assert metrics["outside_changed_pixels_exact"] == 1
    assert metrics["outside_changed_pixels_gt8"] == 0
    assert metrics["outside_max_delta"] == 1


def test_synthetic_case_has_known_text_target_and_background():
    source, reference, box, target = _synthetic_case("gradient")

    assert source.shape == reference.shape
    assert source.shape[:2] == target.shape
    assert box.mask is not None
    assert int(np.count_nonzero(target > 127)) > 0
    assert int(np.count_nonzero(source != reference)) > 0
