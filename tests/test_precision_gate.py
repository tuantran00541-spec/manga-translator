from __future__ import annotations

from scripts.precision_gate import evaluate


def _report():
    return {
        "precision_mode": "mixed",
        "baseline": {"box_recall": 1.0, "mask_boundary_f1": 1.0, "mask_iou": 1.0, "wall_ms": 100.0},
        "candidate": {
            "box_recall": 1.0,
            "mask_boundary_f1": 1.0,
            "mask_iou": 1.0,
            "wall_ms": 90.0,
            "authority_outside_changed": 0,
            "hard_holdout_false_negatives": 0,
        },
        "sensitivity": [{"node": "conv_1", "keep_fp32": True}],
    }


def test_precision_gate_requires_accuracy_and_safety_before_speed():
    result = evaluate(_report())
    assert result["status"] == "pass"
    bad = _report()
    bad["candidate"]["authority_outside_changed"] = 1
    assert evaluate(bad)["status"] == "fail"


def test_precision_gate_blocks_missing_sensitive_layer_evidence():
    report = _report()
    report.pop("sensitivity")
    result = evaluate(report)
    assert result["status"] == "blocked"
