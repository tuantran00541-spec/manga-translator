#!/usr/bin/env python3
"""Dependency-light safety checks for the research quality-gate helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.detector.stroke_refinement import refine_stroke_mask
from migan_onnx_adapter import erase_to_migan_keep_mask
from research_quality_gates import Sample, run_mask


def _write_mask(path: Path, mask: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    encoded.tofile(path)


def _assert_mask_truth_gate(seed: np.ndarray, envelope: np.ndarray, truth: np.ndarray) -> None:
    args = argparse.Namespace(
        save_artifacts=False,
        stroke_min_radius=1,
        stroke_max_radius=6,
        mask_f1_tolerance=0.005,
        mask_fp_tolerance=0.005,
        mask_recall_tolerance=0.02,
    )
    with tempfile.TemporaryDirectory(prefix="mask-gate-sanity-") as temp_dir:
        root = Path(temp_dir)
        _write_mask(root / "seed.png", seed)
        _write_mask(root / "safe.png", envelope)
        _write_mask(root / "truth.png", truth)

        missing_truth = Sample(
            {
                "id": "mask-no-truth",
                "task": "mask",
                "seed_mask": "seed.png",
                "safe_envelope": "safe.png",
            },
            root,
        )
        blocked = run_mask([missing_truth], args, root / "blocked")
        blocked_gate = blocked["gate"]
        assert blocked_gate["eligible_for_next_stage"] is False
        assert blocked_gate["truth_required"] is True
        assert blocked_gate["missing_truth_samples"] == ["mask-no-truth"]
        assert any("truth_mask required" in reason for reason in blocked_gate["reasons"])

        with_truth = Sample(
            {
                "id": "mask-with-truth",
                "task": "mask",
                "seed_mask": "seed.png",
                "truth_mask": "truth.png",
                "safe_envelope": "safe.png",
            },
            root,
        )
        evaluated = run_mask([with_truth], args, root / "evaluated")
        evaluated_gate = evaluated["gate"]
        assert evaluated_gate["missing_truth_samples"] == []
        assert not any("truth_mask required" in reason for reason in evaluated_gate["reasons"])


def main() -> int:
    seed = np.zeros((96, 128), dtype=np.uint8)
    cv2.rectangle(seed, (20, 20), (23, 70), 255, -1)
    cv2.rectangle(seed, (70, 20), (80, 70), 255, -1)
    image = np.full((96, 128, 3), 255, dtype=np.uint8)

    envelope = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    refined, stats = refine_stroke_mask(
        seed,
        image,
        safe_envelope=envelope,
        min_radius=1,
        max_radius=6,
    )
    assert np.all(refined[seed > 127] == 255), "verified seed pixels were removed"
    assert not np.any((refined > 127) & ~(envelope > 127)), "safe envelope was crossed"
    assert stats.max_radius_used <= 6, "configured growth bound was crossed"

    _assert_mask_truth_gate(seed, envelope, refined)

    erase = np.array([[0, 255], [128, 127]], dtype=np.uint8)
    keep = erase_to_migan_keep_mask(erase)
    expected = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    assert np.array_equal(keep, expected), "MI-GAN mask polarity contract is wrong"

    print("research quality-gate invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
