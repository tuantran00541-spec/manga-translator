#!/usr/bin/env python3
"""Dependency-light regression checks for the research mask truth-coverage gate."""
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
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_mask_quality_gate import Sample, evaluate


def _write(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, image)
    assert ok
    buf.tofile(path)


def _args(min_truth_coverage: float) -> argparse.Namespace:
    return argparse.Namespace(
        stroke_min_radius=1,
        stroke_max_radius=6,
        mask_f1_tolerance=0.005,
        mask_fp_tolerance=0.005,
        mask_recall_tolerance=0.02,
        min_truth_coverage=min_truth_coverage,
        save_artifacts=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mask-gate-sanity-") as tmp:
        root = Path(tmp)
        image = np.full((96, 128, 3), 255, dtype=np.uint8)
        seed = np.zeros((96, 128), dtype=np.uint8)
        cv2.rectangle(seed, (20, 20), (23, 70), 255, -1)
        truth = cv2.dilate(
            seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        envelope = cv2.dilate(
            seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        _write(root / "image.png", image)
        _write(root / "seed.png", seed)
        _write(root / "truth.png", truth)
        _write(root / "safe.png", envelope)

        complete = Sample(
            {
                "id": "with-truth",
                "image": "image.png",
                "seed_mask": "seed.png",
                "truth_mask": "truth.png",
                "safe_envelope": "safe.png",
            },
            root,
        )
        missing = Sample(
            {
                "id": "missing-truth",
                "image": "image.png",
                "seed_mask": "seed.png",
                "safe_envelope": "safe.png",
            },
            root,
        )

        no_truth = evaluate([missing, missing], _args(0.0), root / "no-truth")
        assert no_truth["coverage"]["truth_samples"] == 0
        assert no_truth["coverage"]["truth_coverage"] == 0.0
        assert not no_truth["gate"]["eligible_for_next_stage"]
        assert "ground-truth mask evidence missing for all samples" in no_truth["gate"]["reasons"]

        blocked = evaluate([complete, missing], _args(1.0), root / "blocked")
        assert blocked["coverage"]["truth_samples"] == 1
        assert blocked["coverage"]["total_samples"] == 2
        assert blocked["coverage"]["truth_coverage"] == 0.5
        assert not blocked["gate"]["eligible_for_next_stage"]
        assert any(
            "ground-truth mask missing for 1/2 samples" in reason
            for reason in blocked["gate"]["reasons"]
        )

        threshold_ok = evaluate([complete, missing], _args(0.5), root / "threshold")
        assert not any(
            "ground-truth mask missing" in reason
            for reason in threshold_ok["gate"]["reasons"]
        )

        complete_2 = Sample(
            {
                "id": "with-truth-2",
                "image": "image.png",
                "seed_mask": "seed.png",
                "truth_mask": "truth.png",
                "safe_envelope": "safe.png",
            },
            root,
        )
        full = evaluate([complete, complete_2], _args(1.0), root / "full")
        assert full["coverage"]["truth_samples"] == 2
        assert full["coverage"]["truth_coverage"] == 1.0
        assert not any(
            "ground-truth mask missing" in reason for reason in full["gate"]["reasons"]
        )

    print("research mask truth-coverage gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
