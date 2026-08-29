#!/usr/bin/env python3
"""Dependency-light safety checks for the research quality-gate helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.detector.stroke_refinement import refine_stroke_mask
from migan_onnx_adapter import erase_to_migan_keep_mask


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

    erase = np.array([[0, 255], [128, 127]], dtype=np.uint8)
    keep = erase_to_migan_keep_mask(erase)
    expected = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    assert np.array_equal(keep, expected), "MI-GAN mask polarity contract is wrong"

    print("research quality-gate invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
