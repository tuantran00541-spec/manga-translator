#!/usr/bin/env python3
"""Deterministic contracts for ROI-resolution benchmark comparison logic."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_segmenter_roi_resolution import (
    match_residue_hits,
    pixel_amplification,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


baseline = [
    {"x1": 100, "y1": 100, "x2": 180, "y2": 150, "deferred_reason": "post_inpaint_text_residue"},
    {"x1": 300, "y1": 200, "x2": 420, "y2": 260, "deferred_reason": "post_inpaint_text_residue"},
]
candidate = [
    {"x1": 102, "y1": 99, "x2": 182, "y2": 151, "deferred_reason": "post_inpaint_text_residue"},
    {"x1": 302, "y1": 203, "x2": 422, "y2": 263, "deferred_reason": "post_inpaint_text_residue"},
]
check(match_residue_hits(baseline, candidate) == [], "nearby residue hits should match")

missing = match_residue_hits(baseline, candidate[:1])
check(len(missing) == 1, "missing candidate residue must be surfaced")
check(missing[0]["baseline"]["x1"] == 300, "wrong baseline residue reported missing")

ratio = pixel_amplification(model_size=1024, calls=11, source_pixels=817229)
check(14.0 < ratio < 14.2, "fixed-1024 ROI amplification regression")

check(
    pixel_amplification(model_size=640, calls=11, source_pixels=817229) < 5.6,
    "640 model should materially reduce tensor-pixel amplification",
)

print("segmenter ROI resolution benchmark sanity PASS")
