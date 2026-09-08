#!/usr/bin/env python3
"""Phase 8 equivalence/ownership/performance checks for MSER base-candidate caching."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import statistics
import sys
import threading
import time
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.detector.bubble_detector import BubbleBox
from app.detector.recovery import SecondaryTextRecovery


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def raw_regions(groups: int = 24) -> np.ndarray:
    rows = []
    cols = 6
    for group in range(groups):
        gx = 30 + (group % cols) * 160
        gy = 30 + (group // cols) * 150
        for offset in range(4):
            rows.append((gx + offset * 11, gy + (offset % 2), 8, 14))
    return np.asarray(rows, dtype=np.int32)


def signature(boxes):
    return [
        (
            box.x1, box.y1, box.x2, box.y2,
            round(float(box.confidence), 6),
            box.mask_source,
            bool(box.safe_to_inpaint),
            bool(box.ocr_eligible),
            bool(box.needs_review),
            None if box.mask is None else box.mask.tobytes(),
        )
        for box in boxes
    ]


def make_recovery(raw: np.ndarray):
    recovery = SecondaryTextRecovery()
    recovery._extract_primitives = lambda image, gray: raw
    calls = {"seed": 0}

    def seed(gray_crop):
        calls["seed"] += 1
        h, w = gray_crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if h and w:
            y1, y2 = max(0, h // 4), max(1, 3 * h // 4)
            x1, x2 = max(0, w // 4), max(1, 3 * w // 4)
            mask[y1:y2, x1:x2] = 255
        return mask

    recovery._seed_mask = seed
    return recovery, calls


def clear_candidate_cache(recovery):
    recovery._candidate_local = threading.local()


def main() -> int:
    image = np.full((720, 1024, 3), 245, dtype=np.uint8)
    raw = raw_regions()
    recovery, calls = make_recovery(raw)

    first = recovery.detect(image, existing=[])
    check(first, "synthetic recovery fixture produced no candidates")
    first_seed_calls = calls["seed"]
    check(first_seed_calls > 0, "first pass did not build review masks")

    second = recovery.detect(image, existing=[])
    check(signature(second) == signature(first), "cached pass differs from fresh pass")
    check(calls["seed"] == first_seed_calls, "same-image second pass rebuilt seed masks")

    # Returned masks must be clones: caller mutation cannot poison cached evidence.
    masked_index = next((i for i, box in enumerate(first) if box.mask is not None), None)
    check(masked_index is not None, "fixture did not create a cached review mask")
    first[masked_index].mask.fill(0)
    third = recovery.detect(image, existing=[])
    check(np.any(third[masked_index].mask > 0), "caller mutation poisoned cached base mask")

    # Existing-dependent policy must still execute on every invocation.
    blocker = replace(
        third[0],
        safe_to_inpaint=True,
        needs_review=False,
        mask_source="text_segmenter",
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
    )
    cached_filtered = recovery.detect(image, existing=[blocker])
    clear_candidate_cache(recovery)
    fresh_filtered = recovery.detect(image, existing=[blocker])
    check(
        signature(cached_filtered) == signature(fresh_filtered),
        "cached path changed existing-dependent filtering semantics",
    )
    check(len(fresh_filtered) < len(third), "overlapping verified evidence did not re-filter candidates")

    # Cache ownership is exact ndarray identity, not equal pixels/shape.
    seed_before_copy = calls["seed"]
    image_copy = image.copy()
    recovery.detect(image_copy, existing=[])
    check(calls["seed"] > seed_before_copy, "different image object reused base-candidate cache")

    # Microbenchmark the duplicated work targeted by this phase. Fresh runs clear
    # only the base-candidate cache; raw primitive extraction is already cached by
    # Phase 4 and is intentionally outside this comparison.
    recovery, _ = make_recovery(raw)
    recovery.detect(image, existing=[])
    cached_times = []
    for _ in range(12):
        start = time.perf_counter()
        recovery.detect(image, existing=[])
        cached_times.append((time.perf_counter() - start) * 1000.0)

    fresh_times = []
    for _ in range(5):
        clear_candidate_cache(recovery)
        start = time.perf_counter()
        recovery.detect(image, existing=[])
        fresh_times.append((time.perf_counter() - start) * 1000.0)

    cached_ms = statistics.median(cached_times)
    fresh_ms = statistics.median(fresh_times)
    check(
        cached_ms < fresh_ms,
        f"candidate cache did not improve repeated recovery: fresh={fresh_ms:.3f}ms cached={cached_ms:.3f}ms",
    )
    print(
        "phase8 repeated recovery "
        f"fresh={fresh_ms:.3f}ms cached={cached_ms:.3f}ms "
        f"speedup={fresh_ms/max(cached_ms, 1e-9):.2f}x candidates={len(first)}"
    )
    print("backend recovery cache sanity: phase 8 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
