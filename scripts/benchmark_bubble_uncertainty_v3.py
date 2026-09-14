from __future__ import annotations

import hashlib
from pathlib import Path
import statistics
import time

from app.image_io import read_image
import scripts.benchmark_bubble_uncertainty as base


base.OUT = Path("benchmark-results/bubble-uncertainty-v3/report.json")


def _run_precise(detector, paths, indices, warmup_path, label: str) -> dict:
    """Benchmark while preserving raw floating-point audit features.

    CombinedTextDetector.last_metrics() intentionally casts non-*_ms counters to
    int for production telemetry. The uncertainty benchmark needs the original
    fractional Gabor/confidence/geometry values when fitting its rule, otherwise
    thresholds are learned from quantized features and do not reproduce at
    runtime.
    """
    warmup = read_image(warmup_path)
    detector.detect(warmup, parallel=False)
    del warmup

    elapsed: list[float] = []
    rows: dict[str, dict] = {}
    authority: dict[str, str] = {}
    review: dict[str, list] = {}
    counts: dict[str, dict] = {}

    for index in indices:
        image = read_image(paths[index])
        h, w = image.shape[:2]
        started = time.perf_counter()
        boxes = detector.detect(image, parallel=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        elapsed.append(elapsed_ms)

        # Do not go through last_metrics(); preserve the raw audit floats.
        metrics = dict(getattr(detector._metrics_local, "value", {}) or {})
        rows[str(index)] = metrics
        authority[str(index)] = hashlib.sha256(
            base._authority_mask(boxes, h, w).tobytes()
        ).hexdigest()
        review[str(index)] = base._review_signature(boxes)
        counts[str(index)] = {
            "boxes": len(boxes),
            "safe": sum(bool(box.safe_to_inpaint) for box in boxes),
            "review": sum(bool(box.needs_review) for box in boxes),
        }
        del image, boxes

    return {
        "label": label,
        "wall_ms": round(sum(elapsed), 3),
        "mean_ms": round(statistics.mean(elapsed), 3),
        "median_ms": round(statistics.median(elapsed), 3),
        "rows": rows,
        "authority_hashes": authority,
        "review_signatures": review,
        "counts": counts,
    }


base._run = _run_precise


if __name__ == "__main__":
    base.main()
