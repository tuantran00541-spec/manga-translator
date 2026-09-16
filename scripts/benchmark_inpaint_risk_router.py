"""A/B benchmark for the isolated E5 risk-aware inpaint router.

The synthetic cases have an exact clean background.  The control is the
current AdaptiveFastInpainter; the candidate is RiskAwareFastInpainter.  The
candidate may use cheap fill only for low-risk context and must route uncertain
regions to the existing LaMa path.  No production pipeline is changed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
from app.inpaint.risk_aware_inpainter import RiskAwareFastInpainter
from scripts.benchmark_inpaint_accuracy import (
    _box_mask,
    _metric_row,
    _synthetic_case,
)


CASES = ("flat_white", "flat_black", "gradient", "textured", "long_text")


def _prepare(inpainter, workers: int) -> float:
    prepare = getattr(inpainter, "prepare_for_page_workers", None)
    if callable(prepare):
        prepare(max(1, int(workers)))
    started = time.perf_counter()
    inpainter.preload()
    return (time.perf_counter() - started) * 1000.0


def _lama_reference(inpainter, source, authority, box) -> tuple[np.ndarray, float, dict]:
    x1, y1, x2, y2 = inpainter._compute_crop_region(
        box.x1,
        box.y1,
        box.x2,
        box.y2,
        source.shape[1],
        source.shape[0],
    )
    inpainter._begin_metrics(boxes=1)
    started = time.perf_counter()
    output = inpainter._lama_fill(
        source.copy(),
        source[y1:y2, x1:x2],
        authority[y1:y2, x1:x2],
        (x1, y1, x2, y2),
    )
    return output, (time.perf_counter() - started) * 1000.0, inpainter.last_metrics()


def _run_suite(factory, workers: int) -> tuple[dict, float]:
    inpainter = factory()
    preload_ms = _prepare(inpainter, workers)
    rows = {}
    for name in CASES:
        source, reference, box, target = _synthetic_case(name)
        authority = _box_mask(source, box, inpainter)
        started = time.perf_counter()
        output = inpainter.inpaint(source, [box])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows[name] = {
            "elapsed_ms": round(elapsed_ms, 3),
            "inpaint_metrics": inpainter.last_metrics(),
            "quality": _metric_row(source, reference, output, target, authority),
        }
    return rows, preload_ms


def run(out: Path, workers: int) -> dict:
    control, control_preload_ms = _run_suite(AdaptiveFastInpainter, workers)
    candidate, candidate_preload_ms = _run_suite(RiskAwareFastInpainter, workers)

    reference_inpainter = RiskAwareFastInpainter()
    reference_preload_ms = _prepare(reference_inpainter, workers)
    lama_reference = {}
    for name in CASES:
        source, reference, box, target = _synthetic_case(name)
        authority = _box_mask(source, box, reference_inpainter)
        output, elapsed_ms, metrics = _lama_reference(
            reference_inpainter, source, authority, box
        )
        lama_reference[name] = {
            "elapsed_ms": round(elapsed_ms, 3),
            "inpaint_metrics": metrics,
            "quality": _metric_row(source, reference, output, target, authority),
        }

    candidate_quality_regressions = []
    candidate_safety_failures = []
    for name in CASES:
        candidate_quality = candidate[name]["quality"]
        control_quality = control[name]["quality"]
        reference_quality = lama_reference[name]["quality"]
        if candidate_quality["outside_changed_pixels_exact"]:
            candidate_safety_failures.append(name)
        # Non-flat cases are expected to take the LaMa route.  Allow a small
        # measurement tolerance for resize/runtime variation, but never allow
        # an actual residual-class regression over the forced-LaMa reference.
        if name not in {"flat_white", "flat_black"}:
            if candidate_quality["target_mae"] > reference_quality["target_mae"] + 0.5:
                candidate_quality_regressions.append(
                    {
                        "case": name,
                        "candidate_target_mae": candidate_quality["target_mae"],
                        "lama_target_mae": reference_quality["target_mae"],
                    }
                )
            if candidate_quality["target_residual_fraction_gt24"] > (
                reference_quality["target_residual_fraction_gt24"] + 0.02
            ):
                candidate_quality_regressions.append(
                    {
                        "case": name,
                        "candidate_residual_fraction_gt24": candidate_quality[
                            "target_residual_fraction_gt24"
                        ],
                        "lama_residual_fraction_gt24": reference_quality[
                            "target_residual_fraction_gt24"
                        ],
                    }
                )
        # Keep the control comparison in the report even when a candidate is
        # rejected: it is useful evidence for the failure registry.
        candidate[name]["quality_delta_vs_control"] = {
            "target_mae": round(
                candidate_quality["target_mae"] - control_quality["target_mae"],
                4,
            ),
            "residual_fraction_gt24": round(
                candidate_quality["target_residual_fraction_gt24"]
                - control_quality["target_residual_fraction_gt24"],
                6,
            ),
        }

    control_time = sum(float(row["elapsed_ms"]) for row in control.values())
    candidate_time = sum(float(row["elapsed_ms"]) for row in candidate.values())
    result = {
        "benchmark": "inpaint-risk-router-ab-v1",
        "status": "pass"
        if not candidate_safety_failures and not candidate_quality_regressions
        else "fail",
        "source_revision": os.getenv("GITHUB_SHA"),
        "workers": int(workers),
        "cases": list(CASES),
        "control": {
            "class": "AdaptiveFastInpainter",
            "preload_ms": round(control_preload_ms, 3),
            "cases": control,
        },
        "candidate": {
            "class": "RiskAwareFastInpainter",
            "preload_ms": round(candidate_preload_ms, 3),
            "cases": candidate,
        },
        "forced_lama_reference": {
            "preload_ms": round(reference_preload_ms, 3),
            "cases": lama_reference,
        },
        "summary": {
            "control_case_time_ms": round(control_time, 3),
            "candidate_case_time_ms": round(candidate_time, 3),
            "case_time_pct": round(
                (candidate_time / max(1.0, control_time) - 1.0) * 100.0,
                3,
            ),
            "candidate_safety_failures": candidate_safety_failures,
            "candidate_quality_regressions": candidate_quality_regressions,
            "authority_safety": "pass"
            if not candidate_safety_failures
            else "fail",
            "quality_gate": "pass"
            if not candidate_quality_regressions
            else "fail",
            "promotion_eligible": False,
            "promotion_blockers": [
                "synthetic-only evidence; real chapter and hard/holdout gate pending",
                "risk router micro-benchmark is not a production wall-time result",
            ],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("E5_INPAINT_RISK_ROUTER=" + json.dumps(result, ensure_ascii=False), flush=True)
    if candidate_safety_failures or candidate_quality_regressions:
        raise RuntimeError("E5 risk-aware inpaint quality/safety gate failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark-results/inpaint-risk-router/report.json"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("MANGA_INPAINT_ACCURACY_WORKERS", "1")),
    )
    args = parser.parse_args()
    run(args.out, max(1, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
