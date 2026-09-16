"""Gate controlled mixed-precision/QAT evidence for E9.

The gate consumes a report produced by an external calibration/training run;
it never quantizes or replaces a production model itself.  This keeps the
existing FP32 models intact when calibration data or sensitive-node analysis
is incomplete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACCURACY_METRICS = ("box_recall", "mask_boundary_f1", "mask_iou")


def evaluate(
    report: Any,
    *,
    max_accuracy_drop: float = 0.0,
    min_speedup_pct: float = 5.0,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {"status": "blocked", "blockers": ["precision report must be an object"]}
    blockers: list[str] = []
    mode = str(report.get("precision_mode") or "").strip().lower()
    if mode not in {"mixed", "qat", "accuracy_controlled"}:
        blockers.append("precision_mode must be mixed, qat, or accuracy_controlled")
    baseline = report.get("baseline")
    candidate = report.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        blockers.append("baseline and candidate metric objects are required")
        return {"status": "blocked", "blockers": blockers}
    sensitivity = report.get("sensitivity")
    if not isinstance(sensitivity, list) or not sensitivity:
        blockers.append("layer sensitivity evidence is required")
    for metric in ACCURACY_METRICS:
        if metric not in baseline or metric not in candidate:
            blockers.append(f"missing accuracy metric: {metric}")
    if blockers:
        return {"status": "blocked", "blockers": blockers}

    accuracy = {
        metric: float(candidate[metric]) >= float(baseline[metric]) - float(max_accuracy_drop)
        for metric in ACCURACY_METRICS
    }
    safety = int(candidate.get("authority_outside_changed", 1)) == 0
    false_negative = int(candidate.get("hard_holdout_false_negatives", 1)) == 0
    if "wall_ms" not in baseline or "wall_ms" not in candidate:
        blockers.append("baseline/candidate wall_ms is required")
        return {"status": "blocked", "blockers": blockers, "accuracy_gate": accuracy}
    speedup_pct = (1.0 - float(candidate["wall_ms"]) / max(1e-6, float(baseline["wall_ms"]))) * 100.0
    speed = speedup_pct >= float(min_speedup_pct)
    gates = {
        "accuracy": accuracy,
        "safety": safety,
        "hard_holdout_false_negatives": false_negative,
        "speed": speed,
    }
    passed = all(accuracy.values()) and safety and false_negative and speed
    return {
        "status": "pass" if passed else "fail",
        "gates": gates,
        "speedup_pct": round(speedup_pct, 3),
        "sensitivity_nodes": len(sensitivity),
        "thresholds": {
            "max_accuracy_drop": max_accuracy_drop,
            "min_speedup_pct": min_speedup_pct,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.report.is_file():
        return {"status": "blocked", "blockers": [f"precision report missing: {args.report}"]}
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"invalid precision report: {type(exc).__name__}: {exc}"]}
    return evaluate(
        report,
        max_accuracy_drop=args.max_accuracy_drop,
        min_speedup_pct=args.min_speedup_pct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate controlled precision evidence")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--min-speedup-pct", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("E9_PRECISION=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "pass" else 2 if result.get("status") == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
