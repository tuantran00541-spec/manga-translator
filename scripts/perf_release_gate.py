"""Evaluate the E11 full performance release gate.

Every phase report must explicitly say it is promotable.  A passing synthetic
micro-benchmark is not enough: the final gate also needs a real multi-chapter
end-to-end report with safety, memory, and speed evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_PHASES = ("E5", "E6", "E7", "E8", "E9", "E10")


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be an object: {path}")
    return value


def _phase_gate(name: str, report: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if report.get("status") != "pass":
        blockers.append(f"{name} status is {report.get('status')!r}")
    if report.get("promotion_eligible") is not True:
        summary = report.get("summary")
        if isinstance(summary, dict) and summary.get("promotion_eligible") is True:
            pass
        else:
            blockers.append(f"{name} is not explicitly promotion_eligible")

    summary = report.get("summary")
    if isinstance(summary, dict):
        for key in ("candidate_safety_failures", "candidate_quality_regressions"):
            if summary.get(key):
                blockers.append(f"{name} has {key}")
    gates = report.get("gates")
    if name in {"E7", "E8", "E9", "E10"} and not isinstance(gates, dict):
        # These phases use a standardized explicit gate object; without it a
        # result cannot be promoted merely because the top-level status says pass.
        blockers.append(f"{name} has no explicit gates")
    return not blockers, blockers


def evaluate(
    reports: dict[str, dict[str, Any]],
    e2e: dict[str, Any] | None,
    *,
    min_e2e_speedup_pct: float = 15.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    phase_results: dict[str, bool] = {}
    for name in REQUIRED_PHASES:
        report = reports.get(name)
        if report is None:
            phase_results[name] = False
            blockers.append(f"{name} report missing")
            continue
        passed, phase_blockers = _phase_gate(name, report)
        phase_results[name] = passed
        blockers.extend(phase_blockers)

    e2e_gate: dict[str, bool] = {}
    speedup_pct = None
    if e2e is None:
        blockers.append("real multi-chapter E2E report missing")
    else:
        speedup_value = e2e.get("speedup_pct")
        if speedup_value is None:
            control = e2e.get("control")
            candidate = e2e.get("candidate")
            if isinstance(control, dict) and isinstance(candidate, dict):
                control_wall = float(control.get("wall_ms") or 0.0)
                candidate_wall = float(candidate.get("wall_ms") or 0.0)
                if control_wall > 0:
                    speedup_value = (1.0 - candidate_wall / control_wall) * 100.0
        if speedup_value is None:
            blockers.append("E2E report has no speedup_pct or control/candidate wall_ms")
        else:
            speedup_pct = float(speedup_value)
        e2e_gate["safety"] = (
            e2e.get("authority_outside_changed", 1) == 0
            or e2e.get("safety") == "pass"
            or (isinstance(e2e.get("gates"), dict) and e2e["gates"].get("safety") is True)
        )
        if not e2e_gate["safety"]:
            blockers.append("E2E authority safety gate is not zero/pass")
        e2e_gate["memory"] = (
            e2e.get("memory_gate") == "pass"
            or (isinstance(e2e.get("gates"), dict) and e2e["gates"].get("memory") is True)
            or e2e.get("rss_within_cap") is True
        )
        if not e2e_gate["memory"]:
            blockers.append("E2E memory/RSS gate is not explicit pass")
        e2e_gate["speed"] = speedup_pct is not None and speedup_pct >= min_e2e_speedup_pct
        if not e2e_gate["speed"]:
            blockers.append(
                f"E2E speedup {speedup_pct!r} is below {min_e2e_speedup_pct:.1f}%"
            )

    return {
        "status": "pass" if not blockers else "blocked",
        "required_phases": list(REQUIRED_PHASES),
        "phase_gates": phase_results,
        "e2e_gates": e2e_gate,
        "e2e_speedup_pct": round(speedup_pct, 3) if speedup_pct is not None else None,
        "blockers": blockers,
        "thresholds": {"min_e2e_speedup_pct": min_e2e_speedup_pct},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E11 performance release gate")
    parser.add_argument(
        "--report", action="append", default=[], metavar="PHASE=PATH",
        help="phase report, repeated for E5 through E10",
    )
    parser.add_argument("--e2e-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-e2e-speedup-pct", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: dict[str, dict[str, Any]] = {}
    parse_blockers: list[str] = []
    for item in args.report:
        if "=" not in item:
            parse_blockers.append(f"invalid --report {item!r}; use PHASE=PATH")
            continue
        name, raw_path = item.split("=", 1)
        try:
            reports[name.strip().upper()] = _load_report(Path(raw_path))
        except Exception as exc:
            parse_blockers.append(f"cannot load {name}: {type(exc).__name__}: {exc}")
    e2e = None
    if args.e2e_report:
        try:
            e2e = _load_report(args.e2e_report)
        except Exception as exc:
            parse_blockers.append(f"cannot load E2E report: {type(exc).__name__}: {exc}")
    result = evaluate(reports, e2e, min_e2e_speedup_pct=args.min_e2e_speedup_pct)
    result["blockers"] = parse_blockers + list(result.get("blockers") or [])
    if result["blockers"]:
        result["status"] = "blocked"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("E11_RELEASE_GATE=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
