#!/usr/bin/env python3
"""Normalize production profiler JSON into non-overlapping stage attribution.

The production profiler intentionally records many inclusive timers. Those event
sums are useful diagnostic evidence, but adding them together double-counts work.
This module therefore keeps two views separate:

* ``stage_breakdown``: one conservative, non-overlapping page budget derived from
  page ``processing_metrics`` and detector/inpaint sub-metrics.
* ``inclusive_timer_sums``: raw profiler method/event sums, explicitly marked as
  overlapping and never included in the page budget total.

No model runtime is imported here; existing profile JSON can be analyzed offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _num(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _section(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name, {})
    return value if isinstance(value, dict) else {}


def _first_number(*values: Any) -> float:
    for value in values:
        try:
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    return 0.0


def detector_breakdown(metrics: dict[str, Any]) -> dict[str, float | str]:
    detector = _section(metrics, "detector")
    detector_total = _first_number(
        detector.get("total_ms"),
        metrics.get("detect_ms"),
        metrics.get("detector_ms"),
    )
    bubble = _num(detector.get("bubble_model_ms"))
    mser_prefetch = _num(detector.get("focus_prefetch_mser_ms"))
    text_focus = _num(detector.get("text_model_ms"))
    mser_total = _num(detector.get("mser_ms"))
    final_mser = max(0.0, mser_total - mser_prefetch)
    grayscale = _num(detector.get("text_grayscale_fallback_ms"))

    known = bubble + mser_prefetch + text_focus + final_mser + grayscale
    other = max(0.0, detector_total - known)
    return {
        "semantics": "non_overlapping_detector_budget",
        "bubble_model_ms": round(bubble, 3),
        "mser_prefetch_ms": round(mser_prefetch, 3),
        "text_full_focus_ms": round(text_focus, 3),
        "final_mser_ms": round(final_mser, 3),
        "grayscale_fallback_ms": round(grayscale, 3),
        "detector_other_ms": round(other, 3),
        "detector_total_ms": round(detector_total, 3),
    }


def page_stage_breakdown(page: dict[str, Any]) -> dict[str, Any]:
    metrics = page.get("metrics") or page.get("processing_metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    detector = detector_breakdown(metrics)
    auto = _section(metrics, "auto_inpaint")

    read_ms = _first_number(metrics.get("read_ms"), metrics.get("read_decode_ms"))
    auto_total = _first_number(
        metrics.get("auto_inpaint_ms"),
        auto.get("total_ms"),
        metrics.get("inpaint_ms"),
    )
    lama_model = _num(auto.get("lama_model_ms"))
    inpaint_other = max(0.0, auto_total - lama_model)
    write_ms = _first_number(metrics.get("write_ms"), metrics.get("write_encode_ms"))
    page_total = _first_number(metrics.get("total_ms"), metrics.get("wall_ms"))

    known_page = read_ms + _num(detector["detector_total_ms"]) + auto_total + write_ms
    orchestration = max(0.0, page_total - known_page)
    if page_total <= 0.0:
        page_total = known_page

    return {
        "index": page.get("index"),
        "semantics": "non_overlapping_page_budget",
        "read_decode_ms": round(read_ms, 3),
        "detector": detector,
        "auto_inpaint": {
            "total_ms": round(auto_total, 3),
            "lama_model_ms": round(lama_model, 3),
            "non_lama_ms": round(inpaint_other, 3),
            "lama_model_runs": int(auto.get("lama_model_runs", 0) or 0),
            "smart_fill_regions": int(auto.get("smart_fill_regions", 0) or 0),
        },
        "write_encode_ms": round(write_ms, 3),
        "orchestration_other_ms": round(orchestration, 3),
        "page_total_ms": round(page_total, 3),
        "counts": {
            "boxes": int(page.get("boxes", 0) or 0),
            "mask_pixels": int(page.get("mask_pixels", 0) or 0),
            "changed_pixels": int(page.get("changed_pixels", 0) or 0),
        },
    }


def inclusive_timer_sums(run: dict[str, Any]) -> dict[str, Any]:
    summary = run.get("timers", {})
    if not isinstance(summary, dict):
        summary = {}
    rows = {}
    for name, raw in summary.items():
        if not isinstance(raw, dict):
            continue
        rows[str(name)] = {
            "calls": int(raw.get("calls", 0) or 0),
            "sum_ms": round(_num(raw.get("sum_ms")), 3),
            "median_ms": round(_num(raw.get("median_ms")), 3),
            "max_ms": round(_num(raw.get("max_ms")), 3),
        }
    return {
        "semantics": "inclusive_overlapping_diagnostic_timers_do_not_sum",
        "timers": rows,
    }


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    runs_out = []
    for run in report.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        pages = [
            page_stage_breakdown(page)
            for page in (run.get("pages", []) or [])
            if isinstance(page, dict)
        ]
        page_total_sum = round(sum(_num(page["page_total_ms"]) for page in pages), 3)
        runs_out.append(
            {
                "repeat": run.get("repeat"),
                "cold": bool(run.get("cold")),
                "workers": report.get("workers"),
                "wall_ms": round(_num(run.get("wall_ms")), 3),
                "page_total_sum_ms": page_total_sum,
                "parallelism_note": (
                    "page_total_sum_ms may exceed wall_ms when pages run concurrently; "
                    "compare wall_ms across worker configurations"
                ),
                "stage_breakdown": pages,
                "inclusive_timer_sums": inclusive_timer_sums(run),
            }
        )
    return {
        "source_sha": report.get("source_sha"),
        "profile": report.get("profile"),
        "semantics": {
            "stage_breakdown": "non_overlapping budget; residuals are clamped at zero",
            "inclusive_timers": "diagnostic only; method timers overlap and must not be summed",
        },
        "runs": runs_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_report(report)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
