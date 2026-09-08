#!/usr/bin/env python3
"""Deterministic contract tests for Phase 7 profiler stage attribution."""
from __future__ import annotations

from profile_stage_breakdown import analyze_report, detector_breakdown, page_stage_breakdown


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> int:
    metrics = {
        "read_ms": 10.0,
        "detect_ms": 100.0,
        "auto_inpaint_ms": 50.0,
        "write_ms": 5.0,
        "total_ms": 180.0,
        "detector": {
            "total_ms": 100.0,
            "bubble_model_ms": 20.0,
            "focus_prefetch_mser_ms": 10.0,
            "text_model_ms": 40.0,
            "mser_ms": 18.0,
            "text_grayscale_fallback_ms": 7.0,
        },
        "auto_inpaint": {
            "total_ms": 50.0,
            "lama_model_ms": 30.0,
            "lama_model_runs": 2,
            "smart_fill_regions": 3,
        },
    }
    detector = detector_breakdown(metrics)
    check(detector["final_mser_ms"] == 8.0, "final MSER must exclude prefetched pass")
    check(detector["detector_other_ms"] == 15.0, "detector residual attribution is wrong")

    page = page_stage_breakdown(
        {
            "index": 4,
            "metrics": metrics,
            "boxes": 9,
            "mask_pixels": 123,
            "changed_pixels": 100,
        }
    )
    check(page["auto_inpaint"]["non_lama_ms"] == 20.0, "non-LaMa residual is wrong")
    check(page["orchestration_other_ms"] == 15.0, "page residual is wrong")
    check(page["page_total_ms"] == 180.0, "page total changed")
    check(page["counts"]["boxes"] == 9, "page counts were not preserved")

    nested = page_stage_breakdown(
        {
            "index": 5,
            "metrics": {
                "timing_ms": {
                    "read": 4.0, "detect": 100.0, "auto_inpaint": 30.0,
                    "write": 6.0, "total": 145.0,
                },
                "detector": {"total_ms": 100.0},
                "auto_inpaint": {"lama_model_ms": 20.0},
            },
        }
    )
    check(nested["read_decode_ms"] == 4.0, "nested timing read was ignored")
    check(nested["auto_inpaint"]["total_ms"] == 30.0, "nested inpaint timing was ignored")
    check(nested["write_encode_ms"] == 6.0, "nested write timing was ignored")
    check(nested["page_total_ms"] == 145.0, "nested page total was ignored")
    check(nested["orchestration_other_ms"] == 5.0, "nested residual attribution is wrong")

    # Inclusive timer sums must remain a separately labelled diagnostic view.
    report = {
        "source_sha": "abc",
        "profile": "fixture",
        "workers": 2,
        "runs": [
            {
                "repeat": 0,
                "cold": True,
                "wall_ms": 150.0,
                "pages": [{"index": 4, "metrics": metrics}],
                "timers": {
                    "combined.detect": {"calls": 1, "sum_ms": 100.0, "median_ms": 100.0, "max_ms": 100.0},
                    "yolo._nms": {"calls": 2, "sum_ms": 30.0, "median_ms": 15.0, "max_ms": 20.0},
                },
            }
        ],
    }
    analyzed = analyze_report(report)
    run = analyzed["runs"][0]
    check(run["wall_ms"] == 150.0, "run wall changed")
    check(run["page_total_sum_ms"] == 180.0, "page service-time sum changed")
    check(
        run["inclusive_timer_sums"]["semantics"]
        == "inclusive_overlapping_diagnostic_timers_do_not_sum",
        "inclusive timers lost anti-double-count semantics",
    )

    # Malformed/overlapping metrics must never create negative residuals.
    pathological = detector_breakdown(
        {
            "detect_ms": 5.0,
            "detector": {
                "bubble_model_ms": 10.0,
                "focus_prefetch_mser_ms": 4.0,
                "text_model_ms": 10.0,
                "mser_ms": 2.0,
            },
        }
    )
    check(pathological["final_mser_ms"] == 0.0, "derived final MSER became negative")
    check(pathological["detector_other_ms"] == 0.0, "detector residual became negative")

    print("backend profiler attribution sanity: phase 7 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
