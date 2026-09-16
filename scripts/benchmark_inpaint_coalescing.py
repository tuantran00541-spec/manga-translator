"""A/B benchmark for the isolated E6 bounded LaMa coalescer.

The candidate changes only how compatible authorized regions share a LaMa
context crop.  It does not union or expand destructive masks.  Synthetic
backgrounds provide a deterministic clean reference so a one-call candidate
can be compared with the current two-call control on both quality and safety.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.detector.bubble_detector import BubbleBox
from app.inpaint.risk_aware_inpainter import (
    CoalescingRiskAwareInpainter,
    RiskAwareFastInpainter,
)
from scripts.benchmark_inpaint_accuracy import (
    _background,
    _box_mask,
    _metric_row,
    _text_mask,
)


CASE_NAMES = ("gradient_pair", "textured_pair")


def _box(
    coords: tuple[int, int, int, int],
    *,
    lines: list[str],
    scale: float,
) -> BubbleBox:
    x1, y1, x2, y2 = coords
    return BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.99,
        _text_mask(coords, lines, scale),
        source_model="text_segmenter.onnx",
        class_name="text_comic",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )


def _multi_case(name: str) -> tuple[np.ndarray, np.ndarray, list[BubbleBox], np.ndarray]:
    if name == "gradient_pair":
        background = _background("gradient", 640, 960)
        coords = [(100, 130, 300, 350), (340, 130, 540, 350)]
        lines = ["REMOVE THIS TEXT", "KEEP THE GRADIENT"]
        scale = 0.55
    elif name == "textured_pair":
        background = _background("textured", 640, 960)
        coords = [(110, 110, 290, 330), (330, 110, 510, 330)]
        lines = ["REMOVE THIS TEXT", "KEEP THE TEXTURE"]
        scale = 0.55
    else:
        raise ValueError(f"unknown coalescing case {name!r}")

    boxes = [_box(item, lines=lines, scale=scale) for item in coords]
    source = background.astype(np.float32).copy()
    for box in boxes:
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        alpha = box.mask.astype(np.float32) / 255.0
        crop = source[y1:y2, x1:x2]
        crop[:] = crop * (1.0 - alpha[:, :, None]) + 18.0 * alpha[:, :, None]
    source = np.clip(source, 0, 255).astype(np.uint8)

    target = np.zeros(source.shape[:2], dtype=np.uint8)
    for box in boxes:
        target[box.y1:box.y2, box.x1:box.x2] = np.maximum(
            target[box.y1:box.y2, box.x1:box.x2], box.mask
        )
    return source, background, boxes, target


def _authority_mask(
    source: np.ndarray,
    boxes: list[BubbleBox],
    inpainter: RiskAwareFastInpainter,
) -> np.ndarray:
    authority = np.zeros(source.shape[:2], dtype=np.uint8)
    for box in boxes:
        authority = np.maximum(authority, _box_mask(source, box, inpainter))
    return authority


def _prepare(inpainter: RiskAwareFastInpainter, workers: int) -> float:
    prepare = getattr(inpainter, "prepare_for_page_workers", None)
    if callable(prepare):
        prepare(max(1, int(workers)))
    started = time.perf_counter()
    inpainter.preload()
    return (time.perf_counter() - started) * 1000.0


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 3)
    except Exception:
        return None


def _run_suite(factory, workers: int) -> tuple[dict, float]:
    inpainter = factory()
    preload_ms = _prepare(inpainter, workers)
    rows: dict[str, dict] = {}
    for name in CASE_NAMES:
        source, reference, boxes, target = _multi_case(name)
        authority = _authority_mask(source, boxes, inpainter)
        started = time.perf_counter()
        output = inpainter.inpaint(source, boxes)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows[name] = {
            "elapsed_ms": round(elapsed_ms, 3),
            "rss_mb_after": _rss_mb(),
            "inpaint_metrics": inpainter.last_metrics(),
            "quality": _metric_row(
                source, reference, output, target, authority
            ),
        }
    return rows, preload_ms


def _delta(candidate: dict, control: dict) -> dict[str, float]:
    return {
        "target_mae": round(
            float(candidate["target_mae"]) - float(control["target_mae"]), 4
        ),
        "residual_fraction_gt24": round(
            float(candidate["target_residual_fraction_gt24"])
            - float(control["target_residual_fraction_gt24"]),
        6),
        "whole_image_mae": round(
            float(candidate["whole_image_mae"])
            - float(control["whole_image_mae"]),
            4,
        ),
    }


def run(out: Path, workers: int) -> dict:
    control, control_preload_ms = _run_suite(RiskAwareFastInpainter, workers)
    candidate, candidate_preload_ms = _run_suite(
        CoalescingRiskAwareInpainter, workers
    )

    safety_failures: list[str] = []
    quality_regressions: list[dict] = []
    coalesce_merges = 0
    control_runs = 0
    candidate_runs = 0
    for name in CASE_NAMES:
        control_quality = control[name]["quality"]
        candidate_quality = candidate[name]["quality"]
        if candidate_quality["outside_changed_pixels_exact"]:
            safety_failures.append(name)
        if candidate_quality["target_mae"] > control_quality["target_mae"] + 0.75:
            quality_regressions.append(
                {
                    "case": name,
                    "metric": "target_mae",
                    "candidate": candidate_quality["target_mae"],
                    "control": control_quality["target_mae"],
                }
            )
        if candidate_quality["target_residual_fraction_gt24"] > (
            control_quality["target_residual_fraction_gt24"] + 0.03
        ):
            quality_regressions.append(
                {
                    "case": name,
                    "metric": "target_residual_fraction_gt24",
                    "candidate": candidate_quality["target_residual_fraction_gt24"],
                    "control": control_quality["target_residual_fraction_gt24"],
                }
            )
        control_runs += int(
            control[name]["inpaint_metrics"].get("lama_model_runs", 0)
        )
        candidate_metrics = candidate[name]["inpaint_metrics"]
        candidate_runs += int(candidate_metrics.get("lama_model_runs", 0))
        coalesce_merges += int(candidate_metrics.get("coalesce_merges", 0))
        candidate[name]["quality_delta_vs_control"] = _delta(
            candidate_quality, control_quality
        )

    control_time = sum(float(control[name]["elapsed_ms"]) for name in CASE_NAMES)
    candidate_time = sum(float(candidate[name]["elapsed_ms"]) for name in CASE_NAMES)
    result = {
        "benchmark": "inpaint-coalescing-ab-v1",
        "status": "pass"
        if not safety_failures and not quality_regressions and candidate_runs < control_runs
        else "fail",
        "source_revision": os.getenv("GITHUB_SHA"),
        "workers": int(workers),
        "cases": list(CASE_NAMES),
        "control": {
            "class": "RiskAwareFastInpainter",
            "preload_ms": round(control_preload_ms, 3),
            "cases": control,
        },
        "candidate": {
            "class": "CoalescingRiskAwareInpainter",
            "preload_ms": round(candidate_preload_ms, 3),
            "cases": candidate,
        },
        "summary": {
            "control_case_time_ms": round(control_time, 3),
            "candidate_case_time_ms": round(candidate_time, 3),
            "case_time_pct": round(
                (candidate_time / max(1.0, control_time) - 1.0) * 100.0,
                3,
            ),
            "control_lama_model_runs": control_runs,
            "candidate_lama_model_runs": candidate_runs,
            "lama_runs_reduction_pct": round(
                (1.0 - candidate_runs / float(max(1, control_runs))) * 100.0,
                3,
            ),
            "coalesce_merges": coalesce_merges,
            "candidate_safety_failures": safety_failures,
            "candidate_quality_regressions": quality_regressions,
            "authority_safety": "pass" if not safety_failures else "fail",
            "quality_gate": "pass" if not quality_regressions else "fail",
            "run_reduction_gate": "pass"
            if candidate_runs < control_runs
            else "fail",
            "promotion_eligible": False,
            "promotion_blockers": [
                "synthetic-only evidence; real chapter and hard/holdout gate pending",
            ],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("E6_INPAINT_COALESCING=" + json.dumps(result, ensure_ascii=False), flush=True)
    if safety_failures or quality_regressions or candidate_runs >= control_runs:
        raise RuntimeError("E6 inpaint coalescing gate failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark-results/inpaint-coalescing/report.json"),
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
