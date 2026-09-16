"""Benchmark the E10 OCR batch seam against the current per-crop reader.

The input is intentionally an explicit ground-truth file rather than a
confidence-derived pseudo-label.  Example row::

    {"case_id": "p01-b02", "image": "crops/p01-b02.png", "lang": "en",
     "text": "Save the child", "target_mode": "all"}

The script is an offline gate.  It does not change the production OCR profile
and reports ``blocked`` when a labeled dataset or OCR runtime is unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.image_io import read_image
from app.ocr.metrics import summarize_ocr_rows
from app.ocr.multi_lang_ocr import MultiLangOCR


def _load_cases(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("OCR ground truth must contain a non-empty rows array")
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"ground-truth row {index} must be an object")
        image = Path(str(row.get("image") or ""))
        if not image.is_absolute():
            image = path.parent / image
        lang = str(row.get("lang") or "").strip()
        if not lang:
            raise ValueError(f"ground-truth row {index} has no lang")
        if not image.is_file():
            raise ValueError(f"ground-truth image missing: {image}")
        cases.append(
            {
                "case_id": str(row.get("case_id") or f"case-{index:04d}"),
                "image": image,
                "lang": lang,
                "text": str(row.get("text") or ""),
                "target_mode": str(row.get("target_mode") or "all"),
                "partition": str(row.get("partition") or "holdout"),
            }
        )
    return cases


def _read_one(engine: MultiLangOCR, image, case: dict[str, Any]) -> tuple[Any, float]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    started = time.perf_counter()
    result = engine.read_detailed(
        rgb,
        case["lang"],
        target_mode=case["target_mode"],
    )
    return result, (time.perf_counter() - started) * 1000.0


def _run_control(engine: MultiLangOCR, cases: list[dict[str, Any]], images: list[Any]):
    rows: list[dict[str, Any]] = []
    total_ms = 0.0
    for case, image in zip(cases, images):
        result, elapsed_ms = _read_one(engine, image, case)
        total_ms += elapsed_ms
        rows.append(
            {
                "case_id": case["case_id"],
                "reference": case["text"],
                "hypothesis": result.text,
                "model": result.model,
                "quality": result.quality,
                "latency_ms": round(elapsed_ms, 3),
            }
        )
    return rows, total_ms


def _run_batch(
    engine: MultiLangOCR,
    cases: list[dict[str, Any]],
    images: list[Any],
    batch_size: int,
):
    rows: list[dict[str, Any] | None] = [None] * len(cases)
    total_ms = 0.0
    start = 0
    while start < len(cases):
        first = cases[start]
        end = start + 1
        while end < len(cases):
            if (
                cases[end]["lang"] != first["lang"]
                or cases[end]["target_mode"] != first["target_mode"]
                or end - start >= int(batch_size)
            ):
                break
            end += 1
        rgb_images = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for image in images[start:end]
        ]
        started = time.perf_counter()
        results = engine.read_batch(
            rgb_images,
            first["lang"],
            target_mode=first["target_mode"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_ms += elapsed_ms
        if len(results) != end - start:
            raise RuntimeError(
                f"OCR batch returned {len(results)} results for {end - start} inputs"
            )
        per_item_ms = elapsed_ms / float(max(1, end - start))
        for offset, result in enumerate(results):
            case = cases[start + offset]
            rows[start + offset] = {
                "case_id": case["case_id"],
                "reference": case["text"],
                "hypothesis": result.text,
                "model": result.model,
                "quality": result.quality,
                "latency_ms": round(per_item_ms, 3),
            }
        start = end
    return [row for row in rows if row is not None], total_ms


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.ground_truth.is_file():
        return {
            "status": "blocked",
            "benchmark": "ocr-batch-ab-v1",
            "blockers": [f"ground-truth file missing: {args.ground_truth}"],
        }
    try:
        cases = _load_cases(args.ground_truth)
    except Exception as exc:
        return {
            "status": "blocked",
            "benchmark": "ocr-batch-ab-v1",
            "blockers": [f"invalid ground truth: {type(exc).__name__}: {exc}"],
        }

    try:
        images = [read_image(case["image"]) for case in cases]
        engine = MultiLangOCR()
        if args.warmup:
            _read_one(engine, images[0], cases[0])
        control_rows, control_ms = _run_control(engine, cases, images)
        batch_rows, batch_ms = _run_batch(
            engine, cases, images, max(1, int(args.batch_size))
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "benchmark": "ocr-batch-ab-v1",
            "case_count": len(cases),
            "blockers": [f"OCR runtime unavailable: {type(exc).__name__}: {exc}"],
        }

    control_quality = summarize_ocr_rows(control_rows)
    batch_quality = summarize_ocr_rows(batch_rows)
    speedup_pct = (1.0 - batch_ms / float(max(1e-6, control_ms))) * 100.0
    quality_gate = {
        "cer": batch_quality["cer"] <= control_quality["cer"] + args.max_cer_regression,
        "wer": batch_quality["wer"] <= control_quality["wer"] + args.max_wer_regression,
        "line_recall": batch_quality["line_recall"]
        >= control_quality["line_recall"] - args.max_line_recall_drop,
    }
    speed_gate = speedup_pct >= float(args.min_speedup_pct)
    return {
        "status": "pass" if all(quality_gate.values()) and speed_gate else "fail",
        "benchmark": "ocr-batch-ab-v1",
        "dataset": {
            "path": args.ground_truth.as_posix(),
            "case_count": len(cases),
            "partitions": dict(
                sorted(
                    (partition, sum(case["partition"] == partition for case in cases))
                    for partition in {case["partition"] for case in cases}
                )
            ),
        },
        "control": {
            "quality": control_quality,
            "wall_ms": round(control_ms, 3),
            "rows": control_rows,
        },
        "candidate_batch": {
            "quality": batch_quality,
            "wall_ms": round(batch_ms, 3),
            "rows": batch_rows,
            "runtime_metrics": engine.batch_metrics(),
        },
        "quality_gate": quality_gate,
        "speed_gate": speed_gate,
        "speedup_pct": round(speedup_pct, 3),
        "thresholds": {
            "max_cer_regression": args.max_cer_regression,
            "max_wer_regression": args.max_wer_regression,
            "max_line_recall_drop": args.max_line_recall_drop,
            "min_speedup_pct": args.min_speedup_pct,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark guarded OCR batching")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark-results/ocr-batch/report.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cer-regression", type=float, default=0.0)
    parser.add_argument("--max-wer-regression", type=float, default=0.0)
    parser.add_argument("--max-line-recall-drop", type=float, default=0.0)
    parser.add_argument("--min-speedup-pct", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("E10_OCR_BATCH=" + json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report.get("status") == "pass" else 2 if report.get("status") == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
