#!/usr/bin/env python3
"""Fail-closed summary for repeated E0 chapter benchmark artifacts.

This tool intentionally separates *measurement* from promotion.  It accepts
paired compact reports and source/model manifests from multiple fresh runs,
refuses to pool unlike evidence, calculates median/P95 latency, and reports
the E0 P0 gates.  It never marks a candidate production-eligible: the failure
registry and OCR ground truth still require explicit human review.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.benchmarking.manifest import load_manifest, sha256_json


SUMMARY_SCHEMA = "manga-translator.e0-variance.v1"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    result = _number(value, label)
    if not result.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(result)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only evidence fields that must be exact for a valid comparison."""
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("E0 manifest must contain a non-empty cases array")
    source_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"E0 manifest cases[{index}] must be an object")
        source_hash = str(case.get("source_sha256") or "").lower()
        slice_hash = str(case.get("slice_sha256") or "").lower()
        if len(source_hash) != 64 or len(slice_hash) != 64:
            raise ValueError(f"E0 manifest cases[{index}] lacks source/slice SHA-256")
        source_cases.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "source_page": _integer(case.get("source_page"), f"cases[{index}].source_page"),
                "slice_index": _integer(case.get("slice_index"), f"cases[{index}].slice_index"),
                "source_sha256": source_hash,
                "slice_sha256": slice_hash,
            }
        )
    models = manifest.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("E0 manifest must contain model hashes")
    normalized_models: dict[str, str] = {}
    for name, model in models.items():
        if not isinstance(model, Mapping):
            raise ValueError(f"E0 manifest model {name!r} must be an object")
        digest = str(model.get("sha256") or "").lower()
        if len(digest) != 64:
            raise ValueError(f"E0 manifest model {name!r} lacks a SHA-256")
        normalized_models[str(name)] = digest
    return {
        "dataset": str(manifest.get("dataset") or ""),
        "repo_sha": str(manifest.get("repo_sha") or ""),
        "profile": str(manifest.get("profile") or ""),
        "models": normalized_models,
        "parameters": manifest.get("parameters") or {},
        "cases": sorted(source_cases, key=lambda row: row["case_id"]),
    }


def _report_metrics(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    clean = report.get("clean")
    ocr = report.get("ocr")
    accuracy = report.get("inpaint_accuracy")
    if not isinstance(clean, Mapping) or not isinstance(ocr, Mapping) or not isinstance(accuracy, Mapping):
        raise ValueError("compact report needs clean, ocr and inpaint_accuracy objects")
    synthetic = accuracy.get("synthetic")
    real = accuracy.get("real_chapter")
    if not isinstance(synthetic, Mapping) or not isinstance(real, Mapping):
        raise ValueError("inpaint_accuracy needs synthetic and real_chapter objects")
    slices = _integer(report.get("slices"), "report.slices")
    source_images = _integer(report.get("source_images"), "report.source_images")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != slices:
        raise ValueError("report slices do not match the paired E0 manifest")
    unique_sources = {str(case.get("source_sha256") or "") for case in cases}
    if len(unique_sources) != source_images:
        raise ValueError("report source_images do not match the paired E0 manifest")
    if str(report.get("chapter_id") or "") != str((manifest.get("parameters") or {}).get("chapter_id") or ""):
        raise ValueError("report chapter_id does not match the paired E0 manifest")
    if str(report.get("source_url") or "") != str(manifest.get("dataset") or ""):
        raise ValueError("report source_url does not match the paired E0 manifest")
    errors = ocr.get("errors")
    if not isinstance(errors, list):
        raise ValueError("report.ocr.errors must be an array")
    return {
        "run_id": report.get("run_id"),
        "clean_wall_s": _number(clean.get("wall_s"), "clean.wall_s"),
        "clean_slices_per_min": _number(clean.get("slices_per_min"), "clean.slices_per_min"),
        "safe_boxes": _integer(clean.get("safe_boxes"), "clean.safe_boxes"),
        "review_boxes": _integer(clean.get("review_boxes"), "clean.review_boxes"),
        "ocr_planned": _integer(ocr.get("planned"), "ocr.planned"),
        "ocr_completed": _integer(ocr.get("completed"), "ocr.completed"),
        "ocr_errors": len(errors),
        "real_outside_changed_pixels_exact": _integer(
            real.get("outside_changed_pixels_exact"),
            "real_chapter.outside_changed_pixels_exact",
        ),
        "synthetic_outside_changed_pixels_exact": _integer(
            synthetic.get("outside_changed_pixels_exact"),
            "synthetic.outside_changed_pixels_exact",
        ),
        "residue_records": _integer(real.get("residue_records"), "real_chapter.residue_records"),
        "pages_with_residue": _integer(
            real.get("pages_with_residue"), "real_chapter.pages_with_residue"
        ),
    }


def summarize_e0_variance(
    *,
    report_paths: Sequence[Path],
    manifest_paths: Sequence[Path],
    reference_report_path: Path | None = None,
    reference_manifest_path: Path | None = None,
    min_runs: int = 4,
    max_runtime_regression_pct: float = 2.0,
    max_residue_records: int = 30,
) -> dict[str, Any]:
    """Summarize compatible E0 cold/warm reports without granting promotion."""
    if len(report_paths) != len(manifest_paths):
        raise ValueError("--report and --e0-manifest must be supplied in equal counts")
    if not report_paths:
        raise ValueError("at least one paired --report and --e0-manifest is required")
    if min_runs < 1:
        raise ValueError("min_runs must be >= 1")
    if max_runtime_regression_pct < 0:
        raise ValueError("max_runtime_regression_pct must be >= 0")
    if max_residue_records < 0:
        raise ValueError("max_residue_records must be >= 0")

    identities: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for report_path, manifest_path in zip(report_paths, manifest_paths):
        manifest = load_manifest(manifest_path)
        identity = _manifest_identity(manifest)
        report = _load_object(report_path, "compact report")
        identities.append(identity)
        runs.append(_report_metrics(report, manifest))

    expected_identity = identities[0]
    compatible = all(identity == expected_identity for identity in identities[1:])
    clean_walls = [float(run["clean_wall_s"]) for run in runs]
    throughputs = [float(run["clean_slices_per_min"]) for run in runs]
    if (reference_report_path is None) != (reference_manifest_path is None):
        raise ValueError(
            "--reference-report and --reference-e0-manifest must be supplied together"
        )
    reference_wall = None
    reference_compatible = False
    if reference_report_path is not None and reference_manifest_path is not None:
        reference_manifest = load_manifest(reference_manifest_path)
        reference_identity = _manifest_identity(reference_manifest)
        reference = _load_object(reference_report_path, "reference compact report")
        reference_metrics = _report_metrics(reference, reference_manifest)
        reference_wall = float(reference_metrics["clean_wall_s"])
        reference_compatible = reference_identity == expected_identity

    for run in runs:
        if reference_wall is not None:
            run["runtime_regression_pct"] = round(
                (float(run["clean_wall_s"]) / reference_wall - 1.0) * 100.0, 4
            )

    outside_safe = all(
        run["real_outside_changed_pixels_exact"] == 0
        and run["synthetic_outside_changed_pixels_exact"] == 0
        for run in runs
    )
    ocr_complete = all(
        run["ocr_completed"] == run["ocr_planned"] and run["ocr_errors"] == 0
        for run in runs
    )
    residue_gate = all(run["residue_records"] <= max_residue_records for run in runs)
    runtime_gate = reference_compatible and all(
        float(run["runtime_regression_pct"]) <= max_runtime_regression_pct
        for run in runs
    )
    gates = {
        "paired_source_model_config_exact": compatible,
        "enough_cold_warm_runs": len(runs) >= min_runs,
        "outside_authority_zero": outside_safe,
        "ocr_completed_without_errors": ocr_complete,
        "residue_not_above_limit": residue_gate,
        "runtime_reference_provided": reference_wall is not None,
        "runtime_reference_source_model_config_exact": reference_compatible,
        "runtime_not_regressed": runtime_gate,
    }
    all_measured_gates_pass = all(gates.values())
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "review_required" if all_measured_gates_pass else "blocked",
        "promotion_eligible": False,
        "reason": (
            "E0 variance evidence is measured, but human-reviewed failures, OCR ground truth, "
            "and source-separated hard/holdout gates are still required."
        ),
        "inputs": {
            "reports": [str(path) for path in report_paths],
            "e0_manifests": [str(path) for path in manifest_paths],
            "reference_report": str(reference_report_path) if reference_report_path else None,
            "reference_e0_manifest": str(reference_manifest_path) if reference_manifest_path else None,
            "identity_sha256": sha256_json(expected_identity),
        },
        "limits": {
            "min_runs": min_runs,
            "max_runtime_regression_pct": max_runtime_regression_pct,
            "max_residue_records": max_residue_records,
        },
        "gates": gates,
        "variance": {
            "runs": len(runs),
            "clean_wall_s": {
                "min": round(min(clean_walls), 4),
                "median": round(statistics.median(clean_walls), 4),
                "p95": round(_percentile(clean_walls, 0.95), 4),
                "max": round(max(clean_walls), 4),
            },
            "clean_slices_per_min": {
                "min": round(min(throughputs), 4),
                "median": round(statistics.median(throughputs), 4),
                "p95": round(_percentile(throughputs, 0.95), 4),
                "max": round(max(throughputs), 4),
            },
        },
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True, help="compact report.json; repeat per run")
    parser.add_argument("--e0-manifest", type=Path, action="append", required=True, help="paired E0 benchmark-manifest.json; repeat per run")
    parser.add_argument("--reference-report", type=Path, help="compact baseline report for the runtime gate")
    parser.add_argument("--reference-e0-manifest", type=Path, help="paired baseline E0 manifest; required with --reference-report")
    parser.add_argument("--min-runs", type=int, default=4, help="one cold plus this many total runs (default: 4)")
    parser.add_argument("--max-runtime-regression-pct", type=float, default=2.0)
    parser.add_argument("--max-residue-records", type=int, default=30)
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--require-gates", action="store_true", help="return non-zero when a measured gate is blocked")
    args = parser.parse_args()
    try:
        result = summarize_e0_variance(
            report_paths=args.report,
            manifest_paths=args.e0_manifest,
            reference_report_path=args.reference_report,
            reference_manifest_path=args.reference_e0_manifest,
            min_runs=args.min_runs,
            max_runtime_regression_pct=args.max_runtime_regression_pct,
            max_residue_records=args.max_residue_records,
        )
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.require_gates and result["status"] != "review_required":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
