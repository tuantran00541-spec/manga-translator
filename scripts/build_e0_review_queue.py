#!/usr/bin/env python3
"""Build the E0 review queue from one real clean/OCR audit.

The queue is deliberately *not* ground truth.  It freezes the source and
slice hashes, geometry, model/config provenance and automated signals needed
for a reviewer to decide whether a residue or weak OCR result is real.  Every
generated failure case remains ``unreviewed`` and ``unassigned`` so later
hard/holdout gates cannot accidentally train on verifier or confidence noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.benchmarking.failure_registry import (
    FailureRegistry,
    build_failure_case,
    stable_case_id,
)
from app.benchmarking.manifest import build_manifest, sha256_file, write_manifest


REVIEW_QUEUE_SCHEMA = "manga-translator.e0-review-queue.v1"


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _repo_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_local_path(value: object, manifest_path: Path) -> Path:
    raw = Path(str(value or ""))
    if not str(raw):
        raise ValueError("manifest page has no original path")
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, manifest_path.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"manifest original image is unavailable: {raw}")


def _source_hashes(source_index: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    rows = source_index.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source index must contain a non-empty sources array")
    result: dict[int, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"source index sources[{index}] must be an object")
        try:
            source_page = int(row.get("source_page"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source index sources[{index}] has invalid source_page") from exc
        digest = str(row.get("sha256") or row.get("source_sha256") or "").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"source index sources[{index}] has no valid SHA-256")
        if source_page in result:
            raise ValueError(f"source index repeats source_page {source_page}")
        result[source_page] = {
            "sha256": digest,
            "path": str(row.get("path") or ""),
        }
    return result


def _box_snapshot(box: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: box.get(key)
        for key in (
            "id",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "semantic_type",
            "mask_source",
            "safe_to_inpaint",
            "ocr_eligible",
            "needs_review",
            "deferred_reason",
        )
        if key in box
    }


def _ocr_taxonomy(result: Mapping[str, Any]) -> str:
    reason = str(result.get("quality_reason") or "").lower()
    text = str(result.get("text") or "").strip()
    if "edge" in reason:
        return "ocr_edge"
    if not text or "empty" in reason:
        return "ocr_empty"
    if "incomplete" in reason or "partial" in reason:
        return "ocr_partial"
    # Low confidence is an automated signal, not proof of a partial transcript.
    return "other"


def _parse_models(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --model {item!r}; use name=path")
        name, value = item.split("=", 1)
        if not name.strip() or not value.strip():
            raise ValueError(f"invalid --model {item!r}; use name=path")
        result[name.strip()] = value.strip()
    return result


def _audit_crop(
    result: Mapping[str, Any], ocr_report_path: Path
) -> dict[str, Any] | None:
    """Return a hash-verified OCR crop recorded by the audit, when available."""
    raw = result.get("audit_crop")
    if not isinstance(raw, Mapping):
        return None
    relative = str(raw.get("path") or "").strip()
    expected = str(raw.get("sha256") or "").lower()
    if not relative or len(expected) != 64:
        return None
    candidate = Path(relative)
    crop_path = candidate if candidate.is_absolute() else ocr_report_path.parent / candidate
    actual = sha256_file(crop_path)
    if actual != expected:
        raise ValueError(
            f"OCR audit crop is missing or hash-mismatched for box {result.get('box_id')!r}"
        )
    bounds = raw.get("bounds")
    return {
        "path": relative,
        "sha256": actual,
        "bounds": list(bounds) if isinstance(bounds, (list, tuple)) else None,
    }


def build_review_queue(
    *,
    processing_manifest_path: Path,
    source_index_path: Path,
    ocr_report_path: Path,
    out_dir: Path,
    model_paths: Mapping[str, str | Path] | None = None,
    repo_sha: str = "",
    evidence_ref: str = "",
    min_confirmed_ocr_targets: int = 200,
) -> dict[str, Any]:
    """Write a source-hashed E0 queue and return its fail-closed readiness."""
    if min_confirmed_ocr_targets < 1:
        raise ValueError("min_confirmed_ocr_targets must be >= 1")
    processing = _load_object(processing_manifest_path, "processing manifest")
    source_index = _load_object(source_index_path, "source index")
    ocr_report = _load_object(ocr_report_path, "OCR report")
    pages = processing.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("processing manifest must contain a non-empty pages array")
    chapter_id = str(processing.get("chapter_id") or "").strip()
    source_url = str(processing.get("source_url") or "").strip()
    if not chapter_id:
        raise ValueError("processing manifest has no chapter_id")
    if source_index.get("chapter_id") and str(source_index["chapter_id"]) != chapter_id:
        raise ValueError("source index chapter_id differs from processing manifest")
    if source_index.get("source_url") and source_url and str(source_index["source_url"]) != source_url:
        raise ValueError("source index source_url differs from processing manifest")
    source_by_page = _source_hashes(source_index)

    page_metadata: dict[int, dict[str, Any]] = {}
    manifest_cases: list[dict[str, Any]] = []
    box_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            raise ValueError(f"processing manifest pages[{page_index}] must be an object")
        try:
            source_page = int(page.get("source_page"))
            slice_index = int(page.get("slice_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pages[{page_index}] is missing source_page/slice_index") from exc
        source = source_by_page.get(source_page)
        if source is None:
            raise ValueError(f"pages[{page_index}] source_page {source_page} is absent from source index")
        original_path = _resolve_local_path(page.get("original"), processing_manifest_path)
        slice_digest = sha256_file(original_path)
        if slice_digest is None:
            raise ValueError(f"cannot hash manifest original: {original_path}")
        page_metadata[page_index] = {
            "source_page": source_page,
            "slice_index": slice_index,
            "source_sha256": source["sha256"],
            "source_path": source["path"],
            "original": str(page.get("original")),
            "slice_sha256": slice_digest,
            "clean": page.get("clean"),
            "stitch_core": page.get("stitch_core") or {},
            "boxes": [box for box in (page.get("boxes") or []) if isinstance(box, Mapping)],
            "metrics": page.get("processing_metrics") or {},
        }
        manifest_cases.append(
            {
                "case_id": f"{chapter_id}-s{source_page:03d}-slice{slice_index:02d}-{slice_digest[:12]}",
                "partition": "chapter",
                "source_sha256": source["sha256"],
                "source_page": source_page,
                "slice_index": slice_index,
                "slice_sha256": slice_digest,
            }
        )
        for box in page_metadata[page_index]["boxes"]:
            box_id = str(box.get("id") or "").strip()
            if not box_id:
                continue
            if box_id in box_by_id:
                raise ValueError(f"duplicate box id in processing manifest: {box_id}")
            box_by_id[box_id] = (page_index, dict(box))

    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_manifest = build_manifest(
        benchmark="e0-review-queue-v1",
        dataset=source_url or f"chapter:{chapter_id}",
        repo_sha=repo_sha or _repo_sha(),
        model_paths=model_paths or {},
        parameters={
            "chapter_id": chapter_id,
            "source_index_schema": source_index.get("schema"),
            "queue_policy": "unreviewed candidates cannot enter hard/holdout",
            "min_confirmed_ocr_targets": int(min_confirmed_ocr_targets),
        },
        cases=manifest_cases,
        profile="review-candidate",
    )
    manifest_path = out_dir / "benchmark-manifest.json"
    write_manifest(manifest_path, benchmark_manifest)

    registry = FailureRegistry(out_dir / "failure-registry.jsonl")
    review_rows: list[dict[str, Any]] = []
    for page_index, page in page_metadata.items():
        detector_metrics = page["metrics"].get("detector") or {}
        residue_count = int(detector_metrics.get("post_inpaint_residue") or 0)
        if residue_count <= 0:
            continue
        authority_boxes = [
            _box_snapshot(box)
            for box in page["boxes"]
            if bool(box.get("safe_to_inpaint"))
        ]
        case = build_failure_case(
            source_sha256=page["source_sha256"],
            source_page=page["source_page"],
            slice_index=page["slice_index"],
            stage="post_inpaint_verifier",
            taxonomy="residue",
            geometry={
                "page_index": page_index,
                "stitch_core": page["stitch_core"],
                "authority_boxes": authority_boxes,
            },
            partition="unassigned",
            status="unreviewed",
            candidate="e0-residue-verifier-v1",
            baseline={"source_path": page["source_path"], "source_url": source_url},
            observed={"post_inpaint_residue": residue_count},
            evidence={
                "evidence_ref": evidence_ref,
                "processing_manifest": processing_manifest_path.as_posix(),
                "original": page["original"],
                "slice_sha256": page["slice_sha256"],
                "clean": page["clean"],
            },
            metrics={"detector": detector_metrics, "timing": page["metrics"].get("timing_ms") or {}},
            notes="Automated residue candidate; require visual reviewer confirmation before promotion.",
        )
        registry.append(case)
        review_rows.append(
            {
                "case_id": case["case_id"],
                "review_kind": "residue",
                "required_decision": "confirm_residue|dismiss|needs_more_evidence",
                "source_sha256": page["source_sha256"],
                "source_page": page["source_page"],
                "slice_index": page["slice_index"],
                "page_index": page_index,
            }
        )

    results = ocr_report.get("results")
    if not isinstance(results, list):
        raise ValueError("OCR report must contain a results array")
    ocr_samples: list[dict[str, Any]] = []
    reviewable_crop_count = 0
    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise ValueError(f"OCR report results[{result_index}] must be an object")
        box_id = str(result.get("box_id") or "").strip()
        mapping = box_by_id.get(box_id)
        if mapping is None:
            raise ValueError(f"OCR result {result_index} has no mapped box_id: {box_id!r}")
        page_index, box = mapping
        page = page_metadata[page_index]
        crop = _audit_crop(result, ocr_report_path)
        if crop is not None:
            reviewable_crop_count += 1
        sample_id = stable_case_id(
            source_sha256=page["source_sha256"],
            source_page=page["source_page"],
            slice_index=page["slice_index"],
            stage="ocr_ground_truth_sample",
            taxonomy="other",
            geometry={"page_index": page_index, "box": _box_snapshot(box)},
            candidate="e0-ocr-sample-v1",
        )
        ocr_samples.append(
            {
                "sample_id": sample_id,
                "source_sha256": page["source_sha256"],
                "source_page": page["source_page"],
                "slice_index": page["slice_index"],
                "page_index": page_index,
                "box_id": box_id,
                "box": _box_snapshot(box),
                "image": (
                    f"../{crop['path']}" if crop is not None else None
                ),
                "image_sha256": crop["sha256"] if crop is not None else None,
                "crop_bounds": crop["bounds"] if crop is not None else None,
                "lang": str(result.get("lang") or "en"),
                "target_mode": str(
                    box.get("ocr_target_mode") or box.get("target_mode") or "all"
                ),
                "prediction": {
                    key: result.get(key)
                    for key in (
                        "text", "quality", "quality_reason", "confidence", "model",
                        "retry_applied", "elapsed_ms",
                    )
                    if key in result
                },
                "priority": (
                    "high"
                    if str(result.get("quality") or "").lower() in {"review", "reject"}
                    else "normal"
                ),
                "required_decision": "transcribe_exact|dismiss|needs_more_evidence",
            }
        )
        if str(result.get("quality") or "").lower() not in {"review", "reject"}:
            continue
        taxonomy = _ocr_taxonomy(result)
        case = build_failure_case(
            source_sha256=page["source_sha256"],
            source_page=page["source_page"],
            slice_index=page["slice_index"],
            stage="ocr_audit",
            taxonomy=taxonomy,
            geometry={"page_index": page_index, "box": _box_snapshot(box)},
            partition="unassigned",
            status="unreviewed",
            candidate="e0-ocr-audit-v1",
            baseline={"source_path": page["source_path"], "source_url": source_url},
            observed={
                key: result.get(key)
                for key in (
                    "box_id", "text", "quality", "quality_reason", "confidence",
                    "model", "retry_applied", "elapsed_ms",
                )
                if key in result
            },
            evidence={
                "evidence_ref": evidence_ref,
                "ocr_report": ocr_report_path.as_posix(),
                "original": page["original"],
                "slice_sha256": page["slice_sha256"],
            },
            notes="Automated OCR review/reject signal; reviewer must enter exact transcript or dismiss.",
        )
        registry.append(case)
        review_rows.append(
            {
                "case_id": case["case_id"],
                "review_kind": "ocr",
                "required_decision": "transcribe_exact|dismiss|needs_more_evidence",
                "source_sha256": page["source_sha256"],
                "source_page": page["source_page"],
                "slice_index": page["slice_index"],
                "page_index": page_index,
                "box_id": box_id,
                "quality_reason": result.get("quality_reason"),
            }
        )

    ocr_samples_path = out_dir / "ocr-review-samples.json"
    ocr_samples_payload = {
        "schema": "manga-translator.e0-ocr-review-samples.v1",
        "chapter_id": chapter_id,
        "source_url": source_url,
        "rows": ocr_samples,
    }
    ocr_samples_path.write_text(
        json.dumps(ocr_samples_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = registry.summary()
    readiness = {
        "status": "review_required",
        "promotion_eligible": False,
        "blockers": [
            "All generated cases are unreviewed automated candidates.",
            f"Need {int(min_confirmed_ocr_targets)} confirmed OCR transcripts before OCR comparison gates.",
            "Residue candidates need visual confirmation and geometry before hard/holdout assignment.",
        ],
        "queue_cases": len(review_rows),
        "ocr_review_samples": len(ocr_samples),
        "ocr_samples_with_hash_verified_crops": reviewable_crop_count,
        "confirmed_cases": 0,
        "registry": summary,
    }
    queue = {
        "schema": REVIEW_QUEUE_SCHEMA,
        "chapter_id": chapter_id,
        "source_url": source_url,
        "benchmark_manifest": manifest_path.name,
        "failure_registry": registry.path.name,
        "ocr_review_samples": ocr_samples_path.name,
        "rows": review_rows,
        "readiness": readiness,
    }
    queue_path = out_dir / "review-queue.json"
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readiness_path = out_dir / "readiness.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "benchmark": "e0-review-queue-v1",
        "manifest": benchmark_manifest,
        "manifest_path": manifest_path,
        "failure_registry": summary,
        "ocr_review_samples": ocr_samples_payload,
        "review_queue": queue,
        "readiness": readiness,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processing-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--ocr-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--repo-sha", default="")
    parser.add_argument("--evidence-ref", default="")
    parser.add_argument("--min-confirmed-ocr-targets", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_review_queue(
            processing_manifest_path=args.processing_manifest,
            source_index_path=args.source_index,
            ocr_report_path=args.ocr_report,
            out_dir=args.out_dir,
            model_paths=_parse_models(args.model),
            repo_sha=args.repo_sha,
            evidence_ref=args.evidence_ref,
            min_confirmed_ocr_targets=args.min_confirmed_ocr_targets,
        )
    except ValueError as exc:
        print("E0_REVIEW_QUEUE=" + json.dumps({"status": "blocked", "blockers": [str(exc)]}), flush=True)
        return 2
    print(
        "E0_REVIEW_QUEUE="
        + json.dumps(
            {
                "status": report["readiness"]["status"],
                "manifest_sha256": report["manifest"]["manifest_sha256"],
                "failure_registry": report["failure_registry"],
                "queue_cases": report["readiness"]["queue_cases"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
