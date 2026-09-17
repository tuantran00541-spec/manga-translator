#!/usr/bin/env python3
"""Create reviewed E0 evidence and OCR ground truth from explicit decisions.

Automated E0 candidates remain immutable.  A reviewer fills a template that
pins the queue/registry/sample hashes, then this script emits *new* reviewed
artifacts.  It deliberately never marks a benchmark promotable: cross-chapter
hard/holdout evaluation still has to happen in a later gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.benchmarking.failure_registry import PARTITIONS
from app.benchmarking.manifest import sha256_file


DECISIONS_SCHEMA = "manga-translator.e0-review-decisions.v1"
OCR_GROUND_TRUTH_SCHEMA = "manga-translator.ocr-ground-truth.v1"
REVIEWED_REGISTRY_SCHEMA = "manga-translator.e0-reviewed-registry.v1"

_FAILURE_DECISIONS = {
    "residue": frozenset({"confirm_residue", "dismiss", "needs_more_evidence"}),
    "ocr": frozenset({"transcribe_exact", "dismiss", "needs_more_evidence"}),
}
_OCR_DECISIONS = frozenset({"transcribe_exact", "dismiss", "needs_more_evidence"})
_ASSIGNABLE_PARTITIONS = frozenset({"train", "calibration", "hard", "holdout"})


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {index} is not JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {index} must be an object")
        rows.append(row)
    return rows


def _digest(path: Path, label: str) -> str:
    digest = sha256_file(path)
    if not digest:
        raise ValueError(f"cannot hash {label}: {path}")
    return digest


def _unique_ids(rows: list[Mapping[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = str(row.get(field) or "").strip()
        if not identity:
            raise ValueError(f"{label}[{index}] has no {field}")
        if identity in result:
            raise ValueError(f"{label} repeats {field}: {identity}")
        result[identity] = dict(row)
    return result


def _template_inputs(
    review_queue_path: Path,
    failure_registry_path: Path,
    ocr_samples_path: Path,
) -> dict[str, str]:
    return {
        "review_queue_sha256": _digest(review_queue_path, "review queue"),
        "failure_registry_sha256": _digest(failure_registry_path, "failure registry"),
        "ocr_samples_sha256": _digest(ocr_samples_path, "OCR samples"),
    }


def build_review_template(
    *,
    review_queue_path: Path,
    failure_registry_path: Path,
    ocr_samples_path: Path,
) -> dict[str, Any]:
    """Build an intentionally incomplete template for one human reviewer."""
    queue = _load_object(review_queue_path, "review queue")
    samples = _load_object(ocr_samples_path, "OCR samples")
    queue_rows = queue.get("rows")
    sample_rows = samples.get("rows")
    if not isinstance(queue_rows, list):
        raise ValueError("review queue has no rows array")
    if not isinstance(sample_rows, list):
        raise ValueError("OCR samples have no rows array")
    registry_rows = _load_jsonl(failure_registry_path, "failure registry")
    registry_by_id = _unique_ids(registry_rows, "case_id", "failure registry")
    source_hashes = set()
    failure_decisions = []
    for index, row in enumerate(queue_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"review queue rows[{index}] must be an object")
        case_id = str(row.get("case_id") or "").strip()
        if case_id not in registry_by_id:
            raise ValueError(f"review queue references absent failure case: {case_id!r}")
        kind = str(row.get("review_kind") or "")
        if kind not in _FAILURE_DECISIONS:
            raise ValueError(f"review queue case {case_id!r} has unsupported kind {kind!r}")
        source_hashes.add(str(row.get("source_sha256") or ""))
        failure_decisions.append(
            {
                "case_id": case_id,
                "review_kind": kind,
                "decision": "",
                "notes": "",
            }
        )
    ocr_decisions = []
    for index, row in enumerate(sample_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"OCR samples rows[{index}] must be an object")
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"OCR samples rows[{index}] has no sample_id")
        source_hashes.add(str(row.get("source_sha256") or ""))
        ocr_decisions.append(
            {
                "sample_id": sample_id,
                "decision": "",
                "text": "",
                "notes": "",
            }
        )
    return {
        "schema": DECISIONS_SCHEMA,
        "reviewer": "",
        "reviewed_at": "",
        "inputs": _template_inputs(
            review_queue_path, failure_registry_path, ocr_samples_path
        ),
        "source_partitions": {source: "" for source in sorted(source_hashes) if source},
        "failure_decisions": failure_decisions,
        "ocr_decisions": ocr_decisions,
    }


def write_review_template(
    *,
    review_queue_path: Path,
    failure_registry_path: Path,
    ocr_samples_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    template = build_review_template(
        review_queue_path=review_queue_path,
        failure_registry_path=failure_registry_path,
        ocr_samples_path=ocr_samples_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return template


def _validate_inputs(decisions: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    if decisions.get("schema") != DECISIONS_SCHEMA:
        raise ValueError(f"decisions schema must be {DECISIONS_SCHEMA}")
    inputs = decisions.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("decisions has no inputs object")
    for field, value in expected.items():
        if str(inputs.get(field) or "") != value:
            raise ValueError(f"decisions input hash mismatch: {field}")
    if not str(decisions.get("reviewer") or "").strip():
        raise ValueError("decisions reviewer is required")
    if not str(decisions.get("reviewed_at") or "").strip():
        raise ValueError("decisions reviewed_at is required")


def _source_partitions(decisions: Mapping[str, Any], source_hashes: set[str]) -> dict[str, str]:
    raw = decisions.get("source_partitions")
    if not isinstance(raw, Mapping):
        raise ValueError("decisions source_partitions must be an object")
    result: dict[str, str] = {}
    for source_hash in source_hashes:
        partition = str(raw.get(source_hash) or "").strip().lower()
        if partition not in _ASSIGNABLE_PARTITIONS:
            raise ValueError(
                f"source {source_hash} needs one explicit partition from "
                f"{sorted(_ASSIGNABLE_PARTITIONS)}"
            )
        result[source_hash] = partition
    return result


def _decision_index(
    rows: object, identity: str, label: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"decisions {label} must be an array")
    return _unique_ids(
        [row for row in rows if isinstance(row, Mapping)], identity, f"decisions {label}"
    )


def apply_review_decisions(
    *,
    review_queue_path: Path,
    failure_registry_path: Path,
    ocr_samples_path: Path,
    decisions_path: Path,
    out_dir: Path,
    min_confirmed_ocr_targets: int = 200,
) -> dict[str, Any]:
    """Apply explicit reviewer decisions without mutating E0 candidate inputs."""
    if min_confirmed_ocr_targets < 1:
        raise ValueError("min_confirmed_ocr_targets must be >= 1")
    queue = _load_object(review_queue_path, "review queue")
    samples = _load_object(ocr_samples_path, "OCR samples")
    decisions = _load_object(decisions_path, "decisions")
    _validate_inputs(
        decisions,
        _template_inputs(review_queue_path, failure_registry_path, ocr_samples_path),
    )
    registry_rows = _load_jsonl(failure_registry_path, "failure registry")
    registry_by_id = _unique_ids(registry_rows, "case_id", "failure registry")
    queue_by_id = _unique_ids(list(queue.get("rows") or []), "case_id", "review queue")
    sample_by_id = _unique_ids(list(samples.get("rows") or []), "sample_id", "OCR samples")
    failure_decisions = _decision_index(
        decisions.get("failure_decisions"), "case_id", "failure_decisions"
    )
    ocr_decisions = _decision_index(
        decisions.get("ocr_decisions"), "sample_id", "ocr_decisions"
    )

    for case_id in failure_decisions:
        if case_id not in queue_by_id or case_id not in registry_by_id:
            raise ValueError(f"failure decision references unknown case_id: {case_id}")
    for sample_id in ocr_decisions:
        if sample_id not in sample_by_id:
            raise ValueError(f"OCR decision references unknown sample_id: {sample_id}")

    confirmed_sources: set[str] = set()
    for case_id, decision in failure_decisions.items():
        value = str(decision.get("decision") or "").strip()
        kind = str(queue_by_id[case_id].get("review_kind") or "")
        if value not in _FAILURE_DECISIONS.get(kind, frozenset()):
            raise ValueError(f"invalid {kind} decision for {case_id}: {value!r}")
        if value == "confirm_residue" or value == "transcribe_exact":
            confirmed_sources.add(str(registry_by_id[case_id].get("source_sha256") or ""))
    for sample_id, decision in ocr_decisions.items():
        value = str(decision.get("decision") or "").strip()
        if value not in _OCR_DECISIONS:
            raise ValueError(f"invalid OCR decision for {sample_id}: {value!r}")
        if value == "transcribe_exact":
            sample = sample_by_id[sample_id]
            if not sample.get("image") or not sample.get("image_sha256"):
                raise ValueError(f"OCR sample {sample_id} has no hash-verified crop image")
            image_path = (ocr_samples_path.parent / str(sample["image"])).resolve()
            if sha256_file(image_path) != str(sample["image_sha256"]):
                raise ValueError(f"OCR sample {sample_id} crop is missing or hash-mismatched")
            if not isinstance(decision.get("text"), str):
                raise ValueError(f"OCR decision {sample_id} must include text (empty is valid)")
            confirmed_sources.add(str(sample.get("source_sha256") or ""))
    source_partitions = _source_partitions(decisions, {x for x in confirmed_sources if x})

    reviewed_rows: list[dict[str, Any]] = []
    review_log: list[dict[str, Any]] = []
    reviewer = str(decisions["reviewer"]).strip()
    reviewed_at = str(decisions["reviewed_at"]).strip()
    for row in registry_rows:
        case_id = str(row["case_id"])
        decision = failure_decisions.get(case_id)
        if decision is None:
            reviewed_rows.append(row)
            continue
        action = str(decision["decision"]).strip()
        updated = dict(row)
        if action in {"confirm_residue", "transcribe_exact"}:
            updated["status"] = "confirmed"
            updated["partition"] = source_partitions[str(row.get("source_sha256") or "")]
        elif action == "dismiss":
            updated["status"] = "dismissed"
        else:
            updated["status"] = "disputed"
        updated["review"] = {
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "decision": action,
            "notes": str(decision.get("notes") or ""),
        }
        reviewed_rows.append(updated)
        review_log.append({"kind": "failure", "case_id": case_id, **updated["review"]})

    ground_truth_rows: list[dict[str, Any]] = []
    for sample_id, decision in ocr_decisions.items():
        if str(decision["decision"]).strip() != "transcribe_exact":
            continue
        sample = sample_by_id[sample_id]
        source_hash = str(sample.get("source_sha256") or "")
        source_image = (ocr_samples_path.parent / str(sample["image"])).resolve()
        ground_truth_rows.append(
            {
                "case_id": sample_id,
                "image": os.path.relpath(source_image, out_dir),
                "image_sha256": sample["image_sha256"],
                "lang": sample.get("lang") or "en",
                "text": decision["text"],
                "target_mode": sample.get("target_mode") or "all",
                "partition": source_partitions[source_hash],
                "source_sha256": source_hash,
                "source_page": sample.get("source_page"),
                "slice_index": sample.get("slice_index"),
                "box_id": sample.get("box_id"),
                "review": {
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "decision": "transcribe_exact",
                    "notes": str(decision.get("notes") or ""),
                },
            }
        )
        review_log.append(
            {
                "kind": "ocr",
                "sample_id": sample_id,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "decision": "transcribe_exact",
                "notes": str(decision.get("notes") or ""),
            }
        )

    confirmed_failure_count = sum(row.get("status") == "confirmed" for row in reviewed_rows)
    readiness = {
        "status": "review_incomplete",
        "promotion_eligible": False,
        "confirmed_failure_cases": confirmed_failure_count,
        "confirmed_ocr_targets": len(ground_truth_rows),
        "blockers": [
            f"Need {int(min_confirmed_ocr_targets)} confirmed OCR transcripts; got {len(ground_truth_rows)}.",
            "Need a separate untouched source/chapter for holdout before promotion.",
            "Reviewed evidence still requires hard/holdout benchmark gates; no optimizer is promoted here.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    reviewed_registry_path = out_dir / "reviewed-failure-registry.jsonl"
    reviewed_registry_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reviewed_rows),
        encoding="utf-8",
    )
    ground_truth_path = out_dir / "ocr-ground-truth.json"
    ground_truth_path.write_text(
        json.dumps({"schema": OCR_GROUND_TRUTH_SCHEMA, "rows": ground_truth_rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "review-log.json").write_text(
        json.dumps({"schema": REVIEWED_REGISTRY_SCHEMA, "rows": review_log}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "reviewed_registry": reviewed_registry_path,
        "ocr_ground_truth": ground_truth_path,
        "readiness": readiness,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--failure-registry", type=Path, required=True)
    parser.add_argument("--ocr-samples", type=Path, required=True)
    parser.add_argument("--template-out", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--min-confirmed-ocr-targets", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.template_out:
            template = write_review_template(
                review_queue_path=args.review_queue,
                failure_registry_path=args.failure_registry,
                ocr_samples_path=args.ocr_samples,
                out_path=args.template_out,
            )
            print(
                "E0_REVIEW_TEMPLATE="
                + json.dumps(
                    {
                        "failure_decisions": len(template["failure_decisions"]),
                        "ocr_decisions": len(template["ocr_decisions"]),
                    }
                ),
                flush=True,
            )
            return 0
        if not args.decisions or not args.out_dir:
            raise ValueError("--decisions and --out-dir are required when not writing a template")
        result = apply_review_decisions(
            review_queue_path=args.review_queue,
            failure_registry_path=args.failure_registry,
            ocr_samples_path=args.ocr_samples,
            decisions_path=args.decisions,
            out_dir=args.out_dir,
            min_confirmed_ocr_targets=args.min_confirmed_ocr_targets,
        )
    except ValueError as exc:
        print("E0_REVIEW_APPLY=" + json.dumps({"status": "blocked", "blockers": [str(exc)]}), flush=True)
        return 2
    print("E0_REVIEW_APPLY=" + json.dumps(result["readiness"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
