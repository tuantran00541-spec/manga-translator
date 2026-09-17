from __future__ import annotations

import hashlib
import json

from app.benchmarking.failure_registry import FailureRegistry, build_failure_case
from scripts.apply_e0_review_decisions import (
    apply_review_decisions,
    build_review_template,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_apply_review_decisions_creates_ground_truth_without_promoting(tmp_path):
    source_hash = "a" * 64
    registry_path = tmp_path / "failure-registry.jsonl"
    registry = FailureRegistry(registry_path)
    residue = build_failure_case(
        source_sha256=source_hash,
        source_page=0,
        slice_index=0,
        stage="post_inpaint_verifier",
        taxonomy="residue",
        status="unreviewed",
    )
    ocr = build_failure_case(
        source_sha256=source_hash,
        source_page=0,
        slice_index=0,
        stage="ocr_audit",
        taxonomy="ocr_partial",
        status="unreviewed",
    )
    registry.append(residue)
    registry.append(ocr)

    queue_path = tmp_path / "review-queue.json"
    _write_json(
        queue_path,
        {
            "rows": [
                {"case_id": residue["case_id"], "review_kind": "residue", "source_sha256": source_hash},
                {"case_id": ocr["case_id"], "review_kind": "ocr", "source_sha256": source_hash},
            ]
        },
    )
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"review-crop")
    samples_path = tmp_path / "ocr-review-samples.json"
    _write_json(
        samples_path,
        {
            "rows": [
                {
                    "sample_id": "ocr-sample-1",
                    "source_sha256": source_hash,
                    "source_page": 0,
                    "slice_index": 0,
                    "box_id": "box-1",
                    "image": "crop.png",
                    "image_sha256": _sha256(b"review-crop"),
                    "lang": "en",
                    "target_mode": "all",
                }
            ]
        },
    )
    decisions = build_review_template(
        review_queue_path=queue_path,
        failure_registry_path=registry_path,
        ocr_samples_path=samples_path,
    )
    decisions["reviewer"] = "reviewer@example.test"
    decisions["reviewed_at"] = "2026-09-17T16:00:00Z"
    decisions["source_partitions"][source_hash] = "hard"
    decisions["failure_decisions"][0]["decision"] = "confirm_residue"
    decisions["failure_decisions"][1]["decision"] = "transcribe_exact"
    decisions["ocr_decisions"][0]["decision"] = "transcribe_exact"
    decisions["ocr_decisions"][0]["text"] = "Exact transcript"
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, decisions)

    out_dir = tmp_path / "reviewed"
    report = apply_review_decisions(
        review_queue_path=queue_path,
        failure_registry_path=registry_path,
        ocr_samples_path=samples_path,
        decisions_path=decisions_path,
        out_dir=out_dir,
        min_confirmed_ocr_targets=2,
    )

    assert report["readiness"]["promotion_eligible"] is False
    assert report["readiness"]["confirmed_failure_cases"] == 2
    assert report["readiness"]["confirmed_ocr_targets"] == 1
    reviewed_rows = [json.loads(line) for line in (out_dir / "reviewed-failure-registry.jsonl").read_text().splitlines()]
    assert {row["status"] for row in reviewed_rows} == {"confirmed"}
    assert {row["partition"] for row in reviewed_rows} == {"hard"}
    ground_truth = json.loads((out_dir / "ocr-ground-truth.json").read_text())["rows"]
    assert ground_truth[0]["text"] == "Exact transcript"
    assert (out_dir / ground_truth[0]["image"]).resolve() == crop.resolve()


def test_apply_review_decisions_rejects_stale_input_hashes(tmp_path):
    queue = tmp_path / "queue.json"
    registry = tmp_path / "registry.jsonl"
    samples = tmp_path / "samples.json"
    _write_json(queue, {"rows": []})
    registry.write_text("", encoding="utf-8")
    _write_json(samples, {"rows": []})
    decisions = build_review_template(
        review_queue_path=queue,
        failure_registry_path=registry,
        ocr_samples_path=samples,
    )
    decisions["reviewer"] = "reviewer@example.test"
    decisions["reviewed_at"] = "2026-09-17T16:00:00Z"
    decisions["inputs"]["review_queue_sha256"] = "0" * 64
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, decisions)

    try:
        apply_review_decisions(
            review_queue_path=queue,
            failure_registry_path=registry,
            ocr_samples_path=samples,
            decisions_path=decisions_path,
            out_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "input hash mismatch" in str(exc)
    else:
        raise AssertionError("stale decision file was accepted")
