from __future__ import annotations

import json

import pytest

from app.benchmarking.manifest import build_manifest, write_manifest
from scripts.summarize_e0_variance import summarize_e0_variance


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(path, *, source_sha: str = "a" * 64) -> None:
    manifest = build_manifest(
        benchmark="e0-review-queue-v1",
        dataset="https://example.test/chapter",
        repo_sha="same-clean-code",
        model_paths={"bubble": path.parent / "missing.onnx"},
        parameters={"chapter_id": "chapter-a"},
        cases=[
            {
                "case_id": "chapter-a-s000-slice00",
                "partition": "chapter",
                "source_page": 0,
                "slice_index": 0,
                "source_sha256": source_sha,
                "slice_sha256": "b" * 64,
            }
        ],
        profile="review-candidate",
        runner={"python": "3.12"},
    )
    # A real E0 run only compares model files with actual digests.  Keep the
    # fixture equally strict without bringing model bytes into this test.
    manifest["models"]["bubble"]["sha256"] = "c" * 64
    from app.benchmarking.manifest import sha256_json

    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_manifest(path, manifest)


def _report(path, *, wall_s: float, residue_records: int = 30) -> None:
    _write_json(
        path,
        {
            "run_id": path.stem,
            "chapter_id": "chapter-a",
            "source_url": "https://example.test/chapter",
            "source_images": 1,
            "slices": 1,
            "clean": {"wall_s": wall_s, "slices_per_min": 60.0 / wall_s, "safe_boxes": 1, "review_boxes": 2},
            "ocr": {"planned": 1, "completed": 1, "errors": []},
            "inpaint_accuracy": {
                "synthetic": {"outside_changed_pixels_exact": 0},
                "real_chapter": {
                    "outside_changed_pixels_exact": 0,
                    "residue_records": residue_records,
                    "pages_with_residue": 1,
                },
            },
        },
    )


def test_e0_variance_requires_exact_paired_source_model_evidence(tmp_path):
    reports = []
    manifests = []
    for index, wall_s in enumerate((100.0, 101.0, 99.0, 102.0)):
        report = tmp_path / f"report-{index}.json"
        manifest = tmp_path / f"manifest-{index}.json"
        _report(report, wall_s=wall_s)
        _manifest(manifest)
        reports.append(report)
        manifests.append(manifest)
    reference = tmp_path / "reference.json"
    reference_manifest = tmp_path / "reference-manifest.json"
    _report(reference, wall_s=100.0)
    _manifest(reference_manifest)

    result = summarize_e0_variance(
        report_paths=reports,
        manifest_paths=manifests,
        reference_report_path=reference,
        reference_manifest_path=reference_manifest,
    )

    assert result["status"] == "review_required"
    assert result["promotion_eligible"] is False
    assert result["gates"] == {
        "paired_source_model_config_exact": True,
        "enough_cold_warm_runs": True,
        "outside_authority_zero": True,
        "ocr_completed_without_errors": True,
        "residue_not_above_limit": True,
        "runtime_reference_provided": True,
        "runtime_reference_source_model_config_exact": True,
        "runtime_not_regressed": True,
    }
    assert result["variance"]["clean_wall_s"]["median"] == 100.5
    assert result["variance"]["clean_wall_s"]["p95"] == pytest.approx(101.85)


def test_e0_variance_blocks_residue_and_mismatched_source(tmp_path):
    reports = []
    manifests = []
    for index, source_sha in enumerate(("a" * 64, "d" * 64)):
        report = tmp_path / f"report-{index}.json"
        manifest = tmp_path / f"manifest-{index}.json"
        _report(report, wall_s=100.0 + index, residue_records=31)
        _manifest(manifest, source_sha=source_sha)
        reports.append(report)
        manifests.append(manifest)

    result = summarize_e0_variance(
        report_paths=reports,
        manifest_paths=manifests,
        min_runs=2,
    )

    assert result["status"] == "blocked"
    assert result["gates"]["paired_source_model_config_exact"] is False
    assert result["gates"]["residue_not_above_limit"] is False
    assert result["gates"]["runtime_reference_provided"] is False
    assert result["gates"]["runtime_reference_source_model_config_exact"] is False
