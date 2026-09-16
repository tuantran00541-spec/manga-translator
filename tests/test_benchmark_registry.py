from __future__ import annotations

import json

import pytest

from app.benchmarking.failure_registry import (
    FailureRegistry,
    build_failure_case,
    stable_case_id,
)
from app.benchmarking.manifest import (
    build_manifest,
    validate_manifest,
    write_manifest,
)


def test_manifest_is_reproducible_and_validates(tmp_path):
    manifest = build_manifest(
        benchmark="smoke",
        dataset="fixed-16",
        repo_sha="abc123",
        model_paths={"bubble": tmp_path / "missing.onnx"},
        parameters={"threads": 2},
        cases=[{"case_id": "case-1", "partition": "smoke"}],
        runner={"python": "3.12"},
    )
    assert validate_manifest(manifest) == []
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_sha256"] == manifest["manifest_sha256"]


def test_manifest_rejects_duplicate_or_unknown_partition():
    manifest = build_manifest(
        benchmark="smoke",
        dataset="fixed-16",
        repo_sha="abc123",
        cases=[
            {"case_id": "same", "partition": "smoke"},
            {"case_id": "same", "partition": "bad"},
        ],
    )
    errors = validate_manifest(manifest)
    assert any("duplicate case_id" in error for error in errors)
    assert any("partition" in error for error in errors)


def test_failure_case_id_is_stable_and_changes_with_candidate():
    common = {
        "source_sha256": "a" * 64,
        "source_page": 3,
        "slice_index": 1,
        "stage": "detector",
        "taxonomy": "detector_fn",
        "geometry": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
    }
    assert stable_case_id(**common) == stable_case_id(**common)
    assert stable_case_id(**common, candidate="student") != stable_case_id(**common, candidate="router")


def test_failure_registry_deduplicates_without_rewriting_evidence(tmp_path):
    case = build_failure_case(
        source_sha256="b" * 64,
        source_page=1,
        slice_index=0,
        stage="inpaint",
        taxonomy="residue",
        geometry={"x1": 1, "y1": 2, "x2": 8, "y2": 9},
        evidence={"diff": "artifacts/case-1.png"},
    )
    registry = FailureRegistry(tmp_path / "failures.jsonl")
    assert registry.append(case) is True
    assert registry.append(case) is False
    assert len(registry.read()) == 1
    assert registry.summary()["by_taxonomy"] == {"residue": 1}


def test_failure_case_rejects_unknown_taxonomy():
    with pytest.raises(ValueError):
        build_failure_case(
            source_sha256="c" * 64,
            stage="detector",
            taxonomy="made_up",
        )
