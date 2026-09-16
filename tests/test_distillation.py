from __future__ import annotations

import pytest
from types import SimpleNamespace

from app.benchmarking.distillation import (
    assign_source_partitions,
    build_distillation_cases,
    validate_distillation_cases,
)
from app.benchmarking.failure_registry import FailureRegistry, build_failure_case
from app.benchmarking.manifest import validate_manifest
from scripts.build_distillation_manifest import run


def _case(case_id: str, source: str, partition: str = "unassigned"):
    return {
        "case_id": case_id,
        "source_sha256": source,
        "partition": partition,
        "stage": "detector",
        "taxonomy": "detector_fn",
        "status": "confirmed",
    }


def test_distillation_keeps_one_source_in_one_partition():
    rows = [_case("a", "source-a"), _case("b", "source-a")]
    assignments = assign_source_partitions(rows)
    cases = build_distillation_cases(rows)

    assert assignments["source-a"] == cases[0]["partition"]
    assert cases[0]["partition"] == cases[1]["partition"]
    assert validate_distillation_cases(cases) == []


def test_conflicting_explicit_source_partitions_are_blocked():
    with pytest.raises(ValueError, match="conflicting partitions"):
        assign_source_partitions([
            _case("a", "source-a", "hard"),
            _case("b", "source-a", "holdout"),
        ])


def test_distillation_validator_rejects_partition_leakage():
    errors = validate_distillation_cases([
        {"case_id": "a", "source_group": "source-a", "partition": "train"},
        {"case_id": "b", "source_group": "source-a", "partition": "holdout"},
    ])
    assert any("crosses" in error for error in errors)


def test_distillation_manifest_success_recomputes_extended_hash(tmp_path):
    registry_path = tmp_path / "failures.jsonl"
    registry = FailureRegistry(registry_path)
    registry.append(
        build_failure_case(
            source_sha256="a" * 64,
            source_page=1,
            stage="detector",
            taxonomy="detector_fn",
            partition="holdout",
        )
    )
    report = run(
        SimpleNamespace(
            failure_registry=registry_path,
            dataset="fixture",
            repo_sha="abc",
            model=[],
        )
    )
    assert report["status"] == "pass"
    assert validate_manifest(report) == []
