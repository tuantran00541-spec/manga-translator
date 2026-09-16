"""Leakage-safe case preparation for E7/E8 detector distillation.

The module does not train a model.  It turns immutable failure evidence into a
manifest that a later trainer/evaluator can consume while enforcing the most
important data rule: crops from one source page/chapter never land in
different partitions during the same experiment.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


DISTILLATION_PARTITIONS = frozenset({"train", "calibration", "hard", "holdout"})
_EXPLICIT_PARTITIONS = DISTILLATION_PARTITIONS


def source_group(row: Mapping[str, Any]) -> str:
    """Return the immutable identity used to prevent source leakage."""
    source_sha = str(row.get("source_sha256") or "").strip()
    if source_sha:
        return source_sha
    chapter = str(row.get("chapter_id") or "").strip()
    if chapter:
        return f"chapter:{chapter}"
    return ""


def _deterministic_partition(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "calibration"
    if bucket < 95:
        return "hard"
    return "holdout"


def assign_source_partitions(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Assign one stable partition to every source group.

    Explicit failure-registry partitions win. Conflicting explicit labels are
    rejected because silently splitting a source would invalidate the holdout.
    """
    explicit: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = source_group(row)
        if not group:
            continue
        partition = str(row.get("partition") or "unassigned").strip().lower()
        if partition in _EXPLICIT_PARTITIONS:
            explicit[group].add(partition)
    assignments: dict[str, str] = {}
    groups = {source_group(row) for row in rows if source_group(row)}
    for group in sorted(groups):
        labels = explicit.get(group, set())
        if len(labels) > 1:
            raise ValueError(
                f"source group {group!r} has conflicting partitions: {sorted(labels)}"
            )
        assignments[group] = next(iter(labels)) if labels else _deterministic_partition(group)
    return assignments


def build_distillation_cases(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows]
    assignments = assign_source_partitions(source_rows)
    cases: list[dict[str, Any]] = []
    for row in source_rows:
        group = source_group(row)
        if not group:
            raise ValueError(
                f"failure case {row.get('case_id')!r} has no source_sha256/chapter_id"
            )
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("failure case is missing case_id")
        cases.append(
            {
                "case_id": case_id,
                "partition": assignments[group],
                "source_group": group,
                "source_sha256": row.get("source_sha256"),
                "chapter_id": row.get("chapter_id"),
                "source_page": row.get("source_page"),
                "slice_index": row.get("slice_index"),
                "stage": row.get("stage"),
                "taxonomy": row.get("taxonomy"),
                "geometry": row.get("geometry") or {},
                "label_status": row.get("status") or "unreviewed",
                "teacher": {
                    "baseline": row.get("baseline") or {},
                    "observed": row.get("observed") or {},
                },
                "evidence": row.get("evidence") or {},
                "metrics": row.get("metrics") or {},
            }
        )
    return cases


def validate_distillation_cases(cases: Iterable[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id") or "")
        if not case_id:
            errors.append(f"cases[{index}] missing case_id")
        elif case_id in seen:
            errors.append(f"duplicate case_id {case_id}")
        seen.add(case_id)
        group = str(case.get("source_group") or "")
        partition = str(case.get("partition") or "")
        if not group:
            errors.append(f"cases[{index}] missing source_group")
        if partition not in DISTILLATION_PARTITIONS:
            errors.append(f"cases[{index}] has invalid partition {partition!r}")
        if group and partition:
            previous = groups.setdefault(group, partition)
            if previous != partition:
                errors.append(
                    f"source group {group!r} crosses {previous!r}/{partition!r}"
                )
    return errors


def summarize_distillation_cases(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cases = list(cases)
    return {
        "case_count": len(cases),
        "source_group_count": len({str(case.get("source_group")) for case in cases}),
        "by_partition": {
            partition: sum(case.get("partition") == partition for case in cases)
            for partition in sorted(DISTILLATION_PARTITIONS)
        },
        "by_taxonomy": {
            taxonomy: sum(case.get("taxonomy") == taxonomy for case in cases)
            for taxonomy in sorted({str(case.get("taxonomy") or "other") for case in cases})
        },
    }


__all__ = [
    "DISTILLATION_PARTITIONS",
    "assign_source_partitions",
    "build_distillation_cases",
    "source_group",
    "summarize_distillation_cases",
    "validate_distillation_cases",
]
