"""Append-only failure evidence for benchmark and distillation work.

The registry intentionally stores metadata and paths, not image bytes or model
outputs.  Large evidence remains in an artifact directory while this JSONL file
provides a stable, deduplicated index that can be carried between CI runs.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping


FAILURE_REGISTRY_SCHEMA = "manga-translator.failure-registry.v1"
FAILURE_TAXONOMY = frozenset(
    {
        "detector_fn",
        "detector_fp",
        "mask_undercoverage",
        "mask_overreach",
        "seam_duplicate",
        "seam_split",
        "budget_deferred",
        "residue",
        "inpaint_texture_damage",
        "ocr_partial",
        "ocr_edge",
        "ocr_empty",
        "ocr_hallucination",
        "runtime_regression",
        "other",
    }
)
PARTITIONS = frozenset({"unassigned", "train", "calibration", "hard", "holdout"})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _canonical(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def stable_case_id(
    *,
    source_sha256: str,
    source_page: int | None = None,
    slice_index: int | None = None,
    stage: str,
    taxonomy: str,
    geometry: Mapping[str, Any] | None = None,
    candidate: str | None = None,
) -> str:
    """Create a reproducible ID that is independent of artifact filenames."""
    payload = {
        "source_sha256": str(source_sha256),
        "source_page": source_page,
        "slice_index": slice_index,
        "stage": str(stage),
        "taxonomy": str(taxonomy),
        "geometry": _jsonable(geometry or {}),
        "candidate": candidate,
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return f"case-{digest[:20]}"


def build_failure_case(
    *,
    source_sha256: str,
    stage: str,
    taxonomy: str,
    geometry: Mapping[str, Any] | None = None,
    source_page: int | None = None,
    slice_index: int | None = None,
    partition: str = "unassigned",
    status: str = "unreviewed",
    candidate: str | None = None,
    baseline: Mapping[str, Any] | None = None,
    observed: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if taxonomy not in FAILURE_TAXONOMY:
        raise ValueError(f"unknown failure taxonomy: {taxonomy}")
    if partition not in PARTITIONS:
        raise ValueError(f"unknown failure partition: {partition}")
    case_id = stable_case_id(
        source_sha256=source_sha256,
        source_page=source_page,
        slice_index=slice_index,
        stage=stage,
        taxonomy=taxonomy,
        geometry=geometry,
        candidate=candidate,
    )
    return {
        "schema": FAILURE_REGISTRY_SCHEMA,
        "case_id": case_id,
        "source_sha256": str(source_sha256),
        "source_page": source_page,
        "slice_index": slice_index,
        "stage": str(stage),
        "taxonomy": taxonomy,
        "geometry": _jsonable(geometry or {}),
        "partition": partition,
        "status": str(status),
        "candidate": candidate,
        "baseline": _jsonable(baseline or {}),
        "observed": _jsonable(observed or {}),
        "evidence": _jsonable(evidence or {}),
        "metrics": _jsonable(metrics or {}),
        "notes": notes,
    }


@contextmanager
def _locked(handle) -> Iterator[None]:
    """Use an advisory lock when available; remain usable on Windows/CI."""
    try:
        import fcntl  # type: ignore
    except ImportError:
        yield
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FailureRegistry:
    """Append unique cases to a JSONL file without mutating prior evidence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _known_ids(self) -> set[str]:
        if not self.path.is_file():
            return set()
        known: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping) and row.get("case_id"):
                    known.add(str(row["case_id"]))
        return known

    def append(self, case: Mapping[str, Any]) -> bool:
        """Append a case; return False when its stable ID is already present."""
        row = _jsonable(case)
        if not isinstance(row, dict) or not row.get("case_id"):
            raise ValueError("failure case must be an object with case_id")
        if row.get("schema") != FAILURE_REGISTRY_SCHEMA:
            raise ValueError("failure case schema mismatch")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            with _locked(handle):
                known = self._known_ids()
                case_id = str(row["case_id"])
                if case_id in known:
                    return False
                handle.seek(0, os.SEEK_END)
                handle.write(_canonical(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def summary(self) -> dict[str, Any]:
        rows = self.read()
        by_taxonomy: dict[str, int] = {}
        by_partition: dict[str, int] = {}
        for row in rows:
            taxonomy = str(row.get("taxonomy") or "other")
            partition = str(row.get("partition") or "unassigned")
            by_taxonomy[taxonomy] = by_taxonomy.get(taxonomy, 0) + 1
            by_partition[partition] = by_partition.get(partition, 0) + 1
        return {
            "schema": FAILURE_REGISTRY_SCHEMA,
            "total": len(rows),
            "by_taxonomy": dict(sorted(by_taxonomy.items())),
            "by_partition": dict(sorted(by_partition.items())),
        }
