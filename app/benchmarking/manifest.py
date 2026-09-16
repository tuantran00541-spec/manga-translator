"""Versioned benchmark manifests and reproducibility fingerprints.

The production pipeline deliberately does not depend on a particular benchmark
runner.  This module gives every runner the same small manifest contract so a
speed number can be compared with the exact source/model/configuration that
produced it.  It contains no model or image imports and is safe to use in the
dependency-light unit-test suite.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


BENCHMARK_MANIFEST_SCHEMA = "manga-translator.benchmark-manifest.v1"
PARTITIONS = frozenset({"smoke", "hard", "chapter", "holdout"})


def _jsonable(value: Any) -> Any:
    """Convert common scalar/path values without serialising raw image data."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Numpy scalars are intentionally handled without importing numpy.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or ``None`` when a model/source is unavailable."""
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_fingerprint(*, include_hostname: bool = False) -> dict[str, Any]:
    """Return stable-enough runner metadata without exposing environment vars."""
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": __import__("os").cpu_count(),
    }
    if include_hostname:
        result["hostname"] = socket.gethostname()
    return result


def build_manifest(
    *,
    benchmark: str,
    dataset: str,
    repo_sha: str,
    model_paths: Mapping[str, str | Path] | None = None,
    parameters: Mapping[str, Any] | None = None,
    cases: Iterable[Mapping[str, Any]] = (),
    profile: str = "control",
    runner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest that is explicit about source/model/config identity."""
    models = {
        str(name): {
            "path": Path(path).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in (model_paths or {}).items()
    }
    normalized_cases = []
    for case in cases:
        row = _jsonable(case)
        if not isinstance(row, dict):
            raise TypeError("benchmark cases must be mapping objects")
        normalized_cases.append(row)
    manifest = {
        "schema": BENCHMARK_MANIFEST_SCHEMA,
        "benchmark": str(benchmark),
        "dataset": str(dataset),
        "profile": str(profile),
        "repo_sha": str(repo_sha),
        "models": models,
        "parameters": _jsonable(parameters or {}),
        "runner": _jsonable(runner or runtime_fingerprint()),
        "cases": normalized_cases,
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return human-readable contract errors; an empty list means valid."""
    errors: list[str] = []
    if manifest.get("schema") != BENCHMARK_MANIFEST_SCHEMA:
        errors.append("schema must be manga-translator.benchmark-manifest.v1")
    for key in ("benchmark", "dataset", "repo_sha", "profile"):
        if not str(manifest.get(key) or "").strip():
            errors.append(f"missing {key}")
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        errors.append("models must be an object")
    else:
        for name, model in models.items():
            if not isinstance(model, Mapping):
                errors.append(f"models.{name} must be an object")
                continue
            if not str(model.get("path") or "").strip():
                errors.append(f"models.{name}.path is required")
            digest = model.get("sha256")
            if digest is not None and (
                not isinstance(digest, str) or len(digest) != 64
            ):
                errors.append(f"models.{name}.sha256 must be a SHA-256 or null")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        errors.append("cases must be an array")
    else:
        seen: set[str] = set()
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                errors.append(f"cases[{index}] must be an object")
                continue
            partition = str(case.get("partition") or "")
            if partition not in PARTITIONS:
                errors.append(
                    f"cases[{index}].partition must be one of {sorted(PARTITIONS)}"
                )
            case_id = str(case.get("case_id") or "")
            if not case_id:
                errors.append(f"cases[{index}].case_id is required")
            elif case_id in seen:
                errors.append(f"duplicate case_id {case_id}")
            seen.add(case_id)
    expected = manifest.get("manifest_sha256")
    if expected:
        actual = sha256_json(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        if expected != actual:
            errors.append("manifest_sha256 does not match manifest content")
    return errors


def load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark manifest must be a JSON object")
    errors = validate_manifest(value)
    if errors:
        raise ValueError("invalid benchmark manifest: " + "; ".join(errors))
    return value


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid benchmark manifest: " + "; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
