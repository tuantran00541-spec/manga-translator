"""Small, dependency-light helpers for reproducible performance experiments."""

from .failure_registry import FailureRegistry, build_failure_case, stable_case_id
from .manifest import (
    BENCHMARK_MANIFEST_SCHEMA,
    build_manifest,
    load_manifest,
    validate_manifest,
)

__all__ = [
    "BENCHMARK_MANIFEST_SCHEMA",
    "FailureRegistry",
    "build_failure_case",
    "build_manifest",
    "load_manifest",
    "stable_case_id",
    "validate_manifest",
]
