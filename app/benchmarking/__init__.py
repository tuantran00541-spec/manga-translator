"""Small, dependency-light helpers for reproducible performance experiments.

Source-coordinate projection imports detector types and therefore OpenCV.  Do
not load it when callers only need a manifest or failure-registry contract.
"""

from .failure_registry import FailureRegistry, build_failure_case, stable_case_id
from .manifest import (
    BENCHMARK_MANIFEST_SCHEMA,
    build_manifest,
    load_manifest,
    validate_manifest,
)
_LEDGER_EXPORTS = frozenset(
    {
        "SourceCoordinateLedger",
        "SourceTile",
        "clip_box_to_slice",
        "project_boxes_to_slice",
        "source_overlap_pixels",
        "source_tile_key",
    }
)


def __getattr__(name: str):
    """Lazily expose OpenCV-dependent source-coordinate helpers."""
    if name not in _LEDGER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import source_coordinate_ledger

    value = getattr(source_coordinate_ledger, name)
    globals()[name] = value
    return value

__all__ = [
    "BENCHMARK_MANIFEST_SCHEMA",
    "FailureRegistry",
    "build_failure_case",
    "build_manifest",
    "load_manifest",
    "SourceCoordinateLedger",
    "SourceTile",
    "clip_box_to_slice",
    "project_boxes_to_slice",
    "source_overlap_pixels",
    "source_tile_key",
    "stable_case_id",
    "validate_manifest",
]
