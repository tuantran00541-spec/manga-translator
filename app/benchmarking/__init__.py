"""Small, dependency-light helpers for reproducible performance experiments."""

from .failure_registry import FailureRegistry, build_failure_case, stable_case_id
from .manifest import (
    BENCHMARK_MANIFEST_SCHEMA,
    build_manifest,
    load_manifest,
    validate_manifest,
)
from .source_coordinate_ledger import (
    SourceCoordinateLedger,
    SourceTile,
    clip_box_to_slice,
    project_boxes_to_slice,
    source_overlap_pixels,
    source_tile_key,
)

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
