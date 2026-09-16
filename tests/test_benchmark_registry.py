from __future__ import annotations

import json

import numpy as np
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
from app.benchmarking.source_coordinate_ledger import (
    SourceCoordinateLedger,
    SourceTile,
    clip_box_to_slice,
    project_boxes_to_slice,
    source_overlap_pixels,
)
from app.detector.bubble_detector import BubbleBox
from scripts.benchmark_source_coordinate_reuse import _mismatch_taxonomies


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


def test_source_coordinate_projection_clips_mask_without_changing_authority():
    mask = np.arange(24, dtype=np.uint8).reshape(4, 6)
    box = BubbleBox(
        2,
        10,
        8,
        14,
        0.9,
        mask,
        source_model="text_segmenter",
        source_role="text_segmenter",
        semantic_type="free_text",
        safe_to_inpaint=True,
        ocr_eligible=True,
    )
    projected = clip_box_to_slice(box, slice_box=(0, 12, 8, 20))
    assert projected is not None
    assert (projected.x1, projected.y1, projected.x2, projected.y2) == (2, 0, 8, 2)
    assert projected.mask.tolist() == mask[2:4, :].tolist()
    assert projected.safe_to_inpaint is True

    page_boxes = project_boxes_to_slice(
        [box], source_y1=12, source_y2=20, source_width=8
    )
    assert len(page_boxes) == 1
    assert page_boxes[0].mask.shape == (2, 6)


def test_source_coordinate_ledger_shadow_and_overlap_metrics():
    ledger = SourceCoordinateLedger(mode="shadow", max_entries=2)
    tile = SourceTile("source", 0, (0, 0, 10, 10), "detector:v1")
    boxes = [BubbleBox(1, 2, 4, 5, 0.8)]
    assert ledger.lookup(tile) is None
    ledger.store(tile, boxes)
    hit = ledger.lookup(tile)
    assert hit is not None and hit[0] is not boxes[0]
    hit[0].x1 = 99  # defensive copy must not mutate the stored result
    assert ledger.snapshot()["shadow_hits"] == 1
    assert source_overlap_pixels([(0, 0, 10, 10), (0, 5, 10, 15)]) == (200, 150)


def test_source_projection_mismatch_is_distilled_into_fn_fp_and_mask_labels():
    baseline = [
        BubbleBox(0, 0, 4, 4, 0.8, np.ones((4, 4), dtype=np.uint8), source_role="text_segmenter")
    ]
    candidate = [
        BubbleBox(0, 0, 4, 4, 0.8, np.zeros((4, 4), dtype=np.uint8), source_role="text_segmenter"),
        BubbleBox(8, 8, 12, 12, 0.8, source_role="text_segmenter"),
    ]
    assert _mismatch_taxonomies(baseline, candidate) == [
        "detector_fp",
        "mask_undercoverage",
    ]
