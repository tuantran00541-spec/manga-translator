from __future__ import annotations

import hashlib
import json

import pytest

from app.benchmarking.manifest import load_manifest
from scripts.build_e0_review_queue import build_review_queue


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_e0_queue_preserves_source_provenance_and_remains_unreviewed(tmp_path):
    source = tmp_path / "source.webp"
    source.write_bytes(b"source-image")
    slice_image = tmp_path / "slice.png"
    slice_image.write_bytes(b"slice-image")
    source_index = tmp_path / "source-index.json"
    _write_json(
        source_index,
        {
            "schema": "manga-translator.source-index.v1",
            "chapter_id": "chapter-a",
            "source_url": "https://example.test/chapter-a",
            "sources": [
                {"source_page": 0, "path": str(source), "sha256": _sha256(b"source-image")}
            ],
        },
    )
    processing_manifest = tmp_path / "manifest.json"
    _write_json(
        processing_manifest,
        {
            "chapter_id": "chapter-a",
            "source_url": "https://example.test/chapter-a",
            "pages": [
                {
                    "original": str(slice_image),
                    "clean": "clean.png",
                    "source_page": 0,
                    "slice_index": 0,
                    "stitch_core": {"core_source_y1": 0, "core_source_y2": 20},
                    "boxes": [
                        {
                            "id": "box-1",
                            "x1": 1,
                            "y1": 2,
                            "x2": 10,
                            "y2": 12,
                            "safe_to_inpaint": True,
                            "semantic_type": "speech_bubble",
                        }
                    ],
                    "processing_metrics": {"detector": {"post_inpaint_residue": 2}},
                }
            ],
        },
    )
    ocr_report = tmp_path / "ocr.json"
    _write_json(
        ocr_report,
        {
            "results": [
                {
                    "box_id": "box-1",
                    "quality": "review",
                    "quality_reason": "crop-edge-text",
                    "text": "partial",
                    "confidence": 0.61,
                }
            ]
        },
    )

    report = build_review_queue(
        processing_manifest_path=processing_manifest,
        source_index_path=source_index,
        ocr_report_path=ocr_report,
        out_dir=tmp_path / "e0",
        repo_sha="test-sha",
        evidence_ref="test-evidence",
    )

    assert report["readiness"]["status"] == "review_required"
    assert report["readiness"]["promotion_eligible"] is False
    assert report["failure_registry"]["by_taxonomy"] == {"ocr_edge": 1, "residue": 1}
    rows = [json.loads(line) for line in (tmp_path / "e0" / "failure-registry.jsonl").read_text().splitlines()]
    assert {row["status"] for row in rows} == {"unreviewed"}
    assert {row["partition"] for row in rows} == {"unassigned"}
    assert {row["source_sha256"] for row in rows} == {_sha256(b"source-image")}
    assert load_manifest(tmp_path / "e0" / "benchmark-manifest.json")["cases"][0]["partition"] == "chapter"


def test_e0_queue_rejects_page_without_indexed_source_hash(tmp_path):
    image = tmp_path / "slice.png"
    image.write_bytes(b"slice")
    processing = tmp_path / "manifest.json"
    _write_json(
        processing,
        {"chapter_id": "c", "pages": [{"original": str(image), "source_page": 1, "slice_index": 0}]},
    )
    index = tmp_path / "source-index.json"
    _write_json(index, {"chapter_id": "c", "sources": [{"source_page": 0, "sha256": "a" * 64}]})
    ocr = tmp_path / "ocr.json"
    _write_json(ocr, {"results": []})

    with pytest.raises(ValueError, match="absent from source index"):
        build_review_queue(
            processing_manifest_path=processing,
            source_index_path=index,
            ocr_report_path=ocr,
            out_dir=tmp_path / "e0",
        )
