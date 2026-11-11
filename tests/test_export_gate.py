from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.routers.export as export_router


def _blocked(manifest, **_):
    kinds = ["untranslated_story_object"] * 2 + ["story_object_missing_ocr", "unresolved_cleanup_review"]
    return {"ok": False, "blocker_count": len(kinds), "blockers": [{"kind": k, "page_index": 0} for k in kinds]}


def test_unfinished_text_is_summed_up_in_vietnamese(monkeypatch):
    monkeypatch.setattr(export_router, "editorial_preflight", _blocked)
    with pytest.raises(HTTPException) as blocked:
        export_router._editorial_gate_or_409({})
    assert blocked.value.status_code == 409
    assert blocked.value.detail["code"] == "editorial_preflight"
    assert blocked.value.detail["blocker_count"] == 4
    assert blocked.value.detail["message"] == (
        "Chương còn 2 vùng chưa dịch, 1 vùng chưa có chữ gốc, 1 vùng cần xem lại."
    )


def test_a_confirmed_export_goes_through_unfinished_text(monkeypatch):
    monkeypatch.setattr(export_router, "editorial_preflight", _blocked)
    assert export_router._editorial_gate_or_409({}, force=True)["blocker_count"] == 4


def test_export_source_page_accepts_a_numeric_string_and_rejects_a_bool():
    """M10: a source page stored as a string still resolves; a bool is
    not a page number, so it gets the unique negative sentinel."""
    assert export_router._export_source_page({"source_page": "3"}, 0) == 3
    assert export_router._export_source_page({"source_page": 3}, 0) == 3
    assert export_router._export_source_page({"source_page": True}, 0) == -1
    assert export_router._export_source_page({"source_page": "abc"}, 0) == -1
    assert export_router._export_source_page({}, 5) == -6


def test_a_malformed_stitch_core_is_rejected_not_accepted_as_legacy():
    """M9: a single slice carrying a stitch_core key but no usable
    metadata was built with stitch metadata, so its absence is
    corruption, not a legacy chapter without any."""
    items = [{"slice_index": 0, "page_index": 0, "stitch_malformed": True,
              "source_core_range": None, "source_height": None}]
    with pytest.raises(HTTPException) as blocked:
        export_router._validate_stitch_group(0, items)
    assert "malformed stitch metadata" in blocked.value.detail
    assert "slice 1" in blocked.value.detail


def test_a_legacy_single_slice_without_stitch_metadata_still_exports():
    """M9: a single slice with no stitch_core key at all is a legacy
    chapter, not corruption."""
    items = [{"slice_index": 0, "page_index": 0,
              "source_core_range": None, "source_height": None}]
    export_router._validate_stitch_group(0, items)


def test_stitch_validation_errors_name_the_affected_slices():
    """M8: a 409 that does not say which slices are bad sends the
    operator hunting through the whole chapter."""
    items = [{"slice_index": 0, "page_index": 0, "source_core_range": (0, 100), "source_height": 1000},
             {"slice_index": 1, "page_index": 1, "source_core_range": None, "source_height": 1000}]
    with pytest.raises(HTTPException) as blocked:
        export_router._validate_stitch_group(0, items)
    assert "slices [2]" in blocked.value.detail
