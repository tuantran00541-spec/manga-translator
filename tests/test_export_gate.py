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


def test_blocked_export_includes_blocker_locations_and_specific_ai_labels(monkeypatch):
    blocker = {"kind": "unresolved_ai_region", "page_index": 4,
               "region": {"x1": 10, "y1": 20, "x2": 80, "y2": 90}}
    monkeypatch.setattr(export_router, "editorial_preflight", lambda *a, **kw: {
        "ok": False, "blocker_count": 1, "blockers": [blocker]})
    with pytest.raises(HTTPException) as blocked:
        export_router._editorial_gate_or_409({})
    assert blocked.value.detail["blockers"] == [blocker]
    assert "chữ AI phát hiện chưa xử lý" in blocked.value.detail["message"]
