from __future__ import annotations

import json

import numpy as np
import pytest
import requests

from app.visual_qc.cache import load_region_qc_cache, store_region_qc_cache
from app.visual_qc.contact_sheet import ContactSheet, ContactSheetItem
from app.visual_qc.deepseek_region_client import (
    DeepSeekBudgetExceeded,
    DeepSeekRegionQC,
)
from app.visual_qc.regions import QCRegion
from app.visual_qc.schemas import VisualQCChapterRequest


class _Response:
    def __init__(self, body, *, status_code: int = 200, text: str = ""):
        self._body = body
        self.status_code = status_code
        self.text = text
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _region(region_id: str, page_index: int = 0) -> QCRegion:
    return QCRegion(
        page_index=page_index,
        region_id=region_id,
        bbox=(10, 20, 110, 120),
        source_box_ids=(),
        source_kinds=("box",),
        area_ratio=0.1,
        requires_deep_qc=False,
    )


def _sheet(*region_ids: str) -> ContactSheet:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    items = tuple(
        ContactSheetItem(
            region_id=region_id,
            page_index=index,
            source_bbox=(10, 20, 110, 120),
            sheet_bbox=(0, 0, 120, 80),
        )
        for index, region_id in enumerate(region_ids)
    )
    return ContactSheet(image=image, items=items, scale=1.0)


def _body(regions: list[dict], *, prompt_tokens: int = 1000, completion_tokens: int = 100) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"regions": regions})},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def test_deepseek_payload_disables_thinking_and_uses_image_data_url(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response(_body([{"region_id": "P0001-R01", "status": "pass", "issues": []}]))

    monkeypatch.setattr(requests, "post", fake_post)
    client = DeepSeekRegionQC(budget_usd=0.02)
    region = _region("P0001-R01")
    result = client.inspect(
        _sheet(region.region_id),
        {region.region_id: region},
        "secret-key",
        mode="region-clean",
    )

    payload = captured["json"]
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert payload["model"] == "deepseek-v4-flash-vision-exp"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] <= 800
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "Return JSON only" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result[0].status == "pass"

    usage = client.usage_snapshot()
    assert usage["requests"] == 1
    assert usage["prompt_tokens"] == 1000
    assert usage["completion_tokens"] == 100
    assert 0 < usage["estimated_cost_usd"] < usage["budget_usd"]


def test_missing_deepseek_region_is_ambiguous(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(
            _body([{"region_id": "P0001-R01", "status": "pass", "issues": []}])
        ),
    )
    client = DeepSeekRegionQC(budget_usd=0.02)
    first = _region("P0001-R01", 0)
    second = _region("P0002-R01", 1)
    result = client.inspect(
        _sheet(first.region_id, second.region_id),
        {first.region_id: first, second.region_id: second},
        "secret-key",
        mode="region-clean",
    )
    assert [item.status for item in result] == ["pass", "ambiguous"]


def test_budget_guard_blocks_new_request_before_network(monkeypatch):
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(
            _body(
                [{"region_id": "P0001-R01", "status": "pass", "issues": []}],
                prompt_tokens=1000,
                completion_tokens=100,
            )
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = DeepSeekRegionQC(budget_usd=0.005)
    region = _region("P0001-R01")
    client.inspect(
        _sheet(region.region_id),
        {region.region_id: region},
        "secret-key",
        mode="region-clean",
    )
    with pytest.raises(DeepSeekBudgetExceeded):
        client.inspect(
            _sheet(region.region_id),
            {region.region_id: region},
            "secret-key",
            mode="region-clean",
        )
    assert calls == 1


def test_invalid_http_200_body_settles_reservation_conservatively(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response(ValueError("not json")),
    )
    client = DeepSeekRegionQC(budget_usd=0.02)
    region = _region("P0001-R01")
    with pytest.raises(RuntimeError, match="invalid structured response"):
        client.inspect(
            _sheet(region.region_id),
            {region.region_id: region},
            "secret-key",
            mode="region-clean",
        )
    usage = client.usage_snapshot()
    assert usage["requests"] == 1
    assert usage["estimated_cost_usd"] > 0
    assert usage["remaining_budget_usd"] < usage["budget_usd"]


def test_deepseek_cache_does_not_overwrite_legacy_gemini_slot():
    page = {"source_revision": 3, "clean_revision": 7}
    revision = (100, 200, 300)
    store_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="same-model-for-test",
        mode="region-clean",
        clean_file_revision=revision,
        result={"region_id": "P0001-R01", "status": "pass", "issues": []},
        provider="gemini",
    )
    store_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="same-model-for-test",
        mode="region-clean",
        clean_file_revision=revision,
        result={"region_id": "P0001-R01", "status": "flagged", "issues": []},
        provider="deepseek",
    )

    gemini = load_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="same-model-for-test",
        mode="region-clean",
        clean_file_revision=revision,
        provider="gemini",
    )
    deepseek = load_region_qc_cache(
        page,
        region_id="P0001-R01",
        model="same-model-for-test",
        mode="region-clean",
        clean_file_revision=revision,
        provider="deepseek",
    )
    assert gemini["status"] == "pass"
    assert deepseek["status"] == "flagged"
    assert set(page["visual_qc_cache"]) == {"region-clean", "deepseek:region-clean"}


def test_chapter_request_keeps_gemini_default_and_bounds_deepseek_budget():
    request = VisualQCChapterRequest(chapter_id="abc")
    assert request.provider == "gemini"
    assert request.budget_usd == 0.08
    assert VisualQCChapterRequest(
        chapter_id="abc",
        provider="deepseek",
        budget_usd=0.02,
    ).provider == "deepseek"
    with pytest.raises(ValueError):
        VisualQCChapterRequest(chapter_id="abc", provider="deepseek", budget_usd=0.2)
