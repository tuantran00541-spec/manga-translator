import json

from app.translation.deepseek import DeepSeekTranslator, _preflight_cost_usd, _usage_cost_usd


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"translations": {"a": "Xin chào", "b": "Tạm biệt"}},
                            ensure_ascii=False,
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 400,
                "prompt_cache_miss_tokens": 600,
                "completion_tokens": 200,
            },
        }


def test_cost_uses_current_v4_flash_token_classes():
    cost = _usage_cost_usd(
        {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 400,
            "prompt_cache_miss_tokens": 600,
            "completion_tokens": 200,
        }
    )
    assert 0 < cost < 0.001
    assert _preflight_cost_usd([{"text": "hello"}]) > 0


def test_translator_uses_non_thinking_json_and_preserves_ids(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.translation.deepseek.requests.post", fake_post)
    result = DeepSeekTranslator().translate(
        [
            {"id": "a", "page_index": 0, "text": "Hello"},
            {"id": "b", "page_index": 0, "text": "Bye"},
        ],
        api_key="secret",
        source_lang="en",
        target_lang="vi",
        budget_usd=0.02,
    )

    assert result.translations == {"a": "Xin chào", "b": "Tạm biệt"}
    assert captured["json"]["model"] == "deepseek-v4-flash"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["Authorization"] == "Bearer secret"
