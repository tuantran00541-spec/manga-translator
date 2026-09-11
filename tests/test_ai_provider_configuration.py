from __future__ import annotations

import pytest

from app.ai_providers import PROVIDERS, get_provider, validate_model_name
from app.routers import visual_qc as visual_qc_router
from app.routers.translation import TranslateChapterRequest
from app.translation.deepseek import DeepSeekTranslator
from app.visual_qc.schemas import VisualQCChapterRequest


def test_provider_registry_uses_fixed_https_endpoints():
    assert set(PROVIDERS) == {"gemini", "deepseek", "openai", "openrouter", "experiential"}
    assert get_provider("openai").chat_url == "https://api.openai.com/v1/chat/completions"
    assert all(provider.api_base.startswith("https://") for provider in PROVIDERS.values())
    assert get_provider("experiential").models_url == "https://api.experientiallabs.ai/v1/models"


def test_requests_validate_provider_and_model():
    request = VisualQCChapterRequest(
        chapter_id="chapter", provider="openrouter", model="vendor/vision"
    )
    assert request.provider == "openrouter"
    assert request.model == "vendor/vision"
    with pytest.raises(ValueError):
        VisualQCChapterRequest(chapter_id="chapter", provider="unknown")
    with pytest.raises(ValueError):
        TranslateChapterRequest(chapter_id="chapter", provider="gemini")
    with pytest.raises(ValueError):
        validate_model_name("bad\nmodel", default="fallback")


def test_model_listing_normalizes_gemini_names(monkeypatch):
    monkeypatch.setattr(
        visual_qc_router, "get_provider_api_key", lambda provider_id: "secret"
    )

    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {
                "models": [
                    {"name": "models/gemini-flash"},
                    {"name": "models/gemini-pro"},
                ]
            }

    monkeypatch.setattr(
        visual_qc_router.requests, "get", lambda *args, **kwargs: Response()
    )
    assert visual_qc_router.list_provider_models("gemini") == {
        "provider": "gemini",
        "models": ["gemini-flash", "gemini-pro"],
    }


def test_compatible_translation_uses_selected_endpoint(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "model": "vendor/model-a",
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":{"box-1":"Xin chào"}}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("app.translation.deepseek.requests.post", post)
    translator = DeepSeekTranslator(
        "vendor/model-a",
        api_url="https://api.openrouter.example/v1/chat/completions",
        provider_id="openrouter",
        provider_label="OpenRouter",
    )
    result = translator.translate(
        [{"id": "box-1", "page_index": 0, "text": "Hello"}],
        api_key="secret",
        source_lang="en",
        target_lang="vi",
        budget_usd=0.001,
    )
    assert captured["url"] == "https://api.openrouter.example/v1/chat/completions"
    assert captured["json"]["model"] == "vendor/model-a"
    assert "thinking" not in captured["json"]
    assert result.translations == {"box-1": "Xin chào"}
    assert result.estimated_cost_usd == 0.0
