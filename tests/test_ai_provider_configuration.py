from __future__ import annotations

import pytest

from app.ai_providers import (
    PROVIDERS,
    get_provider,
    get_translation_provider,
    validate_model_name,
)
from app.routers import visual_qc as visual_qc_router
from app.routers.translation import TranslateChapterRequest
from app.translation.deepseek import DeepSeekTranslator, OpenAICompatibleTranslator
from app.visual_qc.deepseek_region_client import DeepSeekRegionQC
from app.visual_qc.schemas import VisualQCChapterRequest


def test_provider_registry_uses_fixed_https_endpoints():
    assert set(PROVIDERS) == {"gemini", "deepseek", "openai", "openrouter", "experiential"}
    assert get_provider("openai").chat_url == "https://api.openai.com/v1/chat/completions"
    assert all(provider.api_base.startswith("https://") for provider in PROVIDERS.values())
    assert get_provider("experiential").models_url == "https://api.experientiallabs.ai/v1/models"


def test_provider_contract_matches_current_ui_capabilities():
    gemini = get_provider("gemini")
    deepseek = get_provider("deepseek")
    openrouter = get_provider("openrouter")

    assert gemini.public_capabilities() == {
        "visual_qc": True,
        "translation": False,
        "model_listing": True,
        "cost_tracking": False,
        "image_transport": "inline_base64",
    }
    assert deepseek.public_capabilities()["translation"] is True
    assert deepseek.public_capabilities()["cost_tracking"] is True
    assert deepseek.chat_completion_extras() == {"thinking": {"type": "disabled"}}
    assert openrouter.chat_completion_extras() == {}
    assert get_translation_provider("openai").id == "openai"
    with pytest.raises(ValueError):
        get_translation_provider("gemini")


def test_settings_endpoint_exposes_provider_capabilities(monkeypatch):
    monkeypatch.setattr(
        visual_qc_router,
        "provider_key_status",
        lambda provider_id: {"configured": False, "source": "none"},
    )
    data = visual_qc_router.visual_qc_settings()
    assert data["providers"]["gemini"]["capabilities"]["translation"] is False
    assert data["providers"]["deepseek"]["capabilities"]["cost_tracking"] is True
    assert data["providers"]["openrouter"]["capabilities"]["image_transport"] == "data_url"


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


def test_chapter_qc_keeps_vendor_specific_fields_on_deepseek_only():
    deepseek = DeepSeekRegionQC(
        "deepseek-v4-flash-vision-exp",
        provider_id="deepseek",
        provider_label="DeepSeek",
        chat_url="https://api.deepseek.example/chat/completions",
    )
    openrouter = DeepSeekRegionQC(
        "vendor/vision",
        provider_id="openrouter",
        provider_label="OpenRouter",
        chat_url="https://openrouter.example/api/v1/chat/completions",
    )

    assert deepseek._request_extras == {"thinking": {"type": "disabled"}}
    assert openrouter._request_extras == {}
    assert deepseek.usage_snapshot()["cost_available"] is True
    assert openrouter.usage_snapshot()["cost_available"] is False


def test_generic_translation_defaults_to_selected_provider_contract():
    translator = OpenAICompatibleTranslator(provider_id="openrouter")
    provider = get_provider("openrouter")
    assert translator.model == provider.default_translation_model
    assert translator.api_url == provider.chat_url
    assert translator._request_extras == {}


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


def test_deepseek_translation_keeps_deepseek_request_profile(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "model": "deepseek-v4-flash",
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
    translator = DeepSeekTranslator(provider_id="deepseek")
    translator.translate(
        [{"id": "box-1", "page_index": 0, "text": "Hello"}],
        api_key="secret",
        source_lang="en",
        target_lang="vi",
        budget_usd=0.01,
    )
    assert captured["json"]["thinking"] == {"type": "disabled"}
