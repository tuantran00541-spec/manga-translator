from __future__ import annotations

import pytest

from app import secret_store
from app.ai_providers import PROVIDERS, get_provider, resolve_provider, validate_model_name
from app.routers import visual_qc as visual_qc_router
from app.routers.translation import TranslateChapterRequest
from app.translation.deepseek import DeepSeekTranslator
from app.visual_qc.schemas import VisualQCChapterRequest


def test_provider_registry_uses_fixed_https_endpoints():
    assert set(PROVIDERS) == {"gemini", "deepseek", "openai", "openrouter", "experiential"}
    assert get_provider("openai").chat_url == "https://api.openai.com/v1/chat/completions"
    assert all(provider.api_base.startswith("https://") for provider in PROVIDERS.values())
    assert get_provider("experiential").models_url == "https://api.experientiallabs.ai/v1/models"


def test_custom_provider_is_strictly_openai_compatible():
    provider = resolve_provider(
        "custom-lab",
        label="Custom Lab",
        protocol="openai",
        api_base="https://ai.example.com/v1/",
    )
    assert provider.id == "custom-lab"
    assert provider.builtin is False
    assert provider.chat_url == "https://ai.example.com/v1/chat/completions"
    assert provider.models_url == "https://ai.example.com/v1/models"
    with pytest.raises(ValueError):
        resolve_provider(
            "custom-gemini",
            label="Custom Gemini",
            protocol="gemini",
            api_base="https://ai.example.com/v1beta",
        )
    with pytest.raises(ValueError):
        resolve_provider(
            "custom-http",
            label="Custom HTTP",
            protocol="openai",
            api_base="http://ai.example.com/v1",
        )


def test_requests_accept_runtime_provider_id_and_validate_model():
    request = VisualQCChapterRequest(
        chapter_id="chapter",
        provider="custom-lab",
        provider_label="Custom Lab",
        provider_protocol="openai",
        provider_api_base="https://ai.example.com/v1",
        model="vendor/vision",
    )
    assert request.provider == "custom-lab"
    assert request.model == "vendor/vision"
    inspect = visual_qc_router.VisualQCInspectRuntimeRequest(
        chapter_id="chapter",
        page_index=0,
        provider="custom-lab",
        provider_label="Custom Lab",
        provider_protocol="openai",
        provider_api_base="https://ai.example.com/v1",
        model="vendor/vision",
    )
    assert inspect.provider == "custom-lab"
    translation = TranslateChapterRequest(
        chapter_id="chapter",
        provider="custom-lab",
        provider_label="Custom Lab",
        provider_protocol="openai",
        provider_api_base="https://ai.example.com/v1",
        model="vendor/text",
    )
    assert translation.provider == "custom-lab"
    with pytest.raises(ValueError):
        VisualQCChapterRequest(chapter_id="chapter", provider="Bad Provider")
    with pytest.raises(ValueError):
        validate_model_name("bad\nmodel", default="fallback")


def test_custom_provider_registry_round_trips_without_exposing_key(monkeypatch):
    store = {}

    class FakeKeyringError(Exception):
        pass

    class FakeKeyring:
        @staticmethod
        def get_password(service, account):
            return store.get((service, account))

        @staticmethod
        def set_password(service, account, value):
            store[(service, account)] = value

        @staticmethod
        def delete_password(service, account):
            key = (service, account)
            if key not in store:
                raise FakeKeyringError("not found")
            del store[key]

    monkeypatch.setattr(
        secret_store,
        "_keyring_module",
        lambda: (FakeKeyring, FakeKeyringError),
    )
    monkeypatch.setattr(
        visual_qc_router,
        "list_provider_configs",
        secret_store.list_provider_configs,
    )
    monkeypatch.setattr(
        visual_qc_router,
        "provider_key_status",
        secret_store.provider_key_status,
    )

    secret_store.set_provider_config(
        "custom-lab",
        label="Custom Lab",
        protocol="openai",
        api_base="https://ai.example.com/v1",
    )
    secret_store.set_provider_api_key(
        "custom-lab",
        "super-secret",
        provider_label="Custom Lab",
    )

    configs = secret_store.list_provider_configs()
    assert configs == [{
        "id": "custom-lab",
        "label": "Custom Lab",
        "protocol": "openai",
        "api_base": "https://ai.example.com/v1",
    }]
    snapshot = visual_qc_router.visual_qc_settings()
    assert snapshot["providers"]["custom-lab"]["configured"] is True
    assert snapshot["providers"]["custom-lab"]["builtin"] is False
    assert snapshot["providers"]["custom-lab"]["api_base"] == "https://ai.example.com/v1"
    assert "super-secret" not in repr(snapshot)

    secret_store.delete_provider_api_key("custom-lab", provider_label="Custom Lab")
    secret_store.delete_provider_config("custom-lab")
    assert secret_store.list_provider_configs() == []


def test_model_listing_normalizes_gemini_names(monkeypatch):
    monkeypatch.setattr(
        visual_qc_router, "get_provider_api_key", lambda provider_id, **kwargs: "secret"
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
        "provider_label": "Google Gemini",
        "models": ["gemini-flash", "gemini-pro"],
    }


def test_custom_model_listing_uses_standard_models_endpoint(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        visual_qc_router, "get_provider_api_key", lambda provider_id, **kwargs: "secret"
    )
    monkeypatch.setattr(visual_qc_router, "validate_url", lambda url: url)

    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "vendor/model-a"}, {"id": "vendor/model-b"}]}

    def get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(visual_qc_router.requests, "get", get)
    result = visual_qc_router.list_provider_models(
        "custom-lab",
        provider_label="Custom Lab",
        provider_protocol="openai",
        provider_api_base="https://ai.example.com/v1",
    )
    assert captured["url"] == "https://ai.example.com/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert result["models"] == ["vendor/model-a", "vendor/model-b"]


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
