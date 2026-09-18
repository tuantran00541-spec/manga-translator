from __future__ import annotations

import json

import numpy as np
import pytest

from app.ai_providers import (
    PROVIDERS,
    get_provider,
    resolve_provider,
    validate_model_name,
)
from app.routers import visual_qc as visual_qc_router
from app.routers.translation import TranslateChapterRequest
from app.translation.deepseek import DeepSeekTranslator
from app.visual_qc.schemas import VisualQCChapterRequest


def test_provider_registry_uses_fixed_https_endpoints():
    assert set(PROVIDERS) == {"gemini", "deepseek", "openai", "openrouter", "experiential"}
    assert get_provider("openai").chat_url == "https://api.openai.com/v1/chat/completions"
    assert all(provider.api_base.startswith("https://") for provider in PROVIDERS.values())
    assert get_provider("experiential").models_url == "https://api.experientiallabs.ai/v1/models"


def test_requests_validate_provider_id_shape_and_model():
    request = VisualQCChapterRequest(
        chapter_id="chapter", provider="openrouter", model="vendor/vision"
    )
    assert request.provider == "openrouter"
    assert request.model == "vendor/vision"

    custom = VisualQCChapterRequest(
        chapter_id="chapter", provider="custom-lab", model="vendor/vision"
    )
    assert custom.provider == "custom-lab"

    # Capability/configuration checks happen server-side after a dynamic provider
    # ID is resolved from the secure registry.
    assert TranslateChapterRequest(
        chapter_id="chapter", provider="gemini"
    ).provider == "gemini"

    with pytest.raises(ValueError):
        VisualQCChapterRequest(chapter_id="chapter", provider="Bad Provider!")
    with pytest.raises(ValueError):
        validate_model_name("bad\nmodel", default="fallback")


def test_model_listing_normalizes_gemini_names(monkeypatch):
    monkeypatch.setattr(
        visual_qc_router,
        "get_provider_api_key",
        lambda provider_id, **kwargs: "secret",
    )

    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def close():
            return None

        @staticmethod
        def json():
            return {
                "models": [
                    {"name": "models/gemini-flash"},
                    {"name": "models/gemini-pro"},
                ]
            }

    monkeypatch.setattr(
        visual_qc_router, "safe_get", lambda *args, **kwargs: Response()
    )
    assert visual_qc_router.list_provider_models("gemini") == {
        "provider": "gemini",
        "provider_label": "Google Gemini",
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


def test_custom_provider_is_openai_compatible_and_https_only():
    provider = resolve_provider(
        "custom-lab",
        label="Custom Lab",
        protocol="openai",
        api_base="https://ai.example.com/v1/",
    )
    assert provider.id == "custom-lab"
    assert provider.builtin is False
    assert provider.chat_url == "https://ai.example.com/v1/chat/completions"
    assert provider.supports_visual_qc
    assert provider.supports_translation

    with pytest.raises(ValueError):
        resolve_provider(
            "custom-http",
            label="Custom HTTP",
            protocol="openai",
            api_base="http://ai.example.com/v1",
        )
    with pytest.raises(ValueError):
        resolve_provider(
            "custom-gemini",
            label="Custom Gemini",
            protocol="gemini",
            api_base="https://ai.example.com/v1",
        )


def test_runtime_request_schemas_accept_custom_provider_id_only():
    chapter = VisualQCChapterRequest(
        chapter_id="chapter",
        provider="custom-lab",
        model="vendor/vision",
    )
    inspect = visual_qc_router.VisualQCInspectRequest(
        chapter_id="chapter",
        page_index=0,
        provider="custom-lab",
        model="vendor/vision",
    )
    translation = TranslateChapterRequest(
        chapter_id="chapter",
        provider="custom-lab",
        model="vendor/text",
    )
    assert chapter.provider == inspect.provider == translation.provider == "custom-lab"


def test_custom_provider_registry_round_trips_without_exposing_key(monkeypatch):
    from app import secret_store

    storage = {}

    class FakeKeyringError(Exception):
        pass

    class FakeKeyring:
        @staticmethod
        def get_password(service, account):
            return storage.get((service, account))

        @staticmethod
        def set_password(service, account, value):
            storage[(service, account)] = value

        @staticmethod
        def delete_password(service, account):
            storage.pop((service, account), None)

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
        "top-secret",
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
    custom = snapshot["providers"]["custom-lab"]
    assert custom["configured"] is True
    assert custom["builtin"] is False
    assert custom["api_base"] == "https://ai.example.com/v1"
    assert "top-secret" not in repr(snapshot)


def test_custom_provider_private_remote_is_rejected_before_use():
    provider = resolve_provider(
        "custom-local",
        label="Custom Local",
        protocol="openai",
        api_base="https://127.0.0.1/v1",
    )
    with pytest.raises(Exception):
        visual_qc_router._validate_custom_remote(provider)


def _install_provider_fake_keyring(monkeypatch):
    from app import secret_store

    storage = {}

    class FakeKeyringError(Exception):
        pass

    class FakeKeyring:
        @staticmethod
        def get_password(service, account):
            return storage.get((service, account))

        @staticmethod
        def set_password(service, account, value):
            storage[(service, account)] = value

        @staticmethod
        def delete_password(service, account):
            storage.pop((service, account), None)

    monkeypatch.setattr(
        secret_store,
        "_keyring_module",
        lambda: (FakeKeyring, FakeKeyringError),
    )
    return secret_store, storage


def test_whitespace_environment_key_does_not_shadow_secure_storage(monkeypatch):
    secret_store, _storage = _install_provider_fake_keyring(monkeypatch)
    secret_store.set_provider_api_key("openrouter", "secure-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")

    assert secret_store.get_provider_api_key("openrouter") == "secure-key"
    assert secret_store.provider_key_status("openrouter") == {
        "configured": True,
        "source": "os_secure_storage",
    }


def test_real_environment_key_has_precedence_over_secure_storage(monkeypatch):
    secret_store, _storage = _install_provider_fake_keyring(monkeypatch)
    secret_store.set_provider_api_key("openrouter", "secure-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")

    assert secret_store.get_provider_api_key("openrouter") == "env-key"
    assert secret_store.provider_key_status("openrouter") == {
        "configured": True,
        "source": "environment",
    }


def test_model_listing_uses_bearer_header_without_exposing_key(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        ok = True

        @staticmethod
        def json():
            return {"data": [{"id": "vendor/vision-model"}]}

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(
        visual_qc_router,
        "get_provider_api_key",
        lambda provider_id, **kwargs: "transport-secret",
    )

    def fake_safe_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(visual_qc_router, "safe_get", fake_safe_get)

    models = visual_qc_router.list_provider_models("openrouter")

    assert models["models"] == ["vendor/vision-model"]
    assert captured["headers"]["Authorization"] == "Bearer transport-secret"
    assert "transport-secret" not in json.dumps(models)


def test_openai_compatible_visual_qc_refuses_redirects_and_keeps_key_out_of_body(monkeypatch):
    from app.visual_qc import openai_compatible as visual_qc_client

    captured = {}

    class Response:
        status_code = 200
        ok = True

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"issues":[]}'}}]}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(
        visual_qc_client,
        "_read_image",
        lambda _path: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        visual_qc_client,
        "_encode_for_gemini",
        lambda _image: "ZmFrZS1qcGVn",
    )
    monkeypatch.setattr(visual_qc_client.requests, "post", fake_post)

    qc = visual_qc_client.OpenAICompatibleVisualQC(
        provider_id="openrouter",
        model="vendor/vision-model",
    )
    assert qc.inspect("original.jpg", "clean.jpg", "transport-secret") == []

    assert captured["allow_redirects"] is False
    assert captured["headers"]["Authorization"] == "Bearer transport-secret"
    assert "transport-secret" not in json.dumps(captured["json"])
    assert "thinking" not in captured["json"]


def test_openai_compatible_translation_refuses_redirects_and_keeps_key_out_of_body(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "model": "vendor/text-model",
                "choices": [{
                    "message": {
                        "content": '{"translations":{"box-1":"Xin chào"}}'
                    }
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("app.translation.deepseek.requests.post", fake_post)
    translator = DeepSeekTranslator(
        "vendor/text-model",
        provider_id="openrouter",
    )
    result = translator.translate(
        [{"id": "box-1", "page_index": 0, "text": "Hello"}],
        api_key="transport-secret",
        source_lang="en",
        target_lang="vi",
        budget_usd=0.01,
    )

    assert result.translations == {"box-1": "Xin chào"}
    assert captured["allow_redirects"] is False
    assert captured["headers"]["Authorization"] == "Bearer transport-secret"
    assert "transport-secret" not in json.dumps(captured["json"])
    assert "thinking" not in captured["json"]
