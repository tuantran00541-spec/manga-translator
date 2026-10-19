from __future__ import annotations

import pytest

from app.ai_providers import (
    PROVIDERS,
    resolve_provider,
    validate_model_name,
)
from app.routers import visual_qc as visual_qc_router
from app.routers.translation import TranslateChapterRequest, TranslateVisionPageRequest


def test_provider_registry_uses_fixed_https_endpoints():
    assert set(PROVIDERS) == {"gemini", "deepseek", "openai", "openrouter"}
    assert PROVIDERS["openai"].chat_url == "https://api.openai.com/v1/chat/completions"
    assert all(provider.api_base.startswith("https://") for provider in PROVIDERS.values())


def test_requests_validate_provider_id_shape_and_model():
    request = TranslateChapterRequest(
        chapter_id="chapter", provider="openrouter", model="vendor/vision"
    )
    assert request.provider == "openrouter"
    assert request.model == "vendor/vision"

    custom = TranslateChapterRequest(
        chapter_id="chapter", provider="custom-lab", model="vendor/vision"
    )
    assert custom.provider == "custom-lab"

    assert TranslateChapterRequest(chapter_id="chapter").provider == "deepseek"

    with pytest.raises(ValueError):
        TranslateChapterRequest(chapter_id="chapter", provider="Bad Provider!")
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
    page = TranslateVisionPageRequest(
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
    assert page.provider == translation.provider == "custom-lab"


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


def test_keys_go_to_a_private_file_when_the_machine_has_no_secret_store(monkeypatch, tmp_path):
    import stat

    import keyring
    from keyring.backends import fail

    from app import secret_store

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(secret_store, "_os_store_works", None)
    monkeypatch.setattr(keyring, "get_keyring", lambda: fail.Keyring())
    secret_store.set_provider_api_key("custom-lab", "top-secret", provider_label="Custom Lab")
    path = tmp_path / "manga-translator" / "secrets.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert secret_store.get_provider_api_key("custom-lab", provider_label="Custom Lab") == "top-secret"
    assert secret_store.provider_key_status("custom-lab", provider_label="Custom Lab")["source"] == "local_file"
    secret_store.delete_provider_api_key("custom-lab", provider_label="Custom Lab")
    secret_store.delete_provider_api_key("custom-lab", provider_label="Custom Lab")
    assert secret_store.get_provider_api_key("custom-lab", provider_label="Custom Lab") is None


def test_a_secret_store_that_crashes_while_loading_falls_back_to_the_file(monkeypatch, tmp_path):
    import keyring

    from app import secret_store

    def panic():
        raise BaseException("backend panicked")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(secret_store, "_os_store_works", None)
    monkeypatch.setattr(keyring, "get_keyring", panic)
    secret_store.set_provider_api_key("custom-lab", "k", provider_label="Custom Lab")
    assert (tmp_path / "manga-translator" / "secrets.json").is_file()


@pytest.mark.parametrize("pasted, base", [
    ("https://api.groq.com", "https://api.groq.com/openai/v1"),
    ("api.openai.com/v1/chat/completions", "https://api.openai.com/v1"),
    ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1"),
    ("https://generativelanguage.googleapis.com", "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/", "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("https://api.z.ai/api/paas/v4/chat/completions", "https://api.z.ai/api/paas/v4"),
    ("https://dashscope-intl.aliyuncs.com", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("https://api.perplexity.ai/chat/completions", "https://api.perplexity.ai"),
    ("https://me.openai.azure.com/openai/deployments/gpt/chat/completions?api-version=2024-10-21", "https://me.openai.azure.com/openai/v1"),
    ("https://gw.example.com/custom/v1/models", "https://gw.example.com/custom/v1"),
])
def test_pasted_base_urls_become_the_api_root(pasted, base):
    from app.ai_providers import validate_api_base

    assert validate_api_base(pasted) == base


def test_an_unknown_bare_host_is_probed_for_its_api_root(monkeypatch):
    import requests

    from app.ai_providers import auth_headers

    tried = []

    class Response:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

        def close(self):
            pass

    def fake_get(url, **kwargs):
        tried.append(url)
        if url == "https://gw.example.com/v1/models":
            return Response({"data": [{"id": "m"}]})
        error = requests.HTTPError("404")
        error.response = type("R", (), {"status_code": 404, "headers": {"content-type": "text/html"}})()
        raise error

    monkeypatch.setattr(visual_qc_router, "safe_get", fake_get)
    provider = resolve_provider("gw", label="Gateway", protocol="openai", api_base="https://gw.example.com")
    found = visual_qc_router._discover_api_base(provider, "k")
    assert found.api_base == "https://gw.example.com/v1"
    assert tried == ["https://gw.example.com/models", "https://gw.example.com/v1/models"]
    azure = resolve_provider("az", label="Azure", protocol="openai", api_base="https://me.openai.azure.com")
    assert auth_headers(azure, "k") == {"Authorization": "Bearer k", "api-key": "k"}
    assert auth_headers(PROVIDERS["openai"], "k") == {"Authorization": "Bearer k"}
