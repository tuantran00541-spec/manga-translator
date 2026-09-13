from __future__ import annotations

import json

import numpy as np

import app.secret_store as secret_store
from app.routers import visual_qc as visual_qc_router
from app.schemas import VisualQCKeyRequest
from app.translation.deepseek import OpenAICompatibleTranslator
from app.visual_qc import openai_compatible as visual_qc_client


FAKE_KEY = "sk-test-manga-translator-not-a-real-key"


class FakeKeyringError(Exception):
    pass


class FakeKeyring:
    def __init__(self):
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        key = (service, account)
        if key not in self.values:
            raise FakeKeyringError("password not found")
        del self.values[key]


class JsonResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


def _install_fake_keyring(monkeypatch) -> FakeKeyring:
    fake = FakeKeyring()
    monkeypatch.setattr(secret_store, "_keyring_module", lambda: (fake, FakeKeyringError))
    return fake


def test_fake_key_roundtrip_through_settings_models_visual_qc_and_translation(monkeypatch):
    fake_keyring = _install_fake_keyring(monkeypatch)
    # A whitespace-only environment value must not shadow a valid secure-store key.
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")

    saved = visual_qc_router.save_provider_key(
        "openrouter",
        VisualQCKeyRequest(api_key=f"  {FAKE_KEY}  "),
    )
    assert saved == {
        "configured": True,
        "source": "os_secure_storage",
        "provider": "openrouter",
    }
    assert FAKE_KEY not in json.dumps(saved)
    assert fake_keyring.values[("manga-translator", "ai-provider-openrouter-api-key")] == FAKE_KEY
    assert secret_store.get_provider_api_key("openrouter") == FAKE_KEY

    status = secret_store.provider_key_status("openrouter")
    assert status == {"configured": True, "source": "os_secure_storage"}
    settings = visual_qc_router.visual_qc_settings()
    assert settings["providers"]["openrouter"]["configured"] is True
    assert settings["providers"]["openrouter"]["source"] == "os_secure_storage"
    assert FAKE_KEY not in json.dumps(settings)

    model_call = {}

    def fake_models_get(url, **kwargs):
        model_call.update(url=url, **kwargs)
        return JsonResponse({"data": [{"id": "vendor/vision-model"}]})

    monkeypatch.setattr(visual_qc_router.requests, "get", fake_models_get)
    models = visual_qc_router.list_provider_models("openrouter")
    assert models == {"provider": "openrouter", "models": ["vendor/vision-model"]}
    assert model_call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(models)

    qc_call = {}

    def fake_qc_post(url, **kwargs):
        qc_call.update(url=url, **kwargs)
        return JsonResponse(
            {"choices": [{"message": {"content": '{"issues":[]}'}}]}
        )

    monkeypatch.setattr(visual_qc_client, "_read_image", lambda path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(visual_qc_client, "_encode_for_gemini", lambda image: "ZmFrZS1qcGVn")
    monkeypatch.setattr(visual_qc_client.requests, "post", fake_qc_post)
    qc = visual_qc_client.OpenAICompatibleVisualQC(
        provider_id="openrouter",
        model="vendor/vision-model",
    )
    issues = qc.inspect("original.jpg", "cleaned.jpg", secret_store.get_provider_api_key("openrouter"))
    assert issues == []
    assert qc_call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(qc_call["json"])
    assert "thinking" not in qc_call["json"]

    translation_call = {}

    def fake_translation_post(url, **kwargs):
        translation_call.update(url=url, **kwargs)
        return JsonResponse(
            {
                "model": "vendor/text-model",
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":{"box-1":"Xin chào"}}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )

    monkeypatch.setattr("app.translation.deepseek.requests.post", fake_translation_post)
    translator = OpenAICompatibleTranslator(
        model="vendor/text-model",
        provider_id="openrouter",
    )
    result = translator.translate(
        [{"id": "box-1", "page_index": 0, "text": "Hello"}],
        api_key=secret_store.get_provider_api_key("openrouter"),
        source_lang="en",
        target_lang="vi",
        budget_usd=0.01,
    )
    assert result.translations == {"box-1": "Xin chào"}
    assert translation_call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(translation_call["json"])
    assert "thinking" not in translation_call["json"]

    cleared = visual_qc_router.clear_provider_key("openrouter")
    assert cleared == {"configured": False, "source": "none", "provider": "openrouter"}
    assert secret_store.get_provider_api_key("openrouter") is None
    assert ("manga-translator", "ai-provider-openrouter-api-key") not in fake_keyring.values


def test_real_environment_key_still_has_precedence_over_secure_storage(monkeypatch):
    _install_fake_keyring(monkeypatch)
    secret_store.set_provider_api_key("openrouter", FAKE_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-wins")

    assert secret_store.get_provider_api_key("openrouter") == "env-key-wins"
    assert secret_store.provider_key_status("openrouter") == {
        "configured": True,
        "source": "environment",
    }
