from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import app.secret_store as secret_store
from app.ai_providers import AIProvider, PROVIDERS
from app.routers import visual_qc as visual_qc_router
from app.translation.deepseek import OpenAICompatibleTranslator
from app.visual_qc import openai_compatible as visual_qc_client


FAKE_KEY = "sk-fake-receiver-not-a-real-key"


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
        self.values.pop((service, account), None)


class ReceiverHandler(BaseHTTPRequestHandler):
    server_version = "FakeProvider/1.0"

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - keep CI logs quiet
        return

    def _record(self, body: dict | None = None) -> None:
        self.server.received.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
                "body": body,
            }
        )

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._record()
        if self.path == "/v1/models":
            self._json({"data": [{"id": "receiver/model"}]})
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        body = json.loads(raw.decode("utf-8")) if raw else {}
        self._record(body)
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return

        serialized = json.dumps(body)
        if '"image_url"' in serialized:
            self._json(
                {
                    "model": "receiver/vision",
                    "choices": [{"message": {"content": '{"issues":[]}'}}],
                }
            )
            return

        self._json(
            {
                "model": "receiver/text",
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":{"box-1":"Xin chào từ receiver"}}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }
        )


@contextmanager
def fake_provider_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReceiverHandler)
    server.received = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_fake_key_reaches_real_local_http_receiver(monkeypatch):
    fake_keyring = FakeKeyring()
    monkeypatch.setattr(
        secret_store,
        "_keyring_module",
        lambda: (fake_keyring, FakeKeyringError),
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with fake_provider_server() as (server, base_url):
        monkeypatch.setitem(
            PROVIDERS,
            "openrouter",
            AIProvider(
                id="openrouter",
                label="Fake Receiver",
                protocol="openai",
                api_base=f"{base_url}/v1",
                default_qc_model="receiver/vision",
                default_translation_model="receiver/text",
                env_names=("OPENROUTER_API_KEY",),
                request_profile="standard",
                image_transport="data_url",
                supports_visual_qc=True,
                supports_translation=True,
                tracks_cost=False,
            ),
        )

        secret_store.set_provider_api_key("openrouter", FAKE_KEY)
        assert secret_store.get_provider_api_key("openrouter") == FAKE_KEY

        models = visual_qc_router.list_provider_models("openrouter")
        assert models == {"provider": "openrouter", "models": ["receiver/model"]}

        monkeypatch.setattr(
            visual_qc_client,
            "_read_image",
            lambda path: np.zeros((8, 8, 3), dtype=np.uint8),
        )
        qc = visual_qc_client.OpenAICompatibleVisualQC(
            provider_id="openrouter",
            model="receiver/vision",
        )
        assert qc.inspect(
            "original.jpg",
            "cleaned.jpg",
            secret_store.get_provider_api_key("openrouter"),
        ) == []

        translator = OpenAICompatibleTranslator(
            provider_id="openrouter",
            model="receiver/text",
        )
        translated = translator.translate(
            [{"id": "box-1", "page_index": 0, "text": "Hello"}],
            api_key=secret_store.get_provider_api_key("openrouter"),
            source_lang="en",
            target_lang="vi",
            budget_usd=0.01,
        )
        assert translated.translations == {"box-1": "Xin chào từ receiver"}

        received = list(server.received)  # type: ignore[attr-defined]

    assert [item["method"] for item in received] == ["GET", "POST", "POST"]
    assert [item["path"] for item in received] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert all(item["authorization"] == f"Bearer {FAKE_KEY}" for item in received)

    qc_body = received[1]["body"]
    qc_serialized = json.dumps(qc_body)
    assert "data:image/jpeg;base64," in qc_serialized
    assert FAKE_KEY not in qc_serialized
    assert "thinking" not in qc_body

    translation_body = received[2]["body"]
    assert translation_body["model"] == "receiver/text"
    assert "Hello" in json.dumps(translation_body)
    assert FAKE_KEY not in json.dumps(translation_body)
    assert "thinking" not in translation_body
