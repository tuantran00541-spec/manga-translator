from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api_proxy
from app.agent import client as agent_client
from app.ai_mode import vision_json
from app.ai_providers import PROVIDERS


class _FakeResp:
    def __init__(self, status_code: int, body: dict | str, headers: dict | None = None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body
        self.headers = headers or {}
        self.closed = False

    @property
    def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body, ensure_ascii=False)

    def json(self):
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body

    def close(self):
        self.closed = True


def _sample_img_b64(width: int = 1600, height: int = 1200) -> str:
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture(autouse=True)
def _clean_proxy():
    api_proxy.clear_proxy_logs()
    api_proxy.reset_learned_state()
    yield
    api_proxy.clear_proxy_logs()
    api_proxy.reset_learned_state()


def test_429_and_quota_errors_are_skipped_without_retry():
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        return _FakeResp(429, {"error": {"message": "Rate limit exceeded"}})

    resp, _, meta = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        caller="ai_mode",
        post_fn=fake_post,
    )
    assert resp.status_code == 429
    assert meta.get("skipped_quota") is True
    assert len(calls) == 1

    calls_quota = []

    def fake_quota(url, **kw):
        calls_quota.append(url)
        return _FakeResp(400, {"error": {"message": "You exceeded your current quota / billing balance"}})

    resp2, _, meta2 = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        caller="ai_mode",
        post_fn=fake_quota,
    )
    assert resp2.status_code == 400
    assert meta2.get("skipped_quota") is True
    assert len(calls_quota) == 1
    snap = api_proxy.proxy_snapshot()
    assert snap["stats"]["skipped_quota_calls"] == 2


def test_too_many_tools_is_rescued_and_preserves_core_tools():
    sent_counts = []
    tools = [{"type": "function", "function": {"name": f"extra_tool_{i}", "parameters": {}}} for i in range(40)]
    tools.append({"type": "function", "function": {"name": "read_file", "parameters": {}}})
    tools.append({"type": "function", "function": {"name": "edit_file", "parameters": {}}})

    def fake_post(url, **kw):
        t = kw["json"].get("tools") or []
        sent_counts.append(len(t))
        if len(t) > 10:
            return _FakeResp(400, {"error": {"message": "Too many tools: maximum 10 tools allowed per request"}})
        names = [item["function"]["name"] for item in t]
        assert "read_file" in names and "edit_file" in names
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    resp, final_payload, meta = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "tools": tools},
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        caller="local_proxy",
        post_fn=fake_post,
    )
    assert resp.ok
    assert sent_counts == [42, 10]
    assert len(final_payload["tools"]) == 10
    assert any("Cắt tools 42 → 10" in r for r in meta["rescues_applied"])


def test_agent_client_rescues_tool_count_limit_instead_of_dropping_tools(monkeypatch):
    seen_tools = []

    def fake_post(url, **kw):
        t = kw["json"].get("tools") or []
        seen_tools.append(len(t))
        if len(t) > 2:
            return _FakeResp(400, {"error": {"message": "tools array length must be <= 2 tools"}})
        return _FakeResp(200, {"choices": [{"message": {"content": "done", "tool_calls": []}, "finish_reason": "stop"}], "usage": {}})

    monkeypatch.setattr(agent_client.requests, "post", fake_post)
    specs = [{"name": f"t{i}", "description": "d", "parameters": {"type": "object", "properties": {}}} for i in range(6)]
    out = agent_client.complete(
        PROVIDERS["openai"],
        "sk-test",
        "gpt-4o",
        [{"role": "user", "content": "hi"}],
        tools=specs,
    )
    assert out["text"] == "done"
    assert seen_tools == [6, 2]


def test_too_many_images_splits_vision_json_batches_and_merges(monkeypatch):
    batch_sizes = []

    def fake_post(url, **kw):
        parts = kw["json"]["messages"][0]["content"]
        img_count = sum(1 for p in parts if p.get("type") == "image_url")
        batch_sizes.append(img_count)
        if img_count > 2:
            return _FakeResp(400, {"error": {"message": "Too many images: at most 2 images allowed"}})
        idx_start = sum(batch_sizes[:-1])
        slices = [{"index": idx_start + i + 1, "keep": True} for i in range(img_count)]
        return _FakeResp(200, {
            "choices": [{"message": {"content": json.dumps({"slices": slices})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    monkeypatch.setattr(vision_json.requests, "post", fake_post)
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    images = [(f"SLICE {i}", tiny) for i in range(4)]
    result = vision_json.request_vision_json(PROVIDERS["openai"], "gpt-4o", "sk-test", "scan", images)
    assert batch_sizes == [4, 2, 2]
    assert len(result.data["slices"]) == 4
    assert result.usage["prompt_tokens"] == 200


def test_payload_413_max_tokens_refused_fields_and_malformed_json_are_rescued():
    b64 = _sample_img_b64()
    seen_payloads = []
    steps = [0]

    def fake_post_seq(url, **kw):
        steps[0] += 1
        body = kw["json"]
        seen_payloads.append(body)
        if body.get("max_tokens", 0) > 8192:
            return _FakeResp(400, {"error": {"message": "max_tokens must be <= 8192"}})
        if "response_format" in body:
            return _FakeResp(400, {"error": {"message": "response_format is unsupported"}})
        if steps[0] == 3:
            return _FakeResp(413, {"error": {"message": "Payload too large"}})
        return _FakeResp(200, {"choices": [{"message": {"content": '<think>draft</think>```json\n{"ok": true,```'}}]})

    resp, final_payload, meta = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={
            "model": "custom-m",
            "max_tokens": 1000000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "read"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        },
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        caller="ai_mode",
        post_fn=fake_post_seq,
    )
    assert resp.ok
    assert final_payload["max_tokens"] == 8192
    assert "response_format" not in final_payload
    shrunk_url = final_payload["messages"][0]["content"][1]["image_url"]["url"]
    assert len(shrunk_url) < len(f"data:image/jpeg;base64,{b64}")
    raw_content = resp.json()["choices"][0]["message"]["content"]
    assert api_proxy.repair_json_object(raw_content) == {"ok": True}
    assert len(meta["rescues_applied"]) == 3


def test_local_proxy_router_status_and_chat_completions(monkeypatch):
    from app.main import app
    from app.routers import api_proxy as proxy_router

    monkeypatch.setattr(proxy_router, "get_provider_api_key", lambda *a, **kw: "sk-local")
    monkeypatch.setattr(proxy_router, "list_provider_configs", lambda: [{"id": "custom-gw", "label": "Custom", "protocol": "openai", "api_base": "https://gw.example.com/v1"}])
    monkeypatch.setattr(proxy_router, "_is_loopback_host", lambda host: host == "testclient")

    calls = []

    def fake_post(url, **kw):
        calls.append((url, kw["json"]))
        if len(calls) == 1:
            return _FakeResp(503, {"error": {"message": "Service temporarily unavailable"}})
        return _FakeResp(200, {
            "id": "cmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "```json\n{\"hello\": \"world\",}\n```"}, "finish_reason": "stop"}],
        })

    monkeypatch.setattr(api_proxy.requests, "post", fake_post)
    http = TestClient(app, base_url="http://127.0.0.1")

    res = http.post(
        "/api/proxy/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert json.loads(data["choices"][0]["message"]["content"]) == {"hello": "world"}

    status = http.get("/api/proxy/status").json()
    assert status["enabled"] is True
    assert status["stats"]["rescued_calls"] >= 1
    assert any(e["caller"] == "local_proxy" for e in status["events"])

    cleared = http.post("/api/proxy/logs/clear")
    assert cleared.status_code == 200 and cleared.json()["cleared"] is True
    assert http.get("/api/proxy/status").json()["stats"]["total_calls"] == 0

    models_resp = http.get("/api/proxy/v1/models")
    assert models_resp.status_code == 200, models_resp.text
    model_ids = [row["id"] for row in models_resp.json()["data"]]
    assert "custom-gw:default" in model_ids


def test_auto_wrap_persists_base_url_masked_key_and_preapplies_rules(monkeypatch):
    from app.main import app
    from app.routers import api_proxy as proxy_router

    monkeypatch.setattr(proxy_router, "_is_loopback_host", lambda host: host == "testclient")

    seen_batches = []

    def fake_vision_post(url, **kw):
        parts = kw["json"]["messages"][0]["content"]
        img_count = sum(1 for p in parts if p.get("type") == "image_url")
        seen_batches.append(img_count)
        if img_count > 2:
            return _FakeResp(400, {"error": {"message": "Too many images: at most 2 images allowed"}})
        slices = [{"index": i + 1, "keep": True} for i in range(img_count)]
        return _FakeResp(200, {"choices": [{"message": {"content": json.dumps({"slices": slices})}}]})

    monkeypatch.setattr(vision_json.requests, "post", fake_vision_post)
    tiny = np.zeros((24, 24, 3), dtype=np.uint8)
    images = [(f"SLICE {i}", tiny) for i in range(4)]

    # First call fails once with 4 images -> auto-wraps (api_base, model) with image_cap=2.
    secret_key = "sk-supersecret1234"
    res1 = vision_json.request_vision_json(PROVIDERS["openai"], "vision-wrap-model", secret_key, "scan", images)
    assert len(res1.data["slices"]) == 4
    assert seen_batches == [4, 2, 2]

    # Second call to the same (api_base, model) is already wrapped -> chunks proactively 2+2 on attempt #1!
    res2 = vision_json.request_vision_json(PROVIDERS["openai"], "vision-wrap-model", secret_key, "scan", images)
    assert len(res2.data["slices"]) == 4
    assert seen_batches == [4, 2, 2, 2, 2]

    http = TestClient(app, base_url="http://127.0.0.1")
    snap = http.get("/api/proxy/status").json()
    assert len(snap["wrapped_targets"]) == 1
    wrapped = snap["wrapped_targets"][0]
    assert wrapped["api_base"] == PROVIDERS["openai"].api_base
    assert wrapped["model"] == "vision-wrap-model"
    assert secret_key not in json.dumps(snap)
    assert wrapped["key_hint"].startswith("sk-su") and wrapped["key_hint"].endswith("1234")
    assert any("Ảnh ≤ 2" in rule for rule in wrapped["rules"])

    unwrap_res = http.post("/api/proxy/unwrap", json={"api_base": wrapped["api_base"], "model": "vision-wrap-model"})
    assert unwrap_res.status_code == 200 and unwrap_res.json()["unwrapped"] is True
    assert http.get("/api/proxy/status").json()["wrapped_targets"] == []


def test_explicit_proxy_target_never_reads_keyring_and_blocks_cross_origin(monkeypatch):
    from app.main import app
    from app.routers import api_proxy as proxy_router

    keyring_calls = []
    monkeypatch.setattr(
        proxy_router,
        "get_provider_api_key",
        lambda *a, **kw: keyring_calls.append(a) or "sk-secret-keyring",
    )
    monkeypatch.setattr(proxy_router, "_is_loopback_host", lambda host: host == "testclient")
    monkeypatch.setattr(proxy_router, "validate_url", lambda url: url)
    monkeypatch.setattr(api_proxy, "validate_url", lambda url: url)

    seen = []

    def fake_post(url, **kw):
        seen.append((url, kw["headers"].get("Authorization"), kw["json"]))
        return _FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(api_proxy.requests, "post", fake_post)
    http = TestClient(app, base_url="http://127.0.0.1")

    # Cross-origin and cross-site CSRF must be blocked even on loopback
    assert http.post(
        "/api/proxy/v1/chat/completions",
        headers={"Origin": "https://evil.example"},
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    ).status_code == 403
    assert http.post(
        "/api/proxy/logs/clear",
        headers={"Sec-Fetch-Site": "cross-site"},
    ).status_code == 403
    assert http.post(
        "/api/proxy/v1/chat/completions",
        headers={"Content-Type": "text/plain"},
        content="{}",
    ).status_code == 415

    # Explicit base without caller Bearer token MUST fail 401 and MUST NOT touch keyring
    res_no_auth = http.post(
        "/api/proxy/v1/chat/completions",
        json={
            "base_url": "https://attacker.example/v1",
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert res_no_auth.status_code == 401
    assert keyring_calls == []
    assert seen == []

    # Explicit base with caller Bearer token forwards caller key and strips proxy routing fields
    res_with_auth = http.post(
        "/api/proxy/v1/chat/completions",
        headers={"Authorization": "Bearer caller-owned-key"},
        json={
            "target_base_url": "https://custom.example/v1",
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert res_with_auth.status_code == 200
    assert keyring_calls == []
    assert len(seen) == 1
    assert seen[0][1] == "Bearer caller-owned-key"
    assert "target_base_url" not in seen[0][2]
    assert "provider" not in seen[0][2]


def test_rescue_learning_deferred_until_2xx_and_image_strip_blocked_by_default(monkeypatch):
    monkeypatch.setattr(api_proxy, "validate_url", lambda url: url)
    # Endpoint 404 recovery uses str api_base_candidates, only caches on 2xx, and unwrap clears it
    calls = []

    def fake_404_then_200(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResp(404, {"error": {"message": "Not found"}})
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    from app.ai_providers import AIProvider

    custom = AIProvider(
        id="gw",
        label="GW",
        protocol="openai",
        api_base="https://gw.example",
        default_qc_model="m",
        default_translation_model="m",
        env_names=(),
        request_profile="standard",
        image_transport="data_url",
        builtin=False,
    )
    resp, _, _ = api_proxy.proxy_post(
        url="https://gw.example/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        provider=custom,
        api_key="sk-secret-abc123",
        post_fn=fake_404_then_200,
    )
    assert resp.status_code == 200
    assert calls[1] == "https://gw.example/v1/chat/completions"
    assert api_proxy.unwrap_target("https://gw.example/", "m") is True
    assert "https://gw.example" not in api_proxy._FOUND_CHAT_URLS

    # Failed retries must NOT pollute _WRAPPED_TARGETS / _REFUSED_FIELDS
    def fake_always_fail(url, **kw):
        return _FakeResp(400, {"error": {"message": "temperature unsupported but still failing"}})

    resp_fail, _, _ = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={"model": "fail-m", "temperature": 0.5, "messages": [{"role": "user", "content": "hi"}]},
        provider=PROVIDERS["openai"],
        api_key="sk-secret-abc123",
        post_fn=fake_always_fail,
    )
    assert resp_fail.status_code == 400
    assert api_proxy.proxy_snapshot()["wrapped_targets"] == []

    # allow_image_strip=False (default) must NOT strip images to 0 and hallucinate
    img_payload = {
        "model": "blind-m",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "translate"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _sample_img_b64(64, 64)}},
        ]}],
    }
    resp_blind, _, _ = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload=img_payload,
        provider=PROVIDERS["openai"],
        api_key="sk-secret-abc123",
        post_fn=lambda url, **kw: _FakeResp(400, {"error": {"message": "vision is not supported"}}),
    )
    assert resp_blind.status_code == 400


def test_context_overflow_shrink_tool_sse_and_secret_redaction(monkeypatch):
    from app.downloader import http as dl_http
    from app.routers.api_proxy import _sse_from_completion

    # Context overflow compacts context instead of adjusting max_tokens and keeps tool pairing
    seen_msgs = []

    def fake_overflow(url, **kw):
        seen_msgs.append(kw["json"]["messages"])
        if len(seen_msgs) == 1:
            return _FakeResp(400, {"error": {"message": "maximum context length exceeded (max_tokens too high)"}})
        return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old q1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "f0", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "x" * 600},
        {"role": "user", "content": "old q2"},
        {"role": "assistant", "content": "old a2"},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "recent"},
    ]
    resp, final_payload, meta = api_proxy.proxy_post(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        payload={"model": "m", "max_tokens": 4096, "messages": msgs},
        provider=PROVIDERS["openai"],
        api_key="sk-secret-999999",
        post_fn=fake_overflow,
    )
    assert resp.status_code == 200
    assert any("ngữ cảnh" in r for r in meta["rescues_applied"])
    assert final_payload["max_tokens"] == 4096

    short_tool_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "short1"},
        {"role": "tool", "tool_call_id": "a", "content": "short2"},
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
        {"role": "user", "content": "u3"},
        {"role": "user", "content": "u4"},
        {"role": "user", "content": "u5"},
    ]
    compacted, changed = api_proxy.compact_payload_context({"messages": short_tool_msgs})
    assert changed is True and compacted["messages"][1]["role"] != "tool"

    # _safe_error_detail redacts secret beyond index 600 & query key params
    long_resp = _FakeResp(400, ("a" * 595) + "sk-very-long-secret-tail https://x/?key=AIzaSecret123")
    cleaned = api_proxy._safe_error_detail(long_resp, "sk-very-long-secret-tail")
    assert "sk-very-long-secret-tail" not in cleaned
    assert "AIzaSecret123" not in cleaned

    # _shrink_b64_image returns None when encoding is not strictly smaller
    tiny_b64 = _sample_img_b64(4, 4)
    assert api_proxy._shrink_b64_image(tiny_b64, 1024, 99) is None

    # _sse_from_completion adds index to tool_calls
    import asyncio
    sse = _sse_from_completion({
        "id": "1",
        "choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "a"}}]}, "finish_reason": "tool_calls"}],
    })

    async def _collect():
        return b"".join([chunk async for chunk in sse.body_iterator])

    sse_bytes = asyncio.run(_collect())
    assert b'"index": 0' in sse_bytes and b'"[DONE]"' not in sse_bytes and b"data: [DONE]" in sse_bytes

    # safe_get strips auth headers across cross-origin redirects
    seen_headers = []

    class _DummySession:
        def get(self, url, **kw):
            seen_headers.append((url, dict(kw.get("headers") or {})))
            if len(seen_headers) == 1:
                r = _FakeResp(302, "", headers={"Location": "https://cdn.other.example/img.png"})
                r.raise_for_status = lambda: None
                return r
            r = _FakeResp(200, "ok")
            r.raise_for_status = lambda: None
            return r

    monkeypatch.setattr(dl_http, "_session", lambda: _DummySession())
    monkeypatch.setattr(dl_http, "validate_url", lambda url: url)
    out_r = dl_http.safe_get(
        "https://api.origin.example/start",
        headers={"Authorization": "Bearer secret", "x-goog-api-key": "k2", "Accept": "image/*"},
    )
    assert out_r.status_code == 200
    assert "Authorization" not in seen_headers[1][1]
    assert "x-goog-api-key" not in seen_headers[1][1]
    assert seen_headers[1][1]["Accept"] == "image/*"

