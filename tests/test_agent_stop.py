from __future__ import annotations

import threading
import time
from pathlib import Path

from app.ai_providers import PROVIDERS
from app.agent import client
from app.agent.sandbox import Policy
from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace


def _wait_idle(session, timeout: float = 2.0) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while session.status != "idle" and time.monotonic() < deadline:
        time.sleep(0.02)
    return time.monotonic() - started


def test_stop_aborts_blocking_http_post_immediately(tmp_path: Path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class SlowResp:
        status_code = 200
        headers: dict = {}
        text = ""

        def close(self):
            closed.set()

    def slow_post(url, **kw):
        entered.set()
        release.wait(5.0)
        return SlowResp()

    monkeypatch.setattr(client.requests, "post", slow_post)

    ws = Workspace(tmp_path, Policy("read-only", network=False))
    mgr = AgentSessionManager(tmp_path / "sessions", home=tmp_path)
    session = mgr.create(PROVIDERS["openai"], "sk-test", "m", ws, "bypass", complete=client.complete)
    try:
        session.send("Chạy thử")
        assert entered.wait(2.0), "HTTP post never started"

        t0 = time.monotonic()
        session.stop()
        elapsed = _wait_idle(session, timeout=1.5)

        assert session.status == "idle"
        assert elapsed < 0.8, f"stop took too long: {elapsed:.2f}s"
        assert not any(ev.get("type") == "error" for ev in session.events), session.events
        assert any(ev.get("type") == "notice" and "Đã dừng" in ev.get("text", "") for ev in session.events)
    finally:
        release.set()
        session.close()
    assert closed.wait(1.0), "abandoned response should be closed when background post completes"


def test_stop_aborts_blocked_stream_without_fallback_retry(tmp_path: Path, monkeypatch):
    entered = threading.Event()
    calls = 0

    class HangingStreamResp:
        status_code = 200
        headers: dict = {}
        text = ""
        encoding = "utf-8"

        def __init__(self):
            self._closed = threading.Event()

        def iter_lines(self, decode_unicode=True):
            entered.set()
            yield 'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}'
            self._closed.wait(5.0)
            raise client.requests.exceptions.ConnectionError("socket closed by cancel")

        def close(self):
            self._closed.set()

    def fake_post(url, **kw):
        nonlocal calls
        calls += 1
        return HangingStreamResp()

    monkeypatch.setattr(client.requests, "post", fake_post)

    ws = Workspace(tmp_path, Policy("read-only", network=False))
    mgr = AgentSessionManager(tmp_path / "sessions", home=tmp_path)
    session = mgr.create(PROVIDERS["openai"], "sk-test", "m", ws, "bypass", complete=client.complete)
    try:
        session.send("Chạy thử stream")
        assert entered.wait(2.0), "stream never started"
        time.sleep(0.05)

        session.stop()
        elapsed = _wait_idle(session, timeout=1.5)

        assert session.status == "idle"
        assert elapsed < 0.8, f"stream stop took too long: {elapsed:.2f}s"
        assert calls == 1, f"stop must NOT trigger non-stream fallback retry (got {calls} calls)"
        assert not any(ev.get("type") == "error" for ev in session.events), session.events
        assert any(ev.get("type") == "notice" and "Đã dừng" in ev.get("text", "") for ev in session.events)
    finally:
        session.close()


def test_stop_interrupts_rate_limit_sleep_immediately(tmp_path: Path, monkeypatch):
    entered = threading.Event()

    class RateLimitedResp:
        status_code = 429
        headers = {"Retry-After": "30"}
        text = "rate limit"

        def close(self):
            pass

    def fake_post(url, **kw):
        entered.set()
        return RateLimitedResp()

    monkeypatch.setattr(client.requests, "post", fake_post)
    client._COOLDOWN.clear()
    client._PACE.clear()

    ws = Workspace(tmp_path, Policy("read-only", network=False))
    mgr = AgentSessionManager(tmp_path / "sessions", home=tmp_path)
    session = mgr.create(PROVIDERS["openai"], "sk-test", "m", ws, "bypass", complete=client.complete)
    try:
        session.send("Thử 429")
        assert entered.wait(2.0)
        time.sleep(0.05)

        session.stop()
        elapsed = _wait_idle(session, timeout=1.5)

        assert session.status == "idle"
        assert elapsed < 0.8, f"429 sleep stop took too long: {elapsed:.2f}s"
        assert not any(ev.get("type") == "error" for ev in session.events), session.events
    finally:
        client._COOLDOWN.clear()
        client._PACE.clear()
        session.close()
