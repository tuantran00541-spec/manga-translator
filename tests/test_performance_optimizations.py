from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.session import AgentSessionManager
from app.routers import visual_qc as visual_qc_router


def test_list_provider_models_caches_results_and_supports_refresh(monkeypatch):
    visual_qc_router.clear_models_cache()
    key = {"value": "secret-a"}
    calls = []

    monkeypatch.setattr(
        visual_qc_router,
        "get_provider_api_key",
        lambda provider_id, **kwargs: key["value"],
    )

    class Response:
        ok = True
        status_code = 200

        def __init__(self, models):
            self._models = models

        def close(self):
            return None

        def json(self):
            return {"data": [{"id": model_id} for model_id in self._models]}

    def fake_get(url, **kwargs):
        calls.append(url)
        return Response([f"model-{len(calls)}"])

    monkeypatch.setattr(visual_qc_router, "safe_get", fake_get)

    first = visual_qc_router.list_provider_models("openai")
    second = visual_qc_router.list_provider_models("openai")
    assert first == second == {
        "provider": "openai",
        "provider_label": "OpenAI",
        "models": ["model-1"],
    }
    assert len(calls) == 1

    refreshed = visual_qc_router.list_provider_models("openai", refresh=True)
    assert refreshed["models"] == ["model-2"]
    assert len(calls) == 2

    key["value"] = "secret-b"
    rotated = visual_qc_router.list_provider_models("openai")
    assert rotated["models"] == ["model-3"]
    assert len(calls) == 3

    visual_qc_router.clear_models_cache("openai")
    cleared = visual_qc_router.list_provider_models("openai")
    assert cleared["models"] == ["model-4"]
    assert len(calls) == 4


def test_session_listing_limits_disk_reads_by_mtime_and_caches_meta(tmp_path):
    store_dir = tmp_path / "store"
    sessions_dir = store_dir / "sessions"
    sessions_dir.mkdir(parents=True)

    for idx in range(170):
        sid = f"sess{idx:04d}"
        ts = 1_700_000_000 + idx
        session_file = sessions_dir / f"{sid}.json"
        meta_file = sessions_dir / f"{sid}.meta.json"
        payload = {"id": sid, "title": f"Title {idx}", "model": "gpt-4o", "updated_at": ts, "workspace": str(tmp_path)}
        meta = {"title": f"Title {idx}", "model": "gpt-4o", "updated_at": ts, "workspace": str(tmp_path)}
        session_file.write_text(json.dumps(payload), encoding="utf-8")
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        os.utime(session_file, (ts, ts))
        os.utime(meta_file, (ts, ts))

    mgr = AgentSessionManager(store_dir, home=tmp_path / "home")
    rows = mgr.listing()
    assert len(rows) == 100
    assert rows[0]["id"] == "sess0169"
    assert rows[-1]["id"] == "sess0070"
    assert len(mgr._meta_cache) == 150

    rows_cached = mgr.listing()
    assert rows_cached[0]["id"] == "sess0169"

    # Deleting a session invalidates its listing cache entry cleanly.
    mgr.delete("sess0169")
    assert "sess0169" not in mgr._meta_cache
    rows_after_delete = mgr.listing()
    assert rows_after_delete[0]["id"] == "sess0168"


def test_view_session_reads_saved_snapshot_without_instantiating_session(monkeypatch, tmp_path):
    from app.main import app
    from app.routers import agent as agent_router

    home = tmp_path / "home"
    home.mkdir()
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    store_dir = tmp_path / "store"
    sessions_dir = store_dir / "sessions"
    sessions_dir.mkdir(parents=True)

    sid = "abc12345"
    saved = {
        "id": sid,
        "provider": "openai",
        "model": "gpt-4o",
        "mode": "edits",
        "title": "Saved task",
        "workspace": str(ws_root),
        "sandbox": ["read-only", False],
        "created_at": 10.0,
        "updated_at": 20.0,
        "usage": {"prompt_tokens": 100, "cached_tokens": 50, "completion_tokens": 20},
        "todos": [{"content": "Step 1", "status": "completed"}],
        "events": [{"seq": 1, "type": "user", "text": "hello"}],
    }
    (sessions_dir / f"{sid}.json").write_text(json.dumps(saved), encoding="utf-8")

    mgr = AgentSessionManager(store_dir, home=home)
    create_calls = []
    key_calls = []

    def fail_create(*args, **kwargs):
        create_calls.append((args, kwargs))
        raise AssertionError("view_session must not instantiate AgentSession")

    def fail_keys(*args, **kwargs):
        key_calls.append((args, kwargs))
        raise AssertionError("view_session must not query provider API keys")

    monkeypatch.setattr(mgr, "create", fail_create)
    monkeypatch.setattr(agent_router, "agent_sessions", mgr)
    monkeypatch.setattr(agent_router, "get_provider_api_keys", fail_keys)
    monkeypatch.setattr(agent_router, "is_loopback", lambda host: host == "testclient")

    http = TestClient(app, base_url="http://127.0.0.1")
    head = {"X-Manga-Agent": "1"}
    resp = http.get(f"/api/agent/sessions/{sid}/view", headers=head)
    assert resp.status_code == 200, resp.text
    snap = resp.json()
    assert snap["id"] == sid
    assert snap["saved_only"] is True
    assert snap["provider_label"] == "OpenAI"
    assert snap["sandbox"]["mode"] == "read-only"
    assert snap["stats"]["cache_pct"] == 50
    assert snap["events"] == [{"seq": 1, "type": "user", "text": "hello"}]
    assert create_calls == []
    assert key_calls == []
    assert sid not in mgr.sessions

    missing = http.get("/api/agent/sessions/doesnotexist/view", headers=head)
    assert missing.status_code == 404


def test_frontend_uses_model_cache_parallel_open_and_fragment_batching():
    root = Path(__file__).resolve().parents[1]
    api_js = (root / "app" / "static" / "js" / "api.js").read_text(encoding="utf-8")
    ai_mode_js = (root / "app" / "static" / "js" / "ai-mode.js").read_text(encoding="utf-8")
    agent_js = (root / "app" / "static" / "js" / "agent.js").read_text(encoding="utf-8")

    assert "window.aiModelCache" in api_js
    assert "window.aiModelCache" in ai_mode_js
    assert "window.aiModelCache" in agent_js
    assert "Promise.all" in agent_js
    assert "createDocumentFragment()" in agent_js
    assert "/view`" in agent_js
    assert "hydrateSession" in agent_js
