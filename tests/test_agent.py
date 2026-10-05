import json
import time

import pytest
from fastapi.testclient import TestClient

from app.agent import client
from app.agent.session import AgentSessionManager
from app.agent.tools import ToolError, Workspace
from app.ai_providers import PROVIDERS
from app.routers.agent import is_loopback


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def f():\n    return 1\n\nx = f()\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "big.txt").write_text("needle\n", encoding="utf-8")
    return Workspace(tmp_path)


def test_paths_stay_inside_the_workspace(ws, tmp_path):
    with pytest.raises(ToolError):
        ws.resolve("../outside.txt")
    with pytest.raises(ToolError):
        ws.resolve(str(tmp_path.parent / "x"))
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ToolError):
        ws.run("read_file", {"path": "link/anything"})


def test_list_read_and_search_skip_the_app_data_folder(ws):
    assert "pkg/" in ws.run("list_dir", {}) and "data/" not in ws.run("list_dir", {})
    assert "     2\t    return 1" in ws.run("read_file", {"path": "pkg/a.py", "offset": 2, "limit": 1})
    assert ws.run("search", {"pattern": "needle"}) == "No matches"
    assert ws.run("search", {"pattern": "return", "glob": "*.py"}) == "pkg/a.py:2: return 1"


def test_edit_needs_one_exact_match(ws):
    with pytest.raises(ToolError, match="not found"):
        ws.run("edit_file", {"path": "pkg/a.py", "old_text": "nope", "new_text": "x"})
    ws.run("write_file", {"path": "pkg/b.py", "content": "a\na\n"})
    with pytest.raises(ToolError, match="2 places"):
        ws.run("edit_file", {"path": "pkg/b.py", "old_text": "a", "new_text": "b"})
    ws.run("edit_file", {"path": "pkg/b.py", "old_text": "a", "new_text": "b", "replace_all": True})
    assert ws.run("read_file", {"path": "pkg/b.py"}).count("b") == 2


def test_commands_run_in_the_root_and_report_the_exit_code(ws):
    out = ws.run("run_command", {"command": "python -c \"import os; print(os.listdir('pkg'))\""})
    assert "a.py" in out and "[exit code 0]" in out
    assert "[stopped after 1 s]" in ws.run("run_command", {"command": "python -c \"import time; time.sleep(5)\"", "timeout": 1})


def test_text_tool_calls_parse_and_render_back():
    text, calls = client.parse_text_calls('Đọc file.\n<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>')
    assert text == "Đọc file." and calls[0]["name"] == "read_file" and calls[0]["args"] == {"path": "a"}
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "calls": calls},
               {"role": "tool", "id": calls[0]["id"], "name": "read_file", "content": "body"}]
    native = client.render(history, "sys", False)
    assert native[2]["tool_calls"][0]["function"]["name"] == "read_file" and native[3]["role"] == "tool"
    plain = client.render(history, "sys", True)
    assert "<tool_call>" in plain[2]["content"] and plain[3]["role"] == "user" and "<tool_result" in plain[3]["content"]
    assert "read_file" in plain[0]["content"]


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers = status, body, {}
        self.ok = status < 400
        self.text = json.dumps(body)

    def json(self):
        return self._body


def test_native_calls_and_the_tools_refusal(monkeypatch):
    monkeypatch.setattr(client, "validate_url", lambda url: url)
    sent = []
    reply = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "list_dir", "arguments": "{\"path\": \".\"}"}}]}}], "usage": {"prompt_tokens": 5}}
    monkeypatch.setattr(client.requests, "post", lambda url, **kw: sent.append((url, kw["json"])) or FakeResponse(200, reply))
    turn = client.complete(PROVIDERS["openai"], "k", "m", [], use_tools=True)
    assert turn["calls"] == [{"id": "c1", "name": "list_dir", "args": {"path": "."}}] and "tools" in sent[0][1]
    client.complete(PROVIDERS["gemini"], "k", "m", [], use_tools=False)
    assert sent[1][0].endswith("/openai/chat/completions") and "tools" not in sent[1][1]
    replies = [FakeResponse(429, {"error": {"message": "slow down"}}), FakeResponse(200, reply)]
    waits = []
    monkeypatch.setattr(client.time, "sleep", waits.append)
    monkeypatch.setattr(client.requests, "post", lambda url, **kw: replies.pop(0))
    assert client.complete(PROVIDERS["openai"], "k", "m", [], use_tools=True)["calls"][0]["name"] == "list_dir"
    assert waits == [6.0], "a rate-limited request waits and is sent again"
    monkeypatch.setattr(client.requests, "post", lambda url, **kw: FakeResponse(400, {"error": {"message": "tools are not supported"}}))
    with pytest.raises(client.ToolsUnsupported):
        client.complete(PROVIDERS["openai"], "k", "m", [], use_tools=True)


def scripted(*turns):
    """A fake model that plays the given turns in order and records what it was sent."""
    seen = []

    def complete(provider, key, model, messages, *, use_tools):
        seen.append((messages, use_tools))
        return turns[len(seen) - 1]

    complete.seen = seen
    return complete


def turn(text="", calls=()):
    return {"text": text, "calls": list(calls), "reasoning": "", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def wait_for(session, status, timeout=5.0):
    end = time.time() + timeout
    while session.status != status and time.time() < end:
        time.sleep(0.01)
    assert session.status == status


def test_ask_mode_waits_for_approval_then_writes(ws):
    fake = scripted(turn(calls=[{"id": "w", "name": "write_file", "args": {"path": "new.txt", "content": "hi"}}]), turn("Xong."))
    session = AgentSessionManager().create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    session.send("tạo file")
    wait_for(session, "waiting")
    assert session.pending["name"] == "write_file" and not (ws.root / "new.txt").exists()
    session.decide("allow")
    wait_for(session, "idle")
    assert (ws.root / "new.txt").read_text() == "hi"
    assert [e["type"] for e in session.events] == ["user", "assistant", "approval", "tool", "assistant", "done"]
    assert session.usage == {"prompt_tokens": 2, "completion_tokens": 2}


def test_denied_calls_reach_the_model_and_edits_mode_runs_edits_alone(ws):
    fake = scripted(turn(calls=[{"id": "r", "name": "run_command", "args": {"command": "echo hi"}}]), turn("Ok."))
    session = AgentSessionManager().create(PROVIDERS["openai"], "k", "m", ws, "edits", complete=fake)
    session.send("chạy")
    wait_for(session, "waiting")
    session.decide("deny", "không cần")
    wait_for(session, "idle")
    assert "refused" in session.history[2]["content"] and "không cần" in session.history[2]["content"]
    fake2 = scripted(turn(calls=[{"id": "e", "name": "edit_file", "args": {"path": "pkg/a.py", "old_text": "1", "new_text": "2"}}]), turn("Ok."))
    session.complete = fake2
    session.send("sửa")
    wait_for(session, "idle")
    assert "return 2" in (ws.root / "pkg" / "a.py").read_text()


def test_a_refused_tools_field_switches_the_session_to_text_calls(ws):
    calls = []

    def complete(provider, key, model, messages, *, use_tools):
        calls.append(use_tools)
        if use_tools:
            raise client.ToolsUnsupported("tools not supported")
        return turn("Chào.")

    session = AgentSessionManager().create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=complete)
    session.send("hi")
    wait_for(session, "idle")
    assert calls == [True, False] and session.text_tools
    assert any(e["type"] == "notice" for e in session.events)


@pytest.fixture
def api(monkeypatch, ws):
    from app.main import app
    from app.routers import agent

    manager = AgentSessionManager()
    fake = scripted(turn("Chào bạn."))
    real_create = manager.create
    monkeypatch.setattr(manager, "create", lambda *a, **kw: real_create(*a, **{**kw, "complete": fake}))
    monkeypatch.setattr(agent, "agent_sessions", manager)
    monkeypatch.setattr(agent, "get_provider_api_key", lambda *a, **kw: "secret")
    # The test client calls from the host name "testclient", which stands in for this machine here.
    monkeypatch.setattr(agent, "is_loopback", lambda host: host == "testclient")
    return TestClient(app, base_url="http://127.0.0.1"), str(ws.root)


def test_the_api_needs_the_agent_header_this_machine_and_the_switch(api, monkeypatch):
    http, root = api
    assert http.get("/api/agent/config").status_code == 403
    assert http.get("/api/agent/config", headers={"X-Manga-Agent": "1"}).status_code == 200
    assert is_loopback("127.0.0.1") and is_loopback("::1") and not is_loopback("192.168.1.20")
    monkeypatch.setattr("app.routers.agent.is_loopback", is_loopback)
    assert http.get("/api/agent/config", headers={"X-Manga-Agent": "1"}).status_code == 403, "a caller off this machine is refused"
    monkeypatch.setenv("MANGA_AGENT_MODE", "0")
    assert http.get("/api/agent/config", headers={"X-Manga-Agent": "1"}).status_code == 404


def test_a_session_runs_through_the_api(api):
    http, root = api
    head = {"X-Manga-Agent": "1"}
    created = http.post("/api/agent/sessions", json={"provider": "openai", "model": "m", "workspace": root}, headers=head)
    assert created.status_code == 200, created.text
    sid = created.json()["id"]
    assert http.post(f"/api/agent/sessions/{sid}/messages", json={"text": "hi"}, headers=head).status_code == 200
    for _ in range(200):
        snap = http.get(f"/api/agent/sessions/{sid}?after=0", headers=head).json()
        if snap["status"] == "idle" and snap["events"][-1]["type"] == "done":
            break
        time.sleep(0.01)
    assert [e["text"] for e in snap["events"] if e["type"] == "assistant"] == ["Chào bạn."]
    assert "secret" not in json.dumps(snap)
    assert http.post("/api/agent/sessions", json={"provider": "openai", "model": "m", "workspace": root + "/nope"},
                     headers=head).status_code == 400
    assert http.delete(f"/api/agent/sessions/{sid}", headers=head).status_code == 200
