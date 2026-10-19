import os
import time

import pytest

from app.agent import session as session_module
from app.agent.session import SessionManager, compact
from app.agent.tools import Workspace


def test_tools_stay_inside_the_folder_and_edit_exact_text(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.run("write_file", {"path": "a/b.txt", "content": "one\ntwo\n"}).startswith("Wrote")
    assert "     2\ttwo" in ws.run("read_file", {"path": "a/b.txt"})
    assert ws.run("edit_file", {"path": "a/b.txt", "old_text": "two", "new_text": "three"}) == "Edited a/b.txt"
    assert (tmp_path / "a/b.txt").read_text() == "one\nthree\n"
    assert "exactly once" in ws.run("edit_file", {"path": "a/b.txt", "old_text": "zzz", "new_text": "x"})
    assert "outside the folder" in ws.run("read_file", {"path": "../secret"})
    assert "no tool named" in ws.run("delete_everything", {})


@pytest.mark.skipif(os.name == "nt", reason="the command below is POSIX shell")
def test_commands_run_in_the_folder_without_secret_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "hidden")
    out = Workspace(tmp_path).run("run_command", {"command": "pwd; echo ${SOME_API_KEY:-none}; exit 3"})
    assert str(tmp_path.resolve()) in out and "none" in out and "[exit code 3]" in out


class FakeModel:
    """Replies with the queued answers, recording what the model was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, provider, model, api_key, messages, tools, effort=""):
        self.seen.append(messages)
        return self.replies.pop(0)


def call(name, **args):
    return {"content": "", "calls": [{"id": f"c{time.monotonic_ns()}", "name": name, "args": args}]}


def done(text="Done."):
    return {"content": text, "calls": []}


def wait_for(session, *states):
    for _ in range(200):
        if session.data["status"] in states:
            return
        time.sleep(0.02)
    raise AssertionError(f"status stayed {session.data['status']}")


@pytest.fixture
def manager(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    return SessionManager(tmp_path / "sessions", lambda provider_id: (object(), "key")), work


def test_ask_mode_waits_for_approval_and_a_refusal_reaches_the_model(manager, monkeypatch):
    sessions, work = manager
    model = FakeModel([call("write_file", path="x.txt", content="hi"), done()])
    monkeypatch.setattr(session_module.client, "chat", model)
    s = sessions.create("p", "m", str(work), "ask")
    s.send("make x.txt")
    wait_for(s, "waiting")
    assert s.data["pending"]["name"] == "write_file"
    s.approve(False, "not now")
    wait_for(s, "idle")
    assert not (work / "x.txt").exists()
    assert "not now" in model.seen[1][-1]["content"]


def test_edits_mode_writes_freely_but_asks_before_a_command(manager, monkeypatch):
    sessions, work = manager
    model = FakeModel([call("write_file", path="x.txt", content="hi"), call("run_command", command="ls"), done()])
    monkeypatch.setattr(session_module.client, "chat", model)
    s = sessions.create("p", "m", str(work), "edits")
    s.send("go")
    wait_for(s, "waiting")
    assert (work / "x.txt").read_text() == "hi"
    assert s.data["pending"]["name"] == "run_command"
    s.approve(True)
    wait_for(s, "idle")
    assert "x.txt" in s.data["messages"][-2]["content"]
    assert s.data["messages"][-1] == {"role": "assistant", "content": "Done."}


def test_sessions_are_saved_without_the_key_and_load_again(manager, monkeypatch):
    sessions, work = manager
    (work / "AGENTS.md").write_text("Always answer in Vietnamese.")
    model = FakeModel([done("Xong.")])
    monkeypatch.setattr(session_module.client, "chat", model)
    s = sessions.create("p", "m", str(work))
    s.send("hello")
    wait_for(s, "idle")
    assert "Always answer in Vietnamese." in model.seen[0][0]["content"]
    saved = (sessions.folder / f"{s.id}.json").read_text()
    assert "Xong." in saved and '"key"' not in saved
    fresh = SessionManager(sessions.folder, sessions.resolve)
    assert fresh.get(s.id).data["messages"][-1]["content"] == "Xong."
    assert [item["id"] for item in fresh.list()] == [s.id]


def test_stop_ends_a_turn_waiting_for_approval(manager, monkeypatch):
    sessions, work = manager
    monkeypatch.setattr(session_module.client, "chat", FakeModel([call("run_command", command="ls")]))
    s = sessions.create("p", "m", str(work))
    s.send("go")
    wait_for(s, "waiting")
    s.stop()
    wait_for(s, "idle")
    assert "stopped" in s.data["messages"][-1]["content"]


def test_compact_shortens_old_tool_results_only(monkeypatch):
    monkeypatch.setattr(session_module, "COMPACT_CHARS", 10_000)
    messages = [{"role": "tool", "tool_call_id": str(i), "content": "x" * 5000} for i in range(20)]
    compact(messages)
    assert len(messages[0]["content"]) < 2100
    assert all(len(m["content"]) == 5000 for m in messages[-session_module.KEEP_RECENT:])


def test_agent_api_needs_its_header():
    from fastapi.testclient import TestClient

    from app.main import app

    assert TestClient(app).get("/api/agent/sessions").status_code == 403
