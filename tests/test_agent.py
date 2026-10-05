import json
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import time

import pytest
from fastapi.testclient import TestClient

from app.agent import client, context, mcp, patch, sandbox, skills
from app.agent.session import AgentSessionManager
from app.agent.tools import ToolError, Workspace
from app.ai_providers import PROVIDERS
from app.routers.agent import is_loopback

LANDLOCK = sandbox.backend() == "landlock"


@pytest.fixture
def home(tmp_path):
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def f():\n    return 1\n\nx = f()\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "big.txt").write_text("needle\n", encoding="utf-8")
    return Workspace(root)


def manager(home, store=None):
    return AgentSessionManager(store, home=home)


# Workspace tools.

def test_paths_stay_inside_the_workspace(ws, tmp_path):
    with pytest.raises(ToolError):
        ws.resolve("../outside.txt")
    with pytest.raises(ToolError):
        ws.resolve(str(tmp_path / "x"))
    (ws.root / "link").symlink_to(tmp_path)
    with pytest.raises(ToolError):
        ws.run("read_file", {"path": "link/anything"})


def test_list_read_search_and_glob_skip_the_app_data_folder(ws):
    assert "pkg/" in ws.run("list_dir", {}) and "data/" not in ws.run("list_dir", {})
    assert "     2\t    return 1" in ws.run("read_file", {"path": "pkg/a.py", "offset": 2, "limit": 1})
    assert ws.run("search", {"pattern": "needle"}) == "No matches"
    assert ws.run("search", {"pattern": "return", "glob": "*.py"}) == "pkg/a.py:2: return 1"
    assert ws.run("glob", {"pattern": "**/*.py"}) == "pkg/a.py" and ws.run("glob", {"pattern": "*.txt"}) == "No files"


def test_edit_needs_one_exact_match(ws):
    with pytest.raises(ToolError, match="not found"):
        ws.run("edit_file", {"path": "pkg/a.py", "old_text": "nope", "new_text": "x"})
    ws.run("write_file", {"path": "pkg/b.py", "content": "a\na\n"})
    with pytest.raises(ToolError, match="2 places"):
        ws.run("edit_file", {"path": "pkg/b.py", "old_text": "a", "new_text": "b"})
    ws.run("edit_file", {"path": "pkg/b.py", "old_text": "a", "new_text": "b", "replace_all": True})
    assert ws.run("read_file", {"path": "pkg/b.py"}).count("b") == 2


def test_read_only_sessions_refuse_edits(ws):
    ws.policy = sandbox.Policy("read-only")
    with pytest.raises(ToolError, match="read-only"):
        ws.run("write_file", {"path": "x.txt", "content": "x"})


def test_apply_patch_adds_updates_moves_and_deletes(ws):
    ws.run("write_file", {"path": "gone.txt", "content": "bye\n"})
    summary = ws.run("apply_patch", {"patch": textwrap.dedent("""\
        *** Begin Patch
        *** Add File: pkg/new.py
        +print("hi")
        *** Update File: pkg/a.py
        *** Move to: pkg/b.py
        @@ def f():
        -    return 1
        +    return 2
        *** Delete File: gone.txt
        *** End Patch""")})
    assert summary.splitlines() == ["A pkg/new.py", "M pkg/a.py -> pkg/b.py", "D gone.txt"]
    assert (ws.root / "pkg" / "b.py").read_text() == "def f():\n    return 2\n\nx = f()\n"
    assert not (ws.root / "pkg" / "a.py").exists() and not (ws.root / "gone.txt").exists()


def test_a_patch_that_does_not_match_changes_nothing(ws):
    bad = "*** Begin Patch\n*** Add File: c.py\n+x\n*** Update File: pkg/a.py\n-    return 9\n+    return 2\n*** End Patch"
    with pytest.raises(ToolError, match="not found"):
        ws.run("apply_patch", {"patch": bad})
    assert not (ws.root / "c.py").exists(), "a patch is checked whole before anything is written"
    with pytest.raises(patch.PatchError):
        patch.parse("*** Update File: x\n")


def test_commands_report_exit_codes_and_time_out(ws):
    ws.policy = sandbox.Policy("full-access", True)
    out = ws.run("run_command", {"command": f"{sys.executable} -c \"import os; print(os.listdir('pkg'))\""})
    assert "a.py" in out and "[exit code 0]" in out
    assert "[stopped after 1 s]" in ws.run("run_command", {"command": f"{sys.executable} -c \"import time; time.sleep(5)\"", "timeout": 1})


@pytest.mark.skipif(not LANDLOCK, reason="needs Landlock")
def test_the_sandbox_keeps_writes_in_the_workspace_and_the_network_off(ws, tmp_path):
    # The temp folder is writable on purpose, so the forbidden file goes beside the tests instead.
    folder = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
    try:
        check_sandbox(ws, folder / "outside.txt")
    finally:
        shutil.rmtree(folder)


def check_sandbox(ws, outside):
    py = sys.executable
    out = ws.run("run_command", {"command": f"echo ok > inside.txt && echo bad > {outside}; echo code=$?"})
    assert (ws.root / "inside.txt").read_text() == "ok\n" and not outside.exists() and "code=" in out
    net = ws.run("run_command", {"command": f"{py} -c \"import socket; socket.create_connection(('1.1.1.1', 443), 3)\""})
    assert "Permission" in net or "Errno" in net
    ws.policy = sandbox.Policy("read-only")
    ws.run("run_command", {"command": "echo no > inside.txt"})
    assert (ws.root / "inside.txt").read_text() == "ok\n"
    ws.run("run_command", {"command": f"echo yes > {outside}", "outside_sandbox": True})
    assert outside.read_text() == "yes\n", "an approved escape runs without the sandbox"


# Skills, instructions, commands, hooks.

def write_skill(folder, name, description, body="Do the thing."):
    path = folder / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: >\n  {description}\n---\n{body}\n", encoding="utf-8")
    (path / "scripts").mkdir()
    (path / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")


def test_skills_are_found_listed_and_loaded(ws, home):
    write_skill(ws.root / ".claude" / "skills", "pdf-tools", "Work with PDF files.")
    write_skill(home / ".agents" / "skills", "pdf-tools", "Shadowed user copy.")
    write_skill(home / ".codex" / "skills", "release", "Cut a release.")
    found = skills.discover(ws.root, home)
    assert set(found) == {"pdf-tools", "release"} and found["pdf-tools"].description == "Work with PDF files."
    assert "- release: Cut a release." in skills.catalog(found)
    loaded = skills.load(found["pdf-tools"])
    assert "Do the thing." in loaded and "scripts/run.sh" in loaded


def test_instructions_commands_and_hooks_are_read(ws, home):
    (ws.root / "AGENTS.md").write_text("Use tabs.", encoding="utf-8")
    (home / ".claude").mkdir()
    (home / ".claude" / "CLAUDE.md").write_text("Be brief.", encoding="utf-8")
    assert "Use tabs." in context.instructions(ws.root, home) and "Be brief." in context.instructions(ws.root, home)
    (ws.root / ".claude" / "commands").mkdir(parents=True)
    (ws.root / ".claude" / "commands" / "fix.md").write_text("---\ndescription: Fix an issue\n---\nFix issue $ARGUMENTS now.",
                                                              encoding="utf-8")
    found = context.commands(ws.root, home)
    assert found["fix"]["description"] == "Fix an issue"
    assert context.expand_command(found["fix"]["body"], "42") == "Fix issue 42 now."
    (ws.root / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"PostToolUse": [
        {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": "echo formatted >&2"}]}]}}), encoding="utf-8")
    hooks = context.hooks(ws.root, home)
    assert hooks["workspace"]["PostToolUse"][0]["command"] == "echo formatted >&2"
    assert context.hook_matches("Edit|Write", "edit_file") and not context.hook_matches("Bash", "edit_file")


# The model client.

def test_text_tool_calls_parse_and_render_back():
    text, calls = client.parse_text_calls('Đọc file.\n<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>')
    assert text == "Đọc file." and calls[0]["name"] == "read_file" and calls[0]["args"] == {"path": "a"}
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "calls": calls},
               {"role": "tool", "id": calls[0]["id"], "name": "read_file", "content": "body"}]
    specs = [{"name": "read_file", "description": "Read.", "parameters": {"type": "object", "properties": {}}}]
    native = client.render(history, "sys", False, specs)
    assert native[2]["tool_calls"][0]["function"]["name"] == "read_file" and native[3]["role"] == "tool"
    plain = client.render(history, "sys", True, specs)
    assert "<tool_call>" in plain[2]["content"] and plain[3]["role"] == "user" and "<tool_result" in plain[3]["content"]
    assert "read_file" in plain[0]["content"]


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers = status, body, {}
        self.ok = status < 400
        self.text = json.dumps(body)

    def json(self):
        return self._body


def test_native_calls_rate_limits_and_the_tools_refusal(monkeypatch):
    monkeypatch.setattr(client, "validate_url", lambda url: url)
    sent = []
    reply = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "list_dir", "arguments": "{\"path\": \".\"}"}}]}}], "usage": {"prompt_tokens": 5}}
    monkeypatch.setattr(client.requests, "post", lambda url, **kw: sent.append((url, kw["json"])) or FakeResponse(200, reply))
    spec = [{"name": "list_dir", "description": "", "parameters": {}}]
    turn = client.complete(PROVIDERS["openai"], "k", "m", [], tools=spec)
    assert turn["calls"] == [{"id": "c1", "name": "list_dir", "args": {"path": "."}}] and "tools" in sent[0][1]
    client.complete(PROVIDERS["gemini"], "k", "m", [], tools=None)
    assert sent[1][0].endswith("/openai/chat/completions") and "tools" not in sent[1][1]
    replies = [FakeResponse(429, {"error": {"message": "slow down"}}), FakeResponse(200, reply)]
    waits = []
    monkeypatch.setattr(client.time, "sleep", waits.append)
    monkeypatch.setattr(client.requests, "post", lambda url, **kw: replies.pop(0))
    assert client.complete(PROVIDERS["openai"], "k", "m", [], tools=spec)["calls"][0]["name"] == "list_dir"
    assert waits == [6.0], "a rate-limited request waits and is sent again"
    monkeypatch.setattr(client.requests, "post", lambda url, **kw: FakeResponse(400, {"error": {"message": "tools are not supported"}}))
    with pytest.raises(client.ToolsUnsupported):
        client.complete(PROVIDERS["openai"], "k", "m", [], tools=spec)


# The session loop.

def scripted(*turns):
    """A fake model that plays the given turns in order and records what it was sent."""
    seen = []

    def complete(provider, key, model, messages, *, tools):
        seen.append((messages, tools))
        step = turns[len(seen) - 1]
        return step(messages, tools) if callable(step) else step

    complete.seen = seen
    return complete


def turn(text="", calls=()):
    return {"text": text, "calls": list(calls), "reasoning": "", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def call(tool, **args):
    return {"id": f"{tool}-{len(args)}", "name": tool, "args": args}


def wait_for(session, status, timeout=10.0):
    end = time.time() + timeout
    while session.status != status and time.time() < end:
        time.sleep(0.01)
    assert session.status == status


def test_ask_mode_waits_for_approval_then_writes(ws, home):
    fake = scripted(turn(calls=[call("write_file", path="new.txt", content="hi")]), turn("Xong."))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    session.send("tạo file")
    wait_for(session, "waiting")
    assert session.pending["name"] == "write_file" and not (ws.root / "new.txt").exists()
    session.decide("allow")
    wait_for(session, "idle")
    assert (ws.root / "new.txt").read_text() == "hi"
    assert [e["type"] for e in session.events] == ["user", "assistant", "approval", "tool", "assistant", "done"]
    assert session.usage == {"prompt_tokens": 2, "completion_tokens": 2}


def test_edits_mode_runs_sandboxed_commands_alone_and_asks_to_leave_the_sandbox(ws, home, monkeypatch):
    monkeypatch.setattr(sandbox, "backend", lambda: "landlock")
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "edits")
    assert not session._needs_approval(call("run_command", command="pytest"))
    assert session._needs_approval(call("run_command", command="pip install x", outside_sandbox=True))
    assert not session._needs_approval(call("edit_file", path="a", old_text="a", new_text="b"))
    monkeypatch.setattr(sandbox, "backend", lambda: "none")
    assert session._needs_approval(call("run_command", command="pytest")), "without an OS sandbox commands ask"


def test_denied_calls_reach_the_model(ws, home):
    fake = scripted(turn(calls=[call("run_command", command="echo hi")]), turn("Ok."))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    session.send("chạy")
    wait_for(session, "waiting")
    session.decide("deny", "không cần")
    wait_for(session, "idle")
    assert "refused" in session.history[2]["content"] and "không cần" in session.history[2]["content"]


def test_a_refused_tools_field_switches_the_session_to_text_calls(ws, home):
    calls = []

    def complete(provider, key, model, messages, *, tools):
        calls.append(bool(tools))
        if tools:
            raise client.ToolsUnsupported("tools not supported")
        return turn("Chào.")

    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=complete)
    session.send("hi")
    wait_for(session, "idle")
    assert calls == [True, False] and session.text_tools
    assert any(e["type"] == "notice" for e in session.events)


def test_skills_todos_and_the_read_only_helper(ws, home):
    write_skill(ws.root / ".agents" / "skills", "style", "House style rules.", "Always use tabs.")
    helper_tools = []

    def helper_read(messages, tools):
        helper_tools.append({s["name"] for s in tools or []})
        return turn(calls=[call("read_file", path="pkg/a.py")])

    main = scripted(
        turn(calls=[call("skill", name="style"), call("todo_write", items=[{"content": "read", "status": "in_progress"}])]),
        turn(calls=[call("task", description="find f", prompt="What does f return?")]),
        helper_read, turn("f returns 1."),
        turn("Done."))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=main)
    assert "- style: House style rules." in session.system_prompt()
    session.send("go")
    wait_for(session, "idle")
    outputs = {e["name"]: e["output"] for e in session.events if e["type"] == "tool"}
    assert "Always use tabs." in outputs["skill"] and outputs["todo_write"] == "Plan saved: 0/1 done."
    assert outputs["task"] == "f returns 1." and session.todos == [{"content": "read", "status": "in_progress"}]
    assert helper_tools[0] and not ({"write_file", "run_command", "task", "todo_write"} & helper_tools[0])
    assert [e["state"] for e in session.events if e["type"] == "subagent"] == ["started", "done"]


def test_hooks_block_and_report_only_once_trusted(ws, home, tmp_path):
    (ws.root / ".agents").mkdir()
    (ws.root / ".agents" / "settings.json").write_text(json.dumps({"hooks": {
        "PreToolUse": [{"matcher": "Bash", "command": "echo 'no shell today' >&2; exit 2"}],
        "PostToolUse": [{"matcher": "Write", "command": "echo checked >&2"}]}}), encoding="utf-8")
    ws.policy = sandbox.Policy("full-access", True)
    fake = scripted(turn(calls=[call("run_command", command="echo hi"), call("write_file", path="x.txt", content="x")]), turn("ok"),
                    turn(calls=[call("run_command", command="echo hi"), call("write_file", path="y.txt", content="y")]), turn("ok"))
    session = manager(home, tmp_path / "store").create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    session.send("one")
    wait_for(session, "idle")
    first = [e["output"] for e in session.events if e["type"] == "tool"]
    assert "hi" in first[0] and "checked" not in first[1], "untrusted workspace hooks do not run"
    session.trust_hooks()
    session.send("two")
    wait_for(session, "idle")
    second = [e["output"] for e in session.events if e["type"] == "tool"][2:]
    assert second[0].startswith("Blocked by a hook") and "no shell today" in second[0]
    assert "checked" in second[1]


def test_compact_replaces_old_turns_with_a_summary(ws, home):
    fake = scripted(turn("first answer"), turn("SUMMARY: user asked twice"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    session.send("first")
    wait_for(session, "idle")
    session.history.append({"role": "user", "content": "second"})
    assert session.compact()
    assert session.history[0]["content"].startswith("[Summary of the earlier conversation]\nSUMMARY")
    assert session.history[-1] == {"role": "user", "content": "second"}


def test_slash_commands(ws, home):
    (ws.root / ".agents" / "commands").mkdir(parents=True)
    (ws.root / ".agents" / "commands" / "review.md").write_text("Review $1 carefully.", encoding="utf-8")
    fake = scripted(turn("reviewed"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    assert "/compact" in session.command("/help")["message"] and "/review" in session.command("/help")["message"]
    assert session.command("/mode ask")["message"] == "Quyền: ask" and session.mode == "ask"
    session.command("/sandbox read-only")
    assert session.workspace.policy == sandbox.Policy("read-only", False)
    session.command("/review pkg/a.py")
    wait_for(session, "idle")
    assert session.history[0]["content"] == "Review pkg/a.py carefully."
    with pytest.raises(ValueError):
        session.command("/nope")


# MCP.

FAKE_SERVER = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        if msg["method"] == "initialize":
            result = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}}
        elif msg["method"] == "tools/list":
            result = {"tools": [{"name": "add", "description": "Add two numbers", "annotations": {"readOnlyHint": True},
                                 "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}]}
        elif msg["method"] == "tools/call":
            a = msg["params"]["arguments"]
            result = {"content": [{"type": "text", "text": str(a["a"] + a["b"])}]}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no"}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
''')


def test_mcp_servers_need_trust_in_the_workspace_and_their_tools_run(ws, home, tmp_path):
    (tmp_path / "fake_mcp.py").write_text(FAKE_SERVER, encoding="utf-8")
    (ws.root / ".mcp.json").write_text(json.dumps({"mcpServers": {"calc": {"command": sys.executable,
                                                                           "args": [str(tmp_path / "fake_mcp.py")]}}}), encoding="utf-8")
    fake = scripted(turn(calls=[call("mcp__calc__add", a=2, b=3)]), turn("5"))
    agents = manager(home, tmp_path / "store")
    session = agents.create(PROVIDERS["openai"], "k", "m", ws, "edits", complete=fake)
    session._ensure_mcp()
    assert session.mcp_status["calc"]["state"] == "untrusted" and "mcp__calc__add" not in session.mcp_tools
    session.trust_mcp("calc")
    assert session.mcp_status["calc"]["state"] == "running" and session.mcp_status["calc"]["tools"] == 1
    assert not session._needs_approval(call("mcp__calc__add", a=1, b=1)), "a read-only MCP tool runs alone in edits mode"
    session.send("add")
    wait_for(session, "idle")
    assert [e["output"] for e in session.events if e["type"] == "tool"] == ["5"]
    again = agents.create(PROVIDERS["openai"], "k", "m", Workspace(ws.root), "edits")
    again._ensure_mcp()
    assert again.mcp_status["calc"]["state"] == "running", "trust is remembered for the same config"
    agents.close_all()
    assert mcp.tool_name("my server", "x" * 80) == ("mcp__my_server__" + "x" * 80)[:64]


def test_mcp_config_from_codex_and_claude_files(ws, home):
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('[mcp_servers.docs]\ncommand = "npx"\nargs = ["-y", "docs-mcp"]\n', encoding="utf-8")
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"web": {"type": "http", "url": "https://x.example/mcp"}}}),
                                       encoding="utf-8")
    rows = {r["name"]: r for r in mcp.configured(ws.root, home)}
    assert rows["docs"]["config"]["args"] == ["-y", "docs-mcp"] and rows["web"]["scope"] == "user"


# Persistence and the API.

@pytest.fixture
def api(monkeypatch, ws, home, tmp_path):
    from app.main import app
    from app.routers import agent

    agents = manager(home, tmp_path / "store")
    fake = scripted(turn("Chào bạn."), turn("Lại chào."))
    real_create = agents.create
    monkeypatch.setattr(agents, "create", lambda *a, **kw: real_create(*a, **{**kw, "complete": fake}))
    monkeypatch.setattr(agent, "agent_sessions", agents)
    monkeypatch.setattr(agent, "get_provider_api_key", lambda *a, **kw: "secret")
    # The test client calls from the host name "testclient", which stands in for this machine here.
    monkeypatch.setattr(agent, "is_loopback", lambda host: host == "testclient")
    return TestClient(app, base_url="http://127.0.0.1"), str(ws.root), agents


def test_the_api_needs_the_agent_header_this_machine_and_the_switch(api, monkeypatch):
    http, root, _ = api
    assert http.get("/api/agent/config").status_code == 403
    assert http.get("/api/agent/config", headers={"X-Manga-Agent": "1"}).status_code == 200
    assert is_loopback("127.0.0.1") and is_loopback("::1") and not is_loopback("192.168.1.20")
    monkeypatch.setattr("app.routers.agent.is_loopback", is_loopback)
    assert http.get("/api/agent/config", headers={"X-Manga-Agent": "1"}).status_code == 403, "a caller off this machine is refused"
    monkeypatch.setenv("MANGA_AGENT_MODE", "0")
    assert http.get("/api/agent/config", headers={"X-Manga-Agent": "1"}).status_code == 404


def poll_idle(http, sid, head):
    for _ in range(500):
        snap = http.get(f"/api/agent/sessions/{sid}?after=0", headers=head).json()
        if snap["status"] == "idle" and snap["events"] and snap["events"][-1]["type"] == "done":
            return snap
        time.sleep(0.01)
    raise AssertionError("session did not finish")


def test_a_session_runs_is_saved_and_resumes_through_the_api(api):
    http, root, agents = api
    head = {"X-Manga-Agent": "1"}
    created = http.post("/api/agent/sessions", json={"provider": "openai", "model": "m", "workspace": root,
                                                      "sandbox": "read-only"}, headers=head)
    assert created.status_code == 200, created.text
    assert created.json()["sandbox"]["mode"] == "read-only"
    sid = created.json()["id"]
    assert http.post(f"/api/agent/sessions/{sid}/messages", json={"text": "hi"}, headers=head).status_code == 200
    snap = poll_idle(http, sid, head)
    assert [e["text"] for e in snap["events"] if e["type"] == "assistant"] == ["Chào bạn."]
    assert "secret" not in json.dumps(snap)
    assert http.post(f"/api/agent/sessions/{sid}/messages", json={"text": "/help"}, headers=head).json()["message"]
    agents.sessions.pop(sid).close()
    listed = http.get("/api/agent/sessions", headers=head).json()["sessions"]
    assert listed[0]["id"] == sid and listed[0]["status"] == "saved" and listed[0]["title"] == "hi"
    resumed = http.post(f"/api/agent/sessions/{sid}/resume", headers=head)
    assert resumed.status_code == 200 and resumed.json()["sandbox"]["mode"] == "read-only"
    assert [e["type"] for e in resumed.json()["events"]][:2] == ["user", "assistant"]
    http.post(f"/api/agent/sessions/{sid}/messages", json={"text": "again"}, headers=head)
    poll_idle(http, sid, head)
    assert [h["role"] for h in agents.get(sid).history] == ["user", "assistant", "user", "assistant"]
    assert http.post("/api/agent/sessions", json={"provider": "openai", "model": "m", "workspace": root + "/nope"},
                     headers=head).status_code == 400
    assert http.delete(f"/api/agent/sessions/{sid}", headers=head).status_code == 200
    assert http.post(f"/api/agent/sessions/{sid}/resume", headers=head).status_code == 404
