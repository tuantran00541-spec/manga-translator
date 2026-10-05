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
    assert {n for n, k in found.items() if not k.builtin} == {"pdf-tools", "release"}
    assert found["pdf-tools"].description == "Work with PDF files."
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


# Harness v3: hashline edits, rules, undo, agents, queue, references, memory, plan, goal.

def run_to_idle(session, text="go"):
    session.send(text)
    wait_for(session, "idle")


def tool_outputs(session):
    return [(e["name"], e["ok"], e["output"]) for e in session.events if e["type"] == "tool"]


def test_hashline_edits_check_anchors_and_keep_line_endings(ws):
    (ws.root / "crlf.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    shown = ws.run("read_file", {"path": "crlf.txt", "anchors": True}).splitlines()
    anchor = lambda i: shown[i].split("|")[0]
    ws.run("edit_lines", {"path": "crlf.txt", "edits": [{"op": "replace", "anchor": anchor(1), "text": "TWO\nTWO-B"},
                                                          {"op": "insert_after", "anchor": anchor(2), "text": "four"}]})
    assert (ws.root / "crlf.txt").read_bytes() == b"one\r\nTWO\r\nTWO-B\r\nthree\r\nfour\r\n"
    with pytest.raises(ToolError, match="stale"):
        ws.run("edit_lines", {"path": "crlf.txt", "edits": [{"op": "delete", "anchor": anchor(1)}]})
    fresh = ws.run("read_file", {"path": "crlf.txt", "anchors": True}).splitlines()
    first, second = (line.split("|")[0] for line in fresh[:2])
    with pytest.raises(ToolError, match="same lines"):
        ws.run("edit_lines", {"path": "crlf.txt", "edits": [{"op": "replace", "anchor": first, "end": second, "text": "x"},
                                                              {"op": "delete", "anchor": second}]})
    assert (ws.root / "crlf.txt").read_bytes().startswith(b"one\r\nTWO")


def test_edits_report_syntax_errors_at_once(ws):
    assert "Syntax check failed" in ws.run("write_file", {"path": "bad.py", "content": "def f(:\n"})
    assert "Syntax check" not in ws.run("write_file", {"path": "good.py", "content": "x = 1\n"})
    assert "bad.json" in ws.run("write_file", {"path": "bad.json", "content": "{"})


def test_rules_allow_ask_and_deny_by_pattern(ws, home):
    (home / ".manga-agent").mkdir()
    (home / ".manga-agent" / "settings.json").write_text(json.dumps({"permission": {
        "bash": {"*": "ask", "echo *": "allow", "rm *": "deny"}, "edit": {"notes/*": "allow"}}}), encoding="utf-8")
    (ws.root / ".agents").mkdir()
    (ws.root / ".agents" / "settings.json").write_text(json.dumps({"permission": {"bash": {"git *": "allow"}, "read": {"secret/*": "deny"}}}), encoding="utf-8")
    fake = scripted(turn(calls=[call("run_command", command="echo hi"), call("run_command", command="echo a && rm -rf pkg"),
                                call("write_file", path="notes/a.txt", content="x"), call("read_file", path="secret/k.txt"),
                                call("read_file", path=".env")]), turn("ok"))
    (ws.root / ".env").write_text("KEY=1", encoding="utf-8")
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    session.send("go")
    wait_for(session, "idle")
    results = {i: o for i, (_, _, o) in enumerate(tool_outputs(session))}
    assert "hi" in results[0] and "Denied by a permission rule" in results[1] and (ws.root / "pkg").exists()
    assert (ws.root / "notes" / "a.txt").exists()
    assert "Denied" in results[3] and "Denied" in results[4]
    from app.agent import rules
    loaded = rules.load(ws.root, home)
    assert ("workspace", "bash", "git *", "allow") not in loaded and ("workspace", "read", "secret/*", "deny") in loaded


def test_claude_style_permission_lists_are_read(ws, home):
    from app.agent import rules
    (ws.root / ".claude").mkdir()
    (ws.root / ".claude" / "settings.json").write_text(json.dumps({"permissions": {"allow": ["Bash(git status:*)"], "deny": ["Bash(rm:*)", "Read(.secrets/*)"]}}), encoding="utf-8")
    loaded = rules.load(ws.root, home)
    assert ("workspace", "bash", "rm*", "deny") in loaded and ("workspace", "read", ".secrets/*", "deny") in loaded
    assert not any(r[3] == "allow" and r[0] == "workspace" for r in loaded)


def test_the_same_call_three_times_in_a_row_is_blocked(ws, home):
    same = call("list_dir")
    fake = scripted(turn(calls=[same]), turn(calls=[same]), turn(calls=[same]), turn("stuck"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    outputs = [o for _, _, o in tool_outputs(session)]
    assert "pkg/" in outputs[0] and outputs[2].startswith("Blocked")


def test_undo_restores_edited_files_and_removes_new_ones(ws, home):
    fake = scripted(turn(calls=[call("write_file", path="pkg/a.py", content="changed\n"), call("write_file", path="new.txt", content="n")]),
                    turn("done"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    assert (ws.root / "new.txt").exists()
    assert "pkg/a.py" in session.command("/undo")["message"]
    assert "return 1" in (ws.root / "pkg" / "a.py").read_text() and not (ws.root / "new.txt").exists()
    assert session.history[-2]["content"].startswith("[The user undid")
    assert "Không có gì" in session.command("/undo")["message"]


def test_agent_files_limit_tools_and_a_coder_helper_edits_through_the_parents_approval(ws, home):
    (ws.root / ".agents" / "agents").mkdir(parents=True)
    (ws.root / ".agents" / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews code.\ntools: [Read, Grep]\nmodel: tiny\n---\nYou review code.", encoding="utf-8")
    seen = []

    def reviewer(messages, tools):
        seen.append(({s["name"] for s in tools}, messages[0]["content"]))
        return turn("looks fine")

    def coder(messages, tools):
        return turn(calls=[call("write_file", path="made.txt", content="by coder")])

    fake = scripted(turn(calls=[call("task", description="r", prompt="review", agent="reviewer")]), reviewer,
                    turn(calls=[call("task", description="c", prompt="make a file", agent="coder")]), coder,
                    turn("child done"), turn("all done"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    session.send("go")
    wait_for(session, "waiting")
    assert session.pending["name"] == "write_file" and not (ws.root / "made.txt").exists()
    session.decide("allow")
    wait_for(session, "idle")
    assert seen[0][0] == {"read_file", "search"} and "You review code." in seen[0][1]
    assert (ws.root / "made.txt").read_text() == "by coder"
    assert any(a["name"] == "reviewer" for a in session.snapshot()["agents"])
    session.command("/undo")
    assert not (ws.root / "made.txt").exists()


def test_several_task_calls_in_one_reply_run_together(ws, home):
    import threading
    gate = threading.Barrier(2, timeout=5)

    def helper(messages, tools):
        gate.wait()
        return turn(f"report {messages[1]['content']}")

    fake = scripted(turn(calls=[{"id": "t1", "name": "task", "args": {"description": "a", "prompt": "A"}},
                                {"id": "t2", "name": "task", "args": {"description": "b", "prompt": "B"}}]),
                    helper, helper, turn("merged"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    assert sorted(o for _, _, o in tool_outputs(session)) == ["report A", "report B"]


def test_a_message_sent_while_the_agent_works_joins_the_next_step(ws, home):
    import threading
    release, started = threading.Event(), threading.Event()

    def slow(messages, tools):
        started.set()
        release.wait(5)
        return turn(calls=[call("list_dir")])

    fake = scripted(slow, lambda m, t: turn("seen " + m[-1]["content"]))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    session.send("first")
    assert started.wait(5) and session.send("second, change course") is True
    release.set()
    wait_for(session, "idle")
    assert fake.seen[1][0][-1]["content"] == "second, change course"
    assert [e["queued"] for e in session.events if e["type"] == "user"] == [False, True]


def test_at_references_attach_files_but_not_secrets(ws, home):
    (ws.root / ".env").write_text("KEY=1", encoding="utf-8")
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=scripted(turn("ok")))
    run_to_idle(session, "look at @pkg/a.py, and @.env and @missing.txt")
    sent = session.history[0]["content"]
    assert '<attached path="pkg/a.py">' in sent and "return 1" in sent and "KEY=1" not in sent and "missing" not in sent.split("</attached>")[-1]
    assert session.events[0]["text"] == "look at @pkg/a.py, and @.env and @missing.txt" and session.events[0]["refs"] == ["pkg/a.py"]


def test_memory_notes_persist_into_the_next_session(ws, home):
    fake = scripted(turn(calls=[call("memory", action="add", scope="project", text="Tests run with pytest -q")]), turn("noted"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    later = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=scripted(turn("hi")))
    assert "1. Tests run with pytest -q" in later.system_prompt()
    assert "pytest" in later.command("/memory")["message"]
    later.command("/memory rm project 1")
    assert "Tests run" not in later.system_prompt()


def test_plan_mode_is_read_only_until_the_plan_is_approved(ws, home):
    names = []

    def planning(messages, tools):
        names.append({s["name"] for s in tools})
        return turn(calls=[call("exit_plan_mode", plan="1. change a.py")])

    def building(messages, tools):
        names.append({s["name"] for s in tools})
        return turn(calls=[call("write_file", path="out.txt", content="built")])

    fake = scripted(planning, building, turn("done"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "edits", complete=fake)
    session.command("/plan")
    session.send("add a feature")
    wait_for(session, "waiting")
    assert session.pending["name"] == "exit_plan_mode" and session.pending["args"]["plan"] == "1. change a.py"
    assert "write_file" not in names[0] and "exit_plan_mode" in names[0] and "PLAN MODE" in fake.seen[0][0][0]["content"]
    session.decide("allow")
    wait_for(session, "idle")
    assert "write_file" in names[1] and "exit_plan_mode" not in names[1] and (ws.root / "out.txt").exists()


def test_ask_user_returns_the_answer(ws, home):
    fake = scripted(turn(calls=[call("ask_user", question="Which one?", options=["a", "b"])]), turn("ok"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    session.send("go")
    wait_for(session, "waiting")
    assert session.pending["args"]["options"] == ["a", "b"]
    session.decide("allow", "b")
    wait_for(session, "idle")
    assert tool_outputs(session)[0][2] == "The user answered: b"


def test_a_goal_keeps_the_agent_going_until_it_calls_goal_done(ws, home):
    fake = scripted(turn("step one"), turn("still going"), turn(calls=[call("goal_done", summary="all done")]), turn("final"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    session.command("/goal fix everything")
    wait_for(session, "idle")
    assert session.goal is None and len(fake.seen) == 4
    assert "goal_done" in {s["name"] for s in fake.seen[0][1]} and "Goal: fix everything" in fake.seen[0][0][1]["content"]
    assert sum("not marked done" in m.get("content", "") for m in session.history) == 2


def test_text_tool_calls_survive_the_shapes_open_models_produce():
    text, calls = client.parse_text_calls('Plan.\n<tool_call>\n{"name": "exit_plan_mode", "arguments": {"plan": "1. x"}}')
    assert text == "Plan." and calls[0]["name"] == "exit_plan_mode" and calls[0]["args"] == {"plan": "1. x"}
    _, calls = client.parse_text_calls("<tool_call>\n<function=ask_user>\n<parameter=question>\nWhich?\n</parameter>\n</function>\n</tool_call>")
    assert calls[0]["name"] == "ask_user" and calls[0]["args"] == {"question": "Which?"}
    mangled = '<tool_call>{"name": "task", "arguments": "\n<tool_call>\n{"name": "task", "arguments": {"prompt": "go"}}</tool_call>'
    _, calls = client.parse_text_calls(mangled)
    assert [(c["name"], c["args"]) for c in calls if not c.get("error")] == [("task", {"prompt": "go"})]


def test_rules_also_catch_shell_commands_that_name_a_denied_path():
    from app.agent import rules
    deny = [("default", "read", ".env", "deny"), ("user", "edit", "secret/*", "deny"), ("user", "edit", "*", "allow")]
    path_of = lambda p: p
    for command in ("cat .env", "echo hi > secret/key.txt", "tee secret/a.txt < x"):
        assert rules.check(deny, "run_command", {"command": command}, path_of) == "deny", command
    assert rules.check(deny, "run_command", {"command": "ls -la"}, path_of) is None


# Codex-style helper agents: spawn, wait, send_input, close.

def routed(parent_turns, child_reply):
    """One fake model for a session and its helpers: helpers are told apart by having no spawn_agent tool."""
    state = {"parent": 0}
    lock = __import__("threading").Lock()

    def complete(provider, key, model, messages, *, tools):
        if "spawn_agent" not in {t["name"] for t in tools}:
            return child_reply(messages, tools) if callable(child_reply) else turn(child_reply)
        with lock:
            step = parent_turns[min(state["parent"], len(parent_turns) - 1)]
            state["parent"] += 1
        return step(messages, tools) if callable(step) else step

    return complete


def test_spawn_wait_and_the_duplicate_and_thread_guards(ws, home):
    spawn = lambda m, **kw: {"id": f"s-{m}", "name": "spawn_agent", "args": {"message": m, **kw}}
    fake = routed([turn(calls=[spawn("look at A"), spawn("look at B"), spawn("look at A")]),
                   turn(calls=[{"id": "w", "name": "wait_agent", "args": {}}]), turn("both seen")],
                  lambda messages, tools: turn("report for " + messages[1]["content"]))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    out = [(n, ok, o) for n, ok, o in tool_outputs(session)]
    assert out[0][1] and "ash" in out[0][2] and out[1][1] and "birch" in out[1][2]
    assert not out[2][1] and "already has this exact job" in out[2][2]
    waited = out[3][2]
    assert "ash (explore): completed\nreport for look at A" in waited and "birch (explore): completed\nreport for look at B" in waited
    assert session.usage["prompt_tokens"] >= 3


def test_too_many_running_agents_are_refused(ws, home):
    import threading
    release = threading.Event()
    spawns = [{"id": f"s{i}", "name": "spawn_agent", "args": {"message": f"job {i}"}} for i in range(8)]

    def slow_child(messages, tools):
        release.wait(5)
        return turn("done")

    fake = routed([turn(calls=spawns), lambda m, t: (release.set(), turn("ok"))[1]], slow_child)
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    refused = [o for _, ok, o in tool_outputs(session) if not ok]
    assert len(refused) == 2 and "limit 6" in refused[0]


def test_a_helper_that_finishes_after_the_final_answer_reports_by_itself(ws, home):
    fake = routed([turn(calls=[{"id": "s", "name": "spawn_agent", "args": {"message": "slow job"}}]),
                   turn("I started it and have nothing else."),
                   lambda messages, tools: turn("The helper said: " + messages[-1]["content"])],
                  lambda messages, tools: (time.sleep(0.5), turn("slow result"))[1])
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    notes = [m["content"] for m in session.history if m["role"] == "user" and m["content"].startswith("<subagent_notification>")]
    assert len(notes) == 1 and '"result": "slow result"' in notes[0]
    assert session.events[-2]["text"].startswith("The helper said: <subagent_notification>")


def test_send_input_wakes_a_finished_helper_and_close_agent_releases_it(ws, home):
    fake = routed([turn(calls=[{"id": "s", "name": "spawn_agent", "args": {"message": "first"}}]),
                   turn(calls=[{"id": "w1", "name": "wait_agent", "args": {"ids": ["ash"]}}]),
                   turn(calls=[{"id": "i", "name": "send_input", "args": {"id": "ash", "message": "second"}}]),
                   turn(calls=[{"id": "w2", "name": "wait_agent", "args": {"ids": ["ash"]}}]),
                   turn(calls=[{"id": "c", "name": "close_agent", "args": {"id": "ash"}},
                               {"id": "bad", "name": "wait_agent", "args": {"ids": ["ash"]}}]),
                   turn("done")],
                  lambda messages, tools: turn("answer to " + messages[-1]["content"]))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    outputs = [o for _, _, o in tool_outputs(session)]
    assert "answer to first" in outputs[1] and "answer to second" in outputs[3]
    assert outputs[4].startswith("Closed ash") and "No open agent named 'ash'" in outputs[5]


def test_a_helpers_edit_asks_the_user_even_while_the_parent_waits(ws, home):
    fake = routed([turn(calls=[{"id": "s", "name": "spawn_agent", "args": {"message": "make it", "agent": "coder"}},
                               ]),
                   turn(calls=[{"id": "w", "name": "wait_agent", "args": {}}]), turn("finished")],
                  lambda messages, tools: turn("wrote it") if any(m["role"] == "tool" for m in messages)
                  else turn(calls=[call("write_file", path="helper.txt", content="hi")]))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    session.send("go")
    wait_for(session, "waiting")
    assert session.pending["name"] == "write_file" and session.pending["agent"] == "ash"
    session.decide("allow")
    wait_for(session, "idle")
    assert (ws.root / "helper.txt").read_text() == "hi"


def test_a_tool_call_with_raw_newlines_inside_a_string_is_still_read():
    raw = '<tool_call>{"name": "exit_plan_mode", "arguments": {"plan": "1. a\n2. b"}}</tool_call>'
    _, calls = client.parse_text_calls(raw)
    assert calls[0]["name"] == "exit_plan_mode" and calls[0]["args"] == {"plan": "1. a\n2. b"}


def test_a_model_that_only_writes_unreadable_calls_is_stopped(ws, home):
    junk = turn(calls=[{"id": "x", "name": "", "args": {}, "error": "Unreadable tool call: {"}])
    fake = scripted(*[junk] * 10)
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    run_to_idle(session)
    assert len(fake.seen) == 4 and any(e["type"] == "error" for e in session.events)


# Bundled skills and the GitHub installer.

BUNDLED = {"systematic-debugging", "verification-before-completion", "test-driven-development", "writing-plans", "receiving-code-review",
           "requesting-code-review", "dispatching-parallel-agents", "grilling", "handoff", "research", "diagnosing-bugs", "codebase-design",
           "improve-codebase-architecture", "prototype", "pr", "frontend-design", "webapp-testing", "mcp-builder"}


def test_bundled_skills_load_and_a_users_skill_of_the_same_name_wins(ws, home):
    found = skills.discover(ws.root, home)
    assert BUNDLED <= {n for n, k in found.items() if k.builtin}
    assert found["handoff"].manual and found["improve-codebase-architecture"].manual and not found["systematic-debugging"].manual
    for name in BUNDLED:
        assert "superpowers:" not in skills.load(found[name])
    catalog = skills.catalog(found)
    assert "- systematic-debugging:" in catalog and "- handoff:" not in catalog and "read TodoWrite as todo_write" in catalog
    write_skill(home / ".agents" / "skills", "grilling", "My own grilling.")
    assert skills.discover(ws.root, home)["grilling"].description == "My own grilling." and not skills.discover(ws.root, home)["grilling"].builtin


def test_manual_skills_run_as_slash_commands(ws, home):
    fake = scripted(turn("noted"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake)
    assert any(c["name"] == "handoff" for c in session.snapshot()["commands"])
    session.command("/handoff for the next session")
    wait_for(session, "idle")
    sent = session.history[0]["content"]
    assert "handoff document" in sent and sent.endswith("User input: for the next session")


def fake_repo_zip(files):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))


def test_skills_install_from_a_github_zip(ws, home, monkeypatch):
    from app.agent import skill_install
    archive = fake_repo_zip({
        "repo-main/skills/alpha/SKILL.md": "---\nname: alpha\ndescription: First.\n---\nDo alpha.",
        "repo-main/skills/alpha/notes.md": "extra",
        "repo-main/skills/alpha/../../../escape.txt": "nope",
        "repo-main/skills/Bad Name/SKILL.md": "---\nname: Bad Name\ndescription: x\n---\n",
        "repo-main/other/beta/SKILL.md": "---\nname: beta\ndescription: Second.\n---\n"})
    monkeypatch.setattr(skill_install, "_fetch", lambda owner, repo, ref: archive)
    assert skill_install.parse("obra/superpowers@main/skills") == ("obra", "superpowers", "main", "skills")
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=scripted(turn("x")))
    assert skill_install.install("o/r/skills", home) == ["alpha"]
    assert (home / ".manga-agent" / "skills" / "alpha" / "notes.md").read_text() == "extra"
    assert not (home / ".manga-agent" / "escape.txt").exists() and not (home / "escape.txt").exists()
    assert "alpha" not in session.skills
    assert "Đã cài 1 skill" in session.command("/skills add o/r/skills")["message"] and "alpha" in session.skills
    assert "không thấy SKILL.md" in session.command("/skills add o/r/nothing")["message"]
    assert "dùng dạng" in session.command("/skills add not a spec")["message"]
