"""Plugin API v1: handlers get a ctx, not the session; config, state and events reach plugins."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent import context, sandbox
from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace


class _Prov:
    id = "stub"
    label = "stub"


def _turn(text="", calls=()):
    return {"text": text, "calls": list(calls), "reasoning": "", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _session(tmp_path, source, complete=None, profile=None):
    home, root = tmp_path / "home", tmp_path / "ws"
    (home / ".manga-agent" / "plugins").mkdir(parents=True)
    root.mkdir()
    (home / ".manga-agent" / "plugins" / "probe.py").write_text(source, encoding="utf-8")
    if profile is not None:
        (home / ".manga-agent" / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    mgr = AgentSessionManager(home / ".manga-agent", home=home, trust=context.TrustStore(None))
    s = mgr.create(_Prov(), "secret-key", "m", Workspace(root, sandbox.Policy("workspace-write", False)), "bypass",
                   complete=complete or (lambda *a, **k: _turn("done")))
    s.trust_plugins()
    return s, home, root


def _idle(s, timeout=20):
    deadline = time.time() + timeout
    while s.status != "idle" or not any(e["type"] == "done" for e in s.events):
        assert time.time() < deadline, "session never went idle"
        time.sleep(0.02)


PROBE = '''
def register(api):
    def probe(ctx, args):
        leaks = [n for n in ("api_key", "mode", "registry", "provider", "trust") if hasattr(ctx, n)]
        return f"{ctx.plugin}:{ctx.root.name}:{ctx.read('note.txt')}:{leaks}"
    api.tool({"name": "probe", "description": "probe"}, probe, kind="read")
'''


def test_a_plugin_tool_gets_a_ctx_without_the_key_mode_or_registry(tmp_path):
    s, _, root = _session(tmp_path, PROBE)
    (root / "note.txt").write_text("hi", encoding="utf-8")
    out, ok = s._run_call({"id": "chain-p1", "name": "probe", "args": {}})
    assert ok and out == "probe:ws:hi:[]"


def test_ctx_calls_go_through_the_same_approval(tmp_path):
    source = '''
def register(api):
    api.tool({"name": "remember"}, lambda ctx, a: ctx.call("memory", {"action": "add", "text": "x"})[0])
'''
    s, _, _ = _session(tmp_path, source)
    asked = []
    s._wait_for_decision = lambda call: asked.append(call["name"]) or {"decision": "deny"}
    out, _ = s._run_call({"id": "chain-p2", "name": "remember", "args": {}})
    assert "memory" in asked and "refused" in out


def test_config_comes_from_the_profile_and_state_outlives_the_session(tmp_path):
    source = '''
def register(api):
    def bump(ctx, args):
        n = api.state.get("n", 0) + 1
        api.state.set("n", n)
        return f"{api.config.get('greeting')} {n} v{api.version}"
    api.tool({"name": "bump"}, bump, kind="read")
'''
    s, home, _ = _session(tmp_path, source, profile={"plugins": {"probe": {"greeting": "hello"}}})
    assert s._run_call({"id": "chain-a", "name": "bump", "args": {}})[0] == "hello 1 v1"
    assert s._run_call({"id": "chain-b", "name": "bump", "args": {}})[0] == "hello 2 v1"
    assert json.loads((home / ".manga-agent" / "plugin-state" / "probe.json").read_text())["n"] == 2


def test_lifecycle_events_reach_the_plugin_in_order(tmp_path):
    source = '''
def register(api):
    log = lambda *row: api.state.set("log", api.state.get("log", []) + [list(row)])
    api.on("session_start", lambda ctx: log("start"))
    api.on("before_model", lambda ctx, info: log("before", info["model"]))
    api.on("after_model", lambda ctx, turn: log("after", turn["text"]))
    api.on("turn_end", lambda ctx, text: log("end", text))
'''
    s, home, _ = _session(tmp_path, source)
    s.send("go")
    _idle(s)
    log = json.loads((home / ".manga-agent" / "plugin-state" / "probe.json").read_text())["log"]
    assert log == [["start"], ["before", "m"], ["after", "done"], ["end", "done"]]


def test_a_failing_observer_does_not_break_the_turn(tmp_path):
    source = '''
def register(api):
    api.on("after_model", lambda ctx, turn: 1 / 0)
'''
    s, _, _ = _session(tmp_path, source)
    s.send("go")
    _idle(s)
    assert [e["text"] for e in s.events if e["type"] == "assistant"] == ["done"]


def test_a_plugin_loop_drives_the_session_through_ctx(tmp_path):
    source = '''
def register(api):
    def once(ctx, max_steps):
        ctx.append({"role": "user", "content": "[from the loop]"})
        ctx.default_loop(1)
    api.loop("once", once)
'''
    s, _, _ = _session(tmp_path, source, profile={"loop": "once"})
    s.send("go")
    _idle(s)
    assert any(m.get("content") == "[from the loop]" for m in s.history)
    assert [e["text"] for e in s.events if e["type"] == "assistant"] == ["done"]


def test_ctx_emit_shows_up_as_a_plugin_event(tmp_path):
    source = '''
def register(api):
    api.command("ping", "ping", lambda ctx, args: ctx.emit("pong", n=1) or "ok")
'''
    s, _, _ = _session(tmp_path, source)
    assert s.command("/ping")["message"] == "ok"
    assert [(e["plugin"], e["name"], e["data"]) for e in s.events if e["type"] == "plugin"] == [("probe", "pong", {"n": 1})]


def _plain_session(tmp_path, mode="bypass"):
    home, root = tmp_path / "home", tmp_path / "ws"
    home.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    mgr = AgentSessionManager(home / ".manga-agent", home=home, trust=context.TrustStore(None))
    s = mgr.create(_Prov(), "k", "m", Workspace(root, sandbox.Policy("workspace-write", False)), mode,
                   complete=lambda *a, **k: _turn("done"))
    return s, home, root


GREETER = 'def register(api):\n    api.command("greet", "Greet", lambda ctx, args: "hello " + args)\n'


def test_plugin_write_asks_even_in_bypass_mode_and_a_refusal_writes_nothing(tmp_path):
    s, home, _ = _plain_session(tmp_path)
    asked = []
    s._wait_for_decision = lambda call: asked.append(call["args"]["code"]) or {"decision": "deny"}
    out, ok = s._run_call({"id": "chain-w1", "name": "plugin_write", "args": {"name": "greeter", "code": GREETER}})
    assert asked == [GREETER] and not ok and "refused" in out
    assert not (home / ".manga-agent" / "plugins" / "greeter.py").exists()


def test_an_approved_plugin_is_trusted_loaded_and_reports_its_errors(tmp_path):
    s, home, _ = _plain_session(tmp_path)
    s._wait_for_decision = lambda call: {"decision": "allow"}
    out, ok = s._run_call({"id": "chain-w2", "name": "plugin_write", "args": {"name": "greeter", "code": GREETER}})
    assert ok and "State: loaded" in out and s.command("/greet you")["message"] == "hello you"
    out, _ = s._run_call({"id": "chain-w3", "name": "plugin_write", "args": {"name": "broken", "code": "def register(api):\n    1 / 0\n"}})
    assert "State: failed" in out and "ZeroDivisionError" in out
    out, _ = s._run_call({"id": "chain-w4", "name": "plugin_write", "args": {"name": "greeter", "remove": True}})
    assert "Removed" in out and "greet" not in s.registry.commands


def test_writing_into_a_project_with_untrusted_plugins_does_not_trust_them(tmp_path):
    s, _, root = _plain_session(tmp_path)
    (root / ".agents" / "plugins").mkdir(parents=True)
    (root / ".agents" / "plugins" / "other.py").write_text("def register(api):\n    api.command('evil', 'x', lambda c, a: 'evil')\n", encoding="utf-8")
    s._wait_for_decision = lambda call: {"decision": "allow"}
    out, _ = s._run_call({"id": "chain-w5", "name": "plugin_write", "args": {"name": "mine", "code": GREETER, "scope": "workspace"}})
    assert "not loaded" in out and "evil" not in s.registry.commands and "greet" not in s.registry.commands
    fresh, _, root2 = _plain_session(tmp_path / "fresh")
    fresh._wait_for_decision = lambda call: {"decision": "allow"}
    out, _ = fresh._run_call({"id": "chain-w6", "name": "plugin_write", "args": {"name": "mine", "code": GREETER, "scope": "workspace"}})
    assert "State: loaded" in out and (root2 / ".agents" / "plugins" / "mine.py").is_file()
