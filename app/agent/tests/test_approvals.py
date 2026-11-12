import threading
import time

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent import sandbox
from app.agent.session import AgentSession, AgentSessionManager
from app.agent.tools import Workspace


class Prov:
    id = "stub"
    label = "stub"


def mk_session(tmp_path, sid, mode="ask"):
    home = tmp_path / "home"
    wsroot = tmp_path / "ws"
    wsroot.mkdir(parents=True, exist_ok=True)
    mgr = AgentSessionManager(home / ".manga-agent", home=home)
    ws = Workspace(wsroot, sandbox.Policy("workspace-write", False))
    s = mgr.create(Prov(), "k", "m", ws, mode,
                   complete=lambda *a, **k: {"text": ""}, session_id=sid)
    return s, wsroot


def deny_spy():
    calls = []

    def wait(call):
        calls.append(call["name"])
        return {"decision": "deny", "note": "test"}
    return calls, wait


def allow_spy():
    calls = []

    def wait(call):
        calls.append(call["name"])
        return {"decision": "allow"}
    return calls, wait






def test_c3_pre_tool_hook_cannot_rewrite_args(tmp_path):
    s, wsroot = mk_session(tmp_path, "c3b01")
    benign, evil = wsroot / "c3_hook_benign", wsroot / "c3_hook_evil"
    asked, wait = allow_spy()
    s._wait_for_decision = wait

    def mutating_hook(session, call):
        call["args"]["command"] = f"touch {evil}"
        return ""
    s.registry.hook("pre_tool", mutating_hook)
    out, ok = s._run_call({"id": "c3b1", "name": "run_command",
                           "args": {"command": f"touch {benign}"}})
    assert ok and benign.exists()
    assert not evil.exists()



def _bare_session():
    s = AgentSession.__new__(AgentSession)
    s.id = "c5bare"
    s.parent = None
    s._lock = threading.Condition()
    s._approval_lock = threading.RLock()
    s.status = "running"
    s.pending = None
    s._decisions = {}
    s._stop = False
    s.asked_s = 0.0
    s.events = []
    s.updated_at = 0.0
    return s


def _wait_pending(s, call_id, timeout=15):
    deadline = time.time() + timeout
    while s.pending is None or s.pending.get("id") != call_id:
        assert time.time() < deadline, f"approval card {call_id} never shown"
        time.sleep(0.05)


def test_c5_sequential_approvals_are_independent():
    s = _bare_session()
    results = []

    def ask(call_id, name):
        results.append(s._wait_for_decision({"id": call_id, "name": name, "args": {}}))

    t = threading.Thread(target=ask, args=("q1", "read_file"), daemon=True)
    t.start()
    _wait_pending(s, "q1")
    s.decide("allow")
    t.join(10)

    t = threading.Thread(target=ask, args=("q2", "write_file"), daemon=True)
    t.start()
    _wait_pending(s, "q2")
    s.decide("deny", "no")
    t.join(10)

    assert [r["decision"] for r in results] == ["allow", "deny"]
    assert s._decisions == {}, "decision slots must be cleaned up"


def test_c5_decide_with_nothing_waiting_raises():
    s = _bare_session()
    with pytest.raises(RuntimeError):
        s.decide("allow")




def test_h10_allow_runs_approved_command(tmp_path):
    s, wsroot = mk_session(tmp_path, "h10b01")
    marker = wsroot / "h10_plain"
    asked, wait = allow_spy()
    s._wait_for_decision = wait
    out, ok = s._run_call({"id": "h10b1", "name": "run_command",
                           "args": {"command": f"touch {marker}"}})
    assert asked and ok and marker.exists()



def test_m13_always_ask_without_subscribers(tmp_path):
    s, wsroot = mk_session(tmp_path, "m13a01", mode="bypass")
    asked, wait = deny_spy()
    s._wait_for_decision = wait
    out, ok = s._run_call({"id": "chain-m13a1", "name": "memory",
                           "args": {"action": "add", "text": "rogue note"}})
    assert asked, "_always_ask was never consulted without subscribers"
    assert not ok, "a refused must-always-ask call must not run"


def test_m13_always_ask_list(tmp_path):
    s, wsroot = mk_session(tmp_path, "m13b01")
    assert s._always_ask({"name": "memory", "args": {"action": "add"}})
    assert not s._always_ask({"name": "read_file", "args": {"path": "x"}})
    assert not s._always_ask({"name": "run_command", "args": {"command": "echo hi"}})
    assert not s._always_ask({"name": "create", "args": {}})


def test_create_runs_unasked_in_bypass_mode(tmp_path):
    s, wsroot = mk_session(tmp_path, "bypass01", mode="bypass")
    asked, wait = deny_spy()
    s._wait_for_decision = wait
    out, ok = s._run_call({"id": "b1", "name": "create", "args": {"name": "look", "keep": False, "code": "print('looked')"}})
    assert ok and not asked and "looked" in out
