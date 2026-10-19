"""Approval-waterfall hardening in session.py: each attack must fail closed, and
legitimate flows (allow/deny, no-plugin default, proper plugin payloads) keep working."""
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
    """A real AgentSession in the given approval mode."""
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


# ---- tool/approve waterfall must validate listener returns ----


# ---- tool/execute must not let listeners swap the approved call ----


def test_c3_pre_tool_hook_cannot_rewrite_args(tmp_path):
    """A pre_tool hook rewriting args post-approval changes nothing."""
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


# ---- nested approvals must not steal each other's decisions ----

def _bare_session():
    """A minimal AgentSession exposing the real approval machinery."""
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
    """Two back-to-back approvals each get exactly their own decision."""
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


# ---- approval-card mutations must not change what runs ----


def test_h10_allow_runs_approved_command(tmp_path):
    """Allow runs exactly the approved command."""
    s, wsroot = mk_session(tmp_path, "h10b01")
    marker = wsroot / "h10_plain"
    asked, wait = allow_spy()
    s._wait_for_decision = wait
    out, ok = s._run_call({"id": "h10b1", "name": "run_command",
                           "args": {"command": f"touch {marker}"}})
    assert asked and ok and marker.exists()


# ---- _always_ask applies even with no subscribers ----

def test_m13_always_ask_without_subscribers(tmp_path):
    """In bypass mode with no plugins, `memory` must still ask. (The `chain-`
    id prefix is how created-tool code reaches session tools like memory.)"""
    s, wsroot = mk_session(tmp_path, "m13a01", mode="bypass")
    asked, wait = deny_spy()
    s._wait_for_decision = wait
    out, ok = s._run_call({"id": "chain-m13a1", "name": "memory",
                           "args": {"action": "add", "text": "rogue note"}})
    assert asked, "_always_ask was never consulted without subscribers"
    assert not ok, "a refused must-always-ask call must not run"


def test_m13_always_ask_list(tmp_path):
    """The must-always-ask list keeps its shape."""
    s, wsroot = mk_session(tmp_path, "m13b01")
    assert s._always_ask({"name": "memory", "args": {"action": "add"}})
    assert not s._always_ask({"name": "read_file", "args": {"path": "x"}})
    assert not s._always_ask({"name": "run_command", "args": {"command": "echo hi"}})
    # Its run is gated like a command, so writing a chain is not always-ask; bypass mode runs it unasked.
    assert not s._always_ask({"name": "create", "args": {}})


def test_create_runs_unasked_in_bypass_mode(tmp_path):
    """Bypass mode never stops a chain for approval: no shell tool by default leaves create as the only way to run commands."""
    s, wsroot = mk_session(tmp_path, "bypass01", mode="bypass")
    asked, wait = deny_spy()
    s._wait_for_decision = wait
    out, ok = s._run_call({"id": "b1", "name": "create", "args": {"name": "look", "keep": False, "code": "print('looked')"}})
    assert ok and not asked and "looked" in out
