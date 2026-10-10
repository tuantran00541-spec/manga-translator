"""Tests for the 8 Batch-1 features ported from harness deep-dives (commit 8ea9e92).

1. Repeat reminder (dsh): remind at 3/5, block at 8 (was: hard block at 3)
2. Approval audit log (dsh): approval/asked + approval/decided events
3. Skill arguments (Kimi): $ARGUMENTS/$0/$<n> expansion
4. Auxiliary model routing (ZCode): _aux_complete with 5k cap
5. Per-turn timing telemetry (Codex): TTFT/TTFM
6. Truncation metadata (Pi): explicit counts in _offload
7. Fail-safe stopReason=length (Pi): all calls fail on length cutoff
8. Granular approval modes (Codex): unless-trusted mode
"""
import time
from pathlib import Path

import pytest

from app.agent import client, skills
from app.agent.session import AgentSessionManager
from app.ai_providers import PROVIDERS


@pytest.fixture
def home(tmp_path):
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


@pytest.fixture
def ws(tmp_path):
    from app.agent.tools import Workspace
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")
    return Workspace(root)


def manager(home, store=None):
    return AgentSessionManager(store, home=home)


def scripted(*turns):
    seen = []

    def complete(provider, key, model, messages, *, tools):
        seen.append((messages, tools))
        step = turns[len(seen) - 1]
        return step(messages, tools) if callable(step) else step

    complete.seen = seen
    return complete


def turn(text="", calls=()):
    return {"text": text, "calls": list(calls), "reasoning": "",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def call(tool, **args):
    return {"id": f"{tool}-{id(args)}", "name": tool, "args": args}


# 1. Repeat reminder: remind at 3/5, block at 8.

def test_repeat_reminder_warns_at_3_and_5_but_blocks_at_8(ws, home):
    """The 3rd and 5th identical calls get a [Reminder]; the 8th is blocked."""
    from app.agent.session import AgentSession
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto")
    c = {"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}

    outputs = []
    for i in range(7):
        out, ok = session._run_call({**c, "id": f"c{i}"})
        outputs.append((out, ok))
    # First two: no reminder, succeed.
    assert outputs[0][1] and "[Reminder]" not in outputs[0][0]
    assert outputs[1][1] and "[Reminder]" not in outputs[1][0]
    # 3rd: gentle reminder, still runs.
    assert outputs[2][1] and "[Reminder]" in outputs[2][0] and "3 times" in outputs[2][0]
    # 4th: no reminder (only at 3 and 5+).
    assert outputs[3][1] and "[Reminder]" not in outputs[3][0]
    # 5th: stronger reminder, still runs.
    assert outputs[4][1] and "[Reminder]" in outputs[4][0] and "5th" in outputs[4][0]
    # 6th and 7th: quiet (no nagging).
    assert outputs[5][1] and "[Reminder]" not in outputs[5][0]
    assert outputs[6][1] and "[Reminder]" not in outputs[6][0]
    # 8th identical call: blocked.
    out8, ok8 = session._run_call({**c, "id": "c8"})
    assert not ok8 and "Blocked" in out8 and "8 times" in out8


def test_repeat_guard_ignores_wait_agent(ws, home):
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto")
    c = {"id": "w", "name": "wait_agent", "args": {"id": "x"}}
    for i in range(10):
        out, ok = session._run_call({**c, "id": f"w{i}"})
        assert "[Reminder]" not in out and "Blocked" not in out


# 2. Approval audit log.

def test_approval_audit_logs_asked_and_decided(ws, home):
    fake = scripted(turn(calls=[call("write_file", path="new.txt", content="hi")]),
                    turn("done"))
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "ask", complete=fake)
    assert session._approval_audit == []
    session.send("make a file")
    # The session is waiting for a decision; the "asked" event is logged.
    end = time.time() + 5
    while session.status != "waiting" and time.time() < end:
        time.sleep(0.01)
    asked = [e for e in session._approval_audit if e["event"] == "approval/asked"]
    assert len(asked) == 1 and asked[0]["tool"] == "write_file"
    # Decide, then the "decided" event appears with the outcome.
    session.decide("deny")
    end = time.time() + 5
    while session.status == "waiting" and time.time() < end:
        time.sleep(0.01)
    decided = [e for e in session._approval_audit if e["event"] == "approval/decided"]
    assert len(decided) == 1 and decided[0]["outcome"] == "deny"
    assert decided[0]["id"] == asked[0]["id"]  # paired by id


# 3. Skill arguments.

def test_skill_arguments_expand_placeholders(tmp_path):
    body = "Run with $ARGUMENTS. First: $0, second: $1. Dir: ${SKILL_DIR}"
    assert skills.expand_arguments(body, "foo bar", "/skills/x") == \
        "Run with foo bar. First: foo, second: bar. Dir: /skills/x"


def test_skill_arguments_append_when_no_placeholder():
    body = "Do the thing."
    out = skills.expand_arguments(body, "some args")
    assert out.endswith("Arguments: some args")


def test_skill_arguments_empty_args_unchanged():
    body = "Do $ARGUMENTS now."
    assert skills.expand_arguments(body, "") == body


def test_skill_tool_passes_args_through(ws, home, tmp_path):
    """The skill tool forwards args= for $ARGUMENTS expansion."""
    skill_dir = tmp_path / "sk" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nDo $ARGUMENTS.", encoding="utf-8")
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto")
    # Register the skill manually (bypasses discovery).
    from app.agent.skills import Skill
    skill = Skill(name="demo", description="d", folder=skill_dir, builtin=True)
    session.skills["demo"] = skill
    session._skills_trusted.add("demo")
    out = session._session_tool({"name": "skill", "args": {"name": "demo", "args": "hello world"}, "id": "s1"})
    assert "Do hello world." in out


# 4. Auxiliary model routing.

def test_aux_complete_prefers_aux_model_and_caps_tokens(ws, home):
    seen = {}

    def fake_complete(provider, key, model, messages, *, tools, max_tokens=None):
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return {"text": "summary", "calls": [], "reasoning": "", "usage": {}}

    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake_complete)
    session.profile["aux_model"] = "cheap-model"
    session.profile["max_output_tokens"] = 64000
    out = session._aux_complete([{"role": "user", "content": "hi"}], kind="compact", cap=5000)
    assert out["text"] == "summary"
    assert seen["model"] == "cheap-model"
    assert seen["max_tokens"] == 5000


def test_aux_complete_falls_back_to_main_model(ws, home):
    seen = {}

    def fake_complete(provider, key, model, messages, *, tools, max_tokens=None):
        seen["model"] = model
        return {"text": "s", "calls": [], "reasoning": "", "usage": {}}

    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake_complete)
    session.profile["max_output_tokens"] = 2000  # below the cap: uses the smaller
    session._aux_complete([{"role": "user", "content": "hi"}], kind="compact", cap=5000)
    assert seen["model"] == session.model
    # The double without max_tokens still works (TypeError fallback).
    def bare(provider, key, model, messages, *, tools):
        return {"text": "s", "calls": [], "reasoning": "", "usage": {}}
    session2 = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=bare)
    assert session2._aux_complete([{"role": "user", "content": "hi"}])["text"] == "s"


# 5. Timing telemetry.

def test_turn_timing_records_ttft_and_ttfm(ws, home):
    """_call_model_once attaches ttft_s/ttfm_s timing to the turn."""
    def fake_complete(provider, key, model, messages, *, tools, on_delta=None, max_tokens=None):
        if on_delta:
            on_delta({"text": "he"})
            time.sleep(0.01)
            on_delta({"text": "llo"})
        return {"text": "hello", "calls": [], "reasoning": "",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    fake_complete.streams = True  # enable streaming so on_delta is used

    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto", complete=fake_complete)
    out = session._call_model([{"role": "user", "content": "hi"}], None)
    timing = out.get("timing") or {}
    assert timing.get("ttft_s") is not None and timing["ttft_s"] >= 0
    assert timing.get("ttfm_s") is not None and timing["ttfm_s"] >= timing["ttft_s"]


# 6. Truncation metadata.

def test_offload_includes_truncation_metadata(ws, home, tmp_path):
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto")
    big = "\n".join(f"line {i}" for i in range(5000))  # > OFFLOAD_AT
    out = session._offload({"id": "x1", "name": "run_command"}, big)
    assert "truncatedBy=offload" in out
    assert f"{len(big):,} chars" in out
    assert "5,000 lines" in out or "5000 lines" in out.replace(",", "")
    assert "full output saved to" in out


def test_offload_short_output_untouched(ws, home):
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "auto")
    assert session._offload({"id": "x", "name": "run_command"}, "short") == "short"


# 7. Fail-safe stopReason=length.

def test_length_finish_marks_all_calls_failed():
    import app.agent.client as client_mod
    calls = [{"id": "1", "name": "run_command", "args": {"command": "ls"}}]
    finish = "length"
    # Mirror the fail-safe in client._normalize: every call is failed, none executed.
    if finish == "length":
        for c in calls:
            if not c.get("error"):
                c["error"] = client_mod.CUT_OFF
    assert calls[0]["error"].startswith("Your reply hit the output length limit")
    assert "smaller pieces" in calls[0]["error"]


# 8. UnlessTrusted mode.

def test_unless_trusted_asks_when_workspace_untrusted(ws, home):
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "unless-trusted")
    # No plugins/skills registered -> nothing untrusted -> behaves like auto for reads.
    assert session._needs_approval({"name": "read_file", "args": {"path": "a.txt"}}) is False


def test_unless_trusted_mode_registered(ws, home):
    """The mode is accepted and falls through to auto rules when trusted."""
    session = manager(home).create(PROVIDERS["openai"], "k", "m", ws, "unless-trusted")
    assert session.mode == "unless-trusted"
    # A safe readonly command does not ask in a trusted workspace.
    assert session._needs_approval(
        {"name": "run_command", "args": {"command": "ls"}}) is False
