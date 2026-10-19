"""Concurrency-fix tests for app/agent/session.py.

Each fix has a regression test using forced interleaving (fails on the
unfixed code, passes on the fixed code) plus a normal-operation test
proving the fixed path still works.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest

from app.agent import gitguard, mcp, sandbox
from app.agent.session import AgentSession, AgentSessionManager, TOOLS_LIBRARY
from app.agent.tools import Workspace


# ---- helpers ----

class _Prov:
    id = "stub"
    label = "stub"


@pytest.fixture()
def mgr(tmp_path):
    home = tmp_path / "home"
    return AgentSessionManager(home / ".manga-agent", home=home)


@pytest.fixture()
def ws(tmp_path):
    wsroot = tmp_path / "ws"
    wsroot.mkdir(parents=True, exist_ok=True)
    return Workspace(wsroot, sandbox.Policy("workspace-write", False))


def _make_session(mgr, ws, session_id, complete=None):
    return mgr.create(_Prov(), "k", "m", ws, "ask",
                      complete=complete or (lambda *a, **k: {"text": ""}),
                      session_id=session_id)


def _wait_until(cond, timeout=10.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(interval)
    return bool(cond())


def _quiet_thread_errors():
    """Swallow expected tracebacks from threads that die on a forced race."""
    errors = []
    old = threading.excepthook

    def hook(args):
        errors.append(args)
    threading.excepthook = hook
    return old, errors


# ---- save() must survive non-serializable plugin-emitted data ----

def test_h5_save_survives_nonserializable(mgr, ws):
    """an event carrying a non-JSON value must not break save()."""
    s = _make_session(mgr, ws, "0101020304050607")
    s.history.append({"role": "user", "content": "hi"})
    s.emit("custom", payload=object())  # non-serializable payload
    s.save()
    data = json.loads(s.store.read_text(encoding="utf-8"))
    assert data["id"] == s.id
    assert data["history"][-1]["content"] == "hi"


def test_h5_poisoned_turn_does_not_wedge(mgr, ws):
    """Normal flow: a turn poisoned mid-flight ends idle and takes new work."""
    def blocking_complete(provider, key, model, messages, tools=None, **kw):
        time.sleep(2)
        return {"text": "done", "calls": [], "reasoning": "",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}}

    s = _make_session(mgr, ws, "0102020304050607", complete=blocking_complete)
    assert s.send("do work") is False
    assert _wait_until(lambda: s.status == "running", timeout=5)
    s.emit("custom", payload=object())
    assert _wait_until(lambda: s.status == "idle", timeout=15), \
        f"session wedged in {s.status!r} by a save() crash"
    # A follow-up starts a new turn instead of queueing forever.
    assert s.send("are you there?") is False
    assert _wait_until(lambda: s.status == "idle", timeout=15)
    s.close()


# ---- concurrent save() must not tear the session file ----

def test_h6_concurrent_save_no_tear(mgr, ws):
    """writers hammering save() while readers parse must tear nothing."""
    s = _make_session(mgr, ws, "0301020304050607")
    # large payload: real overlap windows between the writers
    s.history = [{"role": "user", "content": "x" * 20000} for _ in range(80)]
    s.events = [{"seq": i, "type": "tool", "name": "read_file", "ok": True, "output": "y" * 500}
                for i in range(200)]
    s.save()  # prime, so readers never race the very first write
    stop = threading.Event()
    failures, writes = [], [0]

    def writer():
        while not stop.is_set():
            s.save()
            writes[0] += 1

    def reader():
        while not stop.is_set():
            try:
                data = json.loads(s.store.read_text(encoding="utf-8"))
            except ValueError:
                failures.append("unparsable")
            except OSError:
                pass
            else:
                if data.get("id") != s.id:
                    failures.append("id-mismatch")

    threads = ([threading.Thread(target=writer, daemon=True) for _ in range(6)] +
               [threading.Thread(target=reader, daemon=True) for _ in range(2)])
    for t in threads:
        t.start()
    time.sleep(3.5)
    stop.set()
    for t in threads:
        t.join(10)
    assert writes[0] > 10, "writers never ran"
    assert not failures, f"{len(failures)} torn reads ({', '.join(sorted(set(failures)))})"


def test_h6_save_roundtrip(mgr, ws):
    """Normal flow: sequential saves leave a valid session file and meta card."""
    s = _make_session(mgr, ws, "0302020304050607")
    s.title = "hello"
    s.save()
    s.save()
    data = json.loads(s.store.read_text(encoding="utf-8"))
    assert data["id"] == s.id and data["title"] == "hello"
    card = json.loads(s.store.with_name(s.store.stem + ".meta.json").read_text(encoding="utf-8"))
    assert card["title"] == "hello"
    leftovers = [p for p in s.store.parent.glob("*.tmp-*")]
    assert not leftovers, leftovers


# ---- _start_scheduler must not double-start the tick thread ----

def test_h7_start_scheduler_single_thread(mgr, ws):
    """two callers entering the check together start exactly one thread."""
    s = _make_session(mgr, ws, "0701020304050607")
    s.schedules = [{"id": "keep", "prompt": "later", "next_at": time.time() + 3600, "every": 60}]
    s.closed = False
    s._closing = False
    entered = threading.Event()

    class Gate:
        def is_alive(self):
            entered.set()
            time.sleep(0.3)  # let the peer enter the check too
            return False

    s._scheduler = Gate()
    t1 = threading.Thread(target=s._start_scheduler, daemon=True)
    t1.start()
    assert entered.wait(timeout=10)
    t2 = threading.Thread(target=s._start_scheduler, daemon=True)
    t2.start()
    t1.join(10)
    t2.join(10)
    try:
        alive = [t for t in threading.enumerate()
                 if t.name == f"agent-schedule-{s.id}" and t.is_alive()]
        assert len(alive) == 1, f"expected 1 scheduler thread, found {len(alive)}"
    finally:
        s._closing = True
        for t in threading.enumerate():
            if t.name == f"agent-schedule-{s.id}":
                t.join(5)


def test_h7_tick_fires_each_schedule_once(mgr, ws):
    """Normal flow: one _tick fires each due schedule exactly once."""
    s = _make_session(mgr, ws, "0702020304050607")
    sends = []
    s.send = lambda text, **k: sends.append(text) or True
    s.save = lambda: None
    s.schedules = [
        {"id": "once", "prompt": "one-shot", "next_at": time.time() - 5, "every": 0},
        {"id": "rep", "prompt": "recurring", "next_at": time.time() - 5, "every": 60},
    ]
    s.closed = False
    s._closing = False
    s._tick()
    assert sends == ["[Scheduled by you] one-shot", "[Scheduled by you] recurring"]
    assert [t["id"] for t in s.schedules] == ["rep"]
    assert s.schedules[0]["next_at"] > time.time()


# ---- concurrent _start_job must not collide on a job name ----

def test_m5_concurrent_start_no_orphan(mgr, ws):
    """two racing starts must not orphan a job neither can stop."""
    s = _make_session(mgr, ws, "0501020304050607")
    old_hook, errors = _quiet_thread_errors()

    class RacyDict(dict):
        """Timing control only: both threads observe the dict before either assigns."""
        barrier = threading.Barrier(2)

        def __contains__(self, key):
            if isinstance(key, str) and key.startswith("job"):
                try:
                    self.barrier.wait(timeout=3)
                except threading.BrokenBarrierError:
                    pass
            return super().__contains__(key)

    stored = []
    orig_setitem = RacyDict.__setitem__

    def recording_setitem(self, k, v):
        stored.append(v)
        orig_setitem(self, k, v)

    RacyDict.__setitem__ = recording_setitem
    try:
        s.jobs = RacyDict()
        ts = [threading.Thread(
            target=lambda: s._start_job({"command": "sleep 60", "background": True}),
            daemon=True) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(20)
    finally:
        threading.excepthook = old_hook
    try:
        orphans = [j for j in stored if all(j is not v for v in s.jobs.values())]
        assert not orphans, f"{len(orphans)} orphaned job(s): invisible to job_stop/stop()"
        assert len(stored) == len(s.jobs), \
            f"{len(stored)} jobs created but only {len(s.jobs)} reachable"
    finally:
        for j in stored:
            j.stop()


def test_m5_start_stop_job_normal(mgr, ws):
    """Normal flow: a job starts, stops, and another can start after it."""
    s = _make_session(mgr, ws, "0502020304050607")
    msg = s._start_job({"command": "echo hello", "background": True})
    assert "job1" in msg and "job1" in s.jobs
    out = s._session_tool({"name": "job_stop", "args": {"id": "job1"}, "id": "s-1"})
    assert "Stopped" in out
    msg2 = s._start_job({"command": "echo again", "background": True})
    assert "Started job" in msg2
    for j in list(s.jobs.values()):
        j.stop()


# ---- stop() during RPC answers pending .reqs; the worker times out ----

class _NeverJob:
    code = None  # never finishes on its own


def test_m6_shutdown_answers_pending_req(mgr, ws, tmp_path):
    """a .req the server never got to still gets a .res on shutdown,
    so a polling worker wakes up instead of spinning forever."""
    import shutil as _shutil

    s = _make_session(mgr, ws, "0601020304050607")
    rpc = tmp_path / "rpcA"
    rpc.mkdir(parents=True)
    (rpc / "poison.req").write_text("{not json", encoding="utf-8")  # never answerable
    t = threading.Thread(target=s._serve_chain, args=(rpc, _NeverJob()), daemon=True)
    t.start()
    time.sleep(0.3)
    assert t.is_alive()

    seen, res_seen = [], threading.Event()

    def worker():
        # Poll exactly the way the worker's _call does.
        reply = rpc / "poison.res"
        deadline = time.time() + 8
        while not reply.exists() and time.time() < deadline:
            time.sleep(0.02)
        if reply.exists():
            res_seen.set()
        seen.append(reply.exists())

    # The .res lives only between the shutdown sweep and the folder's removal;
    # hold the removal until the worker has observed it (or 10s pass).
    real_rmtree = _shutil.rmtree

    def gated_rmtree(path, ignore_errors=False):
        if path == rpc:
            res_seen.wait(timeout=10)
        return real_rmtree(path, ignore_errors=ignore_errors)

    wt = threading.Thread(target=worker, daemon=True)
    wt.start()
    s._stop = True
    with mock.patch.object(_shutil, "rmtree", gated_rmtree):
        t.join(15)
    wt.join(10)
    s._stop = False
    assert not t.is_alive()
    assert not rpc.exists(), "the rpc folder must still be cleaned up"
    assert seen == [True], "worker kept polling for a .res that never came"


def test_m6_worker_call_eventually_times_out(tmp_path):
    """The worker side of the RPC gives up instead of polling forever."""
    rpc = tmp_path / "rpcB"
    rpc.mkdir(parents=True)

    class FastTime:
        def __init__(self):
            self.now = time.time()

        def time(self):
            self.now += 2000  # jump past the 1800s deadline on the second poll
            return self.now

        def sleep(self, s):
            pass

    ns: dict = {}
    with mock.patch.dict(os.environ, {"AGENT_RPC": str(rpc)}):
        exec(TOOLS_LIBRARY.format(sources="{}", remote="[]"),
             {"_uuid": uuid, "_os": os, "_json": json, "_time": FastTime(),
              "__name__": "__worker__"}, ns)
        result = []

        def worker():
            try:
                ns["_call"]("_chain", {})
            except RuntimeError as exc:
                result.append(str(exc))

        wt = threading.Thread(target=worker, daemon=True)
        wt.start()
        wt.join(10)
        assert not wt.is_alive(), "worker still polling for a .res with no timeout"
        assert result and "never answered" in result[0]


def test_m6_rpc_roundtrip(mgr, ws, tmp_path):
    """Normal flow: a worker's tool call is answered through a live _serve_chain."""
    s = _make_session(mgr, ws, "0602020304050607")
    rpc = tmp_path / "rpcC"
    rpc.mkdir(parents=True)
    t = threading.Thread(target=s._serve_chain, args=(rpc, _NeverJob()), daemon=True)
    t.start()
    try:
        ns: dict = {}
        with mock.patch.dict(os.environ, {"AGENT_RPC": str(rpc)}):
            exec(TOOLS_LIBRARY.format(sources="{}", remote="[]"),
                 {"_uuid": uuid, "_os": os, "_json": json, "_time": time,
                  "__name__": "__worker__"}, ns)
            out = ns["_call"]("_search", {"query": "read file", "limit": 3})
    finally:
        s._stop = True
        t.join(10)
        s._stop = False
    assert isinstance(out, list) and out, out


# ---- stop() catches jobs registered concurrently ----

def test_m7_stop_catches_concurrent_register(tmp_path):
    """a job registered as stop() runs is still stopped, never leaked."""
    s = AgentSession.__new__(AgentSession)
    s._usage_lock = threading.Lock()
    s._lock = threading.Condition()
    s._stop = False
    s.queue = []
    s.goal = None
    s.children = {}
    s.status = "idle"
    s.asked_s = 0.0
    s._root = lambda: s
    root = tmp_path / "ws8"
    root.mkdir(parents=True)
    s.workspace = type("W", (), {"root": root})()

    class GateDict(dict):
        """Block __setitem__ until the stopper thread is parked on the lock."""
        gate = threading.Event()

        def __setitem__(self, k, v):
            assert self.gate.wait(timeout=15), "stop() never ran"
            super().__setitem__(k, v)

    s.jobs = GateDict()

    class FakeOut:
        def join(self, timeout=None):
            return None

        def text(self):
            return ""

    class FakeJob:
        code = 0           # finished...
        alive = True       # ...but a child is still running: kept as a background job
        out = FakeOut()
        started = time.time()
        stopped = False

        def read(self):
            return ""

        def stop(self):
            self.stopped = True

    job = FakeJob()
    guard = gitguard.snapshot(root)
    t = threading.Thread(target=s._await_job, args=(job, 30, guard), daemon=True)
    t.start()
    time.sleep(0.5)  # _await_job is parked in GateDict.__setitem__, holding _usage_lock
    stopper = threading.Thread(target=s.stop, daemon=True)
    stopper.start()
    time.sleep(0.5)  # the stopper is parked on _usage_lock behind it
    GateDict.gate.set()
    t.join(10)
    stopper.join(10)
    assert not t.is_alive() and not stopper.is_alive()
    assert len(s.jobs) == 0, f"job survived stop(): {list(s.jobs)}"
    assert job.stopped, "the concurrently registered job was never stopped"


def test_m7_stop_stops_background_jobs(mgr, ws):
    """Normal flow: stop() stops every background job and empties the dict."""
    s = _make_session(mgr, ws, "0802020304050607")
    s._start_job({"command": "sleep 60", "background": True})
    assert len(s.jobs) == 1
    job = s.jobs["job1"]
    s.stop()
    assert len(s.jobs) == 0
    assert not job.alive


# ---- _ensure_mcp must not start duplicate servers ----

def _write_mcp_json(home: Path):
    mcpdir = home / ".manga-agent"
    mcpdir.mkdir(parents=True, exist_ok=True)
    (mcpdir / "mcp.json").write_text(json.dumps({"mcpServers": {
        "alpha": {"command": "true"}, "beta": {"command": "true"}}}), encoding="utf-8")


class _FakeServer:
    def __init__(self, name, config, cwd, constructions, lock):
        with lock:
            constructions.append(name)
        self.name, self.tools = name, []

    def close(self):
        pass


def _prep_mcp_session(s):
    """A session whose _ensure_mcp runs against fake servers."""
    s._mcp_ready = False
    s.mcp_servers = {}
    s.mcp_status = {}


def test_m10_ensure_mcp_no_duplicate_start(mgr, ws, tmp_path):
    """two callers racing through the check start each server once."""
    _write_mcp_json(tmp_path / "home")
    s = _make_session(mgr, ws, "1301020304050607")
    _prep_mcp_session(s)
    constructions, clock = [], threading.Lock()

    def fake_server(name, config, cwd):
        return _FakeServer(name, config, cwd, constructions, clock)

    # The `self._mcp_ready = True` line inside _ensure_mcp, whatever its number.
    src, start = inspect.getsourcelines(AgentSession._ensure_mcp)
    set_line = start + next(i for i, l in enumerate(src) if "_mcp_ready = True" in l)
    at_set = {"A": threading.Event(), "B": threading.Event()}
    release = threading.Event()

    def make_tracer(name, other):
        def tracer(frame, event, arg):
            if (event == "line" and frame.f_code.co_name == "_ensure_mcp"
                    and frame.f_lineno == set_line):
                at_set[name].set()
                release.wait(timeout=15)
                time.sleep(0.05)  # let the peer arrive before either stores
            return tracer
        return tracer

    # patch once for the whole test: a per-thread patch would restore the mock mid-run
    real_server = mcp.Server
    mcp.Server = fake_server
    try:
        def go(name):
            sys.settrace(make_tracer(name, "B" if name == "A" else "A"))
            try:
                s._ensure_mcp()
            finally:
                sys.settrace(None)

        t1 = threading.Thread(target=go, args=("A",), daemon=True)
        t1.start()
        assert at_set["A"].wait(timeout=10), "thread A never reached the check"
        t2 = threading.Thread(target=go, args=("B",), daemon=True)
        t2.start()
        # B must block on _mcp_lock, never reaching the check
        b_arrived = at_set["B"].wait(timeout=3)
        release.set()
        t1.join(15)
        t2.join(15)
    finally:
        mcp.Server = real_server
    assert sorted(constructions) == ["alpha", "beta"], \
        f"server starts: {constructions} (B reached the check: {b_arrived})"


def test_m10_ensure_mcp_idempotent(mgr, ws, tmp_path):
    """Normal flow: repeated _ensure_mcp calls start each server exactly once."""
    _write_mcp_json(tmp_path / "home")
    s = _make_session(mgr, ws, "1302020304050607")
    _prep_mcp_session(s)
    constructions, clock = [], threading.Lock()

    def fake_server(name, config, cwd):
        return _FakeServer(name, config, cwd, constructions, clock)

    with mock.patch.object(mcp, "Server", fake_server):
        s._ensure_mcp()
        s._ensure_mcp()
    assert sorted(constructions) == ["alpha", "beta"]
    assert set(s.mcp_servers) == {"alpha", "beta"}


# ---- job_stop releases the job's slot at once ----

def test_m11_jobstop_frees_slots(mgr, ws):
    """stopping all jobs frees every MAX_JOBS slot immediately."""
    from app.agent.tools import ToolError

    s = _make_session(mgr, ws, "1501020304050607")
    for _ in range(8):
        s._start_job({"command": "sleep 60", "background": True})
    assert len(s.jobs) == 8
    for i in range(8):
        s._session_tool({"name": "job_stop", "args": {"id": f"job{i + 1}"}, "id": f"s-{i}"})
    assert len(s.jobs) == 0, f"stopped jobs still occupy slots: {sorted(s.jobs)}"
    try:
        msg = s._start_job({"command": "sleep 60", "background": True})
    except ToolError as exc:
        pytest.fail(f"9th job refused although every job is stopped: {exc}")
    assert "job1" in msg
    for j in list(s.jobs.values()):
        j.stop()


# ---- delete() during a turn stays deleted ----

def test_l6_delete_during_turn_stays_deleted(mgr, ws):
    """the orphaned turn thread's save() must not resurrect the file."""
    def blocking_complete(provider, key, model, messages, tools=None, **kw):
        time.sleep(3)
        return {"text": "done", "calls": [], "reasoning": "",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0}}

    s = _make_session(mgr, ws, "1601020304050607", complete=blocking_complete)
    s.send("long task")
    assert _wait_until(lambda: s.status == "running", timeout=5)
    turn_thread = next(t for t in threading.enumerate() if t.name == f"agent-{s.id}")
    mgr.delete(s.id)  # user deletes the session mid-turn
    assert not s.store.is_file(), "session file still present right after delete()"
    turn_thread.join(15)  # the orphaned turn finishes...
    time.sleep(0.5)
    assert not s.store.is_file(), "the orphaned turn thread's save() resurrected the session"


def test_l6_close_idle_session(mgr, ws):
    """Normal flow: closing an idle session is prompt and keeps its file."""
    s = _make_session(mgr, ws, "1602020304050607")
    s.save()
    assert s.store.is_file()
    s.close()
    assert s.store.is_file(), "close() alone must not delete the session file"
    s.save()  # no-op after close: must neither raise nor misbehave


# ---- _count_role counts through the atomic roles.add(cfg) ----

def test_m8_count_role_atomic_budget(mgr, ws, tmp_path):
    """_count_role passes the role's cfg to roles.add: a spend that would push
    past the daily budget is atomically refused instead of racing past it."""
    from app.agent import roles as _roles

    s = _make_session(mgr, ws, "1701020304050607")
    home = tmp_path / "home"
    s.home = home
    s.profile["model_roles"] = {
        "thinker": {"daily_requests": 10, "daily_tokens": 100, "model": "m",
                    "base_url": "http://example.invalid"}}
    _roles._GOOD.clear()
    try:
        s._count_role("thinker", 1, 60)
        assert _roles.today(home)["thinker"] == {"requests": 1, "tokens": 60}
        # 60 + 50 > 100 tokens: the atomic check refuses, nothing more is counted.
        s._count_role("thinker", 1, 50)
        assert _roles.today(home)["thinker"] == {"requests": 1, "tokens": 60}
        # The session's own tally still records the real spend.
        assert s.role_usage["thinker"] == {"requests": 2, "tokens": 110}
    finally:
        _roles._GOOD.clear()


def test_m8_count_role_without_cfg_still_counts(mgr, ws, tmp_path):
    """Names with no budget cfg (e.g. '<source>:<model>' picks) count as before."""
    from app.agent import roles as _roles

    s = _make_session(mgr, ws, "1702020304050607")
    s.home = tmp_path / "home"
    _roles._GOOD.clear()
    try:
        s._count_role("minirouter:some-model", 2, 30)
        assert _roles.today(s.home)["minirouter:some-model"] == {"requests": 2, "tokens": 30}
    finally:
        _roles._GOOD.clear()
