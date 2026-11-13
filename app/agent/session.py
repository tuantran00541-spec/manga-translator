from __future__ import annotations

import atexit
import collections
import copy
import difflib
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.agent import agents, chaincheck, client, codemode, context, gitguard, isolate, mcp, memory, models, registry, repl, roles, rules, sandbox, schema as schemas, skill_install, skills, watch
from app.agent.checkpoint import Checkpoints
from app.ai_providers import auth_headers, resolve_provider
from app.agent.tools import COMMAND_TIMEOUT, KIND, MAX_COMMAND_TIMEOUT, SPECS, ToolError, Workspace, clip
from app.logging_config import logger
from app.agent.session_config import *  # noqa: F401, F403
from app.agent.session_config import _as_function, _signature
from app.agent.session_commands import AgentCommandsMixin
from app.agent.session_tools import AgentToolsMixin
from app.agent.session_chains import AgentChainsMixin
from app.agent.session_subagents import AgentSubagentsMixin
from app.agent.session_loop import AgentLoopMixin

PARALLEL_CALLS = frozenset({"list_dir", "read_file", "search", "glob", "symbols", "web_fetch", "web_search", "task"})


def _harden_process() -> None:
    if platform.system() == "Linux":
        try:
            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
        except (OSError, AttributeError):
            pass


class _Keyed:

    def __init__(self, session: "AgentSession", raw):
        self.session, self.raw = session, raw

    def __call__(self, *args, **kwargs):
        return self.session._complete_keyed(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.raw, name)



class AgentSession(
    AgentCommandsMixin,
    AgentToolsMixin,
    AgentChainsMixin,
    AgentSubagentsMixin,
    AgentLoopMixin,
):

    def __init__(self, session_id: str, provider, api_key: str, model: str, workspace: Workspace, mode: str, *,
                 complete=client.complete, store: Path | None = None, trust: context.TrustStore | None = None,
                 depth: int = 0, home: Path | None = None, parent: "AgentSession | None" = None,
                 agent: agents.Agent | None = None):
        keys = [api_key] if isinstance(api_key, str) else [k for k in api_key if k]
        self.id, self.provider, self.model = session_id, provider, model
        self.api_keys, self.api_key = keys, keys[0] if keys else ""
        self.key_prompt: dict | None = None
        self._key_choice: str | None = None
        self._key_lock = threading.Lock()
        self.workspace, self.mode, self.store, self.depth = workspace, mode, store, depth
        self._raw_complete = complete
        self.complete = _Keyed(self, complete)
        self.trust = trust or context.TrustStore(None)
        self.home, self.parent, self.agent = home, parent, agent
        home_path = home if home is not None else Path.home()
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.todos: list[dict] = []
        self.status = "idle"
        self.text_tools = False
        self.title = ""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        self.stats = {"model_s": 0.0, "tool_s": 0.0}
        self.live: dict | None = None
        self.created_at = self.updated_at = time.time()
        self.pending: dict | None = None
        self._decisions: dict[str, dict | None] = {}
        self._stop = False
        self._lock = threading.Condition()
        self.queue: list[str] = []
        self.tainted = False
        self._skills_loaded: list[str] = []
        self.web_ok: set[str] = set()
        self._schedule_hosts: set[str] = set()
        self._tainted_files: set[str] = set()
        self._inherit_taint = False
        self._taint_latch = False
        self._turn_usage = 0
        self._fallback_at = 0
        self._last_prompt = (0, 0)
        self._after_compact = 0
        self._deadline = self._notice_at = 0.0
        self._clock: tuple[float, float] | None = None
        self.schedules: list[dict] = []
        self._watch = watch.Watch(workspace.root)
        self._repl: repl.Kernel | None = None
        self._branches: dict[str, tuple[Path, dict]] = {}
        self.find_provider = self.list_providers = None
        self._intent_request = 0
        self._answers: list[str] = []
        self._paid_models: set[str] = set()
        self.role = ""
        self.role_usage: dict[str, dict] = {}
        self._before_summary: dict[str, str] = {}
        self._after_summary = self._repeats = 0
        self.asked_s = 0.0
        self._one_offs: dict[str, dict] = {}
        self.chain_budget: dict[str, int] = {}
        self._watch_request = self._request_no = 0
        self._scheduler: threading.Thread | None = None
        self._closing = False
        self._edited = self._plan_nudged = self._continued = self._check_nudged = self._handoff_nudged = False
        self._last_command: str | None = None
        self._failing: dict[str, int] = {}
        self._commands = 0
        self._prewalk = False
        self._interrupt: dict | None = None
        self._pinned: dict[str, str] | None = None
        self.notes = ""
        self._rollover = False
        self._last_request = ""
        self._last_header: tuple | None = None
        self.cache_log: collections.Counter = collections.Counter()
        self._fired: dict[str, int] = {}
        self.plan_mode = False
        self.effort = ""
        self.goal: dict | None = None
        self.active_skills: list[str] = []
        self._recent: list[str] = []
        self._approval_lock = threading.RLock()
        self._spawn_lock = threading.Lock()
        self.children: dict[str, AgentSession] = {}
        self.nick, self.job, self.closed, self.reported, self.mutating = "", None, False, False, False
        self.copy: Path | None = None
        self.base: dict[str, str] | None = None
        self.merge_note = ""
        self.held: str | None = None
        self.counted = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        self.jobs: dict[str, sandbox.Job] = {}
        self._usage_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._sched_lock = threading.Lock()
        self._mcp_lock = threading.Lock()
        self.agents = agents.discover(workspace.root, home_path) if not depth else {}
        self.rules = rules.load(workspace.root, home_path) if not depth else parent.rules
        self.profile = registry.load_profile(workspace.root, home_path) if not depth else parent.profile
        window = int(self.profile["context_window"])
        self.compact_at = int(window * WINDOW_SHARE) if window else int(self.profile["compact_at_tokens"])
        self.max_steps = self.profile["max_steps"]
        self._max_out = int(self.profile["max_output_tokens"])
        self.quirks = models.quirks(model, self.profile["models"])
        self._apply_caps()
        self.disabled = set(self.profile["disable"])
        self.echo_reasoning = provider.id == "deepseek" if self.profile["echo_reasoning"] is None else self.profile["echo_reasoning"]
        if depth:
            self.registry = parent.registry
        else:
            self.registry = self._build_registry(home_path)
        self.checkpoints = parent.checkpoints if depth else Checkpoints()
        self.skills = skills.discover(workspace.root, home)
        self.active_skills = [n for n in self.profile["skills_on"] if n in self.skills] if not depth else []
        self._set_read_roots()
        self.commands = context.commands(workspace.root, home) if not depth else {}
        self.hooks = context.hooks(workspace.root, home) if not depth else parent.hooks
        self.mcp_servers: dict[str, mcp.Server] = {}
        self.mcp_status: dict[str, dict] = {}
        self.mcp_tools: dict[str, tuple[str, dict]] = {}
        self.mcp_kinds: dict[str, str] = {}
        self.created = parent.created if depth else self._load_created()
        self._mcp_stale = False
        self.rate_limited = {"count": 0, "waited_s": 0.0}
        self._plain = False
        self._mcp_ready = depth > 0
        self._started = False

    def _outputs_dir(self) -> Path:
        return (self.home if self.home is not None else Path.home()) / ".manga-agent" / "outputs" / self.id.split("-")[0]

    def _set_read_roots(self) -> None:
        self.workspace.read_roots = [s.folder for s in self.skills.values()] + [self._outputs_dir()]

    @staticmethod
    def _footprint(call: dict) -> tuple[frozenset, frozenset] | None:
        name, args = call["name"], call["args"] if isinstance(call["args"], dict) else {}
        path = os.path.normpath(str(args.get("path") or ".")).lstrip("/")
        if name in PARALLEL_CALLS:
            return (frozenset({path}) if name in ("list_dir", "read_file", "search", "glob", "symbols") else frozenset()), frozenset()
        if name in ("write_file", "edit_file") and args.get("path"):
            return frozenset(), frozenset({path})
        if name == "run_command" and not args.get("background") and not args.get("outside_sandbox") and rules.safe_readonly(str(args.get("command") or "")):
            return frozenset({"."}), frozenset()
        return None


def _failed(output: str) -> bool:
    code = re.findall(r"\[exit code (-?\d+)", output[-400:])
    return bool(code and code[-1] != "0") or "Traceback (most recent call last)" in output[-3000:] or bool(re.search(r"\b\d+ failed\b|^FAILED ", output[-3000:], re.M))


def _alive(job) -> bool:
    return job.alive if hasattr(job, "alive") else job.code is None


class AgentSessionManager:

    def __init__(self, store_dir: Path | None = None, trust: context.TrustStore | None = None, home: Path | None = None):
        self.store_dir, self.home = store_dir, home
        self.find_provider = None
        self.list_providers = None
        self.trust = trust or context.TrustStore(store_dir / "trust.json" if store_dir else None)
        self.sessions: dict[str, AgentSession] = {}
        _harden_process()
        if store_dir is not None:
            sandbox.EXTRA_DENY.append(str(store_dir))
        atexit.register(self.close_all)

    def _path(self, session_id: str) -> Path | None:
        return self.store_dir / "sessions" / f"{session_id}.json" if self.store_dir else None

    def create(self, provider, api_key: str | list[str], model: str, workspace: Workspace, mode: str, complete=client.complete,
               session_id: str | None = None) -> AgentSession:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        idle = sorted((s for s in self.sessions.values() if s.status == "idle"), key=lambda s: s.updated_at)
        while len(self.sessions) >= MAX_SESSIONS and idle:
            self.sessions.pop(idle.pop(0).id).close()
        session_id = session_id or uuid.uuid4().hex[:16]
        session = AgentSession(session_id, provider, api_key, model, workspace, mode, complete=complete,
                               store=self._path(session_id), trust=self.trust, home=self.home)
        session.find_provider, session.list_providers = self.find_provider, self.list_providers
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> AgentSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def saved(self, session_id: str) -> dict:
        path = self._path(session_id)
        if path is None or not path.is_file() or not session_id.isalnum():
            raise KeyError(session_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def listing(self) -> list[dict]:
        rows = {s.id: {"id": s.id, "title": s.title, "model": s.model, "status": s.status, "updated_at": s.updated_at,
                       "workspace": str(s.workspace.root), "live": True} for s in self.sessions.values()}
        folder = self.store_dir / "sessions" if self.store_dir else None
        for path in sorted(folder.glob("*.json")) if folder and folder.is_dir() else []:
            if path.name.endswith(".meta.json") or path.stem in rows:
                continue
            meta = path.with_name(path.stem + ".meta.json")
            try:
                data = json.loads((meta if meta.is_file() else path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows[path.stem] = {"id": path.stem, "title": data.get("title", ""), "model": data.get("model", ""), "status": "saved",
                               "updated_at": data.get("updated_at", 0), "workspace": data.get("workspace", ""), "live": False}
        return sorted(rows.values(), key=lambda r: r["updated_at"], reverse=True)[:100]

    def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            session.close()
        path = self._path(session_id)
        if session_id.isalnum():
            shutil.rmtree((self.home if self.home is not None else Path.home()) / ".manga-agent" / "outputs" / session_id, ignore_errors=True)
        if path is not None and session_id.isalnum() and path.is_file():
            path.unlink()
            path.with_name(path.stem + ".meta.json").unlink(missing_ok=True)
        elif session is None:
            raise KeyError(session_id)

    def close_all(self) -> None:
        for session in list(self.sessions.values()):
            session.close()
