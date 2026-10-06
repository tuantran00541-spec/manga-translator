"""Agent sessions: the tool loop, approvals, hooks, skills, MCP tools, sub-agents, compaction and saved state."""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import platform
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from app.agent import agents, client, context, external, guardian, isolate, mcp, memory, models, registry, rules, sandbox, skill_install, skills
from app.agent.checkpoint import Checkpoints
from app.agent.tools import KIND, SPECS, ToolError, Workspace, clip
from app.logging_config import logger

MODES = ("ask", "edits", "review", "auto")
MAX_STEPS = 300
SUBAGENT_STEPS = 60
MAX_JOBS = 8
URL_TOOLS = ("web_fetch", "web_download")
MASK_KEEP = 12
MASK_BATCH = 6
OFFLOAD_AT = 12_000
MAX_SKILLS_PER_TURN = 3
MAX_AGENT_THREADS = 6
MAX_IMAGES = 3
MCP_DEFER = 15
EMPTY_RETRIES = 3
WRAP_UP = "[The token budget for this turn is used up. Do not call tools. Report in a few paragraphs what you found or built, what is not finished, and what you would do next.]"
EMPTY_NUDGE = "[Your last reply was empty. Continue the task: call a tool or answer the user in text.]"
MAX_CHILDREN = 24
WAIT_DEFAULT = 120
WAIT_MAX = 900
NICKNAMES = ("ash", "birch", "cedar", "elm", "fern", "hazel", "ivy", "juniper", "maple", "oak", "pine", "rowan", "sage", "willow",
             "alder", "beech", "clover", "dahlia", "fir", "holly", "iris", "laurel", "moss", "nettle")
COMPACT_AT = 200_000
MAX_HISTORY_CHARS = 450_000
MAX_SESSIONS = 50
HOOK_TIMEOUT = 60
MAX_REFS = 6
MAX_PARALLEL = 3
PARALLEL_CALLS = frozenset({"list_dir", "read_file", "search", "glob", "symbols", "web_fetch", "web_search", "task"})
DOOM_LOOP = 3
UNREADABLE_TURNS = 4
TOKEN_BUDGET = 10_000_000
URL_RE = re.compile(r"https?://([^\s/:?#]+)")
PLAN_TOOLS = agents.READ_TOOLS | {"todo_write", "task", "ask_user", "memory", "spawn_agent", "wait_agent", "send_input", "close_agent"}
SYSTEM_PROMPT = """You are a coding agent inside the Manga Translator app, working like Claude Code or Codex.
Workspace root: {root} on {system}. Paths are relative to it.
{sandbox}
Work in small verified steps: look first (list_dir, glob, search, read_file), then change files with apply_patch or edit_file
(or edit_lines after read_file with anchors=true), then run the project's tests or the command that proves the change works.
Use todo_write to plan work with several steps and keep it current. Use task for one job you need answered now. To run
several jobs at once call spawn_agent once per job (never the same job twice), then wait_agent; a finished agent also reports to
you by itself. Use ask_user when a decision is the user's, and memory to keep a lasting fact for later sessions.
Do not re-read a file you just changed; the tool reports failure and syntax errors. Fix root causes; keep changes minimal and in the code's style.
End with a short report of what changed, how you checked it, and anything left. Reply in the language the user writes in."""
SESSION_SPECS = {
    "skill": {"name": "skill", "description": "Load a skill's full instructions by name before doing a task it covers.",
              "parameters": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
    "todo_write": {"name": "todo_write", "description": "Replace the task's plan; each item has content and a status of "
                                                       "pending, in_progress or completed. Keep one item in_progress.",
                   "parameters": {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
                       "type": "object", "required": ["content", "status"], "properties": {
                           "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}}}}}},
    "task": {"name": "task", "description": "Send a helper agent off with one job; it returns one report. Give it a complete, "
                                            "standalone instruction.\nAgents:\n{agents}",
             "parameters": {"type": "object", "required": ["description", "prompt"], "properties": {
                 "description": {"type": "string", "description": "A few words naming the job."}, "prompt": {"type": "string"},
                 "agent": {"type": "string", "description": "Which agent; explore by default."}}}},
    "spawn_agent": {"name": "spawn_agent", "description": "Start a helper agent on one job and return at once with its name; it keeps working while you do other "
                                                         "things. Collect its report with wait_agent.\nAgents:\n{agents}",
                    "parameters": {"type": "object", "required": ["message"], "properties": {
                        "message": {"type": "string", "description": "A complete, standalone instruction."}, "agent": {"type": "string"}}}},
    "wait_agent": {"name": "wait_agent", "description": "Wait until the named agents (all unfinished ones when ids is empty) finish and return their reports; "
                                                       "after timeout_s the ones still running are listed as running.",
                   "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}},
                                                                    "timeout_s": {"type": "integer"}}}},
    "send_input": {"name": "send_input", "description": "Send a follow-up message to an agent; a finished agent wakes up and works on it.",
                   "parameters": {"type": "object", "required": ["id", "message"], "properties": {"id": {"type": "string"}, "message": {"type": "string"}}}},
    "close_agent": {"name": "close_agent", "description": "Stop an agent and release it once you no longer need it.",
                    "parameters": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}},
    "ask_user": {"name": "ask_user", "description": "Ask the user a question when a decision is theirs; optional answer choices.",
                 "parameters": {"type": "object", "required": ["question"], "properties": {
                     "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}}},
    "exit_plan_mode": {"name": "exit_plan_mode", "description": "Present your finished plan for the user's approval; editing starts only after they approve.",
                       "parameters": {"type": "object", "required": ["plan"], "properties": {"plan": {"type": "string"}}}},
    "goal_done": {"name": "goal_done", "description": "Mark the user's goal finished and verified, with a short report.",
                  "parameters": {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}}},
    "job_output": {"name": "job_output", "description": "Read what a background job printed since the last read (waiting up to wait_s seconds for more), and whether it is still running.",
                   "parameters": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}, "wait_s": {"type": "integer"}}}},
    "job_stop": {"name": "job_stop", "description": "Stop a background job.",
                 "parameters": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}},
    "tool_search": {"name": "tool_search", "description": "Find a connected (MCP) tool by what it does; the matches become callable on your next turn.",
                    "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}},
    "memory": {"name": "memory", "description": "Keep or drop a lasting note for later sessions: action add, remove or list; scope project or user.",
               "parameters": {"type": "object", "required": ["action"], "properties": {
                   "action": {"type": "string", "enum": ["add", "remove", "list"]}, "scope": {"type": "string", "enum": ["project", "user"]},
                   "text": {"type": "string"}, "index": {"type": "integer", "description": "Note number to remove."}}}},
}
BUILTIN_COMMANDS = {
    "help": "Xem các lệnh", "compact": "Tóm gọn hội thoại để giải phóng chỗ (có thể ghi điều cần giữ)",
    "init": "Viết AGENTS.md mô tả dự án này", "skills": "Xem skill; /skills add CHỦ/REPO[/THƯ-MỤC] cài skill từ GitHub",
    "mcp": "Xem MCP server và công cụ của chúng", "model": "Đổi model: /model TÊN",
    "mode": "Đổi cách duyệt: /mode ask|edits|auto",
    "sandbox": "Đổi sandbox: /sandbox read-only|workspace-write|full-access [net]",
    "clear": "Mở phiên mới", "plan": "Chế độ lập kế hoạch (chỉ đọc đến khi bạn duyệt): /plan [việc] hoặc /plan off",
    "goal": "Giao mục tiêu để agent tự làm nhiều lượt: /goal MỤC TIÊU hoặc /goal off",
    "undo": "Hoàn tác file agent đã sửa ở lượt gần nhất", "memory": "Xem ghi nhớ: /memory, /memory add NỘI DUNG, /memory rm project|user SỐ",
    "agents": "Xem các agent con", "rules": "Xem luật cho phép/hỏi/chặn",
    "plugins": "Xem plugin, tính năng đã tắt và agent ngoài (Codex, Claude Code)",
}
GOAL_PROMPT = ("Goal: {text}\nWork on it across as many steps as needed until it is fully done and verified. "
               "When it is done call goal_done with a short report; if you need a decision from the user call ask_user.")
GOAL_NUDGE = ("The goal is not marked done yet. Keep working on it. If it is finished and verified, call goal_done now; "
              "if you are blocked, call ask_user.")
PLAN_PROMPT = ("PLAN MODE: only read, search and research until the plan is ready, then call exit_plan_mode with the complete plan "
               "(files, steps, checks). Do not change anything until the user approves it.")
REF = re.compile(r"(?<![\w@/])@([^\s@]+)")
INIT_PROMPT = ("Study this repository, then write AGENTS.md at the root with ONLY what an agent cannot work out by reading the code: "
               "commands that are not obvious (install, test, lint, run), conventions the code does not show, and traps that cost time. "
               "No overview, no directory tour, no style advice the linter already enforces; under 40 lines. "
               "If AGENTS.md exists, cut it down the same way instead of adding to it.")
GATE_NUDGE = ("You changed files but have not run anything since. Run the project's tests or the command that proves the change works, "
              "and read the result, or say plainly why no check applies.")


def _harden_process() -> None:
    """Other processes of this user, including a confined command, cannot read this process's memory or environment."""
    if platform.system() == "Linux":
        try:
            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE
        except (OSError, AttributeError):
            pass


class AgentSession:
    """One conversation with one model in one workspace."""

    def __init__(self, session_id: str, provider, api_key: str, model: str, workspace: Workspace, mode: str, *,
                 complete=client.complete, store: Path | None = None, trust: context.TrustStore | None = None,
                 depth: int = 0, home: Path | None = None, parent: "AgentSession | None" = None,
                 agent: agents.Agent | None = None):
        self.id, self.provider, self.api_key, self.model = session_id, provider, api_key, model
        self.workspace, self.mode, self.complete, self.store, self.depth = workspace, mode, complete, store, depth
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
        self._decision: dict | None = None
        self._stop = False
        self._lock = threading.Condition()
        self.queue: list[str] = []
        self._dirty = self._gated = self.tainted = False
        self._skills_loaded: list[str] = []
        self.web_ok: set[str] = set()
        self._turn_usage = 0
        self.plan_mode = False
        self.goal: dict | None = None
        self._recent: list[str] = []
        self._approval_lock = threading.RLock()
        self._spawn_lock = threading.Lock()
        self.children: dict[str, AgentSession] = {}
        self.nick, self.job, self.closed, self.reported, self.mutating = "", None, False, False, False
        self.copy: Path | None = None
        self.base: dict[str, str] | None = None
        self.merge_note = ""
        self.pending: str | None = None
        self.counted = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        self.jobs: dict[str, sandbox.Job] = {}
        self._usage_lock = threading.Lock()
        self.agents = agents.discover(workspace.root, home_path) if not depth else {}
        self.rules = rules.load(workspace.root, home_path) if not depth else parent.rules
        self.profile = registry.load_profile(workspace.root, home_path) if not depth else parent.profile
        self.max_steps = SUBAGENT_STEPS if depth else self.profile["max_steps"]
        self.quirks = models.quirks(model, self.profile["models"])
        self.disabled = set(self.profile["disable"])
        self.echo_reasoning = provider.id == "deepseek" if self.profile["echo_reasoning"] is None else self.profile["echo_reasoning"]
        self.externals = external.available(self.profile["external_agents"]) if not depth and "external" not in self.disabled else {}
        self.registry = self._build_registry(home_path) if not depth else parent.registry
        self.checkpoints = parent.checkpoints if depth else Checkpoints()
        self.skills = skills.discover(workspace.root, home)
        self._set_read_roots()
        self.commands = context.commands(workspace.root, home) if not depth else {}
        self.hooks = context.hooks(workspace.root, home) if not depth else parent.hooks
        self.mcp_servers: dict[str, mcp.Server] = {}
        self.mcp_status: dict[str, dict] = {}
        self.mcp_tools: dict[str, tuple[str, dict]] = {}
        self.mcp_loaded: set[str] = set()
        self._plain = False
        self._mcp_ready = depth > 0

    def _outputs_dir(self) -> Path:
        return (self.home if self.home is not None else Path.home()) / ".manga-agent" / "outputs" / self.id.split("-")[0]

    def _set_read_roots(self) -> None:
        self.workspace.read_roots = [s.folder for s in self.skills.values()] + [self._outputs_dir()]

    # Plugins and feature groups.

    def _build_registry(self, home_path: Path) -> registry.Registry:
        reg = registry.Registry({s["name"] for s in SPECS} | set(SESSION_SPECS))
        if self.externals:
            reg.tool(external.spec(list(self.externals)), lambda session, args: external.run(session, args, self.externals),
                     kind="exec", group="external", always_ask=True)
        registry.load_plugins(reg, self.workspace.root, home_path, self.trust)
        return reg

    def _enabled(self, name: str) -> bool:
        plugin = self.registry.tools.get(name)
        return (plugin.group if plugin else registry.group_of(name)) not in self.disabled

    def trust_plugins(self) -> None:
        home_path = self.home if self.home is not None else Path.home()
        files = registry.plugin_files(self.workspace.root, home_path)
        self.trust.allow(self.workspace.root, "plugins", registry.workspace_digest(files))
        self.registry = self._build_registry(home_path)

    def _plugins_report(self) -> str:
        rows = [f"{p['name']} ({p['scope']}): {p['state']}" + (f" — {p['error']}" if p["error"] else "") for p in self.registry.plugins]
        rows += [f"Agent ngoài: {', '.join(self.externals) or 'không tìm thấy codex hay claude'}",
                 f"Tính năng đã tắt: {', '.join(sorted(self.disabled)) or 'không'}", f"Vòng lặp: {self.profile['loop']}",
                 f"Nhóm có thể tắt trong profile.json: {', '.join(registry.ALL_GROUPS)}"]
        return "\n".join(rows)

    def _hook_call(self, fn, *args) -> str:
        try:
            return str(fn(self, *args) or "")
        except Exception as exc:
            logger.warning("Agent plugin hook failed: {}", exc)
            return ""

    # Events, state and persistence.

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events) + 1, "type": kind, "time": round(time.time(), 3), **data})
            self.updated_at = time.time()
            self._lock.notify_all()

    def snapshot(self, after: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "status": self.status, "provider": self.provider.id, "provider_label": self.provider.label,
                    "model": self.model, "mode": self.mode, "workspace": str(self.workspace.root), "title": self.title,
                    "sandbox": {"mode": self.workspace.policy.mode, "network": self.workspace.policy.network,
                                "backend": sandbox.backend()},
                    "text_tools": self.text_tools, "usage": dict(self.usage), "stats": self._stats(), "live": self._live_view(),
                    "pending": self.pending, "todos": self.todos,
                    "disabled": sorted(self.disabled), "plugins": {"rows": self.registry.plugins, "externals": list(self.externals),
                                                                    "needs_trust": any(p["state"] == "untrusted" for p in self.registry.plugins)},
                    "plan_mode": self.plan_mode, "goal": self.goal and self.goal["text"], "queued": len(self.queue),
                    "agents": [{"name": a.name, "description": a.description} for a in self.agents.values()],
                    "skills": [{"name": s.name, "description": s.description} for s in self.skills.values()],
                    "mcp": list(self.mcp_status.values()), "hooks": self._hook_summary(),
                    "commands": [{"name": k, "description": v} for k, v in self._builtin_commands().items()]
                    + [{"name": k, "description": d} for k, (d, _) in self.registry.commands.items()]
                    + [{"name": s.name, "description": s.description[:120]} for s in self.skills.values() if s.manual]
                    + [{"name": c["name"], "description": c["description"]} for c in self.commands.values()],
                    "events": [e for e in self.events if e["seq"] > after]}

    def _live_view(self) -> dict | None:
        live = self.live
        if not live:
            return None
        return {"text": live["text"][-1500:], "reasoning": live["reasoning"][-600:], "tools": live["tools"]}

    def _spent(self) -> int:
        """Tokens used so far; ones served from the provider's cache are not counted again."""
        return self.usage["prompt_tokens"] - self.usage["cached_tokens"] + self.usage["completion_tokens"]

    def _stats(self) -> dict:
        prompt, cached = self.usage["prompt_tokens"], self.usage["cached_tokens"]
        price = models.prices(self.model, self.profile["prices"])
        cost = None
        if price.get("in") is not None and price.get("out") is not None:
            cost = round(((prompt - cached) * price["in"] + cached * price.get("cached", price["in"]) + self.usage["completion_tokens"] * price["out"]) / 1e6, 4)
        return {"model_seconds": round(self.stats["model_s"], 1), "tool_seconds": round(self.stats["tool_s"], 1),
                "cache_pct": round(100 * cached / prompt) if prompt else 0, "cost": cost}

    def save(self) -> None:
        if self.store is None:
            return
        data = {"id": self.id, "provider": self.provider.id, "model": self.model, "mode": self.mode, "title": self.title,
                "workspace": str(self.workspace.root), "sandbox": [self.workspace.policy.mode, self.workspace.policy.network],
                "created_at": self.created_at, "updated_at": self.updated_at, "usage": self.usage, "todos": self.todos,
                "text_tools": self.text_tools, "plan_mode": self.plan_mode, "goal": self.goal,
                "history": [{k: v for k, v in h.items() if k != "images"} for h in self.history], "events": self.events}
        tmp = self.store.with_suffix(".tmp")
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.store)
        except OSError:
            logger.opt(exception=True).warning("Could not save agent session {}", self.id)

    def restore(self, data: dict) -> None:
        self.history, self.events = list(data.get("history") or []), list(data.get("events") or [])
        self.todos, self.title = list(data.get("todos") or []), str(data.get("title") or "")
        self.usage.update(data.get("usage") or {})
        self.text_tools = bool(data.get("text_tools"))
        self.plan_mode = bool(data.get("plan_mode"))
        goal = data.get("goal")
        self.goal = {"text": str(goal.get("text", "")), "turns": int(goal.get("turns", 0))} if isinstance(goal, dict) else None
        self.created_at = float(data.get("created_at") or self.created_at)

    def set_policy(self, mode: str, network: bool) -> None:
        if mode not in sandbox.MODES:
            raise ValueError(f"sandbox must be one of {', '.join(sandbox.MODES)}")
        self.workspace.policy = sandbox.Policy(mode, bool(network))

    # Starting a turn, approvals and stopping.

    def _builtin_commands(self) -> dict[str, str]:
        return {k: v for k, v in BUILTIN_COMMANDS.items() if registry.COMMAND_GROUPS.get(k) not in self.disabled}

    def _rel(self, path: str) -> str:
        try:
            return self.workspace.rel(self.workspace.resolve(path))
        except ToolError:
            return str(path)

    def _references(self, text: str) -> tuple[str, list[str]]:
        """Files and folders the user named with @path, attached to the message."""
        blocks, names = [], []
        for raw in REF.findall(text):
            name = raw.rstrip(".,;:!?)]}\"'")
            if not name or name in names or len(names) >= MAX_REFS:
                continue
            try:
                path = self.workspace.resolve(name)
            except ToolError:
                continue
            tool = "read_file" if path.is_file() else "list_dir"
            if not path.exists() or rules.check(self.rules, tool, {"path": name}, self._rel) == "deny":
                continue
            try:
                body = self.workspace.run(tool, {"path": name} if tool == "read_file" else {"path": name, "depth": 2})
            except ToolError:
                continue
            names.append(name)
            blocks.append(f'<attached path="{name}">\n{body}\n</attached>')
        return ("\n\n" + "\n".join(blocks)) if blocks else "", names

    def send(self, text: str) -> bool:
        """Start a turn, or queue the message for the running turn; returns True when it was queued."""
        extra, names = self._references(text)
        with self._lock:
            queued = self.status in ("running", "waiting")
            if queued:
                self.queue.append(text + extra)
            else:
                self.status, self._stop = "running", False
        self.title = self.title or text.strip().splitlines()[0][:80]
        self.emit("user", text=text, refs=names, queued=queued)
        if queued:
            return True
        if not self.depth:
            self.checkpoints.begin()
        self._dirty = self._gated = self.tainted = False
        self._skills_loaded = []
        self._turn_usage = self._spent()
        self.web_ok |= {h.lower() for h in URL_RE.findall(text)}
        self.history.append({"role": "user", "content": text + extra})
        threading.Thread(target=self._loop, name=f"agent-{self.id}", daemon=True).start()
        return False

    def command(self, text: str) -> dict:
        """A slash command: built-ins act at once, custom ones become a message to the agent."""
        name, _, args = text.strip()[1:].partition(" ")
        args = args.strip()
        if registry.COMMAND_GROUPS.get(name) in self.disabled:
            raise ValueError(f"/{name} đang tắt trong profile.json")
        if name == "help":
            rows = [f"/{k} — {v}" for k, v in self._builtin_commands().items()] + [f"/{k} — {d}" for k, (d, _) in self.registry.commands.items()]
            return {"message": "\n".join(rows + [f"/{c['name']} — {c['description']}" for c in self.commands.values()])}
        if name == "plugins":
            return {"message": self._plugins_report()}
        if name in self.registry.commands:
            result = self.registry.commands[name][1](self, args)
            return result if isinstance(result, dict) else {"message": str(result)}
        if name == "skills":
            if args.split()[:1] == ["add"]:
                return {"message": self._install_skills(args[3:].strip())}
            rows = [f"{'/' if s.manual else ''}{s.name}{' (có sẵn)' if s.builtin else ''}: {s.description[:150]}" for s in self.skills.values()]
            return {"message": "\n".join(rows) or "Không có skill nào."}
        if name == "mcp":
            self._ensure_mcp()
            rows = [f"{m['name']} ({m['scope']}): {m['state']}" + (f", {m['tools']} công cụ" if m.get("tools") else "")
                    + (f" — {m['error']}" if m.get("error") else "") for m in self.mcp_status.values()]
            return {"message": "\n".join(rows) or "Chưa cấu hình MCP server nào (.mcp.json, ~/.claude.json, ~/.codex/config.toml)."}
        if name == "model" and args:
            self.model = args
            return {"message": f"Model: {args}"}
        if name == "mode" and args in MODES:
            self.mode = args
            return {"message": f"Quyền: {args}"}
        if name == "sandbox" and args:
            parts = args.split()
            self.set_policy(parts[0], len(parts) > 1 and parts[1] in ("net", "network", "on"))
            return {"message": sandbox.Policy(parts[0], self.workspace.policy.network).describe(self.workspace.root)}
        if name == "plan":
            self.plan_mode = args != "off"
            if self.plan_mode and args:
                self.send(args)
            return {"message": "Đang lập kế hoạch: agent chỉ đọc cho tới khi bạn duyệt." if self.plan_mode else "Đã tắt chế độ lập kế hoạch."}
        if name == "goal":
            if args == "off" or (not args and self.goal):
                text, self.goal = (self.goal or {}).get("text"), None if args == "off" else self.goal
                return {"message": "Đã bỏ mục tiêu." if args == "off" else f"Mục tiêu: {text}"}
            if not args:
                return {"message": "Chưa có mục tiêu. Dùng /goal MỤC TIÊU."}
            self.goal = {"text": args[:2000], "turns": 0}
            self.send(GOAL_PROMPT.format(text=args))
            return {"sent": True}
        if name == "undo":
            with self._lock:
                if self.status != "idle":
                    raise RuntimeError("The agent is still working")
            done = [self.workspace.rel(Path(p)) for p in self.checkpoints.undo()]
            self.workspace.seen.clear()
            if not done:
                return {"message": "Không có gì để hoàn tác (chỉ hoàn tác file do công cụ sửa file đã đổi, không gồm lệnh shell)."}
            self.history.append({"role": "user", "content": "[The user undid your last turn's file changes: " + ", ".join(done) + "]"})
            self.history.append({"role": "assistant", "content": "Understood; those files are back as they were.", "calls": []})
            self.emit("notice", text="Đã hoàn tác: " + ", ".join(done))
            return {"message": "Đã hoàn tác " + ", ".join(done)}
        if name == "memory":
            return {"message": self._memory_command(args)}
        if name == "agents":
            return {"message": agents.catalog(self.agents)}
        if name == "rules":
            return {"message": rules.describe(self.rules)}
        if name == "compact":
            with self._lock:
                if self.status != "idle":
                    raise RuntimeError("The agent is still working")
                self.status = "running"
            threading.Thread(target=self._compact_job, args=(args,), daemon=True).start()
            return {"message": "Đang tóm gọn hội thoại…"}
        if name == "init":
            self.send(INIT_PROMPT)
            return {"sent": True}
        if name in self.skills and self.skills[name].manual:
            self.send(skills.load(self.skills[name]) + (f"\n\nUser input: {args}" if args else ""))
            return {"sent": True}
        if name in self.commands:
            self.send(context.expand_command(self.commands[name]["body"], args))
            return {"sent": True}
        raise ValueError(f"Unknown command /{name}; type /help")

    def _install_skills(self, spec: str) -> str:
        home = self.home if self.home is not None else Path.home()
        try:
            names = skill_install.install(spec, home)
        except (ValueError, OSError) as exc:
            return f"Không cài được: {exc}"
        self.skills = skills.discover(self.workspace.root, self.home)
        self._set_read_roots()
        return f"Đã cài {len(names)} skill vào ~/.manga-agent/skills: {', '.join(names)}. Đọc kỹ SKILL.md của chúng trước khi tin."

    def _memory_command(self, args: str) -> str:
        home = self.home if self.home is not None else Path.home()
        parts = args.split(maxsplit=2)
        try:
            if parts[:1] == ["add"] and len(parts) > 1:
                memory.add(home, self.workspace.root, "project", args[4:])
            elif parts[:1] == ["rm"] and len(parts) == 3 and parts[2].isdigit():
                memory.remove(home, self.workspace.root, parts[1], int(parts[2]))
        except ValueError as exc:
            return str(exc)
        return memory.prompt(home, self.workspace.root) or "Chưa có ghi nhớ nào."

    def decide(self, decision: str, note: str = "") -> None:
        with self._lock:
            if self.status != "waiting" or self.pending is None:
                raise RuntimeError("Nothing is waiting for approval")
            self._decision = {"decision": decision, "note": note}
            self._lock.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self.queue.clear()
            self.goal = None
            self._lock.notify_all()
        for child in list(self.children.values()):
            child.pending = None
            child.stop()
        for job in list(self.jobs.values()):
            job.stop()
        self.jobs.clear()

    def close(self) -> None:
        self.stop()
        for child in list(self.children.values()):
            child.close()
            if child.copy is not None:
                shutil.rmtree(child.copy, ignore_errors=True)
        for server in self.mcp_servers.values():
            server.close()
        self.mcp_servers.clear()

    # MCP servers.

    def _ensure_mcp(self) -> None:
        if self._mcp_ready:
            return
        self._mcp_ready = True
        for row in mcp.configured(self.workspace.root, self.home):
            name, digest = row["name"], mcp.config_hash(row["config"])
            status = {"name": name, "scope": row["scope"], "source": row["source"], "digest": digest, "tools": 0, "error": ""}
            self.mcp_status[name] = status
            if row["config"].get("disabled") or row["config"].get("enabled") is False:
                status["state"] = "disabled"
            elif row["scope"] == "workspace" and not self.trust.trusted(self.workspace.root, f"mcp:{name}", digest):
                status["state"] = "untrusted"
            else:
                self._start_mcp(name, row["config"])

    def _start_mcp(self, name: str, config: dict) -> None:
        status = self.mcp_status[name]
        try:
            server = mcp.Server(name, config, self.workspace.root)
        except Exception as exc:
            status.update(state="failed", error=str(exc)[:300])
            return
        self.mcp_servers[name] = server
        for tool in server.tools:
            self.mcp_tools[mcp.tool_name(name, tool["name"])] = (name, tool)
        status.update(state="running", tools=len(server.tools), error="")

    def trust_mcp(self, name: str) -> None:
        self._ensure_mcp()
        status = self.mcp_status.get(name)
        if status is None:
            raise KeyError(name)
        row = next(r for r in mcp.configured(self.workspace.root, self.home) if r["name"] == name)
        self.trust.allow(self.workspace.root, f"mcp:{name}", mcp.config_hash(row["config"]))
        if status["state"] != "running":
            self._start_mcp(name, row["config"])

    # Hooks.

    def _hook_summary(self) -> dict:
        workspace = self.hooks.get("workspace") or {}
        digest = mcp.config_hash(workspace) if workspace else ""
        return {"user": sum(len(v) for v in (self.hooks.get("user") or {}).values()),
                "workspace": sum(len(v) for v in workspace.values()),
                "workspace_trusted": bool(digest) and self.trust.trusted(self.workspace.root, "hooks", digest)}

    def trust_hooks(self) -> None:
        workspace = self.hooks.get("workspace") or {}
        if workspace:
            self.trust.allow(self.workspace.root, "hooks", mcp.config_hash(workspace))

    def _run_hooks(self, event: str, call: dict, output: str | None = None) -> tuple[bool, str]:
        """Matching hooks for one call; a PreToolUse hook exiting 2 blocks the call with its message."""
        rows = list((self.hooks.get("user") or {}).get(event, []))
        if self._hook_summary()["workspace_trusted"]:
            rows += (self.hooks.get("workspace") or {}).get(event, [])
        notes = []
        policy = self.workspace.policy if self.workspace.policy.mode == "full-access" else sandbox.Policy("workspace-write", False)
        payload = {"hook_event_name": event, "tool_name": context.CLAUDE_NAMES.get(call["name"], call["name"]),
                   "tool_input": call["args"], "cwd": str(self.workspace.root), "session_id": self.id}
        if output is not None:
            payload["tool_response"] = output
        for row in rows:
            if not context.hook_matches(row["matcher"], call["name"]):
                continue
            code, err = sandbox.run(row["command"], policy, self.workspace.root, HOOK_TIMEOUT, json.dumps(payload), split=True)
            if code == 2 and event == "PreToolUse":
                return False, f"Blocked by a hook ({row['command']}): {err.strip()}"
            if code != 0 or err.strip():
                notes.append(f"[hook {row['command']} exit {code}] {err.strip()}"[:2000])
        return True, "\n".join(notes)

    # Tools.

    def specs(self) -> list[dict]:
        sees = models.sees_images(self.provider.id, self.model, self.profile["vision"])
        return [row for row in self._all_specs() if self._enabled(row["name"]) and (sees or row["name"] != "view_image")]

    def _all_specs(self) -> list[dict]:
        every = {s["name"] for s in SPECS} | {"skill"}
        if self.depth:
            allowed = self.agent.allowed(every) if self.agent else agents.READ_TOOLS
            rows = [s for s in SPECS if s["name"] in allowed]
            return rows + ([SESSION_SPECS["skill"]] if self.skills and "skill" in allowed else [])
        listing = agents.catalog(self.agents)
        task, spawn = (({**SESSION_SPECS[n], "description": SESSION_SPECS[n]["description"].format(agents=listing)}) for n in ("task", "spawn_agent"))
        rows = list(SPECS)
        if self.skills:
            rows.append(SESSION_SPECS["skill"])
        rows += [SESSION_SPECS["job_output"], SESSION_SPECS["job_stop"]]
        rows += [SESSION_SPECS["todo_write"], SESSION_SPECS["ask_user"], SESSION_SPECS["memory"], task, spawn,
                 SESSION_SPECS["wait_agent"], SESSION_SPECS["send_input"], SESSION_SPECS["close_agent"]]
        rows += [t.spec for t in self.registry.tools.values()]
        if self.plan_mode:
            reads = {n for n, t in self.registry.tools.items() if t.kind == "read"}
            return [r for r in rows if r["name"] in PLAN_TOOLS or r["name"] in reads] + [SESSION_SPECS["exit_plan_mode"]]
        if self.goal:
            rows.append(SESSION_SPECS["goal_done"])
        defer = len(self.mcp_tools) > MCP_DEFER
        if defer:
            names = ", ".join(self.mcp_tools)[:1500]
            rows.append({**SESSION_SPECS["tool_search"], "description": SESSION_SPECS["tool_search"]["description"] + f"\nConnected tools: {names}"})
        for name, (server, tool) in self.mcp_tools.items():
            if defer and name not in self.mcp_loaded:
                continue
            rows.append({"name": name, "description": f"[MCP {server}] {tool.get('description') or tool['name']}"[:1024],
                         "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}}})
        return rows

    def _needs_approval(self, call: dict, verdict: str | None = None) -> bool:
        name = call["name"]
        plugin = self.registry.tools.get(name)
        if plugin and plugin.always_ask:
            return True
        # Leaving the sandbox is never the model's call: it asks even when everything else is automatic.
        if name == "run_command" and call["args"].get("outside_sandbox") and self.workspace.policy.mode != "full-access":
            return True
        if name == "memory":
            return self.mode != "auto" and call["args"].get("action") != "list"
        if self.mode == "auto" or name in SESSION_SPECS or verdict == "allow":
            return False
        if verdict == "ask":
            return True
        if name in URL_TOOLS and self.mode == "edits" and (urlparse(str(call["args"].get("url") or "")).hostname or "") not in self.web_ok:
            return True
        if name in self.mcp_tools:
            read_only = (self.mcp_tools[name][1].get("annotations") or {}).get("readOnlyHint")
            return not (self.mode == "edits" and read_only)
        kind = plugin.kind if plugin else KIND.get(name, "exec")
        if kind == "read":
            return False
        if self.tainted and self.profile["untrusted_guard"] and not (
                name in URL_TOOLS and (urlparse(str(call["args"].get("url") or "")).hostname or "") in self.web_ok):
            return True
        if self.mode in ("ask", "review"):
            return not (name == "run_command" and not call["args"].get("outside_sandbox") and rules.safe_readonly(str(call["args"].get("command") or "")))
        if kind == "exec":
            # In edits mode commands run on their own only inside a working OS sandbox.
            confined = sandbox.backend() != "none" and self.workspace.policy.mode != "full-access"
            return bool(call["args"].get("outside_sandbox")) or not confined
        return False

    def _always_ask(self, call: dict) -> bool:
        plugin = self.registry.tools.get(call["name"])
        return bool(plugin and plugin.always_ask) or call["name"] == "memory" or bool(
            call["name"] == "run_command" and call["args"].get("outside_sandbox") and self.workspace.policy.mode != "full-access")

    def _user_messages(self) -> list[str]:
        root = self
        while root.parent is not None:
            root = root.parent
        return [h["content"] for h in root.history if h["role"] == "user" and not h["content"].startswith(("[", "<"))]

    def _remember(self, category: str, pattern: str) -> None:
        home_path = self.home if self.home is not None else Path.home()
        try:
            rules.save_allow(home_path, category, pattern)
        except OSError as exc:
            logger.warning("Agent session {} could not save a rule: {}", self.id, exc)
            return
        self.rules = rules.load(self.workspace.root, home_path)
        self.emit("notice", text=f"Đã lưu luật: luôn cho phép {category} {pattern}")

    def _wait_for_decision(self, call: dict) -> dict:
        if self.parent is not None:
            return self.parent._wait_for_decision({**call, "agent": self.nick or (self.agent.name if self.agent else "")})
        with self._approval_lock:
            with self._lock:
                previous = self.status
                self.pending, self._decision, self.status = call, None, "waiting"
            self.emit("approval", call=call)
            with self._lock:
                while self._decision is None and not self._stop:
                    self._lock.wait(timeout=1.0)
                decision = self._decision or {"decision": "deny", "note": "stopped"}
                self.pending, self.status = None, previous
            return decision

    def _session_tool(self, call: dict) -> str:
        args = call["args"]
        if call["name"] == "skill":
            found = self.skills.get(str(args.get("name") or ""))
            if found is None:
                raise ToolError(f"No skill named {args.get('name')!r}; known: {', '.join(self.skills) or 'none'}")
            if found.name not in self._skills_loaded:
                if len(self._skills_loaded) >= MAX_SKILLS_PER_TURN:
                    raise ToolError(f"Load at most {MAX_SKILLS_PER_TURN} skills per task; you already loaded {', '.join(self._skills_loaded)}. Use those.")
                self._skills_loaded.append(found.name)
            return skills.load(found)
        if call["name"] == "todo_write":
            items = [{"content": str(i.get("content", ""))[:300], "status": i.get("status") if i.get("status") in
                      ("pending", "in_progress", "completed") else "pending"} for i in args.get("items") or [] if isinstance(i, dict)]
            self.todos = items[:50]
            self.emit("todos", items=self.todos)
            done = sum(i["status"] == "completed" for i in self.todos)
            return f"Plan saved: {done}/{len(self.todos)} done."
        if call["name"] == "ask_user":
            decision = self._wait_for_decision(call)
            answer = decision.get("note", "").strip()
            if decision["decision"] == "deny":
                return "The user did not answer." + (f" Note: {answer}" if answer else "")
            return f"The user answered: {answer}" if answer else "The user answered with no text."
        if call["name"] == "exit_plan_mode":
            decision = self._wait_for_decision(call)
            if decision["decision"] == "deny":
                note = decision.get("note", "").strip()
                return "The user did not approve the plan; revise it." + (f" Their note: {note}" if note else "")
            self.plan_mode = False
            if decision["decision"] == "allow_all":
                self.mode = "auto"
            self.emit("notice", text="Đã duyệt kế hoạch, bắt đầu làm.")
            return "The user approved the plan. Implement it now."
        if call["name"] == "goal_done":
            self.goal = None
            self.emit("notice", text="Mục tiêu đã xong.")
            return "Goal marked done."
        if call["name"] == "memory":
            return self._memory_tool(args)
        if call["name"] == "tool_search":
            words = [w for w in re.split(r"\W+", str(args.get("query") or "").lower()) if w]
            found = [n for n, (server, t) in self.mcp_tools.items()
                     if any(w in f"{n} {t.get('description') or ''}".lower() for w in words)][:8]
            self.mcp_loaded.update(found)
            return "\n".join(f"{n}: {(self.mcp_tools[n][1].get('description') or '')[:200]}" for n in found) + "\nThese tools can be called now." if found else "No connected tool matches."
        if call["name"] == "job_output":
            return self._job_output(str(args.get("id") or ""), args.get("wait_s"))
        if call["name"] == "job_stop":
            job = self._job(str(args.get("id") or ""))
            job.stop()
            return f"Stopped {args.get('id')}. Last output:\n{clip(job.read(), 4000)}"
        if call["name"] == "spawn_agent":
            nick = self._spawn(str(args.get("message") or ""), str(args.get("agent") or "explore"))
            queued = self.children[nick].pending is not None
            lead = f"Queued {nick}: helpers that edit files run one at a time, so it starts when the one before it is done." if queued else f"Started {nick}."
            return f"{lead} Do other work, then call wait_agent with ids [\"{nick}\"]; it also reports to you by itself when done."
        if call["name"] == "wait_agent":
            return self._wait_agents(args.get("ids"), args.get("timeout_s"))
        if call["name"] == "send_input":
            child = self._child(str(args.get("id") or ""))
            child.reported = False
            child.send(str(args.get("message") or ""))
            return f"Sent to {child.nick}; call wait_agent to get its answer."
        if call["name"] == "close_agent":
            child = self._child(str(args.get("id") or ""))
            state = self._state(child)
            child.pending = None
            self._collect(child)
            child.close()
            child.closed = True
            return f"Closed {child.nick} (it was {state})."
        return self._subagent(str(args.get("description") or "task"), str(args.get("prompt") or ""), str(args.get("agent") or "explore"))

    def _job(self, name: str) -> "sandbox.Job":
        job = self.jobs.get(name.strip().lower())
        if job is None:
            raise ToolError(f"No job named {name!r}; running: {', '.join(self.jobs) or 'none'}")
        return job

    def _start_job(self, args: dict) -> str:
        for name in [n for n, j in self.jobs.items() if j.code is not None and not j.read() and time.time() - j.started > 600]:
            self.jobs.pop(name).stop()
        if len(self.jobs) >= MAX_JOBS:
            raise ToolError(f"{MAX_JOBS} background jobs already; stop some with job_stop")
        command = str(args.get("command") or "").strip()
        if not command:
            raise ToolError("command is empty")
        policy = sandbox.Policy("full-access", True) if args.get("outside_sandbox") else self.workspace.policy
        name = f"job{next(i for i in range(1, 100) if f'job{i}' not in self.jobs)}"
        self.jobs[name] = sandbox.Job(command, policy, self.workspace.root)
        time.sleep(1.0)
        job = self.jobs[name]
        state = "running" if job.code is None else f"exited with code {job.code}"
        return f"Started {name} ({state}). Read it with job_output.\n{clip(job.read(), 4000)}".rstrip()

    def _job_output(self, name: str, wait) -> str:
        job = self._job(name)
        try:
            end = time.time() + max(0, min(120, int(wait or 0)))
        except (TypeError, ValueError) as exc:
            raise ToolError("wait_s must be a whole number of seconds") from exc
        while job.code is None and time.time() < end and len(job.out.data) == job.cursor and not self._stop:
            time.sleep(0.25)
        text = job.read()
        state = "running" if job.code is None else f"exited with code {job.code}"
        return f"[{name} {state}]\n{clip(text, 20000) if text else '(no new output)'}"

    def _memory_tool(self, args: dict) -> str:
        home, root, scope = self.home if self.home is not None else Path.home(), self.workspace.root, str(args.get("scope") or "project")
        try:
            if args.get("action") == "add":
                memory.add(home, root, scope, args.get("text") or "")
            elif args.get("action") == "remove":
                memory.remove(home, root, scope, int(args.get("index") or 0))
            elif args.get("action") != "list":
                raise ToolError("action must be add, remove or list")
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return memory.prompt(home, root) or "No notes."

    def _child(self, nick: str) -> "AgentSession":
        child = self.children.get(nick.strip().lower())
        if child is None or child.closed:
            raise ToolError(f"No open agent named {nick!r}; open: {', '.join(n for n, c in self.children.items() if not c.closed) or 'none'}")
        return child

    @staticmethod
    def _state(child: "AgentSession") -> str:
        if child.closed:
            return "closed"
        if child.pending is not None:
            return "queued"
        if child.status != "idle":
            return "running"
        return "errored" if any(e["type"] == "error" for e in child.events[-3:]) else "completed"

    @staticmethod
    def _result(child: "AgentSession") -> str:
        errors = [e["text"] for e in child.events if e["type"] == "error"]
        text = next((e["text"] for e in reversed(child.events) if e["type"] == "assistant" and e["text"]), "")
        return (text or (errors[-1] if errors else "The agent finished without a report.")) + child.merge_note

    def _collect(self, child: "AgentSession") -> None:
        """Fold a finished helper's tokens into this session once and tell the page it is done."""
        if child.reported or child.status != "idle" or child.pending is not None:
            return
        child.reported = True
        self._merge_copy(child)
        self.tainted = self.tainted or child.tainted
        with self._usage_lock:
            for key in self.usage:
                self.usage[key] += child.usage[key] - child.counted[key]
                child.counted[key] = child.usage[key]
        self.emit("subagent", description=child.job[1][:80] if child.job else "", agent=child.agent.name, id=child.nick,
                  state="done", tools=sum(1 for e in child.events if e["type"] == "tool"))

    def _merge_copy(self, child: "AgentSession") -> None:
        """Bring a helper's private copy back into the project and describe the result for the model."""
        if child.copy is None:
            return
        applied, conflicts = isolate.merge(self.workspace.root, child.copy, child.base or {}, child.nick, before=self.checkpoints.save)
        shutil.rmtree(child.copy, ignore_errors=True)
        child.copy = None
        note = f"\n\n[Its work ran in a private copy and was merged: {len(applied)} files brought in" + (f" ({', '.join(applied[:12])})" if applied else "") + "."
        if conflicts:
            note += (f" {len(conflicts)} changed on both sides and were NOT merged: {', '.join(conflicts[:12])}. Its versions are in "
                     f"{isolate.CONFLICTS}/{child.nick}/; merge them by hand, then delete that folder.")
        child.merge_note = note + "]"
        self.emit("notice", text=f"Đã gộp việc của {child.nick}: {len(applied)} file" + (f", {len(conflicts)} xung đột" if conflicts else ""))

    def _spawn(self, message: str, agent_name: str, dedupe: bool = True) -> str:
        with self._spawn_lock:
            return self._spawn_locked(message, agent_name, dedupe)

    def _spawn_locked(self, message: str, agent_name: str, dedupe: bool) -> str:
        if not message.strip():
            raise ToolError("The agent needs a message")
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ToolError(f"No agent named {agent_name!r}; known: {', '.join(self.agents)}")
        job = (agent.name, " ".join(message.split()))
        for nick, other in self.children.items():
            if dedupe and not other.closed and other.job == job:
                raise ToolError(f"{nick} already has this exact job ({self._state(other)}); call wait_agent instead of starting it again")
        running = sum(1 for c in self.children.values() if self._state(c) in ("running", "queued"))
        if running >= MAX_AGENT_THREADS:
            raise ToolError(f"{running} agents are already running (limit {MAX_AGENT_THREADS}); call wait_agent for some of them first")
        if len(self.children) >= MAX_CHILDREN:
            raise ToolError(f"This session already started {MAX_CHILDREN} agents; close finished ones and continue yourself")
        allowed = agent.allowed({s["name"] for s in SPECS} | {"skill"})
        mutating = bool(allowed & (agents.EDIT_TOOLS | {"run_command"}))
        if self.plan_mode and mutating:
            raise ToolError("In plan mode only read-only agents may run")
        busy = next((n for n, c in self.children.items() if c.mutating and self._state(c) in ("running", "queued")), "") if mutating else ""
        nick = next((n for n in NICKNAMES if n not in self.children), f"agent{len(self.children) + 1}")
        copy, base = None, None
        if busy and self.profile["isolate_writers"] and not self.depth:
            # A second editing helper works in its own copy of the project; its changes are merged when its report is collected.
            copy = Path(tempfile.mkdtemp(prefix="agent-copy-"))
            base = isolate.snapshot(self.workspace.root, copy)
            if base is None:
                shutil.rmtree(copy, ignore_errors=True)
                copy = None
            else:
                busy = ""
        helper = Workspace(copy or self.workspace.root, self.workspace.policy if mutating else sandbox.Policy("read-only", False),
                           read_roots=self.workspace.read_roots)
        child = AgentSession(f"{self.id}-{nick}", self.provider, self.api_key, agent.model or self.profile.get("subagent_model") or self.model, helper,
                             self.mode if mutating else "auto", complete=self.complete, depth=self.depth + 1, home=self.home,
                             trust=self.trust, parent=self, agent=agent)
        child.text_tools, child.nick, child.job, child.mutating = self.text_tools, nick, job, mutating and copy is None
        child.copy, child.base = copy, base
        self.children[nick] = child
        self.emit("subagent", description=message[:80], agent=agent.name, id=nick, state="started")
        if busy:
            # Two helpers never edit at once: this one starts when the one before it is done.
            child.pending = message
        else:
            child.send(message)
        return nick

    def _start_queued(self) -> None:
        """Starts the next held-back editing helper once no other editing helper is working."""
        if any(c.mutating and c.pending is None and c.status != "idle" and not c.closed for c in self.children.values()):
            return
        for child in self.children.values():
            if child.pending is not None and not child.closed and not self._stop:
                message, child.pending = child.pending, None
                child.send(message)
                return

    def _wait_agents(self, ids, timeout) -> str:
        names = [str(i).strip().lower() for i in ids] if isinstance(ids, list) and ids else []
        targets = [self._child(n) for n in names] or [c for c in self.children.values() if not c.closed]
        if not targets:
            return "No agents to wait for."
        limit = max(1, min(WAIT_MAX, int(timeout or WAIT_DEFAULT)))
        end = time.time() + limit
        while time.time() < end and not self._stop and any(self._state(c) in ("running", "queued") for c in targets):
            self._start_queued()
            time.sleep(0.25)
        rows, waiting = [], False
        for child in targets:
            state = self._state(child)
            if state in ("running", "queued"):
                waiting = True
                rows.append(f"{child.nick} ({child.agent.name}): still running")
                continue
            self._collect(child)
            rows.append(f"{child.nick} ({child.agent.name}): {state}\n{self._result(child)}")
        return "\n\n".join(rows) + ("\n\n[timed out; call wait_agent again to keep waiting]" if waiting else "")

    def _notifications(self) -> None:
        """Tell the model about helpers that finished on their own, as Codex does with a notification message."""
        self._start_queued()
        for child in list(self.children.values()):
            if not child.closed and not child.reported and child.status == "idle" and child.pending is None:
                self._collect(child)
                note = {"agent_id": child.nick, "agent": child.agent.name, "status": self._state(child), "result": self._result(child)}
                self.history.append({"role": "user", "content": f"<subagent_notification>{json.dumps(note, ensure_ascii=False)}</subagent_notification>"})

    def _subagent(self, description: str, prompt: str, agent_name: str = "explore") -> str:
        """One job, started and waited for at once; the helper is closed afterwards."""
        nick = self._spawn(prompt, agent_name, dedupe=False)
        child = self.children[nick]
        while self._state(child) in ("running", "queued") and not self._stop:
            self._start_queued()
            time.sleep(0.2)
        self._collect(child)
        answer = self._result(child)
        child.close()
        child.closed = True
        return answer

    def _run_call(self, call: dict) -> tuple[str, bool]:
        if call.get("error"):
            return call["error"], False
        known = {s["name"] for s in self.specs()}
        if call["name"] not in known:
            return f"Unknown tool {call['name']!r}; available: {', '.join(sorted(known))}", False
        signature = call["name"] + json.dumps(call["args"], sort_keys=True, default=str)
        self._recent = (self._recent + [signature])[-DOOM_LOOP:]
        if len(self._recent) == DOOM_LOOP and len(set(self._recent)) == 1 and call["name"] != "wait_agent":
            self._recent = []
            self.emit("notice", text=f"Agent gọi lặp {call['name']} {DOOM_LOOP} lần giống hệt; đã chặn.")
            return (f"Blocked: you made this exact {call['name']} call {DOOM_LOOP} times in a row. "
                    "Change your approach or ask the user."), False
        verdict = rules.check(self.rules, call["name"], call["args"], self._rel) if isinstance(call["args"], dict) else None
        if verdict == "deny":
            return f"Denied by a permission rule for {call['name']}. Do not retry it; use another way or ask the user.", False
        # One question at a time: a parallel call on the same host sees the answer instead of asking again.
        with self._approval_lock:
            if self._needs_approval(call, verdict):
                why = "untrusted" if self.tainted and self.mode != "auto" and self.profile["untrusted_guard"] else ""
                reviewed, cleared = "", False
                if self.mode == "review" and not why and not self._always_ask(call):
                    started = time.time()
                    answer, reason = guardian.review(self.complete, self.provider, self.api_key, self.profile["review_model"] or self.model,
                                                     self._user_messages(), call)
                    with self._usage_lock:
                        self.stats["model_s"] += time.time() - started
                    if answer == "allow":
                        self.emit("notice", text=f"Người duyệt cho phép {call['name']}: {reason}")
                        cleared = True
                    else:
                        reviewed = reason
                if not cleared:
                    saved = None if why else rules.remembered(call["name"], call["args"], verdict)
                    extra = {**({"why": why} if why else {}), **({"remember": saved[1]} if saved else {}), **({"reviewer": reviewed} if reviewed else {})}
                    decision = self._wait_for_decision({**call, **extra} if extra else call)
                    if decision["decision"] == "allow_always" and saved:
                        self._remember(*saved)
                    if decision["decision"] == "allow_all":
                        self.mode = "auto"
                        if self.parent is not None:
                            self.parent.mode = "auto"
                    elif decision["decision"] not in ("allow", "allow_always"):
                        note = f" Note from the user: {decision['note']}" if decision.get("note") else ""
                        return f"The user refused this {call['name']} call.{note}", False
                if call["name"] in URL_TOOLS:
                    self.web_ok.add((urlparse(str(call["args"].get("url") or "")).hostname or "").lower())
        if call["name"] not in SESSION_SPECS:
            allowed, message = self._run_hooks("PreToolUse", call)
            if not allowed:
                return message, False
            for fn in self.registry.hooks["pre_tool"]:
                blocked = self._hook_call(fn, call)
                if blocked:
                    return blocked, False
        try:
            if call["name"] in SESSION_SPECS:
                output, ok = self._session_tool(call), True
            elif call["name"] in self.mcp_tools:
                server, tool = self.mcp_tools[call["name"]]
                output, ok = self.mcp_servers[server].call_tool(tool["name"], call["args"])
            elif call["name"] == "run_command" and call["args"].get("background"):
                output, ok = self._start_job(call["args"]), True
            elif call["name"] in self.registry.tools:
                try:
                    output, ok = str(self.registry.tools[call["name"]].handler(self, call["args"])), True
                except ToolError:
                    raise
                except Exception as exc:
                    output, ok = f"Error: {type(exc).__name__}: {exc}", False
            else:
                if KIND.get(call["name"]) == "edit" and isinstance(call["args"], dict):
                    for path in self.workspace.targets(call["name"], call["args"]):
                        self.checkpoints.save(path)
                output, ok = self.workspace.run(call["name"], call["args"]), True
        except ToolError as exc:
            output, ok = f"Error: {exc}", False
        except (OSError, mcp.MCPError) as exc:
            output, ok = f"Error: {type(exc).__name__}: {exc}", False
        name = call["name"]
        if name in ("web_fetch", "web_search", "web_download", "delegate") or name in self.mcp_tools:
            self.tainted = True
        if ok and KIND.get(name) == "edit" and name != "web_download":
            self._dirty = True
        elif name == "run_command":
            self._dirty = False
        if name == "run_command" or name in self.mcp_tools or name in self.registry.tools:
            output = self._offload(call, output)
        if call["name"] not in SESSION_SPECS:
            _, notes = self._run_hooks("PostToolUse", call, output)
            for fn in self.registry.hooks["post_tool"]:
                notes = "\n".join(n for n in (notes, self._hook_call(fn, call, output)) if n)
            if notes:
                output = f"{output}\n{notes}"
        return output, ok

    def _offload(self, call: dict, output: str) -> str:
        """A long output goes to a file the model can read in pieces, instead of losing its middle."""
        if len(output) <= OFFLOAD_AT:
            return output
        folder = self._outputs_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{re.sub(r'[^A-Za-z0-9_-]', '_', str(call['id']))[:40]}-{call['name'][:20]}.txt"
            path.write_text(output, encoding="utf-8")
        except OSError:
            return clip(output)
        return (f"{output[:6000]}\n… [full output saved to {path}: {len(output) - 10000} more characters; "
                f"read it with read_file offset and limit, or search it] …\n{output[-4000:]}")

    # The model turn and the loop.

    def system_prompt(self) -> str:
        parts = [SYSTEM_PROMPT.format(root=self.workspace.root, system=platform.system(),
                                      sandbox=self.workspace.policy.describe(self.workspace.root))]
        if not self.depth:
            parts.append(context.instructions(self.workspace.root, self.home))
            parts.append(memory.prompt(self.home if self.home is not None else Path.home(), self.workspace.root))
            parts += [self._hook_call(fn) for fn in self.registry.prompts]
            if self.plan_mode:
                parts.append(PLAN_PROMPT)
        else:
            parts.append(self.agent.prompt if self.agent else agents.BUILTIN["explore"].prompt)
            parts.append(f"You have at most {SUBAGENT_STEPS} steps; read only what the job needs and send your report well before they run out.")
        if self.quirks.get("prompt_extra"):
            parts.append(str(self.quirks["prompt_extra"]))
        parts.append(skills.catalog(self.skills))
        return "\n\n".join(p for p in parts if p)

    def _on_delta(self, live: dict) -> bool:
        self.live = live
        return self._stop

    def _call_model(self, messages: list[dict], tools: list[dict] | None) -> dict:
        streams = (self.complete is client.complete or getattr(self.complete, "streams", False)) and not self._plain
        started = time.time()
        try:
            extra = {"on_delta": self._on_delta} if streams else {}
            if self.complete is client.complete:
                extra["max_tokens"] = self.profile["max_output_tokens"]
            return self.complete(self.provider, self.api_key, self.model, messages, tools=tools, **extra)
        finally:
            self.stats["model_s"] += time.time() - started

    def _turn(self) -> dict:
        system, specs = self.system_prompt(), self.specs()
        try:
            return self._call_model(client.render(self.history, system, self.text_tools, specs, reasoning=self.echo_reasoning),
                                    None if self.text_tools else specs)
        except client.ToolsUnsupported as exc:
            self.text_tools = True
            self.emit("notice", text=f"Model không nhận gọi công cụ kiểu gốc, chuyển sang gọi công cụ bằng văn bản ({exc}).")
            return self._call_model(client.render(self.history, system, True, specs, reasoning=self.echo_reasoning), None)

    def _size(self) -> int:
        return sum(len(json.dumps(item, ensure_ascii=False)) for item in self.history)

    def _transcript(self, items: list[dict]) -> str:
        rows = []
        for item in items:
            if item["role"] == "user":
                rows.append(f"USER: {item['content']}")
            elif item["role"] == "assistant":
                calls = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:300]})" for c in item.get("calls") or [])
                rows.append(f"ASSISTANT: {item.get('content') or ''}" + (f"\nCALLS: {calls}" if calls else ""))
            else:
                rows.append(f"TOOL {item['name']}: {item['content'][:1500]}")
        return clip("\n\n".join(rows), 200_000)

    def compact(self, focus: str = "") -> bool:
        """Replace everything before the latest user message with a summary the model writes."""
        last_user = max((i for i, item in enumerate(self.history) if item["role"] == "user"), default=0)
        head, tail = self.history[:last_user], self.history[last_user:]
        if not head:
            return False
        ask = ("Summarise this conversation between a user and a coding agent so the agent can continue without it: "
               "the goals, decisions, files read and changed with key details, commands run and results, open problems "
               "and next steps. Be specific and complete." + (f" Focus on: {focus}" if focus else ""))
        messages = [{"role": "system", "content": "You write precise handover summaries."},
                    {"role": "user", "content": f"{ask}\n\n<conversation>\n{self._transcript(head)}\n</conversation>"}]
        summary = self.complete(self.provider, self.api_key, self.profile.get("compact_model") or self.model, messages, tools=None)["text"]
        if not summary:
            return False
        self.history = [{"role": "user", "content": f"[Summary of the earlier conversation]\n{summary}"},
                        {"role": "assistant", "content": "Understood; continuing from that summary.", "calls": []}] + tail
        self.emit("notice", text=f"Đã tóm gọn {len(head)} mục hội thoại cũ.")
        return True

    def _compact_job(self, focus: str) -> None:
        try:
            if not self.compact(focus):
                self.emit("notice", text="Chưa có gì để tóm gọn.")
        except Exception as exc:
            self.emit("error", text=f"Không tóm gọn được: {exc}"[:1000])
        finally:
            with self._lock:
                self.status = "idle"
            self.emit("done")
            self.save()

    def _mask_old(self) -> None:
        """Older tool outputs are dropped in batches, keeping the recent ones; cheaper than summaries and as good."""
        big = [i for i, item in enumerate(self.history) if item["role"] == "tool" and not item.get("masked") and len(item["content"]) > 300]
        old = big[:-MASK_KEEP] if len(big) > MASK_KEEP else []
        if len(old) < MASK_BATCH:
            return
        for i in old:
            item = self.history[i]
            saved = re.search(r"full output saved to (\S+?):", item["content"])
            item["content"] = f"[older {item['name']} output removed to save room" + (f"; the full text is in {saved.group(1)}" if saved else "; run it again if needed") + "]"
            item["masked"] = True

    def _make_room(self) -> None:
        self._mask_old()
        if self._size() > COMPACT_AT and not self.depth:
            try:
                self.compact()
            except Exception as exc:
                logger.warning("Agent session {} could not compact: {}", self.id, exc)
        size = self._size()
        for item in self.history:
            if size <= MAX_HISTORY_CHARS:
                return
            if item["role"] == "tool" and len(item["content"]) > 200:
                size -= len(item["content"]) - 40
                item["content"] = "[older tool output removed to save room]"

    def _drain(self) -> None:
        with self._lock:
            queued, self.queue = self.queue, []
        for text in queued:
            self.history.append({"role": "user", "content": text})
        self._notifications()

    def _more_work(self) -> bool:
        """True when a queued message or an unfinished goal means the turn should go on."""
        with self._lock:
            if self.queue or self._stop:
                return bool(self.queue)
        # Helpers still working: wait for one to finish, then the model reads its report.
        done = lambda: any(not c.closed and not c.reported and c.status == "idle" and c.pending is None for c in self.children.values())
        while any(self._state(c) in ("running", "queued") for c in self.children.values()) and not self._stop and not self.queue:
            self._start_queued()
            if done():
                break
            time.sleep(0.25)
        if done():
            return True
        if self._dirty and not self._gated and self.workspace.policy.mode != "read-only" and any(r["name"] == "run_command" for r in self.specs()):
            self._gated = True
            self.history.append({"role": "user", "content": GATE_NUDGE})
            self.emit("notice", text="Đã sửa file mà chưa chạy kiểm tra nào; nhắc agent chạy kiểm tra.")
            return True
        if self.goal and self.goal["turns"] < self.profile["goal_turns"] and not self.depth:
            self.goal["turns"] += 1
            self.history.append({"role": "user", "content": GOAL_NUDGE})
            self.emit("notice", text=f"Mục tiêu chưa xong, agent làm tiếp ({self.goal['turns']}/{self.profile['goal_turns']}).")
            return True
        return False

    def _run_calls(self, calls: list[dict]) -> None:
        def one(call: dict) -> tuple[str, bool]:
            if self._stop:
                return "Stopped by the user before this call ran.", False
            started = time.time()
            try:
                return self._run_call(call)
            finally:
                with self._usage_lock:
                    self.stats["tool_s"] += time.time() - started

        def record(call: dict, output: str, ok: bool) -> None:
            self.history.append({"role": "tool", "id": call["id"], "name": call["name"], "content": output})
            self.emit("tool", id=call["id"], name=call["name"], ok=ok, output=output)

        # Reads that do not depend on each other run together, up to ten at once; anything that changes things runs alone, in order.
        index = 0
        while index < len(calls):
            end = index + 1
            if calls[index]["name"] in PARALLEL_CALLS:
                while end < len(calls) and calls[end]["name"] in PARALLEL_CALLS:
                    end += 1
            batch = calls[index:end]
            if len(batch) > 1:
                workers = MAX_PARALLEL if any(c["name"] == "task" for c in batch) else 10
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(one, batch))
                for call, (output, ok) in zip(batch, results):
                    record(call, output, ok)
            else:
                record(batch[0], *one(batch[0]))
            index = end

    def _wrap_up(self) -> None:
        """One last reply without tools, so work already done is reported instead of lost."""
        self.history.append({"role": "user", "content": WRAP_UP})
        try:
            turn = self._call_model(client.render(self.history, self.system_prompt(), self.text_tools, [], reasoning=self.echo_reasoning), None)
        except Exception as exc:
            logger.warning("Agent session {} could not wrap up: {}", self.id, exc)
            return
        for key in self.usage:
            self.usage[key] += int(turn["usage"].get(key) or 0)
        self.history.append({"role": "assistant", "content": turn["text"], "calls": []})
        self.emit("assistant", text=turn["text"], reasoning="", calls=[])

    def _show_images(self) -> None:
        """Images the agent asked to see go in as one user message after its tool results; only the latest few stay."""
        images, self.workspace.new_images = self.workspace.new_images, []
        if not images:
            return
        for item in [h for h in self.history if h.get("images")][:-MAX_IMAGES + 1]:
            item["images"] = []
        self.history.append({"role": "user", "content": "Images you asked to see: " + ", ".join(name for name, _ in images),
                             "images": [url for _, url in images]})

    def _steps(self, max_steps: int) -> None:
        """The default step loop: ask the model, run its calls, repeat until it answers without calls."""
        broken = empties = 0
        for _ in range(max_steps):
            if self._stop:
                self.emit("notice", text="Đã dừng.")
                break
            self._drain()
            if self._spent() - self._turn_usage > self.profile["token_budget"]:
                self.emit("error", text=f"Đã dùng quá {self.profile['token_budget']:,} token cho lượt này; dừng. Nhắn tiếp nếu muốn agent làm tiếp.")
                self._wrap_up()
                break
            self._make_room()
            turn = self._turn()
            self.live = None
            for key in self.usage:
                self.usage[key] += int(turn["usage"].get(key) or 0)
            calls = turn["calls"]
            if not turn["text"] and not calls:
                empties += 1
                used = turn["usage"].get("completion_tokens") or 0
                debug = turn.get("debug") or {}
                if empties > EMPTY_RETRIES:
                    self.emit("error", text=f"Model trả lời rỗng {empties} lần liền ({debug}); dừng.")
                    break
                # The model wrote tool calls the server's stream parser lost; a reply without streaming carries them whole.
                if debug.get("finish") == "tool_calls" and not self._plain and self.complete is client.complete:
                    self._plain = True
                    self.emit("notice", text=f"Máy chủ làm mất lệnh gọi công cụ khi stream ({used} token); chuyển sang không stream cho phiên này.")
                else:
                    self.emit("notice", text=f"Model trả lời rỗng ({used} token, {debug}); thử lại {empties}/{EMPTY_RETRIES}.")
                    if empties == 2:
                        self.history.append({"role": "user", "content": EMPTY_NUDGE})
                continue
            empties = 0
            self.history.append({"role": "assistant", "content": turn["text"], "calls": calls,
                                 **({"reasoning": turn["reasoning"][-20000:]} if self.echo_reasoning and turn["reasoning"] else {})})
            self.emit("assistant", text=turn["text"], reasoning=turn["reasoning"][-4000:], calls=calls)
            if not calls:
                if self._more_work():
                    continue
                break
            self._run_calls(calls)
            self._show_images()
            broken = broken + 1 if all(c.get("error") or not c["name"] for c in calls) else 0
            if broken >= UNREADABLE_TURNS:
                self.emit("error", text=f"Model viết {broken} lượt liền lệnh gọi công cụ không đọc được; dừng để khỏi tốn token.")
                break
            self.save()
        else:
            self.emit("notice", text=f"Dừng sau {max_steps} bước; nhắn tiếp để agent làm tiếp.")

    def _loop(self, max_steps: int | None = None) -> None:
        max_steps = max_steps or self.max_steps
        try:
            self._ensure_mcp()
            custom = self.registry.loops.get(self.profile["loop"])
            if self.profile["loop"] != "default" and custom is None:
                self.emit("notice", text=f"Không có vòng lặp {self.profile['loop']!r}; dùng vòng lặp mặc định.")
            (custom or AgentSession._steps)(self, max_steps)
        except Exception as exc:
            logger.opt(exception=True).warning("Agent session {} failed", self.id)
            self.emit("error", text=str(exc)[:1000])
        finally:
            with self._lock:
                again = bool(self.queue) and not self._stop
                if not again:
                    self.status = "idle"
                self._lock.notify_all()
            if again:
                threading.Thread(target=self._loop, name=f"agent-{self.id}", daemon=True).start()
            else:
                self.emit("done")
            self.save()


class AgentSessionManager:
    """Live sessions by id, saved under a folder so they can be reopened after a restart."""

    def __init__(self, store_dir: Path | None = None, trust: context.TrustStore | None = None, home: Path | None = None):
        self.store_dir, self.home = store_dir, home
        self.trust = trust or context.TrustStore(store_dir / "trust.json" if store_dir else None)
        self.sessions: dict[str, AgentSession] = {}
        _harden_process()
        if store_dir is not None:
            sandbox.EXTRA_DENY.append(str(store_dir))
        atexit.register(self.close_all)

    def _path(self, session_id: str) -> Path | None:
        return self.store_dir / "sessions" / f"{session_id}.json" if self.store_dir else None

    def create(self, provider, api_key: str, model: str, workspace: Workspace, mode: str, complete=client.complete,
               session_id: str | None = None) -> AgentSession:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        idle = sorted((s for s in self.sessions.values() if s.status == "idle"), key=lambda s: s.updated_at)
        while len(self.sessions) >= MAX_SESSIONS and idle:
            self.sessions.pop(idle.pop(0).id).close()
        session_id = session_id or uuid.uuid4().hex[:16]
        session = AgentSession(session_id, provider, api_key, model, workspace, mode, complete=complete,
                               store=self._path(session_id), trust=self.trust, home=self.home)
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
            if path.stem in rows:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
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
        if self.home is not None and session_id.isalnum():
            shutil.rmtree(self.home / ".manga-agent" / "outputs" / session_id, ignore_errors=True)
        if path is not None and session_id.isalnum() and path.is_file():
            path.unlink()
        elif session is None:
            raise KeyError(session_id)

    def close_all(self) -> None:
        for session in list(self.sessions.values()):
            session.close()
