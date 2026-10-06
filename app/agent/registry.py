"""What the agent is built from: feature groups that can be switched off, and plugins that add tools, commands, hooks and loops."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Callable

from app.agent import context

# The tool each group switches, so a profile can turn whole features off (shell, web, edits, subagents...).
GROUPS = {
    "list_dir": "files", "read_file": "files", "search": "files", "glob": "files", "symbols": "files", "view_image": "files",
    "write_file": "edit", "edit_file": "edit", "edit_lines": "edit", "apply_patch": "edit",
    "run_command": "shell", "run_script": "shell", "job_output": "shell", "job_input": "shell", "job_stop": "shell", "web_fetch": "web", "web_search": "web", "web_download": "web", "skill": "skills", "todo_write": "todo", "memory": "memory", "ask_user": "ask_user",
    "task": "subagents", "fan_out": "subagents", "oracle": "subagents", "spawn_agent": "subagents", "wait_agent": "subagents", "send_input": "subagents", "close_agent": "subagents",
    "schedule_create": "schedule", "schedule_list": "schedule", "schedule_delete": "schedule", "tool_search": "mcp", "context_notes": "goal", "new_context": "goal", "exit_plan_mode": "plan", "goal_done": "goal", "delegate": "external",
}
COMMAND_GROUPS = {"plan": "plan", "goal": "goal", "undo": "edit", "memory": "memory", "agents": "subagents", "skills": "skills", "mcp": "mcp"}
ALL_GROUPS = sorted(set(GROUPS.values()) | {"mcp"})
REPLACEABLE = frozenset({"list_dir", "read_file", "symbols", "view_image", "search", "glob", "write_file", "edit_file", "edit_lines",
                         "apply_patch", "run_command", "web_fetch", "web_search", "web_download"})
ROLES = ("read", "edit", "exec", "net")
PROFILE_FILES = {"user": (".manga-agent/profile.json",), "workspace": (".agents/profile.json",)}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
EVENTS = ("pre_tool", "post_tool")


def group_of(tool: str) -> str | None:
    return "mcp" if tool.startswith("mcp__") else GROUPS.get(tool)


def load_profile(workspace: Path, home: Path) -> dict:
    """User profile first; a workspace profile may only switch more features off."""
    profile: dict = {"disable": [], "loop": "default", "external_agents": {}, "echo_reasoning": None, "subagent_model": "",
                     "compact_model": "", "untrusted_guard": True, "token_budget": 10_000_000, "max_steps": 300,
                     "models": {}, "prices": {}, "vision": None, "review_model": "", "isolate_writers": True, "max_output_tokens": 8192, "goal_turns": 8, "stream_rules": [], "notes_context": False,
                     "timeout_to_background": True, "compact_at_tokens": 900_000, "context_window": 0, "oracle": False, "fallback_models": [], "advisor": False, "advisor_model": "", "advisor_every": 3, "prewalk_model": "",
                     "replace": {}, "roles": {}}
    for scope, base in (("user", home), ("workspace", workspace)):
        for name in PROFILE_FILES[scope]:
            try:
                data = json.loads((base / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            profile["disable"] += [g for g in data.get("disable") or [] if g in ALL_GROUPS]
            if scope == "user":
                if isinstance(data.get("loop"), str):
                    profile["loop"] = data["loop"]
                if isinstance(data.get("external_agents"), dict):
                    profile["external_agents"].update(data["external_agents"])
                if isinstance(data.get("echo_reasoning"), bool):
                    profile["echo_reasoning"] = data["echo_reasoning"]
                for key in ("subagent_model", "compact_model", "advisor_model", "prewalk_model"):
                    if isinstance(data.get(key), str):
                        profile[key] = data[key].strip()[:100]
                for key in ("models", "prices"):
                    if isinstance(data.get(key), dict):
                        profile[key].update({k: v for k, v in data[key].items() if isinstance(v, dict)})
                if isinstance(data.get("vision"), bool):
                    profile["vision"] = data["vision"]
                if isinstance(data.get("review_model"), str):
                    profile["review_model"] = data["review_model"].strip()[:100]
                if isinstance(data.get("max_steps"), int) and data["max_steps"] > 0:
                    profile["max_steps"] = data["max_steps"]
                if isinstance(data.get("token_budget"), int) and data["token_budget"] > 0:
                    profile["token_budget"] = data["token_budget"]
                if isinstance(data.get("stream_rules"), list):
                    for rule in data["stream_rules"][:10]:
                        try:
                            if isinstance(rule, dict) and isinstance(rule.get("message"), str) and re.compile(str(rule.get("pattern"))):
                                profile["stream_rules"].append({"name": str(rule.get("name") or rule["pattern"])[:40], "pattern": str(rule["pattern"]), "message": rule["message"][:1000]})
                        except re.error:
                            continue
                if isinstance(data.get("fallback_models"), list):
                    profile["fallback_models"] = [m.strip()[:100] for m in data["fallback_models"][:4] if isinstance(m, str) and m.strip()]
                if isinstance(data.get("oracle"), bool):
                    profile["oracle"] = data["oracle"]
                if isinstance(data.get("compact_at_tokens"), int) and (data["compact_at_tokens"] == 0 or 20_000 <= data["compact_at_tokens"] <= 10_000_000):
                    profile["compact_at_tokens"] = data["compact_at_tokens"]
                if isinstance(data.get("context_window"), int) and (data["context_window"] == 0 or 4_000 <= data["context_window"] <= 10_000_000):
                    profile["context_window"] = data["context_window"]
                if isinstance(data.get("timeout_to_background"), bool):
                    profile["timeout_to_background"] = data["timeout_to_background"]
                if isinstance(data.get("advisor"), bool):
                    profile["advisor"] = data["advisor"]
                if isinstance(data.get("advisor_every"), int) and data["advisor_every"] > 0:
                    profile["advisor_every"] = data["advisor_every"]
                if isinstance(data.get("notes_context"), bool):
                    profile["notes_context"] = data["notes_context"]
                if isinstance(data.get("goal_turns"), int) and data["goal_turns"] > 0:
                    profile["goal_turns"] = data["goal_turns"]
                if isinstance(data.get("max_output_tokens"), int) and data["max_output_tokens"] > 0:
                    profile["max_output_tokens"] = data["max_output_tokens"]
                if isinstance(data.get("isolate_writers"), bool):
                    profile["isolate_writers"] = data["isolate_writers"]
                if isinstance(data.get("untrusted_guard"), bool):
                    profile["untrusted_guard"] = data["untrusted_guard"]
                # Any built-in tool can be served by an MCP tool instead, and an MCP tool can be given a role.
                if isinstance(data.get("replace"), dict):
                    profile["replace"].update({k: v for k, v in data["replace"].items()
                                               if k in REPLACEABLE and isinstance(v, str) and v.startswith("mcp__")})
                if isinstance(data.get("roles"), dict):
                    profile["roles"].update({k: v for k, v in data["roles"].items() if str(k).startswith("mcp__") and v in ROLES})
    profile["disable"] = sorted(set(profile["disable"]))
    return profile


@dataclass
class ToolDef:
    spec: dict
    handler: Callable
    kind: str
    group: str
    always_ask: bool = False


class Registry:
    """The slots a plugin fills: tools, slash commands, hooks, prompt text and the step loop."""

    def __init__(self, reserved: set[str]):
        self.reserved = reserved
        self.tools: dict[str, ToolDef] = {}
        self.commands: dict[str, tuple[str, Callable]] = {}
        self.hooks: dict[str, list[Callable]] = {e: [] for e in EVENTS}
        self.prompts: list[Callable] = []
        self.loops: dict[str, Callable] = {}
        self.plugins: list[dict] = []
        self._current = ""

    def tool(self, spec: dict, handler: Callable, kind: str = "exec", group: str | None = None, always_ask: bool = False) -> None:
        """handler(session, args) returns the text the model reads; kind is read, edit, exec or net."""
        name = str(spec.get("name") or "")
        if not NAME_RE.match(name) or name in self.reserved or name in self.tools or kind not in ("read", "edit", "exec", "net"):
            raise ValueError(f"tool {name!r} cannot be registered")
        self.tools[name] = ToolDef(spec, handler, kind, group or self._current or name, always_ask)

    def command(self, name: str, description: str, handler: Callable) -> None:
        """handler(session, args) returns a message string or a dict like command() does."""
        if not NAME_RE.match(name):
            raise ValueError(f"command {name!r} cannot be registered")
        self.commands[name] = (description, handler)

    def hook(self, event: str, fn: Callable) -> None:
        """pre_tool(session, call) returns a message to block the call; post_tool(session, call, output) returns text to append."""
        if event not in EVENTS:
            raise ValueError(f"hook event must be one of {', '.join(EVENTS)}")
        self.hooks[event].append(fn)

    def prompt(self, fn: Callable) -> None:
        self.prompts.append(fn)

    def loop(self, name: str, fn: Callable) -> None:
        """fn(session, max_steps) replaces the step loop; the session API (_turn, _run_calls, history, emit) does the work."""
        self.loops[name] = fn


def plugin_files(workspace: Path, home: Path) -> list[tuple[str, Path]]:
    found = []
    for scope, folder in (("user", home / ".manga-agent" / "plugins"), ("workspace", workspace / ".agents" / "plugins")):
        if folder.is_dir():
            found += [(scope, p) for p in sorted(folder.glob("*.py"))]
    return found


def workspace_digest(files: list[tuple[str, Path]]) -> str:
    h = hashlib.sha256()
    for scope, path in files:
        if scope == "workspace":
            h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def load_plugins(reg: Registry, workspace: Path, home: Path, trust: context.TrustStore) -> None:
    """Python files that define register(api); a workspace's own plugins run only once the user trusts their exact contents."""
    files = plugin_files(workspace, home)
    trusted = trust.trusted(workspace, "plugins", workspace_digest(files))
    for scope, path in files:
        row = {"name": path.stem, "scope": scope, "state": "loaded", "error": ""}
        reg.plugins.append(row)
        if scope == "workspace" and not trusted:
            row["state"] = "untrusted"
            continue
        reg._current = path.stem
        try:
            spec = importlib.util.spec_from_file_location(f"manga_agent_plugin_{scope}_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.register(reg)
        except Exception as exc:
            row.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            reg._current = ""
