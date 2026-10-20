"""What the agent is built from: feature groups that can be switched off, and plugins that add tools, commands, hooks and loops."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Callable

from app.agent import context, plugin_api, roles

# The tool each group switches, so a profile can turn whole features off (shell, web, edits, subagents...).
GROUPS = {
    "list_dir": "files", "read_file": "files", "search": "files", "glob": "files", "symbols": "files", "view_image": "files",
    "write_file": "edit", "edit_file": "edit", "replace_in_files": "edit", "move_file": "edit",
    "delete_file": "edit", "checkpoint": "files",
    "run_command": "shell", "run_script": "shell", "job_output": "shell", "job_input": "shell", "job_stop": "shell", "web_fetch": "web", "web_search": "web", "web_download": "web", "skill": "skills", "mcp_resource": "mcp", "plugin_write": "extensions", "todo_write": "todo", "memory": "memory", "ask_user": "ask_user",
    "task": "subagents", "fan_out": "subagents", "spawn_agent": "subagents", "wait_agent": "subagents", "send_input": "subagents", "close_agent": "subagents",
    "schedule_create": "schedule", "schedule_list": "schedule", "schedule_delete": "schedule", "context_notes": "goal", "new_context": "goal", "exit_plan_mode": "plan", "goal_done": "goal",
   
}
COMMAND_GROUPS = {"plan": "plan", "goal": "goal", "undo": "edit", "memory": "memory", "agents": "subagents", "skills": "skills", "models": "subagents", "intent": "subagents", "mcp": "mcp"}
ALL_GROUPS = sorted(set(GROUPS.values()) | {"mcp"})
REPLACEABLE = frozenset({"list_dir", "read_file", "symbols", "view_image", "search", "glob", "write_file", "edit_file",
                         "run_command", "web_fetch", "web_search", "web_download"})
ROLES = ("read", "edit", "exec", "net")
PROFILE_FILES = {"user": (".manga-agent/profile.json",), "workspace": (".agents/profile.json",)}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
EVENTS = ("pre_tool", "post_tool", "session_start", "before_model", "after_model", "turn_end")


def group_of(tool: str) -> str | None:
    return "mcp" if tool.startswith("mcp__") else GROUPS.get(tool)


def load_profile(workspace: Path, home: Path) -> dict:
    """User profile first; a workspace profile may only switch more features off."""
    profile: dict = {"disable": [], "loop": "default", "echo_reasoning": None, "keep_jobs": False, "subagent_model": "",
                     "compact_model": "", "untrusted_guard": True, "intent": True, "token_budget": 10_000_000, "max_steps": 300,
                     "models": {}, "prices": {}, "vision": None, "isolate_writers": True, "max_output_tokens": 8192, "goal_turns": 8, "stream_rules": [], "notes_context": False,
                     "timeout_to_background": True, "compact_at_tokens": 900_000, "context_window": 0, "fallback_models": [], "prewalk_model": "",
                     "shell_only": False, "replace": {}, "roles": {}, "model_roles": {}, "model_sources": {}, "plugins": {}, "adapters": {}}
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
                if isinstance(data.get("echo_reasoning"), bool) or data.get("echo_reasoning") == "all":
                    profile["echo_reasoning"] = data["echo_reasoning"]
                if isinstance(data.get("keep_jobs"), bool):
                    profile["keep_jobs"] = data["keep_jobs"]
                for key in ("subagent_model", "compact_model", "prewalk_model"):
                    if isinstance(data.get(key), str):
                        profile[key] = data[key].strip()[:100]
                for key in ("models", "prices"):
                    if isinstance(data.get(key), dict):
                        profile[key].update({k: v for k, v in data[key].items() if isinstance(v, dict)})
                if isinstance(data.get("vision"), bool):
                    profile["vision"] = data["vision"]
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
                if isinstance(data.get("compact_at_tokens"), int) and (data["compact_at_tokens"] == 0 or 20_000 <= data["compact_at_tokens"] <= 10_000_000):
                    profile["compact_at_tokens"] = data["compact_at_tokens"]
                if isinstance(data.get("context_window"), int) and (data["context_window"] == 0 or 4_000 <= data["context_window"] <= 10_000_000):
                    profile["context_window"] = data["context_window"]
                if isinstance(data.get("timeout_to_background"), bool):
                    profile["timeout_to_background"] = data["timeout_to_background"]
                if isinstance(data.get("notes_context"), bool):
                    profile["notes_context"] = data["notes_context"]
                if isinstance(data.get("shell_only"), bool) or data.get("shell_only") == "commands":
                    profile["shell_only"] = data["shell_only"]
                if isinstance(data.get("goal_turns"), int) and data["goal_turns"] > 0:
                    profile["goal_turns"] = data["goal_turns"]
                if isinstance(data.get("max_output_tokens"), int) and data["max_output_tokens"] > 0:
                    profile["max_output_tokens"] = data["max_output_tokens"]
                if isinstance(data.get("isolate_writers"), bool):
                    profile["isolate_writers"] = data["isolate_writers"]
                if isinstance(data.get("untrusted_guard"), bool):
                    profile["untrusted_guard"] = data["untrusted_guard"]
                if isinstance(data.get("intent"), bool):
                    profile["intent"] = data["intent"]
                # Any built-in tool can be served by an MCP tool instead, and an MCP tool can be given a role.
                if isinstance(data.get("replace"), dict):
                    profile["replace"].update({k: v for k, v in data["replace"].items()
                                               if k in REPLACEABLE and isinstance(v, str) and v.startswith("mcp__")})
                if isinstance(data.get("adapters"), dict):
                    profile["adapters"].update({str(k): v for k, v in data["adapters"].items() if isinstance(v, str)})
                if isinstance(data.get("plugins"), dict):
                    profile["plugins"].update({k: v for k, v in data["plugins"].items() if isinstance(v, dict)})
                if isinstance(data.get("model_roles"), dict):
                    profile["model_roles"].update(roles.parse(data["model_roles"]))
                if isinstance(data.get("model_sources"), dict):
                    profile["model_sources"].update(roles.parse_sources(data["model_sources"]))
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
        self.providers: dict[str, Callable] = {}
        self.plugins: list[dict] = []
        self._current = ""

    def tool(self, spec: dict, handler: Callable, kind: str = "exec", group: str | None = None, always_ask: bool = False) -> None:
        """handler(session, args) returns the text the model reads; kind is read, edit, exec or net."""
        name = str(spec.get("name") or "")
        kind = spec.get("kind", kind) if kind == "exec" else kind
        if not NAME_RE.match(name):
            raise ValueError(f"tool {name!r} cannot be registered: a name is lower_case letters, digits and _, starting with a letter")
        if name in self.reserved:
            raise ValueError(f"tool {name!r} cannot be registered: a built-in tool has that name")
        if name in self.tools:
            raise ValueError(f"tool {name!r} cannot be registered: plugin {self.tools[name].group} already registered it")
        if kind not in ("read", "edit", "exec", "net"):
            raise ValueError(f"tool {name!r} cannot be registered: kind must be read, edit, exec or net, got {kind!r}; call register(spec, handler, kind)")
        if not callable(handler):
            raise ValueError(f"tool {name!r} cannot be registered: its handler is not a function")
        # The model only takes name, description and a JSON schema; a spec without parameters breaks the request.
        parameters = spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {"type": "object", "properties": {}}
        clean = {"name": name, "description": str(spec.get("description") or name), "parameters": parameters}
        self.tools[name] = ToolDef(clean, handler, kind, group or self._current or name, always_ask)

    def command(self, name: str, description: str, handler: Callable) -> None:
        """handler(session, args) returns a message string or a dict like command() does."""
        if not NAME_RE.match(name):
            raise ValueError(f"command {name!r} cannot be registered")
        self.commands[name] = (description, handler)

    def hook(self, event: str, fn: Callable) -> None:
        """pre_tool returns a message to block the call; post_tool returns text to append; the other events only observe."""
        if event not in EVENTS:
            raise ValueError(f"hook event must be one of {', '.join(EVENTS)}")
        self.hooks[event].append(fn)

    def prompt(self, fn: Callable) -> None:
        self.prompts.append(fn)

    def loop(self, name: str, fn: Callable) -> None:
        """fn(session, max_steps) replaces the step loop; the session API (_turn, _run_calls, history, emit) does the work."""
        self.loops[name] = fn

    def provider(self, name: str, adapter: Callable) -> None:
        """adapter(session, request) answers a model call for a protocol other than OpenAI chat."""
        if not NAME_RE.match(name.replace("-", "_")) or not callable(adapter):
            raise ValueError(f"model adapter {name!r} cannot be registered")
        self.providers[name] = adapter


def load_module_bytes(source: bytes, label: str, filename: str = "<plugin>"):
    """Run plugin source from bytes already read, so the trusted digest covers exactly what runs."""
    spec = importlib.util.spec_from_file_location(label, filename)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


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


def plugin_sources(workspace: Path, home: Path) -> list[tuple[str, Path, bytes | None]]:
    """Every plugin file's bytes, read exactly once.

    The trust digest and the executed code both come from these same bytes, so
    a writer swapping a file between the digest and the load cannot slip
    untrusted code under a trusted digest.
    """
    out: list[tuple[str, Path, bytes | None]] = []
    for scope, path in plugin_files(workspace, home):
        try:
            out.append((scope, path, path.read_bytes()))
        except OSError:
            out.append((scope, path, None))
    return out


def sources_digest(sources: list[tuple[str, Path, bytes | None]]) -> str:
    """Digest of the exact bytes load_plugins is about to execute (single-read).

    Uses the same construction as workspace_digest, so a digest pinned via
    trust_plugins() matches the one checked at load time.
    """
    h = hashlib.sha256()
    for scope, path, data in sources:
        if scope == "workspace" and data is not None:
            h.update(path.name.encode() + b"\0" + data)
    return h.hexdigest()


def user_trusted(trust: context.TrustStore, home: Path, path: Path, data: bytes) -> bool:
    """A user plugin is trusted file by file, for every workspace, by the digest of its exact bytes."""
    return trust.trusted(home, f"plugin-user:{path.name}", hashlib.sha256(data).hexdigest())


def trust_user_plugin(trust: context.TrustStore, home: Path, path: Path, data: bytes) -> None:
    trust.allow(home, f"plugin-user:{path.name}", hashlib.sha256(data).hexdigest())


def load_plugins(reg: Registry, workspace: Path, home: Path, trust: context.TrustStore, config: dict | None = None) -> None:
    """Python files that define register(api); a workspace's own plugins run only once the user trusts their exact contents.

    User-scope plugins need the same explicit trust: an unpinned or changed
    file is reported as "untrusted" and never executed (default-deny, like the
    fixed (g)/(h) flow). The UI surfaces this via needs_trust; trust_plugins()
    pins both scopes.
    """
    sources = plugin_sources(workspace, home)
    trusted_workspace = trust.trusted(workspace, "plugins", sources_digest(sources))
    for scope, path, data in sources:
        row = {"name": path.stem, "scope": scope, "state": "loaded", "error": ""}
        reg.plugins.append(row)
        if data is None:
            row.update(state="failed", error="could not read the plugin file")
            continue
        if scope == "workspace" and not trusted_workspace:
            row["state"] = "untrusted"
            continue
        if scope == "user" and not user_trusted(trust, home, path, data):
            row["state"] = "untrusted"
            continue
        reg._current = path.stem
        try:
            module = load_module_bytes(data, f"manga_agent_plugin_{scope}_{path.stem}", str(path))
            module.register(plugin_api.API(reg, path.stem, (config or {}).get(path.stem), home / ".manga-agent" / "plugin-state"))
        except Exception as exc:
            row.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            reg._current = ""
