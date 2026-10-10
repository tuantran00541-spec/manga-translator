"""What a workspace tells the agent: instructions, hooks, custom slash commands, and which of them the user trusts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading

from app.agent.skills import frontmatter

# Context files that restate what the code shows hurt more than they help, so only a short one is read.
MAX_INSTRUCTIONS = 8_000
# Instruction files the main coding agents read, in the order they are given to the model.
WORKSPACE_INSTRUCTIONS = ("AGENTS.md", "CLAUDE.md", ".claude/CLAUDE.md")
HOME_INSTRUCTIONS = (".manga-agent/AGENTS.md", ".codex/AGENTS.md", ".claude/CLAUDE.md")
WORKSPACE_SETTINGS = (".agents/settings.json", ".claude/settings.json", ".claude/settings.local.json")
HOME_SETTINGS = (".manga-agent/settings.json", ".claude/settings.json")
WORKSPACE_COMMANDS = (".agents/commands", ".claude/commands")
HOME_COMMANDS = (".manga-agent/commands", ".claude/commands", ".codex/prompts")
# Claude Code's tool names, so hooks written for it match the same actions here.
CLAUDE_NAMES = {"run_command": "Bash", "write_file": "Write", "edit_file": "Edit", "apply_patch": "Edit",
                "read_file": "Read", "search": "Grep", "glob": "Glob", "list_dir": "LS", "web_fetch": "WebFetch",
                "todo_write": "TodoWrite", "task": "Task"}
_trust_lock = threading.Lock()


def instructions(workspace: Path, home: Path | None = None) -> str:
    home = home if home is not None else Path.home()
    parts = []
    for path in [home / p for p in HOME_INSTRUCTIONS] + [workspace / p for p in WORKSPACE_INSTRUCTIONS]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            parts.append(f"# Instructions from {path}\n{text[:MAX_INSTRUCTIONS]}")
    return "\n\n".join(parts)


def _hooks(path: Path) -> dict[str, list[dict]]:
    """Hooks from a settings file, in either the flat form or Claude Code's nested form."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    found: dict[str, list[dict]] = {}
    for event, entries in (data.get("hooks") or {}).items() if isinstance(data, dict) else []:
        if event not in ("PreToolUse", "PostToolUse") or not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            commands = [h.get("command") for h in entry.get("hooks") or [] if isinstance(h, dict)] or [entry.get("command")]
            for command in commands:
                if command:
                    found.setdefault(event, []).append({"matcher": str(entry.get("matcher") or ""), "command": str(command),
                                                        "source": str(path)})
    return found


def hooks(workspace: Path, home: Path | None = None) -> dict[str, dict[str, list[dict]]]:
    """Hooks by scope; workspace hooks run only once the user trusts them."""
    home = home if home is not None else Path.home()
    out = {"user": {}, "workspace": {}}
    for scope, paths in (("user", [home / p for p in HOME_SETTINGS]), ("workspace", [workspace / p for p in WORKSPACE_SETTINGS])):
        for path in paths:
            for event, rows in _hooks(path).items():
                out[scope].setdefault(event, []).extend(rows)
    return out


def hook_matches(matcher: str, tool: str) -> bool:
    if matcher in ("", "*"):
        return True
    try:
        return any(re.fullmatch(matcher, name) for name in (tool, CLAUDE_NAMES.get(tool, "")) if name)
    except re.error:
        return False


def commands(workspace: Path, home: Path | None = None) -> dict[str, dict]:
    """Custom slash commands: Markdown prompt templates, workspace ones first."""
    home = home if home is not None else Path.home()
    found: dict[str, dict] = {}
    for folder in [workspace / p for p in WORKSPACE_COMMANDS] + [home / p for p in HOME_COMMANDS]:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.md")):
            name = path.relative_to(folder).with_suffix("").as_posix().replace("/", ":")
            if name in found or not re.fullmatch(r"[A-Za-z0-9_:.-]{1,64}", name):
                continue
            fields, body = frontmatter(path.read_text(encoding="utf-8", errors="replace")[:50_000])
            found[name] = {"name": name, "description": fields.get("description", body.strip().splitlines()[0][:120] if body.strip() else ""),
                           "body": body, "source": str(path)}
    return found


def expand_command(template: str, arguments: str) -> str:
    parts = arguments.split()
    text = template.replace("$ARGUMENTS", arguments)
    for i in range(9, 0, -1):
        text = text.replace(f"${i}", parts[i - 1] if len(parts) >= i else "")
    placed = "$ARGUMENTS" in template or re.search(r"\$[1-9]", template)
    return text if placed or not arguments else f"{text}\n\n{arguments}"


class TrustStore:
    """What the user allowed to start in each workspace: MCP servers and hooks, pinned to their exact config."""

    def __init__(self, path: Path | None):
        self.path = path

    def _read(self) -> dict:
        if self.path is None:
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def trusted(self, workspace: Path, key: str, digest: str) -> bool:
        return self._read().get(str(workspace), {}).get(key) == digest

    def allow(self, workspace: Path, key: str, digest: str) -> None:
        if self.path is None:
            return
        with _trust_lock:
            data = self._read()
            data.setdefault(str(workspace), {})[key] = digest
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
