"""Allow, ask or deny rules for tool calls, matched by glob with the last matching rule winning."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import re

from app.agent import patch as patches

ACTIONS = ("allow", "ask", "deny")
CATEGORY = {"run_command": "bash", "read_file": "read", "list_dir": "read", "search": "read", "glob": "read",
            "write_file": "edit", "edit_file": "edit", "edit_lines": "edit", "apply_patch": "edit",
            "web_fetch": "webfetch", "task": "task", "spawn_agent": "task", "skill": "skill"}
# Claude Code's tool names in its settings.json permission lists.
CLAUDE_CATEGORY = {"Bash": "bash", "Read": "read", "Edit": "edit", "Write": "edit", "WebFetch": "webfetch", "Task": "task"}
SETTINGS = {"user": (".manga-agent/settings.json", ".claude/settings.json"),
            "workspace": (".agents/settings.json", ".claude/settings.json", ".claude/settings.local.json")}
DEFAULT = [("read", ".env", "deny"), ("read", ".env.*", "deny"), ("read", "*.env", "deny"),
           ("edit", ".env", "deny"), ("edit", ".env.*", "deny"), ("edit", "*.env", "deny"),
           ("read", ".env.example", "allow"), ("read", ".env.sample", "allow")]
SPLIT = re.compile(r"&&|\|\||;|\||\n")

Rule = tuple[str, str, str, str]


def _from_opencode(table: dict, scope: str) -> list[Rule]:
    rows = []
    for category, value in table.items():
        if isinstance(value, str) and value in ACTIONS:
            rows.append((scope, str(category), "*", value))
        elif isinstance(value, dict):
            rows += [(scope, str(category), str(pattern), action) for pattern, action in value.items() if action in ACTIONS]
    return rows


def _from_claude(table: dict, scope: str) -> list[Rule]:
    rows = []
    for action in ACTIONS:
        for item in table.get(action) or []:
            match = re.fullmatch(r"(\w+)(?:\((.*)\))?", str(item))
            if not match or match.group(1) not in CLAUDE_CATEGORY:
                continue
            pattern = (match.group(2) or "*").replace(":*", "*").removeprefix("domain:")
            rows.append((scope, CLAUDE_CATEGORY[match.group(1)], pattern, action))
    return rows


def load(workspace: Path, home: Path) -> list[Rule]:
    """Defaults, then user rules, then workspace rules; a workspace can only tighten, never allow."""
    rows: list[Rule] = [("default", *r) for r in DEFAULT]
    for scope, base in (("user", home), ("workspace", workspace)):
        for name in SETTINGS[scope]:
            try:
                data = json.loads((base / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            found = []
            if isinstance(data.get("permission"), dict):
                found += _from_opencode(data["permission"], scope)
            if isinstance(data.get("permissions"), dict):
                found += _from_claude(data["permissions"], scope)
            rows += [r for r in found if scope == "user" or r[3] != "allow"]
    return rows


def describe(rules: list[Rule]) -> str:
    shown = [r for r in rules if r[0] != "default"]
    return "\n".join(f"{a} {c} {p}   ({s})" for s, c, p, a in shown) or "Chưa có luật nào (.agents/settings.json, ~/.manga-agent/settings.json)."


def _match(category: str, pattern: str, subject: str) -> bool:
    if pattern == "*":
        return True
    if pattern.startswith(("~", "$HOME")):
        pattern = str(Path.home()) + pattern.removeprefix("$HOME").removeprefix("~")
    if fnmatch.fnmatchcase(subject, pattern):
        return True
    return category in ("read", "edit") and "/" not in pattern and fnmatch.fnmatchcase(subject.rsplit("/", 1)[-1], pattern)


def _verdict(rules: list[Rule], category: str, subject: str) -> str | None:
    found = None
    for _, rule_category, pattern, action in rules:
        if rule_category in (category, "*") and _match(category, pattern, subject):
            found = action
    return found


def _names_denied_path(rules: list[Rule], command: str) -> bool:
    """Best effort: a command naming a path that an edit or read rule denies is denied too; the sandbox is the real wall."""
    specific = [r for r in rules if r[2] not in ("*", "**")]
    for token in re.split(r"[\s<>|&;()'\"=]+", command):
        token = token.removeprefix("./")
        if token and not token.startswith("-") and any(_verdict(specific, c, token) == "deny" for c in ("edit", "read")):
            return True
    return False


def subjects(name: str, args: dict, path_of) -> list[str]:
    """What the rules for this call match against: commands, paths, URLs or names."""
    if name == "run_command":
        return [part.strip() for part in SPLIT.split(str(args.get("command") or "")) if part.strip()]
    if name == "apply_patch":
        try:
            hunks = patches.parse(str(args.get("patch") or ""))
        except patches.PatchError:
            return []
        return [path_of(p) for h in hunks for p in (h.path, h.move_to) if p]
    if name in ("read_file", "list_dir", "search", "glob", "write_file", "edit_file", "edit_lines"):
        return [path_of(args.get("path") or ".")]
    if name == "web_fetch":
        return [str(args.get("url") or "")]
    if name in ("task", "spawn_agent"):
        return [str(args.get("agent") or "explore")]
    if name == "skill":
        return [str(args.get("name") or "")]
    return []


def check(rules: list[Rule], name: str, args: dict, path_of) -> str | None:
    """deny, ask or allow when the rules decide this call, else None."""
    category = "mcp" if name.startswith("mcp__") else CATEGORY.get(name)
    if category is None:
        return None
    subs = [name] if category == "mcp" else subjects(name, args, path_of)
    verdicts = [_verdict(rules, category, s) for s in subs]
    if name == "run_command" and _names_denied_path(rules, str(args.get("command") or "")):
        verdicts.append("deny")
    if "deny" in verdicts:
        return "deny"
    if "ask" in verdicts:
        return "ask"
    shell_tricks = name == "run_command" and re.search(r"`|\$\(|<\(|>\(", str(args.get("command") or ""))
    if verdicts and all(v == "allow" for v in verdicts):
        return "ask" if shell_tricks else "allow"
    return None
