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
            "web_fetch": "webfetch", "web_download": "webfetch", "web_search": "websearch", "task": "task", "spawn_agent": "task", "skill": "skill"}
# Claude Code's tool names in its settings.json permission lists.
CLAUDE_CATEGORY = {"Bash": "bash", "Read": "read", "Edit": "edit", "Write": "edit", "WebFetch": "webfetch", "Task": "task"}
SETTINGS = {"user": (".manga-agent/settings.json", ".claude/settings.json"),
            "workspace": (".agents/settings.json", ".claude/settings.json", ".claude/settings.local.json")}
DEFAULT = [("read", ".env", "deny"), ("read", ".env.*", "deny"), ("read", "*.env", "deny"),
           ("edit", ".env", "deny"), ("edit", ".env.*", "deny"), ("edit", "*.env", "deny"),
           ("read", ".env.example", "allow"), ("read", ".env.sample", "allow"),
           # Git's hooks and config run commands later, outside any sandbox.
           ("edit", ".git/*", "deny"), ("edit", "*/.git/*", "deny")] + [
    # What can destroy work or leave the project: asked for unless the user's own rules or Auto mode say otherwise.
    ("bash", pattern, "ask") for pattern in (
        "rm *-r*", "rm *-R*", "rm *--recursive*", "git reset --hard*", "git clean*", "git push*", "git config*", "git checkout -- *",
        "git restore*", "git branch -D*", "git stash drop*", "git stash clear*", "sudo *", "kill *", "pkill*", "killall*", "chmod -R*",
        "chown*", "dd *", "mkfs*", "*secret-tool*", "*find-generic-password*", "*find-internet-password*", "*cmdkey*", "*keyring.get_*")]
# Commands that only look; one that is a plain call of these never needs asking about.
SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "pwd", "echo", "grep", "rg", "tree", "stat", "file", "du", "df", "sort", "uniq", "diff",
                 "which", "whoami", "date", "uname", "basename", "dirname", "realpath", "nl", "cut", "tr", "printf", "true", "id", "env"}
SAFE_GIT = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "describe", "shortlog"}
# Interpreters and wrappers run anything, so a saved allow for them is the exact command, never a prefix.
WRAPPERS = {"python", "python3", "node", "bash", "sh", "zsh", "env", "xargs", "eval", "exec", "ssh", "perl", "ruby", "npx", "uv", "uvx", "pip", "npm", "make", "docker", "find", "time", "nohup", "watch"}
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
    if name in ("web_fetch", "web_download"):
        return [str(args.get("url") or "")]
    if name == "web_search":
        return [str(args.get("query") or "")]
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


def safe_readonly(command: str) -> bool:
    """True for a command that only reads: plain calls of look-only programs, no redirects, substitutions or compound tricks beyond pipes."""
    if re.search(r"`|\$\(|<\(|>|<<", command):
        return False
    parts = [p.strip() for p in SPLIT.split(command) if p.strip()]
    for part in parts:
        words = part.split()
        if not words:
            continue
        if words[0] == "git" and len(words) > 1 and words[1] in SAFE_GIT and not any(w.startswith("--output") for w in words):
            continue
        if words[0] == "env" or words[0] not in SAFE_COMMANDS:
            return False
    return bool(parts)


def remembered(name: str, args: dict, verdict: str | None) -> tuple[str, str] | None:
    """The (category, pattern) an 'always allow' would save for this call, or None when it must be asked each time."""
    if verdict in ("ask", "deny") or not isinstance(args, dict):
        return None
    if name == "run_command":
        command = str(args.get("command") or "").strip()
        words = command.split()
        if not words or SPLIT.search(command) or re.search(r"`|\$\(|<\(|>\(|>", command) or args.get("outside_sandbox"):
            return None
        if words[0] in WRAPPERS:
            return "bash", command
        sub = words[1] if len(words) > 1 and re.fullmatch(r"[a-z][\w-]*", words[1]) else ""
        return "bash", f"{words[0]} {sub} *".replace("  ", " ") if sub else f"{words[0]} *"
    if name in ("web_fetch", "web_download"):
        host = re.match(r"https?://([^/\s:?#]+)", str(args.get("url") or ""))
        return ("webfetch", f"https://{host.group(1)}/*") if host else None
    return None


def save_allow(home: Path, category: str, pattern: str) -> None:
    """Add an allow rule to the user's own settings file."""
    path = home / SETTINGS["user"][0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    table = data.setdefault("permission", {})
    section = table.get(category)
    if not isinstance(section, dict):
        section = {"*": section} if isinstance(section, str) else {}
    section[pattern] = "allow"
    table[category] = section
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
