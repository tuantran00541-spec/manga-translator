from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from app.agent.skills import frontmatter

WORKSPACE_DIRS = (".agents/agents", ".claude/agents")
HOME_DIRS = (".manga-agent/agents", ".claude/agents")
READ_TOOLS = frozenset({"list_dir", "read_file", "search", "glob", "symbols", "view_image", "web_fetch", "web_search", "skill"})
EDIT_TOOLS = frozenset({"write_file", "edit_file", "web_download"})
ALIASES = {"read": {"read_file"}, "grep": {"search"}, "glob": {"glob"}, "ls": {"list_dir"}, "bash": {"run_command"},
           "edit": EDIT_TOOLS, "write": EDIT_TOOLS, "multiedit": EDIT_TOOLS, "webfetch": {"web_fetch"}, "websearch": {"web_search"}, "skill": {"skill"}}
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    prompt: str
    tools: frozenset[str] | None = None
    deny: frozenset[str] = field(default_factory=frozenset)
    model: str = ""

    def allowed(self, every: set[str]) -> set[str]:
        return (set(self.tools) if self.tools is not None else set(every)) & every - set(self.deny)


BUILTIN = {
    "explore": Agent("explore", "Read-only researcher: finds code and answers questions about the workspace or the web.",
                     "Role: read-only researcher. Answer the parent's question using workspace/web search and cite exact file paths, line numbers, and symbols. Never suggest or attempt edits.", READ_TOOLS),
    "plan": Agent("plan", "Designs an implementation plan without changing anything or running commands.",
                  "Role: read-only architect. Inspect the relevant files, then return an ordered plan listing exact files/symbols to edit, key risks, and the exact verification check. Never edit files or run commands.", READ_TOOLS),
    "coder": Agent("coder", "Implements one focused change, runs the checks and reports.",
                   "Role: focused implementer. Read relevant code first, apply the smallest clean diff for the assigned task, run the full affected check until PASS (never weaken tests), and report changed files plus check output."),
}


def _names(value: str) -> set[str]:
    out: set[str] = set()
    for raw in re.split(r"[,\s\[\]]+", value):
        raw = raw.strip("\"'")
        if raw:
            out |= ALIASES.get(raw.lower(), {raw})
    return out


def discover(workspace: Path, home: Path) -> dict[str, Agent]:
    found = dict(BUILTIN)
    seen: set[str] = set()
    for folder in [workspace / d for d in WORKSPACE_DIRS] + [home / d for d in HOME_DIRS]:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            try:
                fields, body = frontmatter(path.read_text(encoding="utf-8", errors="replace")[:30_000])
            except OSError:
                continue
            name = (fields.get("name") or path.stem).lower()
            if not NAME_RE.match(name) or name in seen or not fields.get("description"):
                continue
            seen.add(name)
            tools = _names(fields["tools"]) if fields.get("tools") and fields["tools"].strip() != "*" else None
            found[name] = Agent(name, fields["description"][:300], body.strip(), frozenset(tools) if tools is not None else None,
                                frozenset(_names(fields.get("disallowedTools", ""))), fields.get("model", "")[:100])
    return found


def catalog(found: dict[str, Agent]) -> str:
    return "\n".join(f"- {a.name}: {a.description}" for a in found.values())
