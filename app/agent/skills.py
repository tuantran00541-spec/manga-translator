"""Agent Skills (SKILL.md folders) found in the workspace and the user's home, loaded on demand."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import re

from loguru import logger

# Workspace folders first, so a project's own skill wins over a user one with the same name.
WORKSPACE_DIRS = (".agents/skills", ".claude/skills", ".codex/skills")
HOME_DIRS = (".manga-agent/skills", ".agents/skills", ".claude/skills", ".codex/skills")
# Skills shipped with the app come last, so a project's or user's skill of the same name wins.
BUILTIN_DIR = Path(__file__).parent / "builtin_skills"
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
MAX_SKILLS = 200
# A skill's body is third-party markdown injected into the model context; cap what one load can add.
MAX_BODY_CHARS = 32_000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    folder: Path
    manual: bool = False
    builtin: bool = False
    scope: str = "built-in"  # "workspace", "user" or "built-in": where it was discovered
    # True when this workspace skill took the name of a manual (slash-command) skill from the
    # user or built-in scope: typing /name then runs the workspace version instead of the user's.
    shadows_manual: bool = False

    @property
    def file(self) -> Path:
        return self.folder / "SKILL.md"


def digest(skill: Skill) -> str | None:
    """The exact bytes of the skill's SKILL.md, for pinning a trust decision to its content."""
    try:
        return hashlib.sha256(skill.file.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    """The YAML front matter's top-level scalar fields and the body after it."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    fields: dict[str, str] = {}
    key, block = None, []
    for line in text[3:end].splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match and not line.startswith((" ", "\t")):
            if key and block:
                fields[key] = " ".join(block).strip()
            key, value = match.group(1), match.group(2).strip()
            block = []
            if value in ("|", ">", "|-", ">-", ""):
                continue
            fields[key] = value.strip("\"'")
            key = None
        elif key:
            block.append(line.strip())
    if key and block:
        fields[key] = " ".join(block).strip()
    return fields, text[end + 4:].lstrip("\n")


def discover(workspace: Path, home: Path | None = None) -> dict[str, Skill]:
    home = home if home is not None else Path.home()
    found: dict[str, Skill] = {}
    roots = [workspace / d for d in WORKSPACE_DIRS] + [home / d for d in HOME_DIRS] + [BUILTIN_DIR]
    scopes = (["workspace"] * len(WORKSPACE_DIRS) + ["user"] * len(HOME_DIRS) + ["built-in"])
    origin: dict[str, str] = {}
    for scope, root in zip(scopes, roots):
        if not root.is_dir():
            continue
        for skill_file in sorted(root.glob("*/SKILL.md")):
            if len(found) >= MAX_SKILLS:
                return found
            try:
                fields, _ = frontmatter(skill_file.read_text(encoding="utf-8", errors="replace")[:20_000])
            except OSError:
                continue
            name = fields.get("name") or skill_file.parent.name
            description = fields.get("description", "")
            if not (NAME_RE.match(name) and description):
                continue
            if name in found:
                # The nearer scope wins, but say so loudly: a project can
                # silently replace a built-in or user skill just by reusing its
                # name, and its instructions are then loaded into the session.
                logger.warning("Skill {!r} from the {} scope ({}) shadows the {} skill at {}; "
                               "the nearer scope wins and its instructions will be loaded instead.",
                               name, scope, skill_file.parent, origin[name], found[name].folder)
                winner = found[name]
                # The loser is the skill being processed now (from the farther scope); the winner
                # is the nearer one already in found.
                if winner.scope == "workspace" and scope != "workspace" and not winner.shadows_manual:
                    # A workspace skill taking the name of a user's (or built-in) manual skill is the
                    # dangerous case: /name would run the project's instructions instead of the user's.
                    loser_manual = fields.get("disable-model-invocation", "").lower() == "true"
                    if loser_manual:
                        found[name] = replace(winner, shadows_manual=True)
                continue
            found[name] = Skill(name, description[:1024], skill_file.parent.resolve(),
                                fields.get("disable-model-invocation", "").lower() == "true", root == BUILTIN_DIR,
                                scope)
            origin[name] = scope
    return found


def catalog(found: dict[str, Skill]) -> str:
    """The skills the model may load itself; manual ones are only for the user's /commands."""
    rows = "\n".join(f"- {s.name}: {s.description}" for s in found.values() if not s.manual)
    if not rows:
        return ""
    return ("Skills you can load with the skill tool when a task matches one; load it before starting that task, and at most three per task. "
            "Skills are written for other coding agents too: read TodoWrite as todo_write, Task or subagents as task or "
            "spawn_agent with wait_agent, Bash as run_command, Read, Edit and Write as read_file, edit_file and apply_patch, "
            "and asking the user as ask_user.\n" + rows)


def load(skill: Skill) -> str:
    """The skill's instructions with the files it ships, for the model to read next."""
    _, body = frontmatter(skill.file.read_text(encoding="utf-8", errors="replace"))
    if len(body) > MAX_BODY_CHARS:
        # A skill is third-party markdown that lands in the model context; cap it so a
        # multi-megabyte SKILL.md cannot blow up the context window or the bill.
        body = body[:MAX_BODY_CHARS] + f"\n\n[...{len(body) - MAX_BODY_CHARS} chars truncated...]"
    extras = sorted(str(p.relative_to(skill.folder)) for p in skill.folder.rglob("*")
                    if p.is_file() and p.name != "SKILL.md")[:100]
    files = "\n".join(f"- {e}" for e in extras)
    tail = f"\n\nFiles in this skill (read them with read_file using the skill path):\n{files}" if extras else ""
    return f"Skill {skill.name} at {skill.folder}\n\n{body.strip()}{tail}"
