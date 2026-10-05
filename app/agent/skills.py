"""Agent Skills (SKILL.md folders) found in the workspace and the user's home, loaded on demand."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

# Workspace folders first, so a project's own skill wins over a user one with the same name.
WORKSPACE_DIRS = (".agents/skills", ".claude/skills", ".codex/skills")
HOME_DIRS = (".manga-agent/skills", ".agents/skills", ".claude/skills", ".codex/skills")
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
MAX_SKILLS = 200


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    folder: Path

    @property
    def file(self) -> Path:
        return self.folder / "SKILL.md"


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
    roots = [workspace / d for d in WORKSPACE_DIRS] + [home / d for d in HOME_DIRS]
    for root in roots:
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
            if NAME_RE.match(name) and description and name not in found:
                found[name] = Skill(name, description[:1024], skill_file.parent.resolve())
    return found


def catalog(found: dict[str, Skill]) -> str:
    if not found:
        return ""
    rows = "\n".join(f"- {s.name}: {s.description}" for s in found.values())
    return ("Skills you can load with the skill tool when a task matches one; load it before starting that task:\n" + rows)


def load(skill: Skill) -> str:
    """The skill's instructions with the files it ships, for the model to read next."""
    _, body = frontmatter(skill.file.read_text(encoding="utf-8", errors="replace"))
    extras = sorted(str(p.relative_to(skill.folder)) for p in skill.folder.rglob("*")
                    if p.is_file() and p.name != "SKILL.md")[:100]
    files = "\n".join(f"- {e}" for e in extras)
    tail = f"\n\nFiles in this skill (read them with read_file using the skill path):\n{files}" if extras else ""
    return f"Skill {skill.name} at {skill.folder}\n\n{body.strip()}{tail}"
