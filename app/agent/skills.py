from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

WORKSPACE_DIRS = (".agents/skills", ".claude/skills", ".codex/skills")
HOME_DIRS = (".manga-agent/skills", ".agents/skills", ".claude/skills", ".codex/skills")
BUILTIN_DIR = Path(__file__).parent / "builtin_skills"
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_SKIP_WALK_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv"})
MAX_SKILLS = 200
MAX_FRONTMATTER_CHARS = 20_000
MAX_SKILL_CHARS = 262_144
NAMES = (
    "(Tool names here: TodoWrite is todo_write, Task or subagents are task or spawn_agent with wait_agent, Bash is run_command, "
    "Read, Edit and Write are read_file, edit_file and write_file, asking the user is ask_user.)"
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    folder: Path
    manual: bool = False
    builtin: bool = False

    @property
    def file(self) -> Path:
        return self.folder / "SKILL.md"


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n")
    match_fm = _FRONTMATTER_RE.match(normalized)
    if not match_fm:
        return {}, text
    header, body = match_fm.group(1), normalized[match_fm.end() :].lstrip("\n")
    fields: dict[str, str] = {}
    key, block = None, []
    for line in header.splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match and not line.startswith((" ", "\t")):
            if key and block:
                fields[key] = " ".join(block).strip()
            key, value = match.group(1), match.group(2).strip()
            block = []
            if value in ("|", ">", "|-", ">-", ""):
                continue
            fields[key] = _strip_outer_quotes(value)
            key = None
        elif key:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                block.append(stripped)
    if key and block:
        fields[key] = " ".join(block).strip()
    return fields, body


def _read_prefix(path: Path, limit: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(limit)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def discover(workspace: Path, home: Path | None = None) -> dict[str, Skill]:
    home = home if home is not None else Path.home()
    found: dict[str, Skill] = {}
    raw_roots = [workspace / d for d in WORKSPACE_DIRS] + [home / d for d in HOME_DIRS] + [BUILTIN_DIR]
    seen_roots: set[Path] = set()
    builtin_resolved = BUILTIN_DIR.resolve(strict=False)

    for root in raw_roots:
        if not root.is_dir():
            continue
        try:
            root_resolved = root.resolve(strict=True)
        except OSError:
            continue
        if root_resolved in seen_roots:
            continue
        seen_roots.add(root_resolved)
        is_builtin = root_resolved == builtin_resolved
        for skill_file in sorted(root.glob("*/SKILL.md")):
            if len(found) >= MAX_SKILLS:
                return found
            if skill_file.is_symlink() or skill_file.parent.is_symlink():
                if not _is_within(skill_file.parent, root_resolved):
                    continue
            if not _is_within(skill_file, root_resolved):
                continue
            try:
                fields, _ = frontmatter(_read_prefix(skill_file, MAX_FRONTMATTER_CHARS))
            except OSError:
                continue
            name = fields.get("name") or skill_file.parent.name
            description = fields.get("description", "")
            if NAME_RE.match(name) and description and name not in found:
                found[name] = Skill(
                    name,
                    description[:1024],
                    skill_file.parent.resolve(),
                    fields.get("disable-model-invocation", "").lower() == "true",
                    is_builtin,
                )
    return found


def catalog(found: dict[str, Skill], paths: bool = False) -> str:
    shown = [s for s in found.values() if not s.manual]
    if not shown:
        return ""
    if not paths:
        rows = "\n".join(f"- {s.name}: {_short(s.description)}" for s in shown)
        return f"Skills (load with the skill tool before a task that matches, at most three per task):\n{rows}"
    groups: dict[Path, list[Skill]] = {}
    for s in shown:
        groups.setdefault(s.folder.parent, []).append(s)

    def _entry(s: Skill) -> str:
        label = f"{s.name} ({s.folder.name}/SKILL.md)" if s.folder.name != s.name else s.name
        return f"- {label}: {_short(s.description)}"

    rows = "\n".join(
        f"In {root}:\n" + "\n".join(_entry(s) for s in items)
        for root, items in groups.items()
    )
    return f"Skills (read <folder>/<name>/SKILL.md before a task that matches, at most three per task):\n{rows}"


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        prefix = cut.rsplit(" ", 1)[0]
        if len(prefix) >= limit // 2:
            cut = prefix
    return cut + "…"


def active(found: dict[str, Skill], names: list[str]) -> str:
    shown: list[str] = []
    for n in names:
        skill = found.get(n)
        if skill is None:
            continue
        try:
            shown.append(load(skill))
        except OSError:
            continue
    return (
        "Skills the user turned on (follow them for this whole session):\n\n"
        + "\n\n---\n\n".join(shown)
        if shown
        else ""
    )


def load(skill: Skill) -> str:
    _, body = frontmatter(_read_prefix(skill.file, MAX_SKILL_CHARS))
    extras: list[str] = []
    for folder, dirs, names in os.walk(skill.folder):
        dirs[:] = sorted(
            d for d in dirs if d not in _SKIP_WALK_DIRS and not (Path(folder) / d).is_symlink()
        )
        for name in sorted(names):
            if name == "SKILL.md" and Path(folder) == skill.folder:
                continue
            full = Path(folder) / name
            try:
                if not full.is_symlink() and full.is_file():
                    extras.append(str(full.relative_to(skill.folder)))
            except OSError:
                continue
            if len(extras) >= 100:
                break
        if len(extras) >= 100:
            break
    files = "\n".join(f"- {e}" for e in extras)
    tail = (
        f"\n\nFiles in this skill (read them with read_file using the skill path):\n{files}"
        if extras
        else ""
    )
    return f"Skill {skill.name} at {skill.folder}\n{NAMES}\n\n{body.strip()}{tail}"
