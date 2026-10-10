"""Notes the agent keeps between sessions: one file per project and one for the user."""
from __future__ import annotations

import hashlib
from pathlib import Path

MAX_ENTRY = 400
MAX_ENTRIES = 60
MAX_PROMPT_CHARS = 6000
SCOPES = ("project", "user")


def _file(home: Path, workspace: Path, scope: str) -> Path:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
    name = "user.md" if scope == "user" else f"project-{hashlib.sha1(str(workspace).encode()).hexdigest()[:12]}.md"
    return home / ".manga-agent" / "memory" / name


def entries(home: Path, workspace: Path, scope: str) -> list[str]:
    try:
        text = _file(home, workspace, scope).read_text(encoding="utf-8")
    except OSError:
        return []
    return [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]


def _write(home: Path, workspace: Path, scope: str, rows: list[str]) -> None:
    path = _file(home, workspace, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"- {row}\n" for row in rows), encoding="utf-8")


def add(home: Path, workspace: Path, scope: str, text: str) -> int:
    text = " ".join(str(text).split())[:MAX_ENTRY]
    if not text:
        raise ValueError("text is empty")
    rows = entries(home, workspace, scope)
    if text not in rows:
        rows.append(text)
    _write(home, workspace, scope, rows[-MAX_ENTRIES:])
    return len(rows[-MAX_ENTRIES:])


def remove(home: Path, workspace: Path, scope: str, index: int) -> str:
    rows = entries(home, workspace, scope)
    if not 1 <= index <= len(rows):
        raise ValueError(f"no {scope} note number {index}; there are {len(rows)}")
    gone = rows.pop(index - 1)
    _write(home, workspace, scope, rows)
    return gone


def prompt(home: Path, workspace: Path) -> str:
    parts = []
    for scope in SCOPES:
        rows = entries(home, workspace, scope)
        if rows:
            parts.append(f"{scope.capitalize()} notes:\n" + "\n".join(f"{i}. {row}" for i, row in enumerate(rows, 1)))
    text = "\n\n".join(parts)
    return f"# Memory (notes saved from earlier sessions; manage them with the memory tool)\n{text[:MAX_PROMPT_CHARS]}" if text else ""
