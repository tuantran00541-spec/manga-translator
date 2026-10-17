"""Private copies of the workspace for helpers that edit at the same time, and a file-level three-way merge of their work."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

from app.agent.tools import ROOT_SKIP_DIRS, SKIP_DIRS

MAX_FILE_BYTES = 5_000_000
MAX_TOTAL_BYTES = 200_000_000
CONFLICTS = ".agent-conflicts"


def _files(top: Path):
    for folder, dirs, names in os.walk(top):
        at_root = Path(folder) == top
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d != CONFLICTS and not (at_root and d in ROOT_SKIP_DIRS)]
        for name in names:
            path = Path(folder) / name
            if not path.is_symlink():
                yield path


def _hash(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def snapshot(root: Path, dest: Path) -> dict[str, str] | None:
    """Copy the workspace into dest and return each copied file's hash, or None when it is too big to copy."""
    base: dict[str, str] = {}
    total = 0
    for path in _files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            continue
        total += size
        if total > MAX_TOTAL_BYTES:
            return None
        rel = path.relative_to(root).as_posix()
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        base[rel] = _hash(target) or ""
    return base


def merge(root: Path, copy: Path, base: dict[str, str], nick: str, before=None) -> tuple[list[str], list[str]]:
    """Bring the copy's changes into root; a file that changed on both sides is left alone and kept under .agent-conflicts."""
    seen, applied, conflicts = set(), [], []
    now = {p.relative_to(copy).as_posix(): p for p in _files(copy)}

    def put(rel: str, source: Path | None, folder: Path) -> None:
        target = folder / rel
        if before is not None and folder == root:
            before(target)
        if source is None:
            target.unlink(missing_ok=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for rel, path in now.items():
        seen.add(rel)
        mine = _hash(path)
        if base.get(rel) == mine or (rel not in base and mine is None):
            continue
        current = _hash(root / rel)
        if current == mine:
            continue
        if current == base.get(rel):
            put(rel, path, root)
            applied.append(rel)
        else:
            put(rel, path, root / CONFLICTS / nick)
            conflicts.append(rel)
    for rel in base:
        if rel in seen:
            continue
        current = _hash(root / rel)
        if current is None:
            continue
        if current == base[rel]:
            put(rel, None, root)
            applied.append(f"{rel} (deleted)")
        else:
            conflicts.append(f"{rel} (the helper deleted it)")
    return applied, conflicts
