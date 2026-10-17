"""A command may not change a repository's hooks or config: they run later outside the sandbox, so changes are undone."""
from __future__ import annotations

from pathlib import Path


def _watched(root: Path) -> dict[str, bytes]:
    git = root / ".git"
    if not git.is_dir():
        return {}
    found: dict[str, bytes] = {}
    paths = [git / "config", git / "info" / "attributes"]
    hooks = git / "hooks"
    if hooks.is_dir():
        paths += [p for p in hooks.iterdir() if p.is_file() and not p.name.endswith(".sample")]
    for path in paths:
        try:
            found[str(path.relative_to(root))] = path.read_bytes()
        except OSError:
            continue
    return found


def snapshot(root: Path) -> dict[str, bytes]:
    return _watched(root)


def restore(root: Path, before: dict[str, bytes]) -> list[str]:
    """Put back what a command changed; returns what it touched."""
    if not (root / ".git").is_dir():
        return []
    after, touched = _watched(root), []
    for name, data in before.items():
        if after.get(name) != data:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
            touched.append(name)
    for name in after:
        if name not in before:
            (root / name).unlink(missing_ok=True)
            touched.append(name)
    return sorted(touched)
