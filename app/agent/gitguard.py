from __future__ import annotations

from pathlib import Path
import os


def _git_dirs(root: Path) -> tuple[Path, Path] | None:
    git = root / '.git'
    try:
        if git.is_file():
            line = git.read_text(encoding='utf-8').strip()
            if not line.startswith('gitdir: '):
                return None
            git = (root / line[8:]).resolve()
        if not git.is_dir():
            return None
        common_file = git / 'commondir'
        common = (git / common_file.read_text(encoding='utf-8').strip()).resolve() if common_file.is_file() else git
        return git, common
    except OSError:
        return None


def _watched(root: Path) -> dict[str, bytes]:
    dirs = _git_dirs(root)
    if dirs is None:
        return {}
    git, common = dirs
    found: dict[str, bytes] = {}
    paths = [common / "config", git / "config.worktree", common / "info" / "attributes"]
    hooks = common / "hooks"
    if hooks.is_dir():
        paths += [p for p in hooks.iterdir() if p.is_file() and not p.name.endswith(".sample")]
    for path in paths:
        try:
            found[os.path.relpath(path, root)] = path.read_bytes()
        except OSError:
            continue
    return found


def snapshot(root: Path) -> dict[str, bytes]:
    return _watched(root)


def restore(root: Path, before: dict[str, bytes]) -> list[str]:
    if _git_dirs(root) is None:
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
