"""A command may not change a repository's hooks or config: they run later outside the sandbox, so changes are undone."""
from __future__ import annotations

from pathlib import Path


def _git_dir(root: Path) -> Path | None:
    """The workspace's real git dir, following a linked-worktree pointer.

    Mirrors the resolution in sandbox.git_guard_paths so the snapshot/restore
    layer covers the same git dir the Landlock carve denies writes to.
    """
    git = root / ".git"
    if git.is_file():
        try:
            target = git.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not target.startswith("gitdir:"):
            return None
        git = (root / target[len("gitdir:"):].strip()).resolve()
    return git if git.is_dir() else None


def _watched(root: Path) -> dict[str, bytes]:
    git = _git_dir(root)
    if git is None:
        return {}
    found: dict[str, bytes] = {}
    paths = [git / "config", git / "info" / "attributes"]
    # M36: hooks can also live in submodule and worktree git dirs; a hook planted there runs just
    # the same on the user's next commit, so they are watched too.
    hook_dirs = [git / "hooks"]
    modules = git / "modules"
    if modules.is_dir():
        hook_dirs += [p / "hooks" for p in modules.rglob("hooks") if p.is_dir()]
    worktrees = git / "worktrees"
    if worktrees.is_dir():
        hook_dirs += [p / "hooks" for p in worktrees.rglob("hooks") if p.is_dir()]
    for hooks in hook_dirs:
        if hooks.is_dir():
            paths += [p for p in hooks.iterdir() if p.is_file() and not p.name.endswith(".sample")]
    for path in paths:
        try:
            found[str(path.relative_to(root))] = path.read_bytes()
        except (OSError, ValueError):
            # ValueError: path is outside root (e.g. a hooks symlink pointing
            # elsewhere, or a worktree gitdir living outside the workspace).
            continue
    return found


def snapshot(root: Path) -> dict[str, bytes]:
    return _watched(root)


def restore(root: Path, before: dict[str, bytes]) -> list[str]:
    """Put back what a command changed; returns what it touched."""
    if _git_dir(root) is None:
        return []
    after, touched = _watched(root), []
    for name, data in before.items():
        if after.get(name) != data:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
            touched.append(name)
    for name in after:
        if name not in before:
            # L8: do not delete a .git/config (or hooks) that the command legitimately created via
            # "git init" when there was no .git before the snapshot (before == {}). Deleting it
            # breaks the repo the command just created. Only remove files that appeared when a
            # baseline existed to compare against.
            if not before and name in (".git/config", ".git/HEAD"):
                continue
            (root / name).unlink(missing_ok=True)
            touched.append(name)
    return sorted(touched)
