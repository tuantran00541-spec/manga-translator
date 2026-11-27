from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import uuid


@dataclass(frozen=True)
class _FileState:
    data: bytes
    mode: int | None = None


@dataclass(frozen=True)
class GitSnapshot:
    git_dir: Path
    common_dir: Path
    files: dict[Path, _FileState]
    dotgit_file: bytes | None = None

    def __bool__(self) -> bool:
        return bool(self.files) or self.dotgit_file is not None


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(base.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _has_symlink_between(path: Path, base: Path) -> bool:
    current = path
    base_resolved = base.resolve(strict=False)
    while True:
        if current.is_symlink():
            return True
        try:
            if current.resolve(strict=False) == base_resolved:
                return False
        except OSError:
            return True
        if current.parent == current:
            return True
        current = current.parent


def _git_dirs(root: Path) -> tuple[Path, Path, bytes | None] | None:
    dotgit = root / ".git"
    dotgit_file: bytes | None = None
    try:
        if dotgit.is_symlink():
            return None
        if dotgit.is_file():
            raw = dotgit.read_bytes()
            line = raw.decode("utf-8", errors="strict").strip()
            if not line.startswith("gitdir: "):
                return None
            dotgit_file = raw
            target = Path(line[8:].strip())
            git = (target if target.is_absolute() else root / target).resolve(strict=True)
        elif dotgit.is_dir():
            git = dotgit.resolve(strict=True)
        else:
            return None
        if not git.is_dir():
            return None
        common_file = git / "commondir"
        if common_file.is_symlink():
            return None
        if common_file.is_file():
            rel_common = common_file.read_text(encoding="utf-8", errors="strict").strip()
            common_target = Path(rel_common)
            common = (
                common_target if common_target.is_absolute() else git / common_target
            ).resolve(strict=True)
        else:
            common = git
        if not common.is_dir():
            return None
        return git, common, dotgit_file
    except (OSError, UnicodeDecodeError):
        return None


def _watched_paths(git: Path, common: Path) -> list[Path]:
    paths: list[Path] = [
        common / "config",
        git / "config.worktree",
        common / "info" / "attributes",
    ]
    hooks = common / "hooks"
    if not hooks.is_symlink() and hooks.is_dir():
        try:
            for entry in hooks.iterdir():
                if entry.name.endswith(".sample"):
                    continue
                if entry.is_symlink() or entry.is_file():
                    paths.append(entry)
        except OSError:
            pass
    return paths


def _read_regular_file(path: Path, base_dirs: tuple[Path, Path]) -> _FileState | None:
    if path.is_symlink():
        return None
    if not any(_is_within(path, b) and not _has_symlink_between(path, b) for b in base_dirs):
        return None
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return None
        return _FileState(data=path.read_bytes(), mode=stat.S_IMODE(st.st_mode))
    except OSError:
        return None


def _watched(git: Path, common: Path) -> dict[Path, _FileState]:
    found: dict[Path, _FileState] = {}
    base_dirs = (git, common)
    for path in _watched_paths(git, common):
        state = _read_regular_file(path, base_dirs)
        if state is not None:
            found[path] = state
    return found


def _display_name(root: Path, path: Path) -> str:
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return str(path)


def _safe_write_bytes(target: Path, state: _FileState, base_dirs: tuple[Path, Path]) -> None:
    if not any(_is_within(target.parent, b) and not _has_symlink_between(target.parent, b) for b in base_dirs):
        raise OSError(f"Refusing to restore outside guarded git directory: {target}")
    if target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(state.data)
        if state.mode is not None and os.name != "nt":
            try:
                os.chmod(tmp, state.mode)
            except OSError:
                pass
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def snapshot(root: Path) -> GitSnapshot | None:
    dirs = _git_dirs(root)
    if dirs is None:
        return None
    git, common, dotgit_file = dirs
    return GitSnapshot(
        git_dir=git,
        common_dir=common,
        files=_watched(git, common),
        dotgit_file=dotgit_file,
    )


def restore(root: Path, before: GitSnapshot | dict[str, bytes] | None) -> list[str]:
    if not before:
        return []

    touched: list[str] = []
    if isinstance(before, GitSnapshot):
        git, common = before.git_dir, before.common_dir
        expected_files = before.files
        if before.dotgit_file is not None:
            dotgit = root / ".git"
            try:
                if dotgit.is_symlink():
                    dotgit.unlink()
                    dotgit.write_bytes(before.dotgit_file)
                    touched.append(".git")
                elif not dotgit.is_file() or dotgit.read_bytes() != before.dotgit_file:
                    dotgit.write_bytes(before.dotgit_file)
                    touched.append(".git")
            except OSError:
                pass
    else:
        dirs = _git_dirs(root)
        if dirs is None:
            return []
        git, common, _ = dirs
        base_dirs_legacy = (git, common)
        expected_files = {}
        for name, data in before.items():
            candidate = (root / name).resolve(strict=False)
            if any(_is_within(candidate, b) for b in base_dirs_legacy):
                expected_files[candidate] = _FileState(data=data)

    base_dirs = (git, common)
    if not git.is_dir() or not common.is_dir():
        return sorted(touched)

    after_files = _watched(git, common)

    # Restore deleted, modified, or symlink-replaced watched files.
    for path, expected in expected_files.items():
        current = after_files.get(path)
        if path.is_symlink() or current is None or current.data != expected.data:
            _safe_write_bytes(path, expected, base_dirs)
            touched.append(_display_name(root, path))

    # Remove newly added files or symlinks under watched paths (e.g. injected hooks).
    for path in _watched_paths(git, common):
        if path in expected_files:
            continue
        if path.is_symlink() or path.is_file():
            if any(_is_within(path.parent, b) and not _has_symlink_between(path.parent, b) for b in base_dirs):
                path.unlink(missing_ok=True)
                touched.append(_display_name(root, path))

    return sorted(set(touched))
