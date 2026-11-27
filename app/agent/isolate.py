from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from app.agent.tools import ROOT_SKIP_DIRS, SKIP_DIRS

MAX_FILE_BYTES = 5_000_000
MAX_TOTAL_BYTES = 200_000_000
CONFLICTS = ".agent-conflicts"
_NICK_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_safe_rel(rel: str) -> bool:
    if not rel or os.path.isabs(rel) or "\x00" in rel:
        return False
    parts = Path(rel).parts
    return bool(parts) and all(p not in ("", ".", "..") for p in parts)


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _files(top: Path):
    for folder, dirs, names in os.walk(top):
        at_root = Path(folder) == top
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS
            and d != CONFLICTS
            and not (at_root and d in ROOT_SKIP_DIRS)
            and not _is_link_or_junction(Path(folder) / d)
        ]
        for name in names:
            path = Path(folder) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                yield path


def _hash(path: Path) -> str | None:
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return None
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(base.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _has_symlink_component(target: Path, base: Path) -> bool:
    try:
        rel_parts = target.relative_to(base).parts
    except ValueError:
        return True
    cur = base
    for part in rel_parts:
        cur = cur / part
        if _is_link_or_junction(cur):
            return True
    return False


def _atomic_copy_regular(source: Path, target: Path) -> bool:
    try:
        st = source.lstat()
        if not stat.S_ISREG(st.st_mode):
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with source.open("rb") as src_f, tmp.open("wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f, length=65536)
            try:
                os.chmod(tmp, stat.S_IMODE(st.st_mode))
            except OSError:
                pass
            os.replace(tmp, target)
            return True
        except OSError:
            tmp.unlink(missing_ok=True)
            return False
    except OSError:
        return False


def snapshot(root: Path, dest: Path) -> dict[str, str] | None:
    base: dict[str, str] = {}
    total = 0
    for folder, dirs, names in os.walk(root):
        at_root = Path(folder) == root
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS
            and d != CONFLICTS
            and not (at_root and d in ROOT_SKIP_DIRS)
        ]
        for name in dirs + names:
            if _is_link_or_junction(Path(folder) / name):
                return None
        for name in names:
            path = Path(folder) / name
            try:
                st = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                return None
            size = st.st_size
            if size > MAX_FILE_BYTES:
                return None
            total += size
            if total > MAX_TOTAL_BYTES:
                return None
            rel = path.relative_to(root).as_posix()
            if not _is_safe_rel(rel):
                return None
            target = dest / rel
            if not _is_within(target, dest) or not _atomic_copy_regular(path, target):
                return None
            hashed = _hash(target)
            if hashed is None:
                return None
            base[rel] = hashed
    return base


def _sanitize_nick(nick: str) -> str:
    cleaned = str(nick or "").strip()
    if not cleaned or cleaned in (".", "..") or not _NICK_RE.fullmatch(cleaned):
        raise ValueError(f"invalid branch nick: {nick!r}")
    return cleaned


def merge(
    root: Path,
    copy: Path,
    base: dict[str, str],
    nick: str,
    before=None,
) -> tuple[list[str], list[str]]:
    safe_nick = _sanitize_nick(nick)
    seen, applied, conflicts = set(), [], []
    now = {p.relative_to(copy).as_posix(): p for p in _files(copy)}
    conflict_dir = root / CONFLICTS / safe_nick

    def put(rel: str, source: Path | None, folder: Path) -> bool:
        if not _is_safe_rel(rel):
            return False
        target = folder / rel
        if not _is_within(target, folder) or _has_symlink_component(target, folder):
            return False
        if target.exists() and target.is_dir():
            return False
        if before is not None and folder == root:
            before(target)
        try:
            if source is None:
                target.unlink(missing_ok=True)
                return True
            return _atomic_copy_regular(source, target)
        except OSError:
            return False

    for rel, path in now.items():
        if not _is_safe_rel(rel):
            continue
        seen.add(rel)
        mine = _hash(path)
        if mine is None or base.get(rel) == mine:
            continue
        current = _hash(root / rel)
        if current == mine:
            continue
        if current == base.get(rel) and put(rel, path, root):
            applied.append(rel)
        else:
            if put(rel, path, conflict_dir):
                conflicts.append(rel)
            else:
                conflicts.append(f"{rel} (conflict copy failed)")

    for rel in base:
        if rel in seen or not _is_safe_rel(rel):
            continue
        target = root / rel
        if not _is_within(target, root) or _has_symlink_component(target, root):
            continue
        current = _hash(target)
        if current is None:
            continue
        if current == base[rel] and put(rel, None, root):
            applied.append(f"{rel} (deleted)")
        else:
            conflicts.append(f"{rel} (the helper deleted it)")
    return applied, conflicts
