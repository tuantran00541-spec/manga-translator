from __future__ import annotations

import os
from pathlib import Path
import stat

SKIP = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
})
MAX_FILES = 20_000
MAX_FILE_BYTES = 512_000
MAX_TOTAL_BYTES = 64_000_000
UNKNOWN = object()


def _read_capped(path: Path, limit: int = MAX_FILE_BYTES) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            return None
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            data = handle.read(limit + 1)
            return None if len(data) > limit else data
    except OSError:
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


class Watch:
    def __init__(self, root: Path):
        self.root = root
        self.known: dict[str, tuple[int, int]] = {}
        self.content: dict[str, bytes] = {}
        self.ready = False

    def scan(self) -> dict[str, tuple[int, int]]:
        found: dict[str, tuple[int, int]] = {}
        stack = [self.root]
        from app.agent.tools import ROOT_SKIP_DIRS
        while stack and len(found) < MAX_FILES:
            try:
                folder = stack.pop()
                with os.scandir(folder) as entries:
                    for entry in entries:
                        if len(found) >= MAX_FILES:
                            break
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name not in SKIP and not (folder == self.root and entry.name in ROOT_SKIP_DIRS):
                                    stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                if stat.S_ISREG(st.st_mode):
                                    found[entry.path] = (st.st_mtime_ns, st.st_size)
                        except OSError:
                            continue
            except OSError:
                continue
        return found

    def refresh(self) -> None:
        self.known, self.content, total = self.scan(), {}, 0
        for path, (_, size) in self.known.items():
            if size <= MAX_FILE_BYTES and total + size <= MAX_TOTAL_BYTES:
                data = _read_capped(Path(path))
                if data is not None and total + len(data) <= MAX_TOTAL_BYTES:
                    self.content[path] = data
                    total += len(data)
        self.ready = True

    def changes(self) -> list[tuple[Path, object]]:
        now, done = self.scan(), []
        total = sum(len(b) for b in self.content.values())
        for path in sorted(set(now) | set(self.known)):
            if now.get(path) == self.known.get(path):
                continue
            old = None if path not in self.known else self.content.get(path, UNKNOWN)
            done.append((Path(path), old))
            removed = self.content.pop(path, None)
            if removed is not None:
                total -= len(removed)
            if path in now and now[path][1] <= MAX_FILE_BYTES:
                data = _read_capped(Path(path))
                if data is not None:
                    while self.content and total + len(data) > MAX_TOTAL_BYTES:
                        oldest = next(iter(self.content))
                        total -= len(self.content.pop(oldest))
                    if len(data) <= MAX_TOTAL_BYTES:
                        self.content[path] = data
                        total += len(data)
        self._cap_content()
        self.known = now
        return done

    def _cap_content(self) -> None:
        total = sum(len(b) for b in self.content.values())
        if total <= MAX_TOTAL_BYTES:
            return
        for path in list(self.content):
            total -= len(self.content.pop(path))
            if total <= MAX_TOTAL_BYTES:
                break
