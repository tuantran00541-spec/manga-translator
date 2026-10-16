"""Which workspace files a created tool changed, with their old contents kept so /undo still works."""
from __future__ import annotations

import os
from pathlib import Path

SKIP = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache"})
MAX_FILES = 20_000
MAX_FILE_BYTES = 512_000
MAX_TOTAL_BYTES = 64_000_000
UNKNOWN = object()


class Watch:
    def __init__(self, root: Path):
        self.root = root
        self.known: dict[str, tuple[int, int]] = {}
        self.content: dict[str, bytes] = {}
        self.ready = False

    def scan(self) -> dict[str, tuple[int, int]]:
        """(mtime, size) of each file under the root, skipping caches and dependency folders."""
        found: dict[str, tuple[int, int]] = {}
        stack = [self.root]
        while stack and len(found) < MAX_FILES:
            try:
                entries = list(os.scandir(stack.pop()))
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in SKIP:
                            stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        stat = entry.stat(follow_symlinks=False)
                        found[entry.path] = (stat.st_mtime_ns, stat.st_size)
                except OSError:
                    continue
        return found

    def refresh(self) -> None:
        """Take the current files as the baseline, keeping the contents of small ones."""
        self.known, self.content, total = self.scan(), {}, 0
        for path, (_, size) in self.known.items():
            if size <= MAX_FILE_BYTES and total + size <= MAX_TOTAL_BYTES:
                try:
                    self.content[path] = Path(path).read_bytes()
                    total += size
                except OSError:
                    continue
        self.ready = True

    def changes(self) -> list[tuple[Path, object]]:
        """Each file changed since the baseline with its old bytes (None: it is new, UNKNOWN: too big to have kept); the baseline moves on."""
        now, done = self.scan(), []
        for path in sorted(set(now) | set(self.known)):
            if now.get(path) == self.known.get(path):
                continue
            old = None if path not in self.known else self.content.get(path, UNKNOWN)
            done.append((Path(path), old))
            self.content.pop(path, None)
            if path in now and now[path][1] <= MAX_FILE_BYTES:
                try:
                    self.content[path] = Path(path).read_bytes()
                except OSError:
                    pass
        self._cap_content()
        self.known = now
        return done

    def _cap_content(self) -> None:
        """Evict the oldest-kept contents first until the total is back under MAX_TOTAL_BYTES."""
        total = sum(len(b) for b in self.content.values())
        if total <= MAX_TOTAL_BYTES:
            return
        for path in list(self.content):
            total -= len(self.content.pop(path))
            if total <= MAX_TOTAL_BYTES:
                break
