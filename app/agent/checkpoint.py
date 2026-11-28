from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

MAX_TURNS = 30
MAX_FILE_BYTES = 2_000_000
MAX_MARKS = 30
MAX_AUTO_MARKS = 10
MAX_DROPPED = 20
MAX_RAN = 200


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _norm_key(path: Path | str) -> str:
    return os.path.normpath(str(path))


class Checkpoints:
    def __init__(self):
        self.turns: list[dict[str, bytes | None]] = []
        self.skipped: list[set[str]] = []
        self.last_undo_skipped: list[str] = []
        self.first = 0
        self.marks: dict[str, dict] = {}
        self.blobs: dict[str, bytes] = {}
        self.pins: dict[str, str] = {}
        self.dropped: list[dict] = []
        self.ran: list[str] = []
        self.ran_total = 0
        self.auto_no = 0

    def begin(self) -> None:
        self.turns.append({})
        self.skipped.append(set())
        extra = max(0, len(self.turns) - MAX_TURNS)
        if extra:
            del self.turns[:extra]
            del self.skipped[:extra]
            self.first += extra
            self._keep(lambda m: m["turn"] >= self.first)

    def save(self, path: Path) -> None:
        key = _norm_key(path)
        if not self.turns or key in self.turns[-1] or key in self.skipped[-1]:
            return
        try:
            if _is_link(path):
                self.skipped[-1].add(key)
                return
            if path.is_file():
                st = path.lstat()
                if not stat.S_ISREG(st.st_mode):
                    self.skipped[-1].add(key)
                    return
                if st.st_size <= MAX_FILE_BYTES:
                    data = _read(path, max_bytes=MAX_FILE_BYTES)
                    if data is None or data is _TOO_LARGE or data is _READ_FAILED:
                        self.skipped[-1].add(key)
                    else:
                        self.turns[-1][key] = data
                else:
                    self.skipped[-1].add(key)
            elif not path.exists():
                self.turns[-1][key] = None
        except OSError:
            pass

    def remember(self, path: Path, data: bytes | None) -> None:
        key = _norm_key(path)
        if not self.turns or key in self.turns[-1] or key in self.skipped[-1]:
            return
        if data is not None and len(data) > MAX_FILE_BYTES:
            self.skipped[-1].add(key)
            return
        self.turns[-1][key] = data

    def current(self) -> dict[str, bytes | None]:
        return dict(self.turns[-1]) if self.turns else {}

    def undo(self) -> list[str]:
        self.last_undo_skipped = []
        while self.turns and not self.turns[-1] and not self.skipped[-1]:
            self.turns.pop()
            self.skipped.pop()
        if not self.turns:
            self._keep(lambda m: m["turn"] < self.first)
            return []
        skipped = set(self.skipped.pop())
        done = []
        for name, data in self.turns.pop().items():
            if _write(Path(name), data):
                done.append(name)
            else:
                skipped.add(name)
        self.last_undo_skipped = sorted(skipped)
        self._keep(lambda m: m["turn"] < self.first + len(self.turns))
        return done

    def note_ran(self, step: str) -> None:
        self.ran = (self.ran + [step])[-MAX_RAN:]
        self.ran_total += 1

    def ran_since(self, name: str) -> list[str]:
        return self.ran[max(0, len(self.ran) - (self.ran_total - self.marks[name]["ran"])) :]

    def mark(self, name: str, note: str = "", auto: bool = False) -> int:
        if not self.turns:
            self.begin()
        files = {}
        unreadable = set()
        for key in self.turns[-1]:
            p = Path(key)
            if _is_link(p):
                unreadable.add(key)
                continue
            data = _read(p, max_bytes=MAX_FILE_BYTES)
            if data is _TOO_LARGE or data is _READ_FAILED:
                unreadable.add(key)
                continue
            files[key] = None if data is None else self._blob(data)
        self.marks.pop(name, None)
        self.marks[name] = {
            "turn": self.first + len(self.turns) - 1,
            "files": files,
            "unreadable": unreadable,
            "note": note,
            "auto": auto,
            "ran": self.ran_total,
        }
        for kind, cap in ((True, MAX_AUTO_MARKS), (False, MAX_MARKS)):
            names = [n for n, m in self.marks.items() if m["auto"] == kind]
            for old in names[: max(0, len(names) - cap)]:
                del self.marks[old]
        self._keep(lambda m: True)
        return len(files)

    def at(self, name: str) -> dict[str, bytes | None]:
        mark = self.marks[name]
        state: dict[str, bytes | None] = {}
        for turn in self.turns[mark["turn"] - self.first :]:
            for key, data in turn.items():
                if key in mark.get("unreadable", ()):
                    continue
                if key not in state:
                    held = mark["files"][key] if key in mark["files"] else data
                    state[key] = self.blobs[held] if isinstance(held, str) else held
        return state

    def restore(self, state: dict[str, bytes | None], only: str = "") -> list[str]:
        only_norm = os.path.normcase(_norm_key(only)) if only else ""
        done = []
        for name, data in state.items():
            if only_norm and os.path.normcase(_norm_key(name)) != only_norm:
                continue
            path = Path(name)
            existing = _read(path, max_bytes=MAX_FILE_BYTES)
            if existing is _READ_FAILED or existing is _TOO_LARGE or _is_link(path):
                continue
            if existing == data:
                continue
            self.save(path)
            if _write(path, data):
                done.append(name)
        return done

    def drop(self, name: str, why: str) -> None:
        self.dropped = (self.dropped + [{"mark": name, "why": why}])[-MAX_DROPPED:]

    def _blob(self, data: bytes) -> str:
        key = hashlib.sha256(data).hexdigest()
        self.blobs.setdefault(key, data)
        return key

    def _keep(self, alive) -> None:
        self.marks = {n: m for n, m in self.marks.items() if alive(m)}
        held = {h for m in self.marks.values() for h in m["files"].values() if h}
        self.blobs = {h: d for h, d in self.blobs.items() if h in held}


_TOO_LARGE = object()
_READ_FAILED = object()


def _read(path: Path, *, max_bytes: int | None = None):
    try:
        if _is_link(path):
            return _READ_FAILED
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            return _READ_FAILED
        if max_bytes is not None and st.st_size > max_bytes:
            return _TOO_LARGE
        with path.open("rb") as handle:
            if max_bytes is None:
                return handle.read()
            data = handle.read(max_bytes + 1)
            return _TOO_LARGE if len(data) > max_bytes else data
    except FileNotFoundError:
        return None
    except OSError:
        return _READ_FAILED


def _write(path: Path, data: bytes | None) -> bool:
    try:
        if path.is_symlink():
            path.unlink()
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return True
    except OSError:
        return False
