from __future__ import annotations

import hashlib
from pathlib import Path

MAX_TURNS = 30
MAX_FILE_BYTES = 2_000_000
MAX_MARKS = 30
MAX_AUTO_MARKS = 10
MAX_DROPPED = 20
MAX_RAN = 200


class Checkpoints:
    def __init__(self):
        self.turns: list[dict[str, bytes | None]] = []
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
        extra = max(0, len(self.turns) - MAX_TURNS)
        if extra:
            del self.turns[:extra]
            self.first += extra
            self._keep(lambda m: m["turn"] >= self.first)

    def save(self, path: Path) -> None:
        if not self.turns or str(path) in self.turns[-1]:
            return
        try:
            if path.is_file():
                if path.stat().st_size <= MAX_FILE_BYTES:
                    self.turns[-1][str(path)] = path.read_bytes()
            elif not path.exists():
                self.turns[-1][str(path)] = None
        except OSError:
            pass

    def remember(self, path: Path, data: bytes | None) -> None:
        if self.turns and str(path) not in self.turns[-1]:
            self.turns[-1][str(path)] = data

    def current(self) -> dict[str, bytes | None]:
        return dict(self.turns[-1]) if self.turns else {}

    def undo(self) -> list[str]:
        while self.turns and not self.turns[-1]:
            self.turns.pop()
        if not self.turns:
            return []
        done = []
        for name, data in self.turns.pop().items():
            if _write(Path(name), data):
                done.append(name)
        self._keep(lambda m: m["turn"] < self.first + len(self.turns))
        return done

    def note_ran(self, step: str) -> None:
        self.ran = (self.ran + [step])[-MAX_RAN:]
        self.ran_total += 1

    def ran_since(self, name: str) -> list[str]:
        return self.ran[max(0, len(self.ran) - (self.ran_total - self.marks[name]["ran"])):]

    def mark(self, name: str, note: str = "", auto: bool = False) -> int:
        if not self.turns:
            self.begin()
        files = {}
        for key in self.turns[-1]:
            data = _read(Path(key))
            files[key] = None if data is None else self._blob(data)
        self.marks.pop(name, None)
        self.marks[name] = {"turn": self.first + len(self.turns) - 1, "files": files, "note": note, "auto": auto, "ran": self.ran_total}
        for kind, cap in ((True, MAX_AUTO_MARKS), (False, MAX_MARKS)):
            names = [n for n, m in self.marks.items() if m["auto"] == kind]
            for old in names[:max(0, len(names) - cap)]:
                del self.marks[old]
        self._keep(lambda m: True)
        return len(files)

    def at(self, name: str) -> dict[str, bytes | None]:
        mark = self.marks[name]
        state: dict[str, bytes | None] = {}
        for turn in self.turns[mark["turn"] - self.first:]:
            for key, data in turn.items():
                if key not in state:
                    held = mark["files"][key] if key in mark["files"] else data
                    state[key] = self.blobs[held] if isinstance(held, str) else held
        return state

    def restore(self, state: dict[str, bytes | None], only: str = "") -> list[str]:
        done = []
        for name, data in state.items():
            if only and name != only:
                continue
            path = Path(name)
            if _read(path) == data:
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


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _write(path: Path, data: bytes | None) -> bool:
    try:
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return True
    except OSError:
        return False
