"""Per-turn file snapshots, named marks, pinned files and abandoned attempts, so edits can be reviewed and undone."""
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
        self.first = 0  # absolute number of turns[0]
        self.marks: dict[str, dict] = {}
        self.blobs: dict[str, bytes] = {}  # file contents by hash, shared by every mark that holds them
        self.pins: dict[str, str] = {}
        self.dropped: list[dict] = []
        self.ran: list[str] = []  # steps whose effects outside the files cannot be put back
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
        """Remember a file as it was before this turn first changed it."""
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
        """Record a file's contents from before a change seen after the fact, unless this turn already has it."""
        if self.turns and str(path) not in self.turns[-1]:
            self.turns[-1][str(path)] = data

    def current(self) -> dict[str, bytes | None]:
        """Each file the edit tools changed in this request, as it was before the first change (None: it did not exist)."""
        return dict(self.turns[-1]) if self.turns else {}

    def undo(self) -> list[str]:
        """Put the last turn's files back; returns what was restored or removed."""
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
        """Log a step a restore cannot undo, such as a command or a call to an outside server."""
        self.ran = (self.ran + [step])[-MAX_RAN:]
        self.ran_total += 1

    def ran_since(self, name: str) -> list[str]:
        """The logged steps since a mark, oldest first."""
        return self.ran[max(0, len(self.ran) - (self.ran_total - self.marks[name]["ran"])):]

    def mark(self, name: str, note: str = "", auto: bool = False) -> int:
        """Name the state of every file changed so far in this request; returns how many files it holds."""
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
        """Each file changed since the mark, as it was at the mark (None: it did not exist)."""
        mark = self.marks[name]
        state: dict[str, bytes | None] = {}
        for turn in self.turns[mark["turn"] - self.first:]:
            for key, data in turn.items():
                if key not in state:
                    held = mark["files"][key] if key in mark["files"] else data
                    state[key] = self.blobs[held] if isinstance(held, str) else held
        return state

    def restore(self, state: dict[str, bytes | None], only: str = "") -> list[str]:
        """Write files back to the given contents, keeping this turn's undo point first; returns what changed."""
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
        """Keep why an attempt was abandoned, so the next try does not repeat it."""
        self.dropped = (self.dropped + [{"mark": name, "why": why}])[-MAX_DROPPED:]

    def _blob(self, data: bytes) -> str:
        key = hashlib.sha256(data).hexdigest()
        self.blobs.setdefault(key, data)
        return key

    def _keep(self, alive) -> None:
        """Drop the marks that no longer apply and every stored content no mark still holds."""
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
