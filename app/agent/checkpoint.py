"""Per-turn file snapshots so the user can undo what the agent's edit tools changed."""
from __future__ import annotations

from pathlib import Path

MAX_TURNS = 30
MAX_FILE_BYTES = 2_000_000


class Checkpoints:
    def __init__(self):
        self.turns: list[dict[str, bytes | None]] = []

    def begin(self) -> None:
        self.turns.append({})
        del self.turns[:-MAX_TURNS]

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

    def undo(self) -> list[str]:
        """Put the last turn's files back; returns what was restored or removed."""
        while self.turns and not self.turns[-1]:
            self.turns.pop()
        if not self.turns:
            return []
        done = []
        for name, data in self.turns.pop().items():
            path = Path(name)
            try:
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                done.append(name)
            except OSError:
                continue
        return done
