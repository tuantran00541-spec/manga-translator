"""The apply_patch format popularised by Codex: add, delete, update and move files in one patch."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BEGIN, END = "*** Begin Patch", "*** End Patch"
ADD, DELETE, UPDATE, MOVE = "*** Add File: ", "*** Delete File: ", "*** Update File: ", "*** Move to: "
END_OF_FILE = "*** End of File"
GUIDE = """Patch format:
*** Begin Patch
*** Add File: path/new.py
+line of the new file
*** Update File: path/old.py
*** Move to: path/renamed.py   (optional)
@@ def function_the_change_is_in():   (optional anchor line)
 unchanged line kept for context
-line removed
+line added
*** Delete File: path/gone.py
*** End Patch
Give about three unchanged lines around each change so it is found exactly."""


class PatchError(ValueError):
    """A patch that does not parse or does not match the files."""


@dataclass
class Chunk:
    anchors: list[str] = field(default_factory=list)
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    at_end: bool = False


@dataclass
class Hunk:
    kind: str
    path: str
    move_to: str | None = None
    lines: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


def parse(text: str) -> list[Hunk]:
    lines = [line.rstrip("\r") for line in str(text).strip().splitlines()]
    if not lines or lines[0].strip() != BEGIN:
        raise PatchError("The patch must start with '*** Begin Patch'")
    if lines[-1].strip() != END:
        raise PatchError("The patch must end with '*** End Patch'")
    hunks: list[Hunk] = []
    for number, line in enumerate(lines[1:-1], 2):
        stripped = line.strip()
        if stripped.startswith(ADD.strip()):
            hunks.append(Hunk("add", stripped[len(ADD.strip()):].strip()))
        elif stripped.startswith(DELETE.strip()):
            hunks.append(Hunk("delete", stripped[len(DELETE.strip()):].strip()))
        elif stripped.startswith(UPDATE.strip()):
            hunks.append(Hunk("update", stripped[len(UPDATE.strip()):].strip()))
        elif not hunks:
            raise PatchError(f"Line {number}: expected a file header, got {line!r}")
        elif hunks[-1].kind == "add":
            if not line.startswith("+"):
                raise PatchError(f"Line {number}: every line of an added file starts with '+'")
            hunks[-1].lines.append(line[1:])
        elif hunks[-1].kind == "delete":
            if stripped:
                raise PatchError(f"Line {number}: a deleted file takes no lines")
        else:
            hunk = hunks[-1]
            if stripped.startswith(MOVE.strip()) and not hunk.chunks:
                hunk.move_to = stripped[len(MOVE.strip()):].strip()
                continue
            if stripped == END_OF_FILE:
                if not hunk.chunks:
                    raise PatchError(f"Line {number}: '*** End of File' without a change")
                hunk.chunks[-1].at_end = True
                continue
            if line.startswith("@@"):
                anchor = line[2:].strip()
                if not hunk.chunks or hunk.chunks[-1].old or hunk.chunks[-1].new:
                    hunk.chunks.append(Chunk())
                if anchor:
                    hunk.chunks[-1].anchors.append(anchor)
                continue
            if not hunk.chunks:
                hunk.chunks.append(Chunk())
            chunk = hunk.chunks[-1]
            tag, body = (line[:1], line[1:]) if line else (" ", "")
            if tag == " ":
                chunk.old.append(body)
                chunk.new.append(body)
            elif tag == "-":
                chunk.old.append(body)
            elif tag == "+":
                chunk.new.append(body)
            else:
                raise PatchError(f"Line {number}: change lines start with ' ', '-' or '+', got {line!r}")
    if not hunks:
        raise PatchError("The patch changes no file")
    for hunk in hunks:
        if hunk.kind == "update" and not hunk.chunks and not hunk.move_to:
            raise PatchError(f"Update of {hunk.path} has no changes")
    return hunks


def _seek(lines: list[str], pattern: list[str], start: int, at_end: bool) -> int:
    """Where pattern sits in lines from start: exact first, then ignoring trailing, then all outer whitespace."""
    if not pattern:
        return len(lines) if at_end else start
    last = len(lines) - len(pattern)
    order = [last] if at_end else range(start, last + 1)
    for norm in (lambda s: s, str.rstrip, str.strip):
        want = [norm(p) for p in pattern]
        for i in order:
            if i >= start and [norm(x) for x in lines[i:i + len(pattern)]] == want:
                return i
    return -1


def _update(text: str, chunks: list[Chunk], path: str) -> str:
    lines = text.split("\n")
    trailing = lines and lines[-1] == ""
    if trailing:
        lines = lines[:-1]
    cursor = 0
    edits: list[tuple[int, int, list[str]]] = []
    for chunk in chunks:
        for anchor in chunk.anchors:
            found = _seek(lines, [anchor], cursor, False)
            if found < 0:
                raise PatchError(f"{path}: anchor line not found: {anchor!r}")
            cursor = found + 1
        at = _seek(lines, chunk.old, cursor, chunk.at_end)
        if at < 0:
            preview = "\n".join(chunk.old[:6])
            raise PatchError(f"{path}: these lines were not found (read the file again):\n{preview}")
        edits.append((at, len(chunk.old), chunk.new))
        cursor = at + len(chunk.old)
    for at, size, new in reversed(edits):
        lines[at:at + size] = new
    return "\n".join(lines) + ("\n" if trailing or not lines else "")


def apply(text: str, resolve: Callable[[str], Path], rel: Callable[[Path], str]) -> str:
    """Check every hunk against the files first, then write them all; returns a short summary."""
    hunks = parse(text)
    writes: list[tuple[Path, str | None]] = []
    summary = []
    for hunk in hunks:
        target = resolve(hunk.path)
        if hunk.kind == "add":
            if target.exists():
                raise PatchError(f"{hunk.path} already exists; update it instead")
            writes.append((target, "\n".join(hunk.lines) + "\n"))
            summary.append(f"A {rel(target)}")
        elif hunk.kind == "delete":
            if not target.is_file():
                raise PatchError(f"{hunk.path} is not a file")
            writes.append((target, None))
            summary.append(f"D {rel(target)}")
        else:
            if not target.is_file():
                raise PatchError(f"{hunk.path} is not a file")
            new_text = _update(target.read_text(encoding="utf-8"), hunk.chunks, hunk.path)
            if hunk.move_to:
                dest = resolve(hunk.move_to)
                writes += [(dest, new_text), (target, None)]
                summary.append(f"M {rel(target)} -> {rel(dest)}")
            else:
                writes.append((target, new_text))
                summary.append(f"M {rel(target)}")
    for path, content in writes:
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return "\n".join(summary)
