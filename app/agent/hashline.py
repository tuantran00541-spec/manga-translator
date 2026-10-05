"""Hash-anchored line edits: the model names lines by number and a short hash instead of retyping them."""
from __future__ import annotations

import re
import zlib

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
ANCHOR = re.compile(r"^\s*(\d+)#([0-9a-z]{2})")
OPS = ("replace", "delete", "insert_after", "insert_before")
GUIDE = ("Edit lines by anchor. Read the file with anchors=true; each line shows as N#hh|text. "
         "Each edit has op (replace, delete, insert_after or insert_before), anchor (such as 12#a7), "
         "optional end (an anchor, to replace or delete the range anchor..end) and text (the new lines, without anchors).")


class HashlineError(Exception):
    """An anchor edit that cannot be applied; the message tells the model what to read again."""


def tag(line: str) -> str:
    value = zlib.crc32(line.strip().encode("utf-8")) % (36 * 36)
    return DIGITS[value // 36] + DIGITS[value % 36]


def render(lines: list[str], start: int = 1) -> str:
    return "\n".join(f"{start + i}#{tag(line)}|{line}" for i, line in enumerate(lines))


def _anchor(value: object, lines: list[str]) -> int:
    match = ANCHOR.match(str(value or ""))
    if not match:
        raise HashlineError(f"Bad anchor {value!r}; use the N#hh form that read_file shows with anchors=true")
    number, want = int(match.group(1)), match.group(2)
    if not 1 <= number <= len(lines):
        raise HashlineError(f"Anchor {value!r}: the file has {len(lines)} lines; read it again")
    if tag(lines[number - 1]) != want:
        raise HashlineError(f"Anchor {value!r} is stale: line {number} is now {number}#{tag(lines[number - 1])}|{lines[number - 1]}. "
                            "Read the file again with anchors=true")
    return number


def _new_lines(text: object) -> list[str]:
    text = str(text or "")
    if not text:
        return []
    return (text[:-1] if text.endswith("\n") else text).split("\n")


def apply(text: str, edits: list[dict]) -> str:
    """Check every anchor against the file, then apply all edits at once."""
    if not isinstance(edits, list) or not edits:
        raise HashlineError("edits must be a non-empty list")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    trailing = lines[-1] == ""
    if trailing:
        lines.pop()
    spans: list[tuple[int, int, int, list[str]]] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or edit.get("op") not in OPS:
            raise HashlineError(f"Edit {index + 1}: op must be one of {', '.join(OPS)}")
        op, first = edit["op"], _anchor(edit.get("anchor"), lines)
        last = _anchor(edit["end"], lines) if edit.get("end") else first
        if last < first:
            raise HashlineError(f"Edit {index + 1}: end comes before anchor")
        new = _new_lines(edit.get("text"))
        if op in ("insert_after", "insert_before") and (not new or last != first):
            raise HashlineError(f"Edit {index + 1}: {op} needs text and no end")
        if op == "delete":
            new = []
        if op == "insert_after":
            spans.append((first, first, index, new))
        elif op == "insert_before":
            spans.append((first - 1, first - 1, index, new))
        else:
            spans.append((first - 1, last, index, new))
    ordered = sorted(spans, key=lambda s: (s[0], s[1], s[2]))
    for a, b in zip(ordered, ordered[1:]):
        if b[0] < a[1]:
            raise HashlineError(f"Edits {a[2] + 1} and {b[2] + 1} touch the same lines; merge them")
    for start, stop, _, new in reversed(ordered):
        lines[start:stop] = new
    return eol.join(lines) + (eol if trailing or not lines else "")
