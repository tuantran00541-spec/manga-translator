"""Shared by the example runner and the grader: run the engine on one statement and compare its rows with the expected ones."""
import math
import os
import re
import subprocess
import sys

# Removing sqlite3 before the engine starts, so an engine that leans on it fails instead of passing.
NO_SQLITE = ("import runpy, sys; sys.modules['sqlite3'] = None; sys.modules['_sqlite3'] = None; "
             "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")


def ordered(sql: str) -> bool:
    """Whether the outermost query has ORDER BY, outside quotes and parentheses."""
    text, depth, top = re.sub(r"'(?:[^']|'')*'", "''", sql), 0, []
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        top.append(ch if depth == 0 else " ")
    return re.search(r"\border\s+by\b", "".join(top), re.I) is not None


def run(engine: str, data: str, sql: str, block: bool, timeout: float = 20.0) -> tuple[int | None, str]:
    args = [sys.executable, "-c", NO_SQLITE, engine, data, sql] if block else [sys.executable, engine, data, sql]
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=os.path.dirname(os.path.abspath(engine)))
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    return done.returncode, done.stdout


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


def matches(expected: list[list[str]], output: str, in_order: bool) -> bool:
    got = [line.split("\t") for line in output.splitlines()] if output.strip() else []
    if len(got) != len(expected):
        return False
    if in_order:
        return all(len(g) == len(e) and all(_same(a, b) for a, b in zip(g, e)) for g, e in zip(got, expected))
    # Without ORDER BY rows match as a multiset: each expected row takes one equal row once.
    left = list(got)
    for row in expected:
        hit = next((i for i, g in enumerate(left) if len(g) == len(row) and all(_same(a, b) for a, b in zip(g, row))), None)
        if hit is None:
            return False
        left.pop(hit)
    return True
