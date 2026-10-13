"""Reads a chain's code before it runs: tools.<name>(...) calls that cannot work, and in create mode tools that are only a shell."""
from __future__ import annotations

import ast
import difflib
import re

# Methods of the tools object itself, which are not tools the harness serves.
LIBRARY = frozenset({"search", "describe", "intent", "branch", "adopt"})


def check(code: str, known: dict[str, set[str] | None], fixes=lambda name, args: False) -> list[str]:
    """Problems found, one line each; known maps every reachable tool to its argument names, or None when any are accepted.

    fixes(name, argument names) says whether the harness renames such a call into one that works, as it does for other harnesses' names.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "tools"]
    # A call inside a try that catches AttributeError is the chain checking whether a tool exists, so it is left alone.
    probed = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try) and any(_catches(h) for h in t.handlers)
              for stmt in t.body for n in ast.walk(stmt)}
    # A tool made earlier in the same chain with tools.create(name="...") is reachable after it.
    made = {kw.value.value for c in calls if c.func.attr == "create" for kw in c.keywords
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)}
    problems = []
    for call in sorted(calls, key=lambda c: (c.lineno, c.col_offset)):
        name = call.func.attr
        given = [kw.arg for kw in call.keywords if kw.arg is not None]
        if name in LIBRARY or name in made or name.startswith("_") or id(call) in probed:
            continue
        if (name not in known or (known[name] is not None and set(given) - known[name])) and fixes(name, given):
            continue
        if name not in known:
            near = difflib.get_close_matches(name, known, n=3)
            problems.append(f"line {call.lineno}: tools.{name} is not a tool this chain can reach" + (f"; did you mean {', '.join(near)}?" if near else ""))
            continue
        allowed = known[name]
        if allowed is None:
            continue
        for arg in given:
            if arg not in allowed:
                problems.append(f"line {call.lineno}: tools.{name} has no argument {arg!r}; it takes {', '.join(sorted(allowed)) or 'none'}")
    return problems


def _catches(handler: ast.ExceptHandler) -> bool:
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(n is None or (isinstance(n, ast.Name) and n.id in ("AttributeError", "Exception", "BaseException")) for n in names)


# Programs that only read, list or search text; in create mode a tool does that in Python, where it can parse and keep the result.
TEXT_PROGRAMS = frozenset({"cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag", "sed", "awk", "find", "ls", "less", "more", "wc", "tree"})
SHELL_CALLS = frozenset({"run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput", "system", "popen"})


def raw_shell(code: str) -> list[str]:
    """Calls in a created tool that only pass a given command to a shell, or shell out to read and search text."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    params = {a.arg for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
              for a in [*f.args.args, *f.args.kwonlyargs, *([f.args.vararg] if f.args.vararg else [])]}
    problems = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in SHELL_CALLS
                and isinstance(call.func.value, ast.Name) and call.func.value.id in ("subprocess", "os") and call.args):
            continue
        command = call.args[0]
        if isinstance(command, (ast.List, ast.Tuple)) and len(command.elts) >= 3 and _first_words(command) & {"sh", "bash", "zsh"}:
            # ["bash", "-c", command] is the same shell in a list.
            command = command.elts[-1]
        if isinstance(command, ast.JoinedStr) and not any(isinstance(v, ast.Constant) and v.value.strip() for v in command.values):
            command = next((v.value for v in command.values if isinstance(v, ast.FormattedValue)), command)
        if isinstance(command, ast.Name) and command.id in params:
            problems.append(f"line {call.lineno}: the tool hands its {command.id!r} argument straight to a shell, so it is a shell, not a tool")
            continue
        words = _first_words(command)
        found = sorted(words & TEXT_PROGRAMS)
        if found:
            problems.append(f"line {call.lineno}: the tool shells out to {', '.join(found)}; read, list and search files in Python instead")
    return problems


def _first_words(node: ast.AST) -> set[str]:
    """The program names a literal command starts each of its pipeline or && parts with."""
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts and isinstance(node.elts[0], ast.Constant) and isinstance(node.elts[0].value, str):
        return {node.elts[0].value.rsplit("/", 1)[-1]}
    text = node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else \
        "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)) if isinstance(node, ast.JoinedStr) else ""
    words = set()
    for part in re.split(r"&&|\|\||[|;\n]", text):
        tokens = part.strip().split()
        while tokens and ("=" in tokens[0] or tokens[0] in ("sudo", "cd", "time", "env")):
            tokens = tokens[2:] if tokens[0] == "cd" else tokens[1:]
        if tokens:
            words.add(tokens[0].rsplit("/", 1)[-1])
    return words
