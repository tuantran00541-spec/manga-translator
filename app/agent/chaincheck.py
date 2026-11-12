from __future__ import annotations

import ast
import difflib

LIBRARY = frozenset({"search", "describe", "intent", "branch", "adopt"})


def check(code: str, known: dict[str, set[str] | None], fixes=lambda name, args: False) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "tools"]
    probed = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try) and any(_catches(h) for h in t.handlers)
              for stmt in t.body for n in ast.walk(stmt)}
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
