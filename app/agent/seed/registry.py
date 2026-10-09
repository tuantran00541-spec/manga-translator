"""Capability registry: static checks at define-time, restricted execution at call-time.

A capability is Python source that must define ``def main(args: dict) -> str``.
The source is checked *before* it is ever run:

- imports are limited to a whitelist of pure-computation stdlib modules
  (no file, process, socket or introspection access that way),
- a denylist of dangerous builtins (``open``, ``eval``, ``__import__`` …)
  may not even be named,
- dunder attribute access (``x.__class__`` …) is forbidden, closing the
  usual interpreter-escape hatch.

At call time the code runs with a restricted ``__builtins__`` and a single
injected global, ``harness`` — the bound world-access API. ``print`` is
rerouted into the audit log.
"""
from __future__ import annotations

import ast
import threading
from dataclasses import dataclass, field

# Pure computation only: nothing here touches files, processes or sockets.
# (urllib.parse parses; it does not fetch. pathlib/io are deliberately out:
#  Path("/etc/passwd").read_text() would dodge workspace containment.)
_ALLOWED_IMPORTS = frozenset({
    "json", "re", "math", "datetime", "collections", "itertools", "functools",
    "hashlib", "base64", "random", "string", "urllib.parse",
})

_FORBIDDEN_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "breakpoint", "input", "help", "exit", "quit",
    "memoryview", "property", "classmethod", "staticmethod",
})

_SAFE_BUILTINS = {
    name: getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "bytes", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hash", "hex",
        "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "zip", "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "StopIteration",
        "NotImplementedError", "ZeroDivisionError",
    )
    if hasattr(__builtins__, name)
}


class CapabilityError(ValueError):
    """The capability source or call is rejected."""


def check_source(code: str) -> None:
    """Reject capability source that reaches outside the allowed surface."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CapabilityError(f"syntax error: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORTS:
                    raise CapabilityError(f"import of {alias.name!r} is not allowed in capability code")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _ALLOWED_IMPORTS:
                raise CapabilityError(f"import from {node.module!r} is not allowed in capability code")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise CapabilityError(f"use of {node.id!r} is not allowed in capability code")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise CapabilityError(f"dunder attribute access ({node.attr!r}) is not allowed in capability code")


@dataclass
class Capability:
    name: str
    description: str
    parameters: dict
    code: str
    version: int = 1
    persisted: bool = False


class Registry:
    """The model's tool surface: capabilities it defined this session."""

    def __init__(self, harness, audit):
        self._harness = harness
        self._audit = audit
        self._caps: dict[str, Capability] = {}

    def names(self) -> list[str]:
        return sorted(self._caps)

    def define(self, name: str, description: str, parameters: dict | None,
               code: str, persisted: bool = False) -> str:
        if not name.replace("_", "").isalnum() or not name:
            raise CapabilityError(f"bad capability name: {name!r}")
        if name == "define":
            raise CapabilityError("'define' is the seed tool and cannot be redefined")
        check_source(code)
        compile(code, f"<capability {name}>", "exec")  # syntax, again, for the record
        old = self._caps.get(name)
        version = (old.version + 1) if old else 1
        self._caps[name] = Capability(
            name=name, description=description or "",
            parameters=parameters or {"type": "object", "properties": {}},
            code=code, version=version, persisted=persisted,
        )
        self._audit("capability.define", f"{name} v{version}" + (" (persisted)" if persisted else ""))
        action = "redefined" if old else "defined"
        return f"{action} capability '{name}' (v{version}). It is now callable as a tool."

    def specs(self) -> list[dict]:
        """OpenAI-style function specs for the current tool surface."""
        return [
            {"name": c.name, "description": f"{c.description} (capability v{c.version})",
             "parameters": c.parameters}
            for c in self._caps.values()
        ]

    def call(self, name: str, args: dict, timeout: int = 60) -> str:
        cap = self._caps.get(name)
        if cap is None:
            known = ", ".join(self.names()) or "(none yet — use define first)"
            raise CapabilityError(f"no capability named {name!r}; defined: {known}")
        self._audit("capability.call", f"{name} {str(args)[:200]}")
        return _execute(cap.code, args or {}, self._harness, self._audit, timeout)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import json
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _execute(code: str, args: dict, harness, audit, timeout: int) -> str:
    compiled = compile(code, "<capability>", "exec")
    safe = dict(_SAFE_BUILTINS)

    def _print(*values):
        audit("capability.print", " ".join(str(v) for v in values))

    safe["print"] = _print
    # The ONLY contact with the outside world. No imports needed for it.
    namespace = {"__builtins__": safe, "harness": harness}
    box: dict = {}

    def target():
        try:
            exec(compiled, namespace)
            main = namespace.get("main")
            if not callable(main):
                box["error"] = "capability must define: def main(args: dict)"
            else:
                box["result"] = main(args)
        except Exception as exc:  # noqa: BLE001 — the model sees the error text
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return "error: capability timed out"
    if "error" in box:
        return f"error: {box['error']}"
    return _stringify(box.get("result"))
