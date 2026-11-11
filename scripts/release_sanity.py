from __future__ import annotations

import ast
from collections import defaultdict
import importlib.util
from pathlib import Path
import re

# Test folders under app are left out, as tests/ at the root is.
PYTHON_PATHS = sorted([*(p for p in Path("app").rglob("*.py") if "tests" not in p.parts), Path("run.py")])
JS_PATHS = sorted(Path("app/static/js").rglob("*.js"))
HTML_PATHS = sorted(Path("app/templates").rglob("*.html"))


def _fail(title: str, failures: list[str]) -> None:
    if not failures:
        return
    print(title)
    for failure in failures:
        print(f"  {failure}")
    raise SystemExit(1)


def check_unused_python_imports() -> None:
    failures: list[str] = []
    for path in PYTHON_PATHS:
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        exported: set[str] = set()
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                continue
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                exported.update(
                    item.value
                    for item in value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )

        for node in ast.walk(tree):
            aliases: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                aliases = [
                    (alias.asname or alias.name.split(".")[0], alias.name)
                    for alias in node.names
                ]
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                aliases = [
                    (alias.asname or alias.name, f"{node.module or ''}.{alias.name}")
                    for alias in node.names
                    if alias.name != "*"
                ]
            else:
                continue

            source_line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
            if "noqa" in source_line.lower():
                continue
            for local_name, imported_name in aliases:
                if local_name not in used and local_name not in exported:
                    failures.append(
                        f"{path}:{node.lineno}: unused import {local_name} <- {imported_name}"
                    )

    _fail("Unused imports found:", failures)
    print("Unused import check OK")


def check_loguru_usage() -> None:
    methods = {
        "trace", "debug", "info", "success", "warning", "error",
        "critical", "exception",
    }
    placeholder = re.compile(r"%(?:[-+0-9.]*[sdif])")
    failures: list[str] = []

    for path in PYTHON_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr not in methods
                or not node.args
            ):
                continue

            receiver = node.func.value
            is_logger = isinstance(receiver, ast.Name) and receiver.id == "logger"
            if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
                is_logger = (
                    isinstance(receiver.func.value, ast.Name)
                    and receiver.func.value.id == "logger"
                )
            message = node.args[0]
            if is_logger and any(
                keyword.arg == "exc_info" for keyword in node.keywords
            ):
                failures.append(
                    f"{path}:{node.lineno}: use logger.opt(exception=True), not exc_info"
                )
            if (
                is_logger
                and len(node.args) > 1
                and isinstance(message, ast.Constant)
                and isinstance(message.value, str)
                and placeholder.search(message.value)
            ):
                failures.append(
                    f"{path}:{node.lineno}: Loguru call uses %-style placeholders"
                )

    _fail("Invalid Loguru usage found:", failures)
    print("Loguru usage check OK")


def check_top_level_python_reachability() -> None:
    source_by_path = {
        path: path.read_text(encoding="utf-8")
        for path in PYTHON_PATHS
    }
    all_source = "\n".join(source_by_path.values())
    failures: list[str] = []

    for path, source in source_by_path.items():
        if path.name == "__init__.py":
            continue
        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if getattr(node, "decorator_list", None):
                continue
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            occurrences = len(
                re.findall(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                    all_source,
                )
            )
            if occurrences == 1:
                failures.append(
                    f"{path}:{node.lineno}: {type(node).__name__} {name} has no textual reference"
                )

    _fail("Unreachable top-level definitions found:", failures)
    print("Top-level definition reachability OK")


def _module_name(path: Path) -> str:
    if path.name == "__init__.py":
        return ".".join(path.parent.parts)
    return ".".join(path.with_suffix("").parts)


def check_python_module_reachability() -> None:
    # Skills bundled for the Agent carry their own scripts, and tests are run by pytest; neither are app modules.
    app_paths = sorted(p for p in Path("app").rglob("*.py") if "builtin_skills" not in p.parts and "tests" not in p.parts)
    existing = {
        _module_name(path)
        for path in app_paths
        if path.name != "__init__.py"
    }
    incoming: dict[str, list[str]] = defaultdict(list)

    for path in [*app_paths, Path("run.py")]:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        current_module = _module_name(path) if path != Path("run.py") else "run"
        package = current_module if path.name == "__init__.py" else current_module.rpartition(".")[0]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in existing:
                        incoming[alias.name].append(f"{path}:{node.lineno}")
                continue
            if not isinstance(node, ast.ImportFrom):
                continue

            if node.level:
                if not package:
                    continue
                relative = "." * node.level + (node.module or "")
                try:
                    base = importlib.util.resolve_name(relative, package)
                except (ImportError, ValueError):
                    continue
            else:
                base = node.module or ""

            if base in existing:
                incoming[base].append(f"{path}:{node.lineno}")
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                if candidate in existing:
                    incoming[candidate].append(f"{path}:{node.lineno}")

    entrypoints = {"app.main"}
    failures = [
        f"{module} has no incoming runtime import"
        for module in sorted(existing)
        if module not in incoming and module not in entrypoints
    ]
    _fail("Orphan Python runtime modules found:", failures)
    print(f"Python module reachability OK: {len(existing)} modules")


def check_exact_python_helper_duplication() -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in PYTHON_PATHS:
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if len(node.body) < 3:
                continue
            normalized = ast.dump(
                ast.Module(body=node.body, type_ignores=[]),
                annotate_fields=True,
                include_attributes=False,
            )
            if len(normalized) < 700:
                continue
            groups[normalized].append(f"{path}:{node.lineno} ({node.name})")

    duplicates = [locations for locations in groups.values() if len(locations) > 1]
    failures = [" <> ".join(locations) for locations in duplicates]
    _fail("Substantial exact duplicate Python function bodies found:", failures)
    print("Substantial exact Python helper duplication check OK")


def check_duplicate_api_routes() -> None:
    methods = {"get", "post", "put", "patch", "delete", "options", "head"}
    seen: dict[tuple[str, str], str] = {}
    failures: list[str] = []

    for path in sorted(Path("app/routers").glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prefix = ""
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "router" for target in targets):
                continue
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            if not (isinstance(func, ast.Name) and func.id == "APIRouter"):
                continue
            for keyword in value.keywords:
                if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    prefix = keyword.value.value

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                if not (isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == "router"):
                    continue
                method = decorator.func.attr.lower()
                if method not in methods:
                    continue
                route_value = decorator.args[0] if decorator.args else next(
                    (kw.value for kw in decorator.keywords if kw.arg in {"path", "route"}),
                    None,
                )
                if not isinstance(route_value, ast.Constant) or not isinstance(route_value.value, str):
                    continue
                route_path = prefix + route_value.value
                key = (method.upper(), route_path)
                location = f"{path}:{node.lineno} ({node.name})"
                previous = seen.get(key)
                if previous:
                    failures.append(
                        f"Duplicate route {key[0]} {key[1]}: {previous} <> {location}"
                    )
                else:
                    seen[key] = location

    _fail("Duplicate API routes found:", failures)
    print(f"Route uniqueness OK: {len(seen)} API routes")


def _defined_names(tree: ast.AST) -> set[str]:
    """Every name a module defines textually: assignments, arguments,
    imports, defs/classes, except/match bindings. A name used but never
    defined anywhere in its module is undefined (the ch70 class:
    `_rectangle_fallback_allowed` was called at two sites after round 9
    deleted its `def`)."""
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchAs) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            defined.add(node.rest)
    return defined


def check_undefined_python_names() -> None:
    import builtins

    allowed = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__package__", "__spec__",
        "__loader__", "__cached__", "__builtins__", "__path__",
    }
    failures: list[str] = []
    for path in PYTHON_PATHS:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        defined = _defined_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined and node.id not in allowed:
                    failures.append(f"{path}:{node.lineno}: undefined name {node.id}")

    _fail("Undefined Python names found:", failures)
    print("Undefined Python name check OK")


_TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def check_dead_code_after_terminator() -> None:
    """Any statement after a return/raise/break/continue in the same
    block can never run. At the round-9 head the deleted function's body
    survived exactly like this (mask_builder.py:120-127)."""
    failures: list[str] = []

    def scan_block(body: list[ast.stmt], path: Path) -> None:
        terminator: str | None = None
        for stmt in body:
            if terminator is not None:
                failures.append(
                    f"{path}:{stmt.lineno}: unreachable code after {terminator}"
                )
            if isinstance(stmt, _TERMINATORS):
                terminator = type(stmt).__name__.lower()

    def visit(node: ast.AST, path: Path) -> None:
        for _field, value in ast.iter_fields(node):
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, ast.stmt) for item in value)
            ):
                scan_block(value, path)
        for child in ast.iter_child_nodes(node):
            visit(child, path)

    for path in PYTHON_PATHS:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        visit(tree, path)

    _fail("Unreachable code found:", failures)
    print("Unreachable code check OK")


def check_javascript_function_reachability() -> None:
    source_by_path = {
        path: path.read_text(encoding="utf-8")
        for path in JS_PATHS
    }
    runtime_source = "\n".join(
        [*source_by_path.values(), *(path.read_text(encoding="utf-8") for path in HTML_PATHS)]
    )
    patterns = (
        (
            "function",
            re.compile(
                r"(?m)^\s*(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
            ),
        ),
        (
            "arrow binding",
            re.compile(
                r"(?m)^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*"
                r"(?:async\s*)?(?:\([^;\n]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>"
            ),
        ),
        (
            "function binding",
            re.compile(
                r"(?m)^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*"
                r"(?:async\s*)?function(?:\s+[A-Za-z_$][A-Za-z0-9_$]*)?\s*\("
            ),
        ),
    )
    failures: list[str] = []
    seen_declarations: set[tuple[Path, int, str]] = set()

    for path, source in source_by_path.items():
        for kind, pattern in patterns:
            for match in pattern.finditer(source):
                name = match.group(1)
                line = source.count("\n", 0, match.start()) + 1
                key = (path, line, name)
                if key in seen_declarations:
                    continue
                seen_declarations.add(key)
                occurrences = len(
                    re.findall(
                        rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])",
                        runtime_source,
                    )
                )
                if occurrences == 1:
                    failures.append(
                        f"{path}:{line}: {kind} {name} has no textual reference"
                    )

    _fail("Unreachable JavaScript function bindings found:", failures)
    print("JavaScript function reachability OK")


def main() -> None:
    check_unused_python_imports()
    check_loguru_usage()
    check_top_level_python_reachability()
    check_python_module_reachability()
    check_exact_python_helper_duplication()
    check_duplicate_api_routes()
    check_undefined_python_names()
    check_dead_code_after_terminator()
    check_javascript_function_reachability()


if __name__ == "__main__":
    main()
