from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import re

PYTHON_PATHS = sorted([*Path("app").rglob("*.py"), Path("run.py")])
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


def check_named_javascript_reachability() -> None:
    source_by_path = {
        path: path.read_text(encoding="utf-8")
        for path in JS_PATHS
    }
    runtime_source = "\n".join(
        [*source_by_path.values(), *(path.read_text(encoding="utf-8") for path in HTML_PATHS)]
    )
    declaration = re.compile(
        r"(?m)^\s*(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
    )
    failures: list[str] = []

    for path, source in source_by_path.items():
        for match in declaration.finditer(source):
            name = match.group(1)
            occurrences = len(
                re.findall(
                    rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])",
                    runtime_source,
                )
            )
            if occurrences == 1:
                line = source.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path}:{line}: function {name} has no textual reference"
                )

    _fail("Unreachable named JavaScript functions found:", failures)
    print("Named JavaScript function reachability OK")


def main() -> None:
    check_unused_python_imports()
    check_top_level_python_reachability()
    check_exact_python_helper_duplication()
    check_duplicate_api_routes()
    check_named_javascript_reachability()


if __name__ == "__main__":
    main()
