from __future__ import annotations

import ast
import difflib

LIBRARY = frozenset({"search", "describe", "intent", "branch", "adopt"})
_TRY_NODES = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)
_SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _catches_any(handler: ast.ExceptHandler, targets: tuple[str, ...]) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in targets for n in names)


def _binds_tools_in_scope(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        all_args = [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *([args.vararg] if args.vararg else []),
            *([args.kwarg] if args.kwarg else []),
        ]
        if any(a.arg == "tools" for a in all_args):
            return True
    elif isinstance(node, ast.Lambda):
        args = node.args
        all_args = [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *([args.vararg] if args.vararg else []),
            *([args.kwarg] if args.kwarg else []),
        ]
        if any(a.arg == "tools" for a in all_args):
            return True
    body = getattr(node, "body", None)
    if not isinstance(body, list):
        return False
    stack = list(body)
    while stack:
        cur = stack.pop()
        if isinstance(cur, _SCOPE_BOUNDARIES):
            if getattr(cur, "name", None) == "tools":
                return True
            continue
        if isinstance(cur, ast.Name) and isinstance(cur.ctx, ast.Store) and cur.id == "tools":
            return True
        if isinstance(cur, (ast.Import, ast.ImportFrom)):
            for alias in cur.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound == "tools":
                    return True
        if isinstance(cur, ast.ExceptHandler) and cur.name == "tools":
            return True
        stack.extend(ast.iter_child_nodes(cur))
    return False


def _collect_calls_and_probes(
    tree: ast.AST,
) -> tuple[list[ast.Call], set[int], set[int]]:
    calls: list[ast.Call] = []
    probed_attr: set[int] = set()
    probed_type: set[int] = set()

    def visit(
        node: ast.AST,
        *,
        shadowed: bool,
        in_attr_try: bool,
        in_type_try: bool,
    ) -> None:
        if isinstance(node, _SCOPE_BOUNDARIES):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    visit(dec, shadowed=shadowed, in_attr_try=in_attr_try, in_type_try=in_type_try)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for default in [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]:
                    visit(default, shadowed=shadowed, in_attr_try=in_attr_try, in_type_try=in_type_try)
            inner_shadowed = shadowed or _binds_tools_in_scope(node)
            inner_children = (
                [node.body] if isinstance(node, ast.Lambda) else list(node.body)
            )
            for child in inner_children:
                visit(child, shadowed=inner_shadowed, in_attr_try=False, in_type_try=False)
            return

        if isinstance(node, _TRY_NODES):
            catches_attr = any(
                _catches_any(h, ("AttributeError", "Exception", "BaseException"))
                for h in node.handlers
            )
            catches_type = any(
                _catches_any(h, ("TypeError", "Exception", "BaseException"))
                for h in node.handlers
            )
            for stmt in node.body:
                visit(
                    stmt,
                    shadowed=shadowed,
                    in_attr_try=in_attr_try or catches_attr,
                    in_type_try=in_type_try or catches_type,
                )
            for handler in node.handlers:
                visit(handler, shadowed=shadowed, in_attr_try=in_attr_try, in_type_try=in_type_try)
            for stmt in (*node.orelse, *node.finalbody):
                visit(stmt, shadowed=shadowed, in_attr_try=in_attr_try, in_type_try=in_type_try)
            return

        if (
            not shadowed
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tools"
        ):
            calls.append(node)
            if in_attr_try:
                probed_attr.add(id(node))
            if in_type_try:
                probed_type.add(id(node))

        for child in ast.iter_child_nodes(node):
            visit(child, shadowed=shadowed, in_attr_try=in_attr_try, in_type_try=in_type_try)

    module_shadowed = _binds_tools_in_scope(tree)
    visit(tree, shadowed=module_shadowed, in_attr_try=False, in_type_try=False)
    return calls, probed_attr, probed_type


def _created_tool_name(call: ast.Call) -> str | None:
    if call.func.attr != "create":
        return None
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def check(
    code: str,
    known: dict[str, set[str] | None],
    fixes=lambda name, args: False,
) -> list[str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []

    calls, probed_attr, probed_type = _collect_calls_and_probes(tree)
    made: set[str] = set()
    problems: list[str] = []
    all_names = set(known) | LIBRARY

    for call in sorted(calls, key=lambda c: (c.lineno, c.col_offset)):
        created = _created_tool_name(call)
        if created:
            made.add(created)
        name = call.func.attr
        given = [kw.arg for kw in call.keywords if kw.arg is not None]
        if name in LIBRARY or name in made or name.startswith("_"):
            continue
        missing_tool = name not in known
        bad_args = (not missing_tool) and known[name] is not None and bool(set(given) - known[name])
        if missing_tool and id(call) in probed_attr:
            continue
        if bad_args and id(call) in probed_type:
            continue
        if (missing_tool or bad_args) and fixes(name, given):
            continue
        if missing_tool:
            near = difflib.get_close_matches(name, all_names, n=3)
            problems.append(
                f"line {call.lineno}: tools.{name} is not a tool this chain can reach"
                + (f"; did you mean {', '.join(near)}?" if near else "")
            )
            continue
        allowed = known[name]
        if allowed is None:
            continue
        for arg in given:
            if arg not in allowed:
                problems.append(
                    f"line {call.lineno}: tools.{name} has no argument {arg!r}; "
                    f"it takes {', '.join(sorted(allowed)) or 'none'}"
                )
    return problems
