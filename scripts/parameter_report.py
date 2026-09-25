from __future__ import annotations

import argparse
import ast
import json
from numbers import Number
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import parameters


def parameter_snapshot() -> dict[str, Number | bool | str]:
    result: dict[str, Number | bool | str] = {}
    for name, value in vars(parameters).items():
        if name.isupper() and not name.startswith("_") and isinstance(value, (bool, int, float, str)):
            result[name] = value
    return dict(sorted(result.items()))


def environment_overrides() -> dict[str, str]:
    tree = ast.parse(Path(parameters.__file__).read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if isinstance(function, ast.Name) and function.id.startswith("_env_"):
            result[node.targets[0].id] = node.value.args[0].value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the effective Manga Translator tuning parameters."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of an aligned table",
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="list only the settings that can be overridden with environment variables",
    )
    args = parser.parse_args()

    values = parameter_snapshot()
    overrides = environment_overrides()
    if args.env_only:
        values = {name: value for name, value in values.items() if name in overrides}
    if args.json:
        print(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    width = max((len(name) for name in values), default=0)
    for name, value in values.items():
        suffix = f"  [{overrides[name]}]" if name in overrides else ""
        print(f"{name:<{width}} = {value}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
