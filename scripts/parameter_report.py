from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.parameters import parameter_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the effective Manga Translator tuning parameters."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of an aligned table",
    )
    args = parser.parse_args()

    parameters = parameter_snapshot()
    if args.json:
        print(json.dumps(parameters, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    width = max((len(name) for name in parameters), default=0)
    for name, value in parameters.items():
        print(f"{name:<{width}} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
