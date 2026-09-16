#!/usr/bin/env python3
"""Create or validate the reproducibility manifest used by perf experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.benchmarking.manifest import build_manifest, load_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a minimal benchmark manifest")
    init.add_argument("output", type=Path)
    init.add_argument("--benchmark", required=True)
    init.add_argument("--dataset", required=True)
    init.add_argument("--repo-sha", required=True)
    init.add_argument("--profile", default="control")
    init.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="model path; may be repeated",
    )
    init.add_argument("--parameter", action="append", default=[], metavar="NAME=VALUE")

    validate = sub.add_parser("validate", help="validate an existing manifest")
    validate.add_argument("manifest", type=Path)

    args = parser.parse_args()
    if args.command == "validate":
        manifest = load_manifest(args.manifest)
        print(json.dumps({"status": "pass", "manifest_sha256": manifest.get("manifest_sha256")}, indent=2))
        return 0

    models = {}
    for item in args.model:
        if "=" not in item:
            parser.error("--model must use NAME=PATH")
        name, path = item.split("=", 1)
        if not name or not path:
            parser.error("--model must use NAME=PATH")
        models[name] = path
    parameters = {}
    for item in args.parameter:
        if "=" not in item:
            parser.error("--parameter must use NAME=VALUE")
        name, value = item.split("=", 1)
        if not name:
            parser.error("--parameter must use NAME=VALUE")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        parameters[name] = value
    manifest = build_manifest(
        benchmark=args.benchmark,
        dataset=args.dataset,
        repo_sha=args.repo_sha,
        profile=args.profile,
        model_paths=models,
        parameters=parameters,
    )
    write_manifest(args.output, manifest)
    print(json.dumps({"status": "pass", "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
