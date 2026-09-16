"""Build the leakage-safe E7/E8 teacher/student case manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.benchmarking.distillation import (
    build_distillation_cases,
    summarize_distillation_cases,
    validate_distillation_cases,
)
from app.benchmarking.failure_registry import FailureRegistry
from app.benchmarking.manifest import build_manifest, sha256_json, write_manifest


def _repo_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.failure_registry.is_file():
        return {
            "status": "blocked",
            "benchmark": "detector-distillation-manifest-v1",
            "blockers": [f"failure registry missing: {args.failure_registry}"],
        }
    rows = FailureRegistry(args.failure_registry).read()
    if not rows:
        return {
            "status": "blocked",
            "benchmark": "detector-distillation-manifest-v1",
            "blockers": ["failure registry has no valid JSONL cases"],
        }
    try:
        cases = build_distillation_cases(rows)
    except Exception as exc:
        return {
            "status": "blocked",
            "benchmark": "detector-distillation-manifest-v1",
            "blockers": [f"cannot assign source partitions: {type(exc).__name__}: {exc}"],
        }
    errors = validate_distillation_cases(cases)
    if errors:
        return {
            "status": "blocked",
            "benchmark": "detector-distillation-manifest-v1",
            "blockers": errors,
            "summary": summarize_distillation_cases(cases),
        }
    model_paths = {}
    for item in args.model:
        if "=" not in item:
            return {"status": "blocked", "blockers": [f"invalid --model {item!r}; use name=path"]}
        name, path = item.split("=", 1)
        model_paths[name.strip()] = path.strip()
    manifest = build_manifest(
        benchmark="detector-distillation-manifest-v1",
        dataset=args.dataset,
        repo_sha=args.repo_sha or _repo_sha(),
        model_paths=model_paths,
        parameters={
            "source_group_partitioning": "sha256-deterministic-v1",
            "teacher": "production-full-stack",
            "student_heads": ["bubble", "text", "stroke"],
            "authority_policy": "student-proposal-only-until-mask-gate",
        },
        cases=cases,
        profile="teacher-labels",
    )
    manifest["status"] = "pass"
    manifest["distillation_summary"] = summarize_distillation_cases(cases)
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E7/E8 distillation manifest")
    parser.add_argument("--failure-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="failure-registry")
    parser.add_argument("--repo-sha", default="")
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # A successful result is a normal validated benchmark manifest. A blocked
    # result remains inspectable and is intentionally not disguised as data.
    if report.get("status") == "pass":
        write_manifest(args.out, report)
    else:
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("E7_E8_DISTILLATION=" + json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
