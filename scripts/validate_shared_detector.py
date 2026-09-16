"""Validate the E8 shared-backbone multi-head contract.

This is a contract/gate tool, not a model trainer.  It prevents a future
multi-head ONNX candidate from accidentally turning proposal or stroke heads
into destructive mask authority before the existing mask-quality verifier has
approved the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_HEADS = {
    "bubble": "proposal",
    "text": "segmentation",
    "stroke": "segmentation",
}


def validate_contract(contract: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["contract must be an object"]
    if contract.get("schema") != "manga-translator.shared-detector-contract.v1":
        errors.append("schema must be manga-translator.shared-detector-contract.v1")
    input_spec = contract.get("input")
    if not isinstance(input_spec, dict):
        errors.append("input must be an object")
    else:
        if input_spec.get("name") != "images":
            errors.append("shared input must be named images")
        if input_spec.get("dtype") != "float32":
            errors.append("shared input must be float32")
        if list(input_spec.get("shape") or []) != [1, 3, 1024, 1024]:
            errors.append("shared input shape must be [1,3,1024,1024]")
    heads = contract.get("heads")
    if not isinstance(heads, dict):
        return errors + ["heads must be an object"]
    if set(heads) != set(REQUIRED_HEADS):
        errors.append(f"heads must be exactly {sorted(REQUIRED_HEADS)}")
    for name, task in REQUIRED_HEADS.items():
        head = heads.get(name)
        if not isinstance(head, dict):
            errors.append(f"head {name!r} must be an object")
            continue
        if not str(head.get("output") or "").strip():
            errors.append(f"head {name!r} needs an output name")
        if head.get("task") != task:
            errors.append(f"head {name!r} task must be {task!r}")
        if bool(head.get("destructive_authority")):
            errors.append(
                f"head {name!r} cannot declare destructive authority; use the existing mask gate"
            )
        if head.get("requires_mask_gate") is not True:
            errors.append(f"head {name!r} must require the mask gate")
    if contract.get("authority_policy") != "existing-text-segmenter-mask-gate":
        errors.append("authority_policy must be existing-text-segmenter-mask-gate")
    return errors


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.contract.is_file():
        return {"status": "blocked", "blockers": [f"contract missing: {args.contract}"]}
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "blocked", "blockers": [f"invalid contract: {type(exc).__name__}: {exc}"]}
    errors = validate_contract(contract)
    if errors:
        return {"status": "fail", "contract_errors": errors}
    if not args.benchmark_report:
        return {
            "status": "blocked",
            "contract": contract,
            "blockers": ["E8 accuracy/latency benchmark report not supplied"],
        }
    if not args.benchmark_report.is_file():
        return {
            "status": "blocked",
            "contract": contract,
            "blockers": [f"benchmark report missing: {args.benchmark_report}"],
        }
    report = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    gates = report.get("gates") if isinstance(report, dict) else None
    if not isinstance(gates, dict):
        return {
            "status": "blocked",
            "contract": contract,
            "blockers": ["E8 report has no explicit gates"],
        }
    required = {
        "head_parity": bool(gates.get("head_parity")),
        "hard_holdout_false_negatives": int(gates.get("hard_holdout_false_negatives", 1)) == 0,
        "authority_safety": int(gates.get("authority_outside_changed", 1)) == 0,
        "speed": bool(gates.get("speed")),
    }
    return {
        "status": "pass" if all(required.values()) else "fail",
        "contract": contract,
        "benchmark_report": args.benchmark_report.as_posix(),
        "gates": required,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate E8 shared-detector contract")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("E8_SHARED_DETECTOR=" + json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report.get("status") == "pass" else 2 if report.get("status") == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
