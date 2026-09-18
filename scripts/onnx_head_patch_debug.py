#!/usr/bin/env python3
"""Diagnose whether stale internal value_info blocks the specialized YOLO head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import onnx

from scripts.onnx_head_patch_probe import specialize, exercise


WATCH = {
    "/model.22/Concat_4_output_0",
    "/model.22/Split_output_0",
    "/model.22/dfl/Reshape_output_0",
    "/model.22/dfl/Reshape_1_output_0",
    "/model.22/Slice_output_0",
    "/model.22/Slice_1_output_0",
    "/model.22/Sub_output_0",
    "/model.22/Add_1_output_0",
    "/model.22/Concat_5_output_0",
    "/model.22/Mul_2_output_0",
}


def shape_of(v):
    dims = []
    for d in v.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(str(d.dim_param))
        else:
            dims.append(None)
    return dims


def capture_value_info(model):
    rows = {}
    for v in list(model.graph.value_info) + list(model.graph.output):
        if v.name in WATCH:
            rows[v.name] = shape_of(v)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/text_segmenter.onnx"))
    p.add_argument("--output-dir", type=Path, default=Path("benchmark-results/onnx-surgery"))
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 512])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {"candidates": []}
    for size in args.sizes:
        base = args.output_dir / f"debug_{size}.onnx"
        specialize(args.model, base, size)
        model = onnx.load(str(base), load_external_data=True)
        row = {
            "size": size,
            "value_info_count": len(model.graph.value_info),
            "watched_before_infer": capture_value_info(model),
        }

        try:
            inferred = onnx.shape_inference.infer_shapes(
                model,
                check_type=True,
                strict_mode=True,
                data_prop=True,
            )
            row["shape_inference_success"] = True
            row["watched_after_infer"] = capture_value_info(inferred)
        except Exception as exc:
            row["shape_inference_success"] = False
            row["shape_inference_error"] = str(exc)
            row["shape_inference_traceback_tail"] = traceback.format_exc().splitlines()[-12:]

        stripped = onnx.load(str(base), load_external_data=True)
        del stripped.graph.value_info[:]
        stripped_path = args.output_dir / f"text_segmenter_{size}_head_patched_no_value_info.onnx"
        onnx.checker.check_model(stripped)
        onnx.save(stripped, str(stripped_path))
        row["stripped_execution"] = exercise(stripped_path, size, repeats=2)
        report["candidates"].append(row)

    output = args.output_dir / "head-patch-debug.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
