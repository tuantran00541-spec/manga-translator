#!/usr/bin/env python3
"""Throwaway feasibility probe for static-input ONNX graph surgery.

This does not modify production models. It copies text_segmenter.onnx, changes
only the declared image input H/W, and asks ONNX checker plus ORT/OpenVINO to
load and execute the candidate. The goal is to discover whether the exported
YOLOv8m-seg graph is genuinely shape-flexible or contains baked 1024 constants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort


def declared_shape(value_info):
    out = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            out.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            out.append(str(dim.dim_param))
        else:
            out.append(None)
    return out


def mutate_input_hw(source: Path, target: Path, size: int) -> dict:
    model = onnx.load(str(source), load_external_data=True)
    image = model.graph.input[0]
    before = declared_shape(image)
    dims = image.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise RuntimeError(f"expected NCHW detector input, got {before}")
    dims[2].dim_param = ""
    dims[2].dim_value = int(size)
    dims[3].dim_param = ""
    dims[3].dim_value = int(size)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    reloaded = onnx.load(str(target), load_external_data=True)
    onnx.checker.check_model(reloaded)
    return {
        "declared_input_before": before,
        "declared_input_after": declared_shape(reloaded.graph.input[0]),
        "node_count": len(reloaded.graph.node),
        "initializer_count": len(reloaded.graph.initializer),
    }


def provider_configs():
    available = set(ort.get_available_providers())
    configs = []
    if "CPUExecutionProvider" in available:
        configs.append(("cpu", ["CPUExecutionProvider"]))
    if "OpenVINOExecutionProvider" in available:
        configs.append((
            "openvino",
            [
                (
                    "OpenVINOExecutionProvider",
                    {
                        "device_type": "CPU",
                        "precision": "FP32",
                        "num_of_threads": "2",
                        "num_streams": "1",
                    },
                ),
                "CPUExecutionProvider",
            ],
        ))
    return configs


def exercise(model_path: Path, size: int, repeats: int) -> dict:
    result = {
        "model": model_path.name,
        "size": int(size),
        "providers": {},
    }
    for label, providers in provider_configs():
        row = {}
        try:
            started = time.perf_counter()
            session = ort.InferenceSession(str(model_path), providers=providers)
            row["session_load_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["effective_providers"] = list(session.get_providers())
            meta = session.get_inputs()[0]
            row["runtime_input_shape"] = list(meta.shape)
            row["runtime_output_shapes"] = [list(o.shape) for o in session.get_outputs()]
            x = np.zeros((1, 3, size, size), dtype=np.float32)
            feed = {meta.name: x}

            started = time.perf_counter()
            outputs = session.run(None, feed)
            row["first_run_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["actual_output_shapes"] = [list(np.asarray(o).shape) for o in outputs]

            timings = []
            for _ in range(max(1, repeats)):
                started = time.perf_counter()
                session.run(None, feed)
                timings.append((time.perf_counter() - started) * 1000)
            row["warm_runs_ms"] = [round(v, 3) for v in timings]
            row["warm_median_ms"] = round(float(np.median(timings)), 3)
            row["success"] = True
        except Exception as exc:
            row["success"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            row["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        result["providers"][label] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/text_segmenter.onnx"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/onnx-surgery"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 640, 512])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.model),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "candidates": [],
    }

    for size in args.sizes:
        if size == 1024:
            candidate = args.model
            mutation = {
                "declared_input_before": None,
                "declared_input_after": None,
                "baseline": True,
            }
        else:
            candidate = args.output_dir / f"text_segmenter_{size}_declared.onnx"
            try:
                mutation = mutate_input_hw(args.model, candidate, size)
                mutation["success"] = True
            except Exception as exc:
                mutation = {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                }
        row = {"size": size, "mutation": mutation}
        if size == 1024 or mutation.get("success"):
            row["execution"] = exercise(candidate, size, args.repeats)
        report["candidates"].append(row)

    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    baseline = next(c for c in report["candidates"] if c["size"] == 1024)
    baseline_ok = any(
        p.get("success") for p in baseline["execution"]["providers"].values()
    )
    if not baseline_ok:
        raise RuntimeError("baseline 1024 model did not execute; probe environment invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
