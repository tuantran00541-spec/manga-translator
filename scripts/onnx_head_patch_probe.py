#!/usr/bin/env python3
"""Throwaway probe: specialize the static YOLOv8m-seg ONNX head for a new square input.

Weights and convolution graph stay unchanged. We patch only input/output metadata
and the five size-dependent detection-head initializers: DFL reshape shapes,
anchor points, and per-anchor strides.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort


RESHAPE0 = "/model.22/dfl/Constant_output_0"
RESHAPE1 = "/model.22/dfl/Constant_1_output_0"
ANCHOR0 = "/model.22/Constant_12_output_0"
ANCHOR1 = "/model.22/Constant_13_output_0"
STRIDES = "/model.22/Constant_15_output_0"


def set_shape(value_info, dims):
    shape = value_info.type.tensor_type.shape.dim
    if len(shape) != len(dims):
        raise RuntimeError(f"rank mismatch for {value_info.name}")
    for slot, value in zip(shape, dims):
        slot.dim_param = ""
        slot.dim_value = int(value)


def replace_initializer(model, name, array):
    for index, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.asarray(array), name=name)
            )
            return
    raise KeyError(name)


def make_anchors(size):
    points = []
    strides = []
    for stride in (8, 16, 32):
        side = size // stride
        y, x = np.meshgrid(
            np.arange(side, dtype=np.float32) + 0.5,
            np.arange(side, dtype=np.float32) + 0.5,
            indexing="ij",
        )
        points.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        strides.append(np.full(side * side, float(stride), dtype=np.float32))
    anchor = np.concatenate(points, axis=0).T[None, :, :].astype(np.float32)
    stride_tensor = np.concatenate(strides, axis=0)[None, :].astype(np.float32)
    return anchor, stride_tensor


def specialize(source: Path, target: Path, size: int):
    if size % 32:
        raise ValueError("YOLO input must be divisible by 32")
    model = onnx.load(str(source), load_external_data=True)
    anchor, stride_tensor = make_anchors(size)
    count = int(anchor.shape[-1])

    set_shape(model.graph.input[0], [1, 3, size, size])
    set_shape(model.graph.output[0], [1, 37, count])
    set_shape(model.graph.output[1], [1, 32, size // 4, size // 4])

    replace_initializer(
        model, RESHAPE0, np.asarray([1, 4, 16, count], dtype=np.int64)
    )
    replace_initializer(
        model, RESHAPE1, np.asarray([1, 4, count], dtype=np.int64)
    )
    replace_initializer(model, ANCHOR0, anchor)
    replace_initializer(model, ANCHOR1, anchor.copy())
    replace_initializer(model, STRIDES, stride_tensor)

    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return {
        "candidate_count": count,
        "anchor_shape": list(anchor.shape),
        "stride_shape": list(stride_tensor.shape),
        "stride_counts": {
            "8": int((size // 8) ** 2),
            "16": int((size // 16) ** 2),
            "32": int((size // 32) ** 2),
        },
    }


def configs():
    available = set(ort.get_available_providers())
    rows = []
    if "CPUExecutionProvider" in available:
        rows.append(("cpu", ["CPUExecutionProvider"]))
    if "OpenVINOExecutionProvider" in available:
        rows.append((
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
    return rows


def exercise(path: Path, size: int, repeats: int):
    out = {}
    for label, providers in configs():
        row = {}
        try:
            t0 = time.perf_counter()
            session = ort.InferenceSession(str(path), providers=providers)
            row["session_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            row["effective_providers"] = list(session.get_providers())
            inp = session.get_inputs()[0]
            row["input_shape"] = list(inp.shape)
            row["declared_outputs"] = [list(x.shape) for x in session.get_outputs()]
            x = np.zeros((1, 3, size, size), dtype=np.float32)

            t0 = time.perf_counter()
            outputs = session.run(None, {inp.name: x})
            row["first_run_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            row["actual_outputs"] = [list(np.asarray(x).shape) for x in outputs]

            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                session.run(None, {inp.name: x})
                times.append((time.perf_counter() - t0) * 1000)
            row["warm_runs_ms"] = [round(x, 3) for x in times]
            row["warm_median_ms"] = round(float(np.median(times)), 3)
            row["success"] = True
        except Exception as exc:
            row["success"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            row["traceback_tail"] = traceback.format_exc().splitlines()[-15:]
        out[label] = row
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/text_segmenter.onnx"))
    p.add_argument("--output-dir", type=Path, default=Path("benchmark-results/onnx-surgery"))
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 512])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source": str(args.model),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "candidates": [],
    }
    for size in args.sizes:
        target = args.output_dir / f"text_segmenter_{size}_head_patched.onnx"
        row = {"size": size}
        try:
            row["patch"] = specialize(args.model, target, size)
            row["patch"]["success"] = True
            row["execution"] = exercise(target, size, args.repeats)
        except Exception as exc:
            row["patch"] = {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-15:],
            }
        report["candidates"].append(row)

    output = args.output_dir / "head-patch-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
