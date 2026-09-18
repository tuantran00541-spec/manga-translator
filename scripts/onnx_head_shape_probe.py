#!/usr/bin/env python3
"""Inspect static shape/anchor artifacts baked into the YOLOv8 segmentation head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TARGET = 21504


def value_info_shape(value_info):
    out = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            out.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            out.append(str(dim.dim_param))
        else:
            out.append(None)
    return out


def small_tensor(tensor):
    try:
        arr = np.asarray(numpy_helper.to_array(tensor))
    except Exception:
        return None
    row = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "size": int(arr.size),
    }
    flat = arr.reshape(-1)
    if arr.size <= 64:
        row["values"] = flat.tolist()
    elif np.issubdtype(arr.dtype, np.number):
        row["min"] = float(np.min(flat))
        row["max"] = float(np.max(flat))
        row["sample"] = flat[:12].tolist()
    return row


def contains_target(value):
    if value is None:
        return False
    if isinstance(value, dict):
        if TARGET in value.get("shape", []):
            return True
        vals = value.get("values")
        if isinstance(vals, list) and TARGET in vals:
            return True
    return False


def constant_payload(node):
    rows = []
    for attr in node.attribute:
        item = {"name": attr.name, "type": int(attr.type)}
        if attr.type == onnx.AttributeProto.INT:
            item["int"] = int(attr.i)
        elif attr.type == onnx.AttributeProto.INTS:
            item["ints"] = [int(v) for v in attr.ints]
        elif attr.type == onnx.AttributeProto.TENSOR:
            item["tensor"] = small_tensor(attr.t)
        rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/text_segmenter.onnx"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/onnx-surgery/head-shapes.json"))
    args = parser.parse_args()

    model = onnx.load(str(args.model), load_external_data=True)
    producer = {}
    for node in model.graph.node:
        for output in node.output:
            producer[output] = node

    suspicious_initializers = []
    for init in model.graph.initializer:
        row = {"name": init.name, **(small_tensor(init) or {})}
        if TARGET in row.get("shape", []):
            row["reason"] = "dimension_21504"
            suspicious_initializers.append(row)
        elif row.get("size", 0) <= 64 and TARGET in row.get("values", []):
            row["reason"] = "value_21504"
            suspicious_initializers.append(row)

    suspicious_constants = []
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        attrs = constant_payload(node)
        hit = False
        for attr in attrs:
            if attr.get("int") == TARGET or TARGET in attr.get("ints", []):
                hit = True
            if contains_target(attr.get("tensor")):
                hit = True
        if hit:
            suspicious_constants.append({
                "name": node.name,
                "outputs": list(node.output),
                "attributes": attrs,
            })

    head_nodes = []
    for node in model.graph.node:
        if "/model.22" not in (node.name or ""):
            continue
        head_nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "constant_attributes": constant_payload(node) if node.op_type == "Constant" else None,
        })

    focus_names = {"/model.22/dfl/Reshape", "/model.22/Concat_6"}
    focus = []
    for node in model.graph.node:
        if node.name not in focus_names:
            continue
        row = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "input_producers": [],
        }
        for inp in node.input:
            p = producer.get(inp)
            if p is None:
                continue
            prow = {
                "tensor": inp,
                "name": p.name,
                "op_type": p.op_type,
                "inputs": list(p.input),
                "outputs": list(p.output),
            }
            if p.op_type == "Constant":
                prow["constant_attributes"] = constant_payload(p)
            row["input_producers"].append(prow)
        focus.append(row)

    report = {
        "model": str(args.model),
        "ir_version": int(model.ir_version),
        "opsets": [{"domain": o.domain, "version": int(o.version)} for o in model.opset_import],
        "inputs": [{"name": v.name, "shape": value_info_shape(v)} for v in model.graph.input],
        "outputs": [{"name": v.name, "shape": value_info_shape(v)} for v in model.graph.output],
        "suspicious_initializers": suspicious_initializers,
        "suspicious_constants": suspicious_constants,
        "focus": focus,
        "head_node_count": len(head_nodes),
        "head_nodes": head_nodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "inputs": report["inputs"],
        "outputs": report["outputs"],
        "suspicious_initializers": suspicious_initializers,
        "suspicious_constants": suspicious_constants,
        "focus": focus,
        "head_node_count": len(head_nodes),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
