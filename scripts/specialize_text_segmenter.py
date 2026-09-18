#!/usr/bin/env python3
"""Build a lower-resolution text_segmenter ONNX from the validated 1024 export.

The trained weights and convolution graph are unchanged. Ultralytics' static
YOLOv8m-seg export bakes the candidate count, anchor grid and stride vector into
the detection head, so changing only the declared input shape is invalid. This
utility specializes those size-dependent tensors for another square input.

This is a build/provisioning tool, not a runtime dependency. Production falls
back to text_segmenter.onnx when the specialized artifact is absent or invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper


SOURCE_SIZE = 1024
SOURCE_CANDIDATES = 21_504
RESHAPE0 = "/model.22/dfl/Constant_output_0"
RESHAPE1 = "/model.22/dfl/Constant_1_output_0"
ANCHOR0 = "/model.22/Constant_12_output_0"
ANCHOR1 = "/model.22/Constant_13_output_0"
STRIDES = "/model.22/Constant_15_output_0"


def _shape(value_info) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return result


def _set_shape(value_info, dims: list[int]) -> None:
    shape = value_info.type.tensor_type.shape.dim
    if len(shape) != len(dims):
        raise ValueError(
            f"{value_info.name} rank mismatch: {len(shape)} != {len(dims)}"
        )
    for slot, value in zip(shape, dims):
        slot.dim_param = ""
        slot.dim_value = int(value)


def _initializer_index(model) -> dict[str, int]:
    return {item.name: index for index, item in enumerate(model.graph.initializer)}


def _replace_initializer(model, index_by_name: dict[str, int], name: str, array) -> None:
    index = index_by_name.get(name)
    if index is None:
        raise ValueError(f"Required YOLO head initializer is missing: {name}")
    model.graph.initializer[index].CopyFrom(
        numpy_helper.from_array(np.asarray(array), name=name)
    )


def _make_anchors(size: int) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    for stride in (8, 16, 32):
        side = size // stride
        y, x = np.meshgrid(
            np.arange(side, dtype=np.float32) + 0.5,
            np.arange(side, dtype=np.float32) + 0.5,
            indexing="ij",
        )
        points.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        strides.append(
            np.full(side * side, float(stride), dtype=np.float32)
        )
    anchors = np.concatenate(points, axis=0).T[None, :, :].astype(np.float32)
    stride_tensor = np.concatenate(strides, axis=0)[None, :].astype(np.float32)
    return anchors, stride_tensor


def _validate_source(model) -> None:
    inputs = list(model.graph.input)
    outputs = {item.name: item for item in model.graph.output}
    if len(inputs) != 1 or inputs[0].name != "images":
        raise ValueError("Expected one detector input named 'images'")
    actual_input = _shape(inputs[0])
    if actual_input != [1, 3, SOURCE_SIZE, SOURCE_SIZE]:
        raise ValueError(
            f"Expected validated 1024 text segmenter input; got {actual_input}"
        )
    if set(outputs) != {"output0", "output1"}:
        raise ValueError(
            f"Expected text segmenter outputs output0/output1; got {sorted(outputs)}"
        )
    if _shape(outputs["output0"]) != [1, 37, SOURCE_CANDIDATES]:
        raise ValueError(
            "Unexpected detector head layout; refusing to patch an unknown export"
        )
    if _shape(outputs["output1"]) != [1, 32, 256, 256]:
        raise ValueError(
            "Unexpected mask prototype layout; refusing to patch an unknown export"
        )

    names = _initializer_index(model)
    required = (RESHAPE0, RESHAPE1, ANCHOR0, ANCHOR1, STRIDES)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing validated YOLO head tensors: {missing}")

    expected_shapes = {
        RESHAPE0: [4],
        RESHAPE1: [3],
        ANCHOR0: [1, 2, SOURCE_CANDIDATES],
        ANCHOR1: [1, 2, SOURCE_CANDIDATES],
        STRIDES: [1, SOURCE_CANDIDATES],
    }
    for name, expected in expected_shapes.items():
        arr = numpy_helper.to_array(model.graph.initializer[names[name]])
        if list(arr.shape) != expected:
            raise ValueError(
                f"Unexpected initializer shape for {name}: {list(arr.shape)}"
            )


def specialize_text_segmenter(source: Path, output: Path, size: int) -> dict:
    size = int(size)
    if size < 256 or size > SOURCE_SIZE or size % 32:
        raise ValueError("Target size must be a multiple of 32 in [256, 1024]")
    if source.resolve() == output.resolve():
        raise ValueError("Refusing to overwrite the source model in place")

    model = onnx.load(str(source), load_external_data=True)
    _validate_source(model)

    anchors, stride_tensor = _make_anchors(size)
    candidate_count = int(anchors.shape[-1])
    index_by_name = _initializer_index(model)

    _set_shape(model.graph.input[0], [1, 3, size, size])
    outputs = {item.name: item for item in model.graph.output}
    _set_shape(outputs["output0"], [1, 37, candidate_count])
    _set_shape(outputs["output1"], [1, 32, size // 4, size // 4])

    _replace_initializer(
        model,
        index_by_name,
        RESHAPE0,
        np.asarray([1, 4, 16, candidate_count], dtype=np.int64),
    )
    _replace_initializer(
        model,
        index_by_name,
        RESHAPE1,
        np.asarray([1, 4, candidate_count], dtype=np.int64),
    )
    _replace_initializer(model, index_by_name, ANCHOR0, anchors)
    _replace_initializer(model, index_by_name, ANCHOR1, anchors.copy())
    _replace_initializer(model, index_by_name, STRIDES, stride_tensor)

    # The static export contains inferred intermediate shapes for 1024. They are
    # advisory metadata rather than weights/ops; leaving them stale can make
    # compilers reject an otherwise-correct specialized graph.
    del model.graph.value_info[:]

    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=output.stem + ".",
        suffix=".tmp.onnx",
        dir=str(output.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        onnx.save(model, str(temp_path))
        reloaded = onnx.load(str(temp_path), load_external_data=True)
        onnx.checker.check_model(reloaded)
        if _shape(reloaded.graph.input[0]) != [1, 3, size, size]:
            raise RuntimeError("Specialized model input shape changed during save")
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "source": str(source),
        "output": str(output),
        "size": size,
        "candidate_count": candidate_count,
        "prototype_shape": [1, 32, size // 4, size // 4],
        "stride_counts": {
            "8": (size // 8) ** 2,
            "16": (size // 16) ** 2,
            "32": (size // 32) ** 2,
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/text_segmenter.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/text_segmenter_640.onnx"),
    )
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    report = specialize_text_segmenter(args.source, args.output, args.size)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
