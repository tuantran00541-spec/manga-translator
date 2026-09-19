#!/usr/bin/env python3
"""Build an accuracy-first INT8 QDQ variant of the residue-only 640 segmenter.

This is an experimental/build-time tool. It does not change production model
selection. The resulting ONNX is intended to be copied over the retry-only
text_segmenter_640.onnx inside an isolated benchmark lane.

Calibration preprocessing intentionally mirrors YoloDetector._preprocess:
BGR->RGB, exact square letterbox with fill=114, float32 /255, NCHW batch 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LETTERBOX_VALUE = 114
HEAD_SUBGRAPH_INPUTS = (
    "/model.22/Concat_3",
    "/model.22/Concat_6",
    "/model.22/Concat_24",
    "/model.22/Concat_5",
    "/model.22/Concat_4",
)
HEAD_SUBGRAPH_OUTPUT = "/model.22/Concat_29"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_paths(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Refusing to overwrite the source model in place")


def select_calibration_images(
    raw_dir: Path,
    *,
    start_index: int,
    subset_size: int,
) -> list[Path]:
    start_index = max(0, int(start_index))
    subset_size = int(subset_size)
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if not raw_dir.is_dir():
        raise ValueError(f"Calibration directory does not exist: {raw_dir}")

    images = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    selected = images[start_index : start_index + subset_size]
    if not selected:
        raise ValueError(
            f"No calibration images in {raw_dir} from index {start_index}"
        )
    return selected


def preprocess_calibration_image(
    image: np.ndarray,
    input_size: int,
) -> np.ndarray:
    input_size = int(input_size)
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if image is None or image.ndim < 2:
        raise ValueError("calibration image must be a non-empty array")

    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("calibration image dimensions must be positive")

    nominal = min(input_size / src_w, input_size / src_h)
    resized_w = max(1, min(input_size, int(src_w * nominal)))
    resized_h = max(1, min(input_size, int(src_h * nominal)))
    pad_x = (input_size - resized_w) // 2
    pad_y = (input_size - resized_h) // 2

    rgb = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3 and image.shape[2] == 3
        else image
    )
    resized = cv2.resize(rgb, (resized_w, resized_h))
    canvas = np.full(
        (input_size, input_size, 3),
        LETTERBOX_VALUE,
        dtype=np.uint8,
    )
    canvas[
        pad_y : pad_y + resized_h,
        pad_x : pad_x + resized_w,
    ] = resized
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


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


def _candidate_count(size: int) -> int:
    return sum((size // stride) ** 2 for stride in (8, 16, 32))


def validate_source_contract(model, *, input_size: int) -> None:
    inputs = list(model.graph.input)
    outputs = {item.name: item for item in model.graph.output}
    if len(inputs) != 1 or inputs[0].name != "images":
        raise ValueError("Expected one detector input named 'images'")
    if _shape(inputs[0]) != [1, 3, input_size, input_size]:
        raise ValueError(
            f"Expected static {input_size}x{input_size} segmenter input; "
            f"got {_shape(inputs[0])}"
        )
    if set(outputs) != {"output0", "output1"}:
        raise ValueError(
            f"Expected text segmenter outputs output0/output1; got {sorted(outputs)}"
        )

    candidates = _candidate_count(input_size)
    expected0 = [1, 37, candidates]
    expected1 = [1, 32, input_size // 4, input_size // 4]
    if _shape(outputs["output0"]) != expected0:
        raise ValueError(
            f"Unexpected detector head shape: {_shape(outputs['output0'])}; "
            f"expected {expected0}"
        )
    if _shape(outputs["output1"]) != expected1:
        raise ValueError(
            f"Unexpected mask prototype shape: {_shape(outputs['output1'])}; "
            f"expected {expected1}"
        )


def _ignored_scope(nncf, model):
    # Intel's official YOLOv8-seg accuracy-control example keeps several
    # numerically sensitive ops plus the final head subgraph in floating point.
    # Our YOLOv8m-seg export normally has the same model.22 tensor names. If an
    # exporter changes those names, keep the type-level protection rather than
    # refusing to run the experiment.
    tensor_names = {
        name
        for node in model.graph.node
        for name in tuple(node.input) + tuple(node.output)
        if name
    }
    subgraphs = []
    if all(name in tensor_names for name in HEAD_SUBGRAPH_INPUTS) and (
        HEAD_SUBGRAPH_OUTPUT in tensor_names
    ):
        subgraphs.append(
            nncf.Subgraph(
                inputs=list(HEAD_SUBGRAPH_INPUTS),
                outputs=[HEAD_SUBGRAPH_OUTPUT],
            )
        )
    return nncf.IgnoredScope(
        types=["Mul", "Sub", "Sigmoid"],
        subgraphs=subgraphs,
    )


def quantize_text_segmenter_int8(
    source: Path,
    output: Path,
    raw_dir: Path,
    *,
    input_size: int = 640,
    start_index: int = 32,
    subset_size: int = 64,
) -> dict:
    validate_model_paths(source, output)
    if not source.is_file():
        raise ValueError(f"Source model does not exist: {source}")

    import nncf
    import onnx

    model = onnx.load(str(source), load_external_data=True)
    validate_source_contract(model, input_size=input_size)
    calibration_paths = select_calibration_images(
        raw_dir,
        start_index=start_index,
        subset_size=subset_size,
    )
    input_name = model.graph.input[0].name

    def transform_fn(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read calibration image: {path}")
        return {input_name: preprocess_calibration_image(image, input_size)}

    calibration_dataset = nncf.Dataset(calibration_paths, transform_fn)
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        subset_size=len(calibration_paths),
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        fast_bias_correction=False,
        ignored_scope=_ignored_scope(nncf, model),
    )
    validate_source_contract(quantized_model, input_size=input_size)
    onnx.checker.check_model(quantized_model)

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=output.stem + ".",
        suffix=".tmp.onnx",
        dir=str(output.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        onnx.save(quantized_model, str(temp_path))
        reloaded = onnx.load(str(temp_path), load_external_data=True)
        validate_source_contract(reloaded, input_size=input_size)
        onnx.checker.check_model(reloaded)
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)

    source_bytes = source.stat().st_size
    output_bytes = output.stat().st_size
    return {
        "source": str(source),
        "output": str(output),
        "input_size": int(input_size),
        "start_index": int(start_index),
        "calibration_images": len(calibration_paths),
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "size_ratio": output_bytes / max(1, source_bytes),
        "nncf_version": getattr(nncf, "__version__", "unknown"),
        "onnx_version": getattr(onnx, "__version__", "unknown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("models/text_segmenter_640.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/text_segmenter_640_int8.onnx"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--start-index", type=int, default=32)
    parser.add_argument("--subset-size", type=int, default=64)
    args = parser.parse_args()

    report = quantize_text_segmenter_int8(
        args.source,
        args.output,
        args.raw_dir,
        input_size=args.input_size,
        start_index=args.start_index,
        subset_size=args.subset_size,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
