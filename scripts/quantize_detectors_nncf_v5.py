from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import nncf
import numpy as np
import onnx


def _dataset(tensor_paths: list[Path], input_name: str) -> nncf.Dataset:
    def transform(path: Path):
        canvas = np.load(path, allow_pickle=False)
        if canvas.ndim != 3 or canvas.shape[2] != 3:
            raise ValueError(f"Unexpected calibration tensor shape {canvas.shape}: {path}")
        blob = canvas.astype(np.float32, copy=False) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return {input_name: blob}

    return nncf.Dataset(tensor_paths, transform)


def _quantize_one(model_path: Path, tensor_paths: list[Path]) -> dict:
    started = time.perf_counter()
    fp32_size = model_path.stat().st_size
    model = onnx.load(model_path.as_posix())
    if not model.graph.input:
        raise RuntimeError(f"Model has no inputs: {model_path}")
    input_name = model.graph.input[0].name
    dataset = _dataset(tensor_paths, input_name)
    quantized = nncf.quantize(
        model,
        dataset,
        subset_size=len(tensor_paths),
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
        fast_bias_correction=True,
    )
    temp_path = model_path.with_suffix(".int8.tmp.onnx")
    onnx.save_model(quantized, temp_path.as_posix())
    checked = onnx.load(temp_path.as_posix())
    onnx.checker.check_model(checked)
    int8_size = temp_path.stat().st_size
    os.replace(temp_path, model_path)
    return {
        "name": model_path.name,
        "input_name": input_name,
        "fp32_size_bytes": fp32_size,
        "int8_size_bytes": int8_size,
        "size_ratio": round(int8_size / fp32_size, 6) if fp32_size else None,
        "quantization_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    tensor_paths = sorted(Path(args.calibration_dir).glob("*.npy"))
    if not tensor_paths:
        raise RuntimeError(f"No calibration tensors in {args.calibration_dir}")

    started = time.perf_counter()
    rows = []
    for raw_path in args.models:
        model_path = Path(raw_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        rows.append(_quantize_one(model_path, tensor_paths))

    report = {
        "status": "pass",
        "nncf_version": getattr(nncf, "__version__", "unknown"),
        "onnx_version": getattr(onnx, "__version__", "unknown"),
        "preset": "performance",
        "target_device": "CPU",
        "calibration_tensors": len(tensor_paths),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "models": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
