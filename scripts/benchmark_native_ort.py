#!/usr/bin/env python3
"""Compare Python and C++ ONNX Runtime calls on identical detector/LaMa tensors.

This is a wrapper-overhead probe, not a production migration.  Both paths use
CPUExecutionProvider with fixed thread settings, then the script checks every
float output before reporting a speed difference.  Image preprocessing,
detector postprocessing and the inpainting safety policy stay in Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_LETTERBOX_VALUE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("detector", "lama"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ort-root", required=True, type=Path,
                        help="Unpacked onnxruntime-linux-x64-<version> directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=positive_int, default=7)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--lama-size", type=positive_int, default=512)
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def make_session(model: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # Match app.ort_utils' conservative CPU path rather than benchmarking a
    # larger cache footprint that production intentionally does not allow.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(model), sess_options=options,
                                providers=["CPUExecutionProvider"])


def detector_tensor(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("input image is empty")
    input_size = int(DETECTOR_INPUT_SIZE)
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, min(input_size, int(width * scale)))
    resized_height = max(1, min(input_size, int(height * scale)))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_width, resized_height))
    canvas = np.full((input_size, input_size, 3), DETECTOR_LETTERBOX_VALUE,
                     dtype=np.uint8)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return np.ascontiguousarray(
        (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    )


def lama_tensors(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    canvas = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image_tensor = np.ascontiguousarray(
        (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    )
    mask = np.zeros((size, size), dtype=np.float32)
    margin = max(16, size // 4)
    mask[margin:size - margin, margin:size - margin] = 1.0
    return image_tensor, np.ascontiguousarray(mask[None, None])


def prepare_inputs(args: argparse.Namespace, session: ort.InferenceSession) -> dict[str, np.ndarray]:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {args.image}")
    inputs = session.get_inputs()
    if args.kind == "detector":
        if len(inputs) != 1:
            raise RuntimeError(f"detector expected one input, got {len(inputs)}")
        return {inputs[0].name: detector_tensor(image)}
    if len(inputs) != 2:
        raise RuntimeError(f"LaMa expected two inputs, got {len(inputs)}")
    image_tensor, mask_tensor = lama_tensors(image, args.lama_size)
    return {inputs[0].name: image_tensor, inputs[1].name: mask_tensor}


def run_python(session: ort.InferenceSession, feeds: dict[str, np.ndarray],
               warmup: int, runs: int) -> tuple[list[np.ndarray], dict[str, float]]:
    names = [output.name for output in session.get_outputs()]
    for _ in range(warmup):
        session.run(names, feeds)
    durations: list[float] = []
    outputs: list[np.ndarray] = []
    for _ in range(runs):
        started = time.perf_counter()
        outputs = session.run(names, feeds)
        durations.append((time.perf_counter() - started) * 1000.0)
    return outputs, {
        "runs": runs,
        "warmup": warmup,
        "min_ms": min(durations),
        "median_ms": percentile(durations, 0.5),
        "p95_ms": percentile(durations, 0.95),
    }


def compile_native(args: argparse.Namespace, binary: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "native" / "ort_native_benchmark.cc"
    include = args.ort_root / "include"
    library = args.ort_root / "lib"
    if not (include / "onnxruntime_cxx_api.h").is_file():
        raise RuntimeError(f"missing C++ header in --ort-root: {include}")
    if not (library / "libonnxruntime.so").is_file():
        raise RuntimeError(f"missing C++ library in --ort-root: {library}")
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("g++ or c++ is required for the native probe")
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler, "-std=c++17", "-O3", "-DNDEBUG", str(source),
        f"-I{include}", f"-L{library}", "-lonnxruntime",
        f"-Wl,-rpath,{library}", "-o", str(binary),
    ]
    subprocess.run(command, check=True)


def write_manifest(feeds: dict[str, np.ndarray], work_dir: Path) -> Path:
    manifest = work_dir / "inputs.tsv"
    rows: list[str] = []
    for index, (name, value) in enumerate(feeds.items()):
        tensor = np.ascontiguousarray(value, dtype=np.float32)
        data_path = work_dir / f"input_{index}.f32"
        tensor.tofile(data_path)
        shape = ",".join(str(int(dimension)) for dimension in tensor.shape)
        rows.append(f"{name}\t{shape}\t{data_path.resolve()}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def run_native(args: argparse.Namespace, manifest: Path, work_dir: Path) -> dict[str, Any]:
    binary = work_dir / "ort_native_benchmark"
    compile_native(args, binary)
    command = [
        str(binary), "--model", str(args.model.resolve()), "--manifest", str(manifest),
        "--output-dir", str(work_dir.resolve()), "--warmup", str(args.warmup),
        "--runs", str(args.runs), "--intra-op", str(args.threads),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=True)
    prefix = "NATIVE_ORT_RESULT "
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError(f"native probe did not return a result: {completed.stdout}\n{completed.stderr}")


def compare_outputs(reference: list[np.ndarray], work_dir: Path) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    all_exact = True
    for index, expected in enumerate(reference):
        path = work_dir / f"cpp_output_{index}.f32"
        actual = np.fromfile(path, dtype=np.float32).reshape(expected.shape)
        exact = bool(np.array_equal(expected, actual))
        all_exact = all_exact and exact
        checked.append({
            "index": index,
            "shape": list(expected.shape),
            "exact": exact,
            "max_abs_diff": float(np.max(np.abs(expected - actual))),
        })
    return {"all_exact": all_exact, "outputs": checked}


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    args.out.mkdir(parents=True, exist_ok=True)
    session = make_session(args.model, args.threads)
    feeds = prepare_inputs(args, session)
    reference, python_result = run_python(session, feeds, args.warmup, args.runs)
    work_dir = args.out / "native"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(feeds, work_dir)
    native_result = run_native(args, manifest, work_dir)
    output_check = compare_outputs(reference, work_dir)
    python_ms = float(python_result["median_ms"])
    native_ms = float(native_result["median_ms"])
    report = {
        "benchmark": "native-ort-wrapper-probe",
        "kind": args.kind,
        "model": args.model.name,
        "provider": "CPUExecutionProvider",
        "threads": args.threads,
        "scope": "Session::Run only; excludes image pre/postprocessing and safe pipeline policy",
        "python": python_result,
        "cpp": native_result,
        "output_check": output_check,
        "median_speedup": python_ms / native_ms if native_ms else None,
        "decision": (
            "probe-only: do not migrate production without same-provider end-to-end and quality evidence"
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if output_check["all_exact"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
