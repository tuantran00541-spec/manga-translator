"""ONNX Runtime helpers tuned for multi-page thread pools."""

from __future__ import annotations

import os

import onnxruntime as ort


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 2)


def make_session(model_path, *, intra_op_threads: int | None = None) -> ort.InferenceSession:
    """
    Create an InferenceSession optimized for concurrent page workers.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    if intra_op_threads is None:
        cpu = _cpu_count()
        intra_op_threads = 8 if cpu >= 8 else (2 if cpu >= 4 else 1)
    opts.intra_op_num_threads = max(1, int(intra_op_threads))
    opts.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
