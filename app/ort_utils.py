"""ONNX Runtime helpers tuned for CPU-only multi-page processing."""

from __future__ import annotations

import os

import onnxruntime as ort


_THREAD_ENV = "MANGA_ORT_INTRA_OP_THREADS"
_DEFAULT_HIGH_CPU_THREADS = 8


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 2)


def _default_intra_op_threads() -> int:
    """Pick a conservative default for CPU-only concurrent page workers.

    The project processes pages concurrently, so deliberately avoid making
    the ORT pool unbounded on high-core machines. Eight threads is the
    benchmarked production target; smaller CPUs get smaller pools.
    """
    cpu = _cpu_count()
    if cpu >= 8:
        return _DEFAULT_HIGH_CPU_THREADS
    if cpu >= 4:
        return 2
    return 1


def _configured_intra_op_threads() -> int:
    """Return the default, with an explicit environment override."""
    raw = os.environ.get(_THREAD_ENV, "").strip()
    if not raw:
        return _default_intra_op_threads()

    try:
        value = int(raw)
    except ValueError:
        return _default_intra_op_threads()

    return max(1, value)


def make_session(model_path, *, intra_op_threads: int | None = None) -> ort.InferenceSession:
    """Create an ONNX Runtime CPU session for concurrent page workers."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.intra_op_num_threads = max(
        1,
        int(intra_op_threads) if intra_op_threads is not None else _configured_intra_op_threads(),
    )
    opts.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
