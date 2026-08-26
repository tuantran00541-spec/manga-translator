from __future__ import annotations

import os

import onnxruntime as ort


_THREAD_ENV = "MANGA_ORT_INTRA_OP_THREADS"
_CPU_ARENA_ENV = "MANGA_ORT_CPU_MEM_ARENA"
_MEM_PATTERN_ENV = "MANGA_ORT_MEM_PATTERN"
_DEFAULT_HIGH_CPU_THREADS = 8


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 2)


def _default_intra_op_threads() -> int:
    cpu = _cpu_count()
    if cpu >= 8:
        return _DEFAULT_HIGH_CPU_THREADS
    if cpu >= 4:
        return 2
    return 1


def _configured_intra_op_threads() -> int:
    raw = os.environ.get(_THREAD_ENV, "").strip()
    if not raw:
        return _default_intra_op_threads()

    try:
        value = int(raw)
    except ValueError:
        return _default_intra_op_threads()

    return max(1, value)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def make_session(
    model_path,
    *,
    intra_op_threads: int | None = None,
    enable_cpu_mem_arena: bool | None = None,
    enable_mem_pattern: bool | None = None,
) -> ort.InferenceSession:
    """Create a CPU-only ONNX Runtime session with bounded allocator retention.

    Manga Translator commonly keeps several independent ONNX sessions alive at
    once (bubble detector, text segmenter and LaMa).  ORT's CPU arena and memory
    pattern cache can each retain large peak allocations per session, which is a
    poor trade-off on the supported low-memory CPU path.  They therefore default
    to disabled, while environment/argument overrides remain available for
    machines where throughput matters more than peak RSS.
    """

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = (
        _env_flag(_CPU_ARENA_ENV, False)
        if enable_cpu_mem_arena is None
        else bool(enable_cpu_mem_arena)
    )
    opts.enable_mem_pattern = (
        _env_flag(_MEM_PATTERN_ENV, False)
        if enable_mem_pattern is None
        else bool(enable_mem_pattern)
    )
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
