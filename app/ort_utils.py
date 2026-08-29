from __future__ import annotations

import os
import threading

import onnxruntime as ort


_THREAD_ENV = "MANGA_ORT_INTRA_OP_THREADS"
_CPU_ARENA_ENV = "MANGA_ORT_CPU_MEM_ARENA"
_MEM_PATTERN_ENV = "MANGA_ORT_MEM_PATTERN"
_SERIALIZE_ENV = "MANGA_ORT_SERIALIZE_INFERENCE"
_DEFAULT_HIGH_CPU_THREADS = 4
_ORT_INFERENCE_LOCK = threading.RLock()


def _cpu_count() -> int:
    """Return CPU capacity visible to the process, including cgroup quota."""
    host = max(1, os.cpu_count() or 2)
    try:
        raw = open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8").read().strip().split()
        if len(raw) >= 2 and raw[0] != "max":
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                host = min(host, max(1, (quota + period - 1) // period))
    except (OSError, ValueError):
        pass
    return host


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


def _drop_model_file_cache_hint(model_path) -> None:
    """Best-effort release of model file pages after ORT has parsed the model."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        with open(model_path, "rb") as model_file:
            os.posix_fadvise(model_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, TypeError, ValueError):
        pass


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


class _SerializedSession:
    """Thin proxy for sessions whose workspace must not overlap another run."""

    def __init__(self, session: ort.InferenceSession):
        self._session = session

    def run(self, *args, **kwargs):
        with _ORT_INFERENCE_LOCK:
            return self._session.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def make_session(
    model_path,
    *,
    intra_op_threads: int | None = None,
    enable_cpu_mem_arena: bool | None = None,
    enable_mem_pattern: bool | None = None,
    serialize_inference: bool | None = None,
):
    """Create a CPU-only ONNX Runtime session for the low-memory path.

    ORT's CPU arena and memory-pattern cache retain large peak allocations for
    Manga Translator's detector/segmenter/LaMa sessions, so both are disabled by
    default. Inference serialization is opt-in per session: detector sessions and
    the fixed 512px LaMa compatibility fallback can serialize their workspaces,
    while the preferred dynamic LaMa is allowed to overlap the bounded two-page
    schedule.  The conservative per-session thread cap prevents that overlap from
    oversubscribing common desktop CPUs.
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

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    _drop_model_file_cache_hint(model_path)
    should_serialize = (
        _env_flag(_SERIALIZE_ENV, False)
        if serialize_inference is None
        else bool(serialize_inference)
    )
    return _SerializedSession(session) if should_serialize else session
