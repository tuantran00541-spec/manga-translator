from __future__ import annotations

import os
import threading

import onnxruntime as ort


_THREAD_ENV = "MANGA_ORT_INTRA_OP_THREADS"
_CPU_ARENA_ENV = "MANGA_ORT_CPU_MEM_ARENA"
_MEM_PATTERN_ENV = "MANGA_ORT_MEM_PATTERN"
_SERIALIZE_ENV = "MANGA_ORT_SERIALIZE_INFERENCE"
_DEFAULT_HIGH_CPU_THREADS = 8
_ORT_INFERENCE_LOCK = threading.RLock()


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


class _SerializedSession:
    """Thin proxy that prevents high-memory ORT inference from overlapping.

    Page workers may still decode images, build masks and write outputs in
    parallel. Only ``InferenceSession.run`` is serialized, which avoids two
    detector/LaMa workspaces peaking at the same time under the supported
    workers=2 CPU path. All other session APIs are delegated transparently.
    """

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
    """Create a CPU-only ONNX Runtime session for the low-memory production path.

    Manga Translator keeps several ONNX sessions alive at once (bubble detector,
    text segmenter and LaMa). ORT's CPU arena/memory pattern cache can retain
    large peak allocations per session, while concurrent session.run calls can
    make those peaks overlap. The low-memory defaults therefore disable arena
    retention and mem-pattern caching and serialize only the high-memory ORT run
    calls. Environment/argument overrides remain available for benchmarking on
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

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    should_serialize = (
        _env_flag(_SERIALIZE_ENV, True)
        if serialize_inference is None
        else bool(serialize_inference)
    )
    return _SerializedSession(session) if should_serialize else session
