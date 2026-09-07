from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import onnxruntime as ort

from app.parameters import (
    ORT_HIGH_CPU_THRESHOLD,
    ORT_HIGH_CPU_THREADS,
    ORT_INTER_OP_THREADS,
    ORT_MEDIUM_CPU_THRESHOLD,
    ORT_MEDIUM_CPU_THREADS,
)


_THREAD_ENV = "MANGA_ORT_INTRA_OP_THREADS"
_CPU_ARENA_ENV = "MANGA_ORT_CPU_MEM_ARENA"
_MEM_PATTERN_ENV = "MANGA_ORT_MEM_PATTERN"
_SERIALIZE_ENV = "MANGA_ORT_SERIALIZE_INFERENCE"
_PROVIDER_ENV = "MANGA_ORT_PROVIDER"
_REQUIRE_PROVIDER_ENV = "MANGA_ORT_REQUIRE_PROVIDER"
_OPENVINO_SCOPE_ENV = "MANGA_ORT_OPENVINO_SCOPE"
_OPENVINO_THREADS_ENV = "MANGA_ORT_OPENVINO_THREADS"
_OPENVINO_STREAMS_ENV = "MANGA_ORT_OPENVINO_STREAMS"
_OPENVINO_CACHE_ENV = "MANGA_ORT_OPENVINO_CACHE_DIR"
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
    if cpu >= ORT_HIGH_CPU_THRESHOLD:
        return ORT_HIGH_CPU_THREADS
    if cpu >= ORT_MEDIUM_CPU_THRESHOLD:
        return ORT_MEDIUM_CPU_THREADS
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


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
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


def _openvino_selected(model_path) -> bool:
    provider = os.environ.get(_PROVIDER_ENV, "cpu").strip().lower()
    if provider not in {"openvino", "ov"}:
        return False
    scope = os.environ.get(_OPENVINO_SCOPE_ENV, "all").strip().lower()
    if scope in {"", "all"}:
        return True
    name = Path(model_path).name.lower()
    if scope in {"detector", "detectors"}:
        return "bubble" in name or "text_segmenter" in name
    return False


def _openvino_provider_options() -> dict[str, str]:
    threads = _positive_env_int(_OPENVINO_THREADS_ENV, _configured_intra_op_threads())
    streams = _positive_env_int(_OPENVINO_STREAMS_ENV, 1)
    config: dict[str, dict[str, str]] = {
        "CPU": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32",
            "NUM_STREAMS": str(streams),
            "INFERENCE_NUM_THREADS": str(threads),
        }
    }
    cache_dir = os.environ.get(_OPENVINO_CACHE_ENV, "").strip()
    if cache_dir:
        config["CPU"]["CACHE_DIR"] = cache_dir
        config["CPU"]["CACHE_MODE"] = "OPTIMIZE_SPEED"
    return {
        "device_type": "CPU",
        "load_config": json.dumps(config, separators=(",", ":")),
    }


def _provider_stack(model_path):
    use_openvino = _openvino_selected(model_path)
    if not use_openvino:
        return ["CPUExecutionProvider"], False

    available = set(ort.get_available_providers())
    if "OpenVINOExecutionProvider" not in available:
        if _env_flag(_REQUIRE_PROVIDER_ENV, False):
            raise RuntimeError(
                "MANGA_ORT_PROVIDER=openvino requested, but OpenVINOExecutionProvider "
                f"is unavailable; providers={sorted(available)}"
            )
        return ["CPUExecutionProvider"], False

    return [
        ("OpenVINOExecutionProvider", _openvino_provider_options()),
        "CPUExecutionProvider",
    ], True


class _SerializedSession:
    """Thin proxy for sessions whose workspace must not overlap another run."""

    def __init__(self, session: ort.InferenceSession):
        self._session = session
        self._timing_local = threading.local()

    def run(self, *args, **kwargs):
        wait_started_at = time.perf_counter()
        with _ORT_INFERENCE_LOCK:
            lock_wait_ms = (time.perf_counter() - wait_started_at) * 1000.0
            run_started_at = time.perf_counter()
            try:
                return self._session.run(*args, **kwargs)
            finally:
                self._timing_local.value = {
                    "global_lock_wait_ms": lock_wait_ms,
                    "model_run_ms": (time.perf_counter() - run_started_at) * 1000.0,
                }

    def last_run_timing(self) -> dict[str, float]:
        timing = getattr(self._timing_local, "value", {})
        return {
            "global_lock_wait_ms": float(timing.get("global_lock_wait_ms", 0.0)),
            "model_run_ms": float(timing.get("model_run_ms", 0.0)),
        }

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
    """Create a CPU inference session with an opt-in OpenVINO turbo path.

    The default stays the low-memory CPUExecutionProvider behavior. Experiments
    can set MANGA_ORT_PROVIDER=openvino and scope it to detector models while
    keeping a CPU fallback. OpenVINO performs its own graph/kernel fusion, so ORT
    graph optimization is disabled for those sessions as recommended upstream.
    """

    providers, use_openvino = _provider_stack(model_path)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if use_openvino
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
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
        int(intra_op_threads)
        if intra_op_threads is not None
        else _configured_intra_op_threads(),
    )
    opts.inter_op_num_threads = ORT_INTER_OP_THREADS

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=providers,
    )
    _drop_model_file_cache_hint(model_path)
    should_serialize = (
        _env_flag(_SERIALIZE_ENV, False)
        if serialize_inference is None
        else bool(serialize_inference)
    )
    return _SerializedSession(session) if should_serialize else session
