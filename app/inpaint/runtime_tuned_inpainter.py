from __future__ import annotations

from collections import OrderedDict
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from app.config import LAMA_DYNAMIC_MODEL
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.logging_config import logger
from app.model_contracts import decode_lama_output
from app.ort_utils import _configured_intra_op_threads
from app.parameters import ORT_INTER_OP_THREADS


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    return max(int(minimum), min(int(maximum), int(value)))


def _runtime_session_options(
    *,
    threads: int,
    arena: bool,
    mem_pattern: bool,
    dynamic_block_base: int,
) -> ort.SessionOptions:
    """Build an FP32 CPU session without changing LaMa weights or graph semantics."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = bool(arena)
    opts.enable_mem_pattern = bool(mem_pattern)
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = max(1, int(ORT_INTER_OP_THREADS))
    if int(dynamic_block_base) > 0:
        opts.add_session_config_entry(
            "session.dynamic_block_base", str(int(dynamic_block_base))
        )
    return opts


class _TensorEntry:
    def __init__(self, session, image_name: str, mask_name: str, output_name: str, h: int, w: int):
        self.image = np.empty((1, 3, h, w), dtype=np.float32)
        self.mask = np.empty((1, 1, h, w), dtype=np.float32)
        self.mask_bool = np.empty((h, w), dtype=np.bool_)
        self.output = np.empty((1, 3, h, w), dtype=np.float32)
        self.binding = session.io_binding()
        self.binding.bind_cpu_input(image_name, self.image)
        self.binding.bind_cpu_input(mask_name, self.mask)
        self.binding.bind_output(
            output_name,
            device_type="cpu",
            device_id=0,
            element_type=np.float32,
            shape=self.output.shape,
            buffer_ptr=self.output.ctypes.data,
        )


class LamaTensorPool:
    """Small LRU of exact-shape CPU tensors and I/O bindings for dynamic LaMa."""

    def __init__(self, capacity: int = 4):
        self.capacity = max(1, int(capacity))
        self._entries: OrderedDict[tuple[int, int], _TensorEntry] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def get(self, session, image_name: str, mask_name: str, output_name: str, h: int, w: int):
        key = (int(h), int(w))
        entry = self._entries.pop(key, None)
        hit = entry is not None
        if entry is None:
            entry = _TensorEntry(session, image_name, mask_name, output_name, key[0], key[1])
        self._entries[key] = entry
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return entry, hit

    @staticmethod
    def fill(entry: _TensorEntry, canvas: np.ndarray, mask_canvas: np.ndarray) -> None:
        # Base LaMa path converts BGR uint8 -> RGB float32 / 255. Fill the same
        # NCHW values directly into reusable buffers to avoid temporary RGB,
        # astype, transpose and contiguous allocations.
        np.divide(canvas[:, :, 2], 255.0, out=entry.image[0, 0], casting="unsafe")
        np.divide(canvas[:, :, 1], 255.0, out=entry.image[0, 1], casting="unsafe")
        np.divide(canvas[:, :, 0], 255.0, out=entry.image[0, 2], casting="unsafe")
        np.greater(mask_canvas, 127, out=entry.mask_bool)
        np.copyto(entry.mask[0, 0], entry.mask_bool, casting="unsafe")


class RuntimeTunedFastInpainter(FastInpainter):
    """Isolated benchmark candidate for lossless CPU runtime tuning.

    This class intentionally does not replace production session creation. It
    exists on the perf branch so arena/memory-pattern, thread scheduling and
    tensor reuse can be measured independently before any production port.
    """

    def __init__(self):
        super().__init__()
        self._runtime_arena = _env_bool("MANGA_LAMA_RUNTIME_ARENA", False)
        self._runtime_mem_pattern = _env_bool("MANGA_LAMA_RUNTIME_MEM_PATTERN", False)
        requested_threads = _env_int("MANGA_LAMA_RUNTIME_THREADS", 0, 0, 64)
        self._runtime_threads = requested_threads or _configured_intra_op_threads()
        self._runtime_dynamic_block_base = _env_int(
            "MANGA_LAMA_RUNTIME_DYNAMIC_BLOCK_BASE", 0, 0, 64
        )
        self._runtime_tensor_pool_enabled = _env_bool(
            "MANGA_LAMA_RUNTIME_TENSOR_POOL", False
        )
        pool_capacity = _env_int("MANGA_LAMA_RUNTIME_TENSOR_POOL_CAPACITY", 4, 1, 16)
        self._runtime_tensor_pool = LamaTensorPool(pool_capacity)
        self._runtime_session_active = False

    def runtime_config(self) -> dict[str, int | bool]:
        return {
            "arena": self._runtime_arena,
            "mem_pattern": self._runtime_mem_pattern,
            "threads": int(self._runtime_threads),
            "dynamic_block_base": int(self._runtime_dynamic_block_base),
            "tensor_pool": self._runtime_tensor_pool_enabled,
            "tensor_pool_capacity": self._runtime_tensor_pool.capacity,
            "runtime_session_active": self._runtime_session_active,
        }

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        super()._begin_metrics(boxes=boxes)
        self._metrics_local.value.update(
            {
                "tensor_pool_hits": 0,
                "tensor_pool_misses": 0,
            }
        )

    def _ensure_session(self) -> None:
        if self.session is not None:
            return
        # The experiment targets lama-manga-dynamic.onnx only. If the dynamic
        # model is unavailable, preserve the validated production fallback.
        if not self._prefer_dynamic or not LAMA_DYNAMIC_MODEL.is_file():
            return super()._ensure_session()

        with self._session_lock:
            if self.session is not None:
                return
            started = time.perf_counter()
            with self._session_state_lock:
                self._session_load_state = "loading"
                self._session_load_ms = None
                self._session_load_error = None
            try:
                opts = _runtime_session_options(
                    threads=self._runtime_threads,
                    arena=self._runtime_arena,
                    mem_pattern=self._runtime_mem_pattern,
                    dynamic_block_base=self._runtime_dynamic_block_base,
                )
                session = ort.InferenceSession(
                    str(LAMA_DYNAMIC_MODEL),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._configure_loaded_session(
                    session,
                    LAMA_DYNAMIC_MODEL,
                    expected_dynamic=True,
                )
                self._runtime_session_active = True
                self._runtime_tensor_pool.clear()
            except Exception as exc:
                load_ms = round((time.perf_counter() - started) * 1000.0, 1)
                with self._session_state_lock:
                    self._session_load_state = "failed"
                    self._session_load_ms = load_ms
                    self._session_load_error = type(exc).__name__
                raise

            load_ms = round((time.perf_counter() - started) * 1000.0, 1)
            with self._session_state_lock:
                self._session_load_state = "ready"
                self._session_load_ms = load_ms
                self._session_load_error = None
            logger.info(
                "Runtime-tuned LaMa ready in {:.1f} ms config={}",
                load_ms,
                self.runtime_config(),
            )

    def _run_lama(self, canvas: np.ndarray, mask_canvas: np.ndarray) -> np.ndarray:
        self._ensure_session()
        if not self.dynamic_lama or not self._runtime_tensor_pool_enabled:
            return super()._run_lama(canvas, mask_canvas)

        self._metric_add("lama_model_runs")
        h, w = canvas.shape[:2]
        entry, hit = self._runtime_tensor_pool.get(
            self.session,
            self.image_input,
            self.mask_input,
            self.output_name,
            h,
            w,
        )
        self._metric_add("tensor_pool_hits" if hit else "tensor_pool_misses")
        self._runtime_tensor_pool.fill(entry, canvas, mask_canvas)

        model_started_at = time.perf_counter()
        self.session.run_with_iobinding(entry.binding)
        self._metric_add(
            "lama_model_ms",
            round((time.perf_counter() - model_started_at) * 1000.0),
        )
        painted_rgb = decode_lama_output(entry.output, self.lama_contract)
        return cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)
