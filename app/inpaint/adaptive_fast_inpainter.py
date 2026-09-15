from __future__ import annotations

import ctypes
import gc
import time

import onnxruntime as ort

from app.config import LAMA_DYNAMIC_MODEL, LAMA_MODEL
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.inpaint.lama_runtime_policy import (
    LamaRuntimeProfile,
    select_lama_runtime_profile,
)
from app.logging_config import logger
from app.ort_utils import _drop_model_file_cache_hint, make_session
from app.parameters import FIXED_LAMA_CONCURRENT_INFERENCE, ORT_INTER_OP_THREADS


class AdaptiveFastInpainter(FastInpainter):
    """FastInpainter with CPU LaMa runtime policy tuned to page concurrency.

    The image/mask/model path is unchanged. Only ONNX Runtime scheduling and
    allocator settings differ, so the candidate remains suitable for pixel-level
    A/B comparison against FastInpainter.
    """

    def __init__(self):
        super().__init__()
        self._runtime_profile: LamaRuntimeProfile = select_lama_runtime_profile(1)
        self._loaded_runtime_signature: tuple[int, int, bool, bool] | None = None

    def runtime_profile_status(self) -> dict:
        status = self._runtime_profile.as_dict()
        status["loaded_signature"] = (
            list(self._loaded_runtime_signature)
            if self._loaded_runtime_signature is not None
            else None
        )
        status["session_loaded"] = bool(self.session is not None)
        return status

    @staticmethod
    def _trim_process_heap() -> None:
        try:
            libc = ctypes.CDLL(None)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim(0)
        except Exception:
            pass

    def _release_dynamic_session_locked(self) -> None:
        old_session = self.session
        self.session = None
        self.image_input = None
        self.mask_input = None
        self.output_name = None
        self.lama_contract = None
        self.dynamic_lama = False
        self._session_run_count = 0
        self._loaded_runtime_signature = None
        if old_session is not None:
            del old_session
            gc.collect()
            self._trim_process_heap()

    def prepare_for_page_workers(self, page_workers: int) -> dict:
        """Select the session profile before page worker threads begin.

        A loaded dynamic session is rebuilt only if a session-affecting setting
        changes. Memory byte counters themselves are diagnostic and do not cause
        churn when the arena decision remains the same.
        """
        desired = select_lama_runtime_profile(page_workers)
        desired_signature = desired.session_signature()
        with self._session_lock:
            if (
                self.session is not None
                and self.dynamic_lama
                and self._loaded_runtime_signature is not None
                and self._loaded_runtime_signature != desired_signature
            ):
                logger.info(
                    "Reconfiguring dynamic LaMa runtime {} -> {}",
                    self._loaded_runtime_signature,
                    desired_signature,
                )
                self._release_dynamic_session_locked()
            self._runtime_profile = desired
        return desired.as_dict()

    @staticmethod
    def _session_options(profile: LamaRuntimeProfile) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = bool(profile.enable_cpu_mem_arena)
        opts.enable_mem_pattern = bool(profile.enable_mem_pattern)
        opts.intra_op_num_threads = max(1, int(profile.intra_op_threads))
        opts.inter_op_num_threads = max(1, int(ORT_INTER_OP_THREADS))
        if int(profile.dynamic_block_base) > 0:
            opts.add_session_config_entry(
                "session.dynamic_block_base",
                str(int(profile.dynamic_block_base)),
            )
        return opts

    def _ensure_session(self) -> None:
        if self.session is not None:
            return
        if not self._prefer_dynamic or not LAMA_DYNAMIC_MODEL.is_file():
            return super()._ensure_session()

        with self._session_lock:
            if self.session is not None:
                return

            profile = self._runtime_profile
            load_started_at = time.perf_counter()
            with self._session_state_lock:
                self._session_load_state = "loading"
                self._session_load_ms = None
                self._session_load_error = None

            logger.info(
                "Preparing adaptive LaMa {} profile={}",
                LAMA_DYNAMIC_MODEL,
                profile.as_dict(),
            )
            try:
                try:
                    session = ort.InferenceSession(
                        str(LAMA_DYNAMIC_MODEL),
                        sess_options=self._session_options(profile),
                        providers=["CPUExecutionProvider"],
                    )
                    _drop_model_file_cache_hint(LAMA_DYNAMIC_MODEL)
                    self._configure_loaded_session(
                        session,
                        LAMA_DYNAMIC_MODEL,
                        expected_dynamic=True,
                    )
                    self._loaded_runtime_signature = profile.session_signature()
                except Exception:
                    logger.exception(
                        "Failed adaptive dynamic LaMa {}; falling back to {}",
                        LAMA_DYNAMIC_MODEL,
                        LAMA_MODEL,
                    )
                    session = make_session(
                        LAMA_MODEL,
                        serialize_inference=not FIXED_LAMA_CONCURRENT_INFERENCE,
                    )
                    self._configure_loaded_session(
                        session,
                        LAMA_MODEL,
                        expected_dynamic=False,
                    )
                    self._loaded_runtime_signature = None
            except Exception as exc:
                load_ms = round((time.perf_counter() - load_started_at) * 1000.0, 1)
                with self._session_state_lock:
                    self._session_load_state = "failed"
                    self._session_load_ms = load_ms
                    self._session_load_error = type(exc).__name__
                raise

            load_ms = round((time.perf_counter() - load_started_at) * 1000.0, 1)
            with self._session_state_lock:
                self._session_load_state = "ready"
                self._session_load_ms = load_ms
                self._session_load_error = None
            logger.info(
                "Adaptive inpaint model {} ready in {:.1f} ms (dynamic={}, profile={})",
                self.lama_model_path,
                load_ms,
                self.dynamic_lama,
                self.runtime_profile_status(),
            )
