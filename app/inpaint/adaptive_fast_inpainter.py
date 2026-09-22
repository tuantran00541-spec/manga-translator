from __future__ import annotations

import ctypes
import gc
import time

import cv2
import numpy as np
import onnxruntime as ort

from app.config import LAMA_DYNAMIC_MODEL, LAMA_MODEL
from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
import app.inpaint.fast_lama_inpainter as fast_lama
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.inpaint.lama_runtime_policy import (
    LamaRuntimeProfile,
    select_lama_runtime_profile,
)
from app.logging_config import logger
from app.ort_utils import _drop_model_file_cache_hint, make_session
from app.parameters import FIXED_LAMA_CONCURRENT_INFERENCE, ORT_INTER_OP_THREADS


_DENSE_AUTHORITY_MIN_CONTRAST = fast_lama._env_float(
    "MANGA_DENSE_AUTHORITY_MIN_CONTRAST", 180.0, 64.0, 255.0
)


class AdaptiveFastInpainter(FastInpainter):
    """ FastInpainter with CPU LaMa runtime policy tuned to page concurrency. """

    def __init__(self):
        super().__init__()
        self._runtime_profile: LamaRuntimeProfile = select_lama_runtime_profile(1)
        self._loaded_runtime_signature: tuple[int, int, bool, bool] | None = None

    def _try_stroke_authority_fill(
        self,
        image: np.ndarray,
        box: BubbleBox,
        protected_regions: list[dict] | None,
    ) -> bool:
        """Use a validated smooth surface when dense stroke refinement is too wide.

        FastInpainter first attempts the normal stroke-only authority-ring route.
        Some outlined lettering legitimately occupies more than the conservative
        stroke-fraction cap, even though the surrounding bubble is an extremely
        clean smooth gradient. Sending that dense detector authority to LaMa can
        create a visible block/facet. For that narrow case, fit the same robust
        quadratic surface from *outside* the full authority and paint only pixels
        already owned by that authority.

        The full-authority fallback is additionally restricted to genuinely high-
        contrast lettering. Muted or colored SFX can have an equally smooth ring
        while still exposing the dense authority boundary when reconstructed
        wholesale; those cases stay on the established LaMa path.
        """
        if super()._try_stroke_authority_fill(image, box, protected_regions):
            return True

        if (
            not fast_lama._STROKE_REFINE_ENABLED
            or not self._bubble_candidate(box)
            or box.semantic_type != "speech_bubble"
            or box.source_role != "text_segmenter"
        ):
            return False

        h, w = image.shape[:2]
        pad = int(fast_lama._BUBBLE_FASTPATH_PAD)
        x1 = max(0, int(box.x1) - pad)
        y1 = max(0, int(box.y1) - pad)
        x2 = min(w, int(box.x2) + pad)
        y2 = min(h, int(box.y2) + pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return False

        crop_box = (x1, y1, x2, y2)
        crop = image[y1:y2, x1:x2]
        local_box = self._local_box(box, x1, y1)
        authority_mask = build_mask((y2 - y1, x2 - x1), [local_box], crop)
        authority_mask = self._subtract_protected_regions(
            authority_mask,
            crop_box,
            protected_regions,
        )
        authority = authority_mask > 127
        ys, xs = np.nonzero(authority)
        authority_pixels = int(xs.size)
        if authority_pixels < 64:
            return False

        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        tight_area = max(1, (bx2 - bx1) * (by2 - by1))
        occupancy = authority_pixels / float(tight_area)
        if occupancy < fast_lama._STROKE_REFINE_DENSE_FRACTION_MIN:
            return False

        ring = self._mask_ring(authority_mask, fast_lama._BUBBLE_FASTPATH_RING)
        if int(np.count_nonzero(ring)) < 96:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray_f = gray.astype(np.float32, copy=False)
        background = float(np.median(gray_f[ring]))
        inside = gray_f[authority]
        low = float(np.percentile(inside, 10.0))
        high = float(np.percentile(inside, 90.0))
        dominant_contrast = max(background - low, high - background)
        if dominant_contrast < _DENSE_AUTHORITY_MIN_CONTRAST:
            self._metric_add("dense_authority_contrast_rejects")
            return False

        painted = self._bubble_gradient_fill_from_authority_ring(
            crop,
            authority_mask,
            authority_mask,
        )
        if painted is None:
            return False

        target = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = np.where(
            authority[:, :, None],
            painted,
            target,
        )
        self._metric_add("dense_authority_gradient_regions")
        self._metric_add("dense_authority_gradient_pixels", authority_pixels)
        self._metric_add("bubble_fast_fill_regions")
        self._metric_add("bubble_fast_fill_gradient_regions")
        self._metric_add("bubble_fast_fill_pixels", authority_pixels)
        return True

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
        """ Select the session profile before page worker threads begin. """
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
