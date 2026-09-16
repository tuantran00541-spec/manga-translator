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
    """FastInpainter with CPU LaMa runtime policy tuned to page concurrency.

    The image/mask/model path is unchanged. Only ONNX Runtime scheduling and
    allocator settings differ, so the candidate remains suitable for pixel-level
    A/B comparison against FastInpainter.
    """

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
        contrast lettering. Real-page auditing showed that muted/colored SFX can
        have an equally smooth ring but still expose the dense authority boundary
        when reconstructed wholesale; those cases must stay on LaMa.
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

        # A smooth ring alone is not enough to justify repainting the whole dense
        # authority: muted/colored SFX can pass the surface validator but leave a
        # visible authority-shaped patch. Require the detector-owned region to
        # contain a strong luminance separation from its immediate background.
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
        opts.intra_op_num_threads = max(1, int(profile.intra_op_threads))
        opts.inter_op_num_threads = max(1, int(ORT_INTER_OP_THREADS))
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_cpu_mem_arena = bool(profile.enable_cpu_mem_arena)
        opts.enable_mem_pattern = bool(profile.enable_mem_pattern)
        return opts

    def _ensure_session(self) -> None:
        if self.session is not None:
            return
        model_path = self.model_path
        if not model_path.exists():
            raise RuntimeError(f"LaMa model not found: {model_path}")

        with self._session_lock:
            if self.session is not None:
                return
            profile = self._runtime_profile
            logger.info(
                "Preparing adaptive LaMa {} profile={}",
                model_path,
                profile.as_dict(),
            )
            started = time.perf_counter()
            self.session = make_session(
                model_path,
                providers=["CPUExecutionProvider"],
                session_options=self._session_options(profile),
            )
            self._loaded_runtime_signature = profile.session_signature()
            self._session_run_count = 0
            self._inspect_model_contract()
            logger.info(
                "Adaptive inpaint model {} ready in {:.1f} ms (dynamic={}, profile={})",
                model_path,
                (time.perf_counter() - started) * 1000.0,
                self.dynamic_lama,
                self.runtime_profile_status(),
            )

    def release_session(self) -> None:
        with self._session_lock:
            self._release_dynamic_session_locked()

    def close(self) -> None:
        self.release_session()

    def _run_session(self, feeds: dict) -> list:
        self._ensure_session()
        assert self.session is not None
        if not self.dynamic_lama:
            return super()._run_session(feeds)
        return self.session.run([self.output_name], feeds)

    def _fixed_model_concurrency_limit(self) -> int:
        return max(1, int(FIXED_LAMA_CONCURRENT_INFERENCE))

    def _dynamic_model_path(self):
        return LAMA_DYNAMIC_MODEL

    def _fixed_model_path(self):
        return LAMA_MODEL

    def _drop_model_cache_hint(self) -> None:
        _drop_model_file_cache_hint(self.model_path)
