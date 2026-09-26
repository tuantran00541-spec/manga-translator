import gc
import os
import threading
import time

import numpy as np
import cv2
from app.config import LAMA_MODEL, LAMA_DYNAMIC_MODEL
from app.detector.bubble_detector import BubbleBox
from app.inpaint.clustering import cluster_boxes, compute_crop_region, compute_manual_crop_region, split_oversized_cluster_area
from app.detector.mask_builder import build_mask
from app.logging_config import logger
from app.model_contracts import decode_lama_output, validate_lama_session
from app.ort_utils import make_session
import app.parameters as _params
from app.parameters import (
    DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM,
    DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS,
    FIXED_LAMA_RECYCLE_MEMORY_LIMIT_BYTES,
    FIXED_LAMA_CONCURRENT_INFERENCE,
    FIXED_LAMA_SESSION_MAX_RUNS,
    FIXED_LAMA_TILE_ASPECT,
    INPAINT_NATIVE_TILE_EDGE_DENSITY_MIN,
    INPAINT_NATIVE_TILE_ENABLED,
    INPAINT_NATIVE_TILE_MASK_AREA_MIN,
    INPAINT_SIZE,
    MANUAL_DILATION_SCALE,
    MANUAL_FEATHER_RADIUS,
    MANUAL_MAX_DILATION,
    MANUAL_MIN_DILATION,
    MANUAL_TILE_OVERLAP,
    SMART_FILL_BLACK_EDGE_DENSITY_MAX,
    SMART_FILL_BLACK_LEVEL,
    SMART_FILL_BLACK_RATIO_MIN,
    SMART_FILL_BLACK_STD_MAX,
    SMART_FILL_CANNY_HIGH,
    SMART_FILL_CANNY_LOW,
    SMART_FILL_CHROMA_STD_MAX,
    SMART_FILL_CLEAN_RING_MARGIN,
    SMART_FILL_CONTEXT_MARGIN_FACTOR,
    SMART_FILL_EDGE_DENSITY_MAX,
    SMART_FILL_FULL_STD_MAX,
    SMART_FILL_MIDTONE_MAX,
    SMART_FILL_MIDTONE_MIN,
    SMART_FILL_MIDTONE_STD_MAX,
    SMART_FILL_RING_PIXELS_MIN,
    SMART_FILL_SURFACE_MIN_RANGE,
    SMART_FILL_SURFACE_RESIDUAL_MAX,
    SMART_FILL_SURFACE_SEED_TOL,
    SMART_FILL_WHITE_LEVEL,
    SMART_FILL_WHITE_RATIO_MIN,
    SMART_FILL_WHITE_STD_MAX,
    USE_DYNAMIC_LAMA,
)

_FIXED_LAMA_RECYCLE_ENV = "MANGA_FIXED_LAMA_SESSION_RECYCLE"


def _optional_env_flag(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _tight_cgroup_memory_limit() -> bool:
    if os.name != "posix":
        return False

    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            raw = open(path, "r", encoding="utf-8").read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if 0 < limit <= FIXED_LAMA_RECYCLE_MEMORY_LIMIT_BYTES:
            return True
    return False


def _should_recycle_fixed_session() -> bool:
    override = _optional_env_flag(_FIXED_LAMA_RECYCLE_ENV)
    if override is not None:
        return override
    return _tight_cgroup_memory_limit()


class Inpainter:
    def __init__(self):
        self._prefer_dynamic = USE_DYNAMIC_LAMA and LAMA_DYNAMIC_MODEL.is_file()
        self._session_lock = threading.RLock()
        self._session_state_lock = threading.Lock()
        self._metrics_local = threading.local()
        self._session_run_count = 0
        self._session_load_state = "idle"
        self._session_load_ms = None
        self._session_load_error = None
        self.session = None
        self.image_input = None
        self.mask_input = None
        self.dynamic_lama = False
        self.lama_contract = None
        self.output_name = None
        self._serialize_fixed_inference = not FIXED_LAMA_CONCURRENT_INFERENCE
        self.lama_model_path = LAMA_DYNAMIC_MODEL if self._prefer_dynamic else LAMA_MODEL
        self._recycle_fixed_session = False

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        self._metrics_local.value = {
            "boxes": max(0, int(boxes)),
            "clusters": 0,
            "skipped_clusters": 0,
            "split_clusters": 0,
            "smart_fill_regions": 0,
            "lama_regions": 0,
            "lama_model_runs": 0,
            "lama_model_ms": 0,
            "lama_native_single_regions": 0,
            "lama_tiled_regions": 0,
            "session_lock_wait_ms": 0,
            "ort_global_lock_wait_ms": 0,
            "mask_components": 0,
        }

    def _metric_add(self, name: str, amount: int = 1) -> None:
        metrics = getattr(self._metrics_local, "value", None)
        if metrics is None:
            self._begin_metrics()
            metrics = self._metrics_local.value
        metrics[name] = int(metrics.get(name, 0)) + int(amount)

    def last_metrics(self) -> dict[str, int]:
        metrics = getattr(self._metrics_local, "value", {})
        return {str(name): int(value) for name, value in metrics.items()}

    @property
    def session_loaded(self) -> bool:
        return self.session is not None

    @property
    def serialized_inference(self) -> bool:
        if not self.session_loaded:
            return bool(
                not self._prefer_dynamic and self._serialize_fixed_inference
            )
        return bool(not self.dynamic_lama and self._serialize_fixed_inference)

    def session_load_status(self) -> dict:
        with self._session_state_lock:
            return {
                "state": self._session_load_state,
                "load_ms": self._session_load_ms,
                "failed": self._session_load_error is not None,
            }

    def preload(self) -> None:
        self._ensure_session()

    def _configure_loaded_session(
        self,
        session,
        model_path,
        *,
        expected_dynamic: bool,
    ) -> None:
        contract = validate_lama_session(
            session,
            dynamic=bool(expected_dynamic),
            fixed_size=INPAINT_SIZE,
        )
        self.session = session
        self.lama_contract = contract
        self.image_input = contract.image_input_name
        self.mask_input = contract.mask_input_name
        self.output_name = contract.output_name
        self.dynamic_lama = contract.dynamic
        self._serialize_fixed_inference = bool(
            not contract.dynamic and type(session).__name__ == "_SerializedSession"
        )
        self.lama_model_path = model_path
        self._session_run_count = 0
        self._recycle_fixed_session = (
            self._serialize_fixed_inference and _should_recycle_fixed_session()
        )

    def _ensure_session(self) -> None:
        if self.session is not None:
            return
        with self._session_lock:
            if self.session is not None:
                return

            prefer_dynamic = self._prefer_dynamic
            model_path = LAMA_DYNAMIC_MODEL if prefer_dynamic else LAMA_MODEL
            load_started_at = time.perf_counter()
            with self._session_state_lock:
                self._session_load_state = "loading"
                self._session_load_ms = None
                self._session_load_error = None
            logger.info("Preparing inpaint model {}", model_path)
            try:
                try:
                    session = make_session(
                        model_path,
                        serialize_inference=(
                            not prefer_dynamic
                            and not FIXED_LAMA_CONCURRENT_INFERENCE
                        ),
                    )
                except Exception:
                    if not prefer_dynamic:
                        raise
                    logger.exception(
                        "Failed to load dynamic LaMa model {}; falling back to {}",
                        LAMA_DYNAMIC_MODEL,
                        LAMA_MODEL,
                    )
                    model_path = LAMA_MODEL
                    session = make_session(
                        model_path,
                        serialize_inference=not FIXED_LAMA_CONCURRENT_INFERENCE,
                    )

                self._configure_loaded_session(
                    session,
                    model_path,
                    expected_dynamic=(model_path == LAMA_DYNAMIC_MODEL),
                )
            except Exception as exc:
                load_ms = round(
                    (time.perf_counter() - load_started_at) * 1000.0,
                    1,
                )
                with self._session_state_lock:
                    self._session_load_state = "failed"
                    self._session_load_ms = load_ms
                    self._session_load_error = type(exc).__name__
                raise

            load_ms = round(
                (time.perf_counter() - load_started_at) * 1000.0,
                1,
            )
            with self._session_state_lock:
                self._session_load_state = "ready"
                self._session_load_ms = load_ms
                self._session_load_error = None
            logger.info(
                "Inpaint model {} ready in {:.1f} ms (dynamic={})",
                self.lama_model_path,
                load_ms,
                self.dynamic_lama,
            )

    def _recycle_fixed_session_if_needed(self) -> None:
        if (
            self.session is None
            or self.dynamic_lama
            or not self._recycle_fixed_session
            or self._session_run_count < FIXED_LAMA_SESSION_MAX_RUNS
        ):
            return

        old_session = self.session
        self.session = None
        del old_session
        gc.collect()
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim(0)
        except Exception:
            pass

        session = make_session(self.lama_model_path, serialize_inference=True)
        self._configure_loaded_session(
            session, self.lama_model_path, expected_dynamic=False
        )

    @staticmethod
    def _subtract_protected_regions(
        local_mask: np.ndarray,
        crop_box: tuple[int, int, int, int],
        protected_regions: list[dict] | None,
    ) -> np.ndarray:
        if local_mask is None or not protected_regions:
            return local_mask
        cx1, cy1, cx2, cy2 = (int(value) for value in crop_box)
        clipped = local_mask.copy()
        for region in protected_regions:
            if not isinstance(region, dict):
                continue
            try:
                rx1, rx2 = sorted((int(region.get("x1", 0)), int(region.get("x2", 0))))
                ry1, ry2 = sorted((int(region.get("y1", 0)), int(region.get("y2", 0))))
            except (TypeError, ValueError):
                continue
            ix1, iy1 = max(cx1, rx1), max(cy1, ry1)
            ix2, iy2 = min(cx2, rx2), min(cy2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            clipped[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = 0
        return clipped

    def inpaint(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
        *,
        protected_regions: list[dict] | None = None,
    ) -> np.ndarray:
        self._begin_metrics(boxes=len(boxes))
        if not boxes:
            return image.copy()

        result = image.copy()
        h, w = image.shape[:2]
        raw_clusters = cluster_boxes(boxes)
        clusters: list[list[BubbleBox]] = []
        for cluster in raw_clusters:
            parts = split_oversized_cluster_area(cluster, w, h)
            if len(parts) > 1:
                self._metric_add("split_clusters", len(parts) - 1)
            clusters.extend(parts)
        self._metrics_local.value["clusters"] = len(clusters)

        for cluster in clusters:
            x1 = min(b.x1 for b in cluster)
            y1 = min(b.y1 for b in cluster)
            x2 = max(b.x2 for b in cluster)
            y2 = max(b.y2 for b in cluster)

            crop_box = compute_crop_region(x1, y1, x2, y2, w, h)

            cx1, cy1, cx2, cy2 = crop_box
            local_boxes = []
            for b in cluster:
                local_box = BubbleBox(
                    b.x1 - cx1,
                    b.y1 - cy1,
                    b.x2 - cx1,
                    b.y2 - cy1,
                    b.confidence,
                    b.mask,
                    source_model=b.source_model,
                    class_id=b.class_id,
                    class_name=b.class_name,
                    semantic_type=b.semantic_type,
                    mask_source=b.mask_source,
                    safe_to_inpaint=bool(b.safe_to_inpaint),
                    ocr_eligible=bool(b.ocr_eligible),
                    needs_review=bool(b.needs_review),
                    source_role=b.source_role,
                    deferred_reason=b.deferred_reason,
                )
                if bool(getattr(b, "allow_rectangle_fallback", False)):
                    local_box.allow_rectangle_fallback = True
                local_boxes.append(local_box)
            crop_img = image[cy1:cy2, cx1:cx2]
            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)
            local_mask = self._subtract_protected_regions(
                local_mask,
                crop_box,
                protected_regions,
            )

            result = self._smart_paint_region(result, local_mask, crop_box)

        return result

    def inpaint_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        force_lama: bool = False,
    ) -> np.ndarray:
        self._begin_metrics()
        if mask is None or not np.any(mask > 127):
            return image.copy()

        binary_mask = (mask > 127).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        self._metrics_local.value["mask_components"] = max(0, int(num_labels) - 1)

        result = image.copy()
        h, w = image.shape[:2]

        for label in range(1, num_labels):
            x, y, bbox_w, bbox_h, area = (int(v) for v in stats[label])
            if area <= 0 or bbox_w <= 0 or bbox_h <= 0:
                continue

            scale = max(1, min(bbox_w, bbox_h))
            kernel_size = int(
                np.clip(
                    round(scale * MANUAL_DILATION_SCALE) * 2 + 1,
                    MANUAL_MIN_DILATION,
                    MANUAL_MAX_DILATION,
                )
            )
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            radius = kernel_size // 2
            rx1 = max(0, x - radius)
            ry1 = max(0, y - radius)
            rx2 = min(w, x + bbox_w + radius)
            ry2 = min(h, y + bbox_h + radius)
            component_roi = (labels[ry1:ry2, rx1:rx2] == label).astype(np.uint8) * 255
            if not np.any(component_roi > 127):
                continue
            dilated_roi = cv2.dilate(component_roi, kernel, iterations=1)

            dys, dxs = np.where(dilated_roi > 127)
            if len(dys) == 0:
                continue
            gx1 = rx1 + int(dxs.min())
            gy1 = ry1 + int(dys.min())
            gx2 = rx1 + int(dxs.max())
            gy2 = ry1 + int(dys.max())
            crop_box = compute_manual_crop_region(gx1, gy1, gx2, gy2, w, h)
            cx1, cy1, cx2, cy2 = crop_box

            local_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
            ix1 = max(cx1, rx1)
            iy1 = max(cy1, ry1)
            ix2 = min(cx2, rx2)
            iy2 = min(cy2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            local_mask[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = (
                dilated_roi[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1]
            )

            result = self._smart_paint_region(
                result,
                local_mask,
                crop_box,
                feather=True,
                force_lama=force_lama,
            )

        return result

    @staticmethod
    def _smart_fill_color(crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray | None:
        mask_bool = local_mask > 127
        if not np.any(mask_bool):
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        non_mask = ~mask_bool
        if not np.any(non_mask):
            return None

        margin = max(1, int(SMART_FILL_CLEAN_RING_MARGIN))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        ring = (cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0) & non_mask
        if int(np.count_nonzero(ring)) < SMART_FILL_RING_PIXELS_MIN:
            return None

        context_margin = margin * SMART_FILL_CONTEXT_MARGIN_FACTOR
        context_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_margin * 2 + 1, context_margin * 2 + 1),
        )
        context = (
            cv2.dilate(mask_bool.astype(np.uint8), context_kernel) > 0
        ) & non_mask

        ring_gray = gray[ring]
        context_gray = gray[context]
        ring_pixels = crop[ring]
        context_std = float(context_gray.std())

        chroma_safe = True
        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
            context_ab = lab[context, 1:3].astype(np.float32, copy=False)
            ring_chroma_std = float(np.max(ring_ab.std(axis=0))) if ring_ab.size else 0.0
            context_chroma_std = float(np.max(context_ab.std(axis=0))) if context_ab.size else 0.0
            chroma_safe = bool(
                ring_chroma_std <= SMART_FILL_CHROMA_STD_MAX
                and context_chroma_std <= SMART_FILL_CHROMA_STD_MAX
            )

        edges = cv2.Canny(
            gray,
            SMART_FILL_CANNY_LOW,
            SMART_FILL_CANNY_HIGH,
            L2gradient=True,
        ) > 0
        context_edge_density = float(edges[context].mean())

        white_ratio = float((ring_gray > SMART_FILL_WHITE_LEVEL).mean())
        black_ratio = float((ring_gray < SMART_FILL_BLACK_LEVEL).mean())
        ring_std = float(ring_gray.std())
        median_gray = float(np.median(ring_gray))

        if (
            chroma_safe
            and white_ratio >= SMART_FILL_WHITE_RATIO_MIN
            and ring_std <= SMART_FILL_WHITE_STD_MAX
            and context_std <= SMART_FILL_FULL_STD_MAX
            and context_edge_density <= SMART_FILL_EDGE_DENSITY_MAX
        ):
            white_pixels = ring_pixels[ring_gray > SMART_FILL_WHITE_LEVEL]
            if len(white_pixels):
                return np.median(white_pixels, axis=0).astype(np.uint8)

        if (
            chroma_safe
            and black_ratio >= SMART_FILL_BLACK_RATIO_MIN
            and ring_std <= SMART_FILL_BLACK_STD_MAX
            and context_std <= SMART_FILL_BLACK_STD_MAX
            and context_edge_density <= SMART_FILL_BLACK_EDGE_DENSITY_MAX
        ):
            black_pixels = ring_pixels[ring_gray < SMART_FILL_BLACK_LEVEL]
            if len(black_pixels):
                return np.median(black_pixels, axis=0).astype(np.uint8)

        if (
            chroma_safe
            and SMART_FILL_MIDTONE_MIN <= median_gray <= SMART_FILL_MIDTONE_MAX
            and ring_std <= SMART_FILL_MIDTONE_STD_MAX
            and context_std <= SMART_FILL_MIDTONE_STD_MAX
            and context_edge_density <= SMART_FILL_BLACK_EDGE_DENSITY_MAX
        ):
            return np.median(ring_pixels, axis=0).astype(np.uint8)

        return None

    @staticmethod
    def _quadratic_design(
        xs: np.ndarray,
        ys: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        x = (xs.astype(np.float32) + 0.5) / max(1.0, float(width))
        y = (ys.astype(np.float32) + 0.5) / max(1.0, float(height))
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=1)

    @classmethod
    def _smart_fill_surface(
        cls,
        crop: np.ndarray,
        local_mask: np.ndarray,
        fill_color: np.ndarray,
    ) -> np.ndarray | None:
        mask_bool = local_mask > 127
        margin = max(1, int(SMART_FILL_CLEAN_RING_MARGIN))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        ring = (cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0) & ~mask_bool
        ys, xs = np.nonzero(ring)
        if xs.size < max(SMART_FILL_RING_PIXELS_MIN, 24) or crop.ndim != 3:
            return None
        values = crop[ring].astype(np.float32)
        height, width = crop.shape[:2]
        design = cls._quadratic_design(xs, ys, width, height)
        inliers = np.max(np.abs(values - fill_color.astype(np.float32)), axis=1) <= SMART_FILL_SURFACE_SEED_TOL
        coef = None
        for _ in range(3):
            if int(np.count_nonzero(inliers)) < max(24, design.shape[1] * 4):
                return None
            coef, *_ = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)
            residual = np.max(np.abs(design @ coef - values), axis=1)
            refined = residual <= SMART_FILL_SURFACE_RESIDUAL_MAX * 2.0
            if np.array_equal(refined, inliers):
                break
            inliers = refined
        fit = design[inliers] @ coef
        rms = float(np.sqrt(np.mean((fit - values[inliers]) ** 2)))
        if rms > SMART_FILL_SURFACE_RESIDUAL_MAX:
            return None
        my, mx = np.nonzero(mask_bool)
        surface = cls._quadratic_design(mx, my, width, height) @ coef
        low = values[inliers].min(axis=0) - 1.0
        high = values[inliers].max(axis=0) + 1.0
        surface = np.clip(surface, low, high)
        if float(np.max(np.ptp(surface, axis=0))) < SMART_FILL_SURFACE_MIN_RANGE:
            return None
        return np.clip(np.rint(surface), 0, 255).astype(np.uint8)

    def _smart_paint_region(
        self,
        image: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple,
        feather: bool = False,
        force_lama: bool = False,
    ) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop = image[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 4 or crop_w < 4:
            return image

        mask_bool = local_mask > 127
        if not np.any(mask_bool):
            return image

        fill_color = None if force_lama else self._smart_fill_color(crop, local_mask)
        if fill_color is not None:
            self._metric_add("smart_fill_regions")
            filled = crop.copy()
            surface = self._smart_fill_surface(crop, local_mask, fill_color)
            if surface is not None:
                self._metric_add("smart_fill_gradient_regions")
                filled[mask_bool] = surface
            else:
                filled[mask_bool] = fill_color
            image[cy1:cy2, cx1:cx2] = filled
            return image

        self._metric_add("lama_regions")
        return self._lama_fill(image, crop, local_mask, crop_box, feather=feather)

    def _lama_fill(self, image: np.ndarray, crop: np.ndarray, local_mask: np.ndarray, crop_box: tuple, feather: bool = False) -> np.ndarray:
        self._ensure_session()
        cx1, cy1, cx2, cy2 = crop_box
        crop_h, crop_w = crop.shape[:2]

        max_dim = max(crop_h, crop_w)
        min_dim = max(1, min(crop_h, crop_w))
        aspect = max_dim / min_dim
        long_crop = (
            max_dim > INPAINT_SIZE
            and aspect >= FIXED_LAMA_TILE_ASPECT
        )
        texture_tiling = False
        if (
            INPAINT_NATIVE_TILE_ENABLED
            and max_dim > DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM
            and int(np.count_nonzero(local_mask > 127)) >= INPAINT_NATIVE_TILE_MASK_AREA_MIN
        ):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            edges = cv2.Canny(gray, SMART_FILL_CANNY_LOW, SMART_FILL_CANNY_HIGH)
            context = local_mask <= 127
            edge_density = float(np.mean(edges[context] > 0)) if np.any(context) else 0.0
            texture_tiling = edge_density >= INPAINT_NATIVE_TILE_EDGE_DENSITY_MIN

        crop_pixels = int(crop_h * crop_w)
        dynamic_native_ok = bool(
            self.dynamic_lama
            and not feather
            and max_dim <= DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM
            and crop_pixels <= DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS
        )
        if dynamic_native_ok:
            self._metric_add("lama_native_single_regions")
            painted = self._lama_fill_single(crop, local_mask)
        elif long_crop or texture_tiling or (feather and max_dim > INPAINT_SIZE):
            self._metric_add("lama_tiled_regions")
            painted = self._lama_fill_tiled(crop, local_mask)
        else:
            painted = self._lama_fill_single(crop, local_mask)

        original_crop = image[cy1:cy2, cx1:cx2]
        if _params.INPAINT_GRAIN_RESTORE:
            painted = self._restore_grain(painted, crop, local_mask)
        if feather:
            core = local_mask > 127
            alpha = core.astype(np.float32)
            if MANUAL_FEATHER_RADIUS > 0:
                k = MANUAL_FEATHER_RADIUS * 2 + 1
                feathered = cv2.GaussianBlur(alpha, (k, k), 0)
                alpha = np.where(core, 1.0, feathered)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            blended = painted.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1.0 - alpha)
            image[cy1:cy2, cx1:cx2] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        else:
            mask_3d = (local_mask > 127)[:, :, None]
            image[cy1:cy2, cx1:cx2] = np.where(mask_3d, painted, original_crop)
        return image

    @staticmethod
    def _restore_grain(painted: np.ndarray, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        """Add back fine grain LaMa smooths away, sampled from the ring around the hole.

        Only where the surroundings have grain (high-pass std above a floor); flat
        paper and bubbles are left alone. Values come from the ring itself, so the
        grain keeps its strength and tone, and the result is deterministic.
        """
        hole = local_mask > 127
        count = int(np.count_nonzero(hole))
        if count < 16 or painted.shape[:2] != crop.shape[:2]:
            return painted
        ring_px = int(_params.INPAINT_GRAIN_RING_PX)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
        ring = (cv2.dilate(hole.astype(np.uint8), kernel) > 0) & ~hole
        if int(np.count_nonzero(ring)) < 64:
            return painted
        source = crop.astype(np.float32)
        high = source - cv2.GaussianBlur(source, (0, 0), 1.2)
        ring_high = high[ring]
        ring_std = float(ring_high.std())
        if ring_std < float(_params.INPAINT_GRAIN_MIN_STD):
            return painted
        filled = painted.astype(np.float32)
        own_std = float((filled - cv2.GaussianBlur(filled, (0, 0), 1.2))[hole].std())
        gain = float(np.sqrt(max(0.0, ring_std ** 2 - own_std ** 2)) / ring_std)
        if gain <= 0.05:
            return painted
        rng = np.random.default_rng(count * 7919 + int(ring_high.size))
        grain = ring_high[rng.integers(0, ring_high.shape[0], size=count)]
        filled[hole] += gain * grain
        return np.clip(np.rint(filled), 0, 255).astype(np.uint8)

    @staticmethod
    def _resize_mask_preserve_support(
        mask: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        width, height = max(1, int(width)), max(1, int(height))
        source = (mask > 127).astype(np.uint8) * 255
        src_h, src_w = source.shape[:2]
        if (src_w, src_h) == (width, height):
            return source
        if width < src_w or height < src_h:
            coverage = cv2.resize(
                source.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            return (coverage > 0.0).astype(np.uint8) * 255
        return cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)

    def _lama_fill_single(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        if self.dynamic_lama:
            return self._lama_fill_single_dynamic(crop, local_mask)
        return self._lama_fill_single_fixed(crop, local_mask)

    def _lama_fill_single_dynamic(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        crop_h, crop_w = crop.shape[:2]

        scale = min(
            1.0,
            DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM / max(crop_h, crop_w),
        )
        new_h = max(1, int(round(crop_h * scale)))
        new_w = max(1, int(round(crop_w * scale)))

        if new_h != crop_h or new_w != crop_w:
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_resized = self._resize_mask_preserve_support(local_mask, new_w, new_h)
        else:
            crop_resized = crop
            mask_resized = local_mask

        canvas_h = max(8, ((new_h + 7) // 8) * 8)
        canvas_w = max(8, ((new_w + 7) // 8) * 8)
        pad_y = (canvas_h - new_h) // 2
        pad_x = (canvas_w - new_w) // 2
        pad_bottom = canvas_h - new_h - pad_y
        pad_right = canvas_w - new_w - pad_x

        canvas = cv2.copyMakeBorder(
            crop_resized, pad_y, pad_bottom, pad_x, pad_right, cv2.BORDER_REPLICATE
        )
        # The padding repeats the edge pixels, so it must repeat the edge mask
        # too: text cut by the crop edge (a slice boundary, a tile edge) would
        # otherwise be fed to LaMa as known context and painted back.
        mask_canvas = cv2.copyMakeBorder(
            mask_resized, pad_y, pad_bottom, pad_x, pad_right, cv2.BORDER_REPLICATE
        )

        painted_full = self._run_lama(canvas, mask_canvas)
        painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

        if new_h == crop_h and new_w == crop_w:
            return painted_crop
        return cv2.resize(painted_crop, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

    def _lama_fill_single_fixed(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        crop_h, crop_w = crop.shape[:2]
        scale = INPAINT_SIZE / max(crop_h, crop_w)
        new_h = max(1, int(round(crop_h * scale)))
        new_w = max(1, int(round(crop_w * scale)))
        pad_y = (INPAINT_SIZE - new_h) // 2
        pad_x = (INPAINT_SIZE - new_w) // 2

        crop_resized = cv2.resize(
            crop,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        pad_bottom = INPAINT_SIZE - new_h - pad_y
        pad_right = INPAINT_SIZE - new_w - pad_x
        canvas = cv2.copyMakeBorder(
            crop_resized, pad_y, pad_bottom, pad_x, pad_right, cv2.BORDER_REPLICATE
        )

        mask_resized = self._resize_mask_preserve_support(local_mask, new_w, new_h)
        mask_canvas = cv2.copyMakeBorder(
            mask_resized, pad_y, pad_bottom, pad_x, pad_right, cv2.BORDER_REPLICATE
        )

        painted_full = self._run_lama(canvas, mask_canvas)
        painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

        interpolation = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_CUBIC
        return cv2.resize(painted_crop, (crop_w, crop_h), interpolation=interpolation)

    def _run_lama(self, canvas: np.ndarray, mask_canvas: np.ndarray) -> np.ndarray:
        self._ensure_session()
        self._metric_add("lama_model_runs")
        crop_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_blob = np.ascontiguousarray(
            (crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )
        mask_blob = np.ascontiguousarray(
            (mask_canvas > 127).astype(np.float32)[None, None]
        )
        feed = {self.image_input: img_blob, self.mask_input: mask_blob}

        if self.dynamic_lama or not self._serialize_fixed_inference:
            model_started_at = time.perf_counter()
            output = self.session.run([self.output_name], feed)[0]
            self._metric_add(
                "lama_model_ms",
                round((time.perf_counter() - model_started_at) * 1000.0),
            )
            if not self.dynamic_lama:
                self._session_run_count += 1
        else:
            lock_started_at = time.perf_counter()
            with self._session_lock:
                self._metric_add(
                    "session_lock_wait_ms",
                    round((time.perf_counter() - lock_started_at) * 1000.0),
                )
                self._recycle_fixed_session_if_needed()
                model_started_at = time.perf_counter()
                output = self.session.run([self.output_name], feed)[0]
                measured_model_ms = (
                    time.perf_counter() - model_started_at
                ) * 1000.0
                timing_provider = getattr(self.session, "last_run_timing", None)
                run_timing = (
                    timing_provider()
                    if callable(timing_provider)
                    else {
                        "global_lock_wait_ms": 0.0,
                        "model_run_ms": measured_model_ms,
                    }
                )
                self._metric_add(
                    "ort_global_lock_wait_ms",
                    round(run_timing["global_lock_wait_ms"]),
                )
                self._metric_add(
                    "lama_model_ms",
                    round(run_timing["model_run_ms"]),
                )
                self._session_run_count += 1

        painted_rgb = decode_lama_output(output, self.lama_contract)
        return cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)

    def _lama_fill_tiled(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        tile = INPAINT_SIZE
        overlap = min(MANUAL_TILE_OVERLAP, tile // 4)
        step = tile - overlap

        output = np.zeros((h, w, 3), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        y_starts = self._tile_starts(h, tile, step)
        x_starts = self._tile_starts(w, tile, step)
        for y0 in y_starts:
            y1 = min(h, y0 + tile)
            for x0 in x_starts:
                x1 = min(w, x0 + tile)
                tile_img = crop[y0:y1, x0:x1]
                tile_mask = local_mask[y0:y1, x0:x1]
                tile_h, tile_w = tile_img.shape[:2]
                wy = self._tile_weight(tile_h, overlap, y0 > 0, y1 < h)
                wx = self._tile_weight(tile_w, overlap, x0 > 0, x1 < w)
                weight = wy[:, None] * wx[None, :]

                if not np.any(tile_mask > 127):
                    output[y0:y1, x0:x1] += tile_img.astype(np.float32) * weight[:, :, None]
                    weights[y0:y1, x0:x1] += weight
                    continue

                tile_painted = self._lama_fill_single(tile_img, tile_mask)
                output[y0:y1, x0:x1] += tile_painted.astype(np.float32) * weight[:, :, None]
                weights[y0:y1, x0:x1] += weight

        weights = np.maximum(weights, 1e-6)
        return np.clip(np.rint(output / weights[:, :, None]), 0, 255).astype(np.uint8)

    @staticmethod
    def _tile_starts(length: int, tile: int, step: int) -> list[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, max(1, length - tile + 1), step))
        last = length - tile
        if starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _tile_weight(length: int, overlap: int, has_before: bool, has_after: bool) -> np.ndarray:
        weight = np.ones(length, dtype=np.float32)
        if overlap <= 0:
            return weight
        ramp = np.linspace(0.0, 1.0, min(overlap, length), dtype=np.float32)
        if has_before:
            weight[:len(ramp)] = np.minimum(weight[:len(ramp)], ramp)
        if has_after:
            weight[-len(ramp):] = np.minimum(weight[-len(ramp):], ramp[::-1])
        return weight







