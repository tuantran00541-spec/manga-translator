import gc
import os
import threading
import time

import numpy as np
import cv2
from app.config import LAMA_MODEL, LAMA_DYNAMIC_MODEL
from app.detector.bubble_detector import BubbleBox, MAX_BOX_AREA_RATIO
from app.detector.mask_builder import build_mask
from app.logging_config import logger
from app.ort_utils import make_session
from app.parameters import (
    DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM,
    FIXED_LAMA_RECYCLE_MEMORY_LIMIT_BYTES,
    FIXED_LAMA_CONCURRENT_INFERENCE,
    FIXED_LAMA_SESSION_MAX_RUNS,
    FIXED_LAMA_TILE_ASPECT,
    INPAINT_CLUSTER_MAX_DIM,
    INPAINT_CLUSTER_GROUP_HEIGHT_FACTOR,
    INPAINT_CLUSTER_LINE_OVERLAP_MIN,
    INPAINT_CLUSTER_PADDING,
    INPAINT_CLUSTER_SPLIT_COUNT,
    INPAINT_CLUSTER_SPLIT_HEIGHT_FACTOR,
    INPAINT_CROP_LONG_ASPECT_THRESHOLD,
    INPAINT_CROP_PADDING,
    INPAINT_SIZE,
    MANUAL_CROP_PADDING,
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
    SMART_FILL_CLEAN_RING_MARGIN,
    SMART_FILL_CONTEXT_MARGIN_FACTOR,
    SMART_FILL_EDGE_DENSITY_MAX,
    SMART_FILL_FULL_STD_MAX,
    SMART_FILL_MIDTONE_MAX,
    SMART_FILL_MIDTONE_MIN,
    SMART_FILL_MIDTONE_STD_MAX,
    SMART_FILL_RING_PIXELS_MIN,
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
    """Return True only for Linux-style memory cgroups with a small hard cap.

    The fixed-session recycle workaround exists for the 4 GiB acceptance
    container. Recreating a ~200 MB ONNX session every four calls is harmful on
    normal desktop installs, especially Windows, so do not enable it merely
    because the fixed model is selected.
    """
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
        self._metrics_local = threading.local()
        self._session_run_count = 0
        self.session = None
        self.image_input = None
        self.mask_input = None
        self.dynamic_lama = False
        self._serialize_fixed_inference = not FIXED_LAMA_CONCURRENT_INFERENCE
        self.lama_model_path = LAMA_DYNAMIC_MODEL if self._prefer_dynamic else LAMA_MODEL
        self._recycle_fixed_session = False

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        self._metrics_local.value = {
            "boxes": max(0, int(boxes)),
            "clusters": 0,
            "skipped_clusters": 0,
            "smart_fill_regions": 0,
            "lama_regions": 0,
            "lama_model_runs": 0,
            "lama_model_ms": 0,
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
        """Return counters from the current worker thread's latest inpaint call."""
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

    def _configure_loaded_session(self, session, model_path) -> None:
        inputs = session.get_inputs()
        image_shape = inputs[0].shape
        dynamic_lama = any(
            isinstance(dim, str) or dim is None for dim in image_shape[2:4]
        )
        self.session = session
        self.image_input = inputs[0].name
        self.mask_input = inputs[1].name
        self.dynamic_lama = dynamic_lama
        self._serialize_fixed_inference = bool(
            not dynamic_lama and type(session).__name__ == "_SerializedSession"
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

            self._configure_loaded_session(session, model_path)
            logger.info(
                "Loaded inpaint model {} lazily (dynamic={})",
                self.lama_model_path,
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
        self._configure_loaded_session(session, self.lama_model_path)

    def inpaint(self, image: np.ndarray, boxes: list[BubbleBox]) -> np.ndarray:
        self._begin_metrics(boxes=len(boxes))
        if not boxes:
            return image.copy()

        result = image.copy()
        h, w = image.shape[:2]
        clusters = self._cluster_boxes(boxes)
        self._metrics_local.value["clusters"] = len(clusters)

        for cluster in clusters:
            x1 = min(b.x1 for b in cluster)
            y1 = min(b.y1 for b in cluster)
            x2 = max(b.x2 for b in cluster)
            y2 = max(b.y2 for b in cluster)

            if len(cluster) > 1 and (x2 - x1) * (y2 - y1) > w * h * MAX_BOX_AREA_RATIO:
                self._metric_add("skipped_clusters")
                logger.warning(f"Skipping multi-box cluster ({len(cluster)} boxes) at ({x1}, {y1}, {x2}, {y2}): area exceeds MAX_BOX_AREA_RATIO")
                continue

            crop_box = self._compute_crop_region(x1, y1, x2, y2, w, h)

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
                )
                if bool(getattr(b, "allow_rectangle_fallback", False)):
                    local_box.allow_rectangle_fallback = True
                local_boxes.append(local_box)
            crop_img = image[cy1:cy2, cx1:cx2]
            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)

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
            crop_box = self._compute_manual_crop_region(gx1, gy1, gx2, gy2, w, h)
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
            white_ratio >= SMART_FILL_WHITE_RATIO_MIN
            and ring_std <= SMART_FILL_WHITE_STD_MAX
            and context_std <= SMART_FILL_FULL_STD_MAX
            and context_edge_density <= SMART_FILL_EDGE_DENSITY_MAX
        ):
            white_pixels = ring_pixels[ring_gray > SMART_FILL_WHITE_LEVEL]
            if len(white_pixels):
                return np.median(white_pixels, axis=0).astype(np.uint8)

        if (
            black_ratio >= SMART_FILL_BLACK_RATIO_MIN
            and ring_std <= SMART_FILL_BLACK_STD_MAX
            and context_std <= SMART_FILL_BLACK_STD_MAX
            and context_edge_density <= SMART_FILL_BLACK_EDGE_DENSITY_MAX
        ):
            black_pixels = ring_pixels[ring_gray < SMART_FILL_BLACK_LEVEL]
            if len(black_pixels):
                return np.median(black_pixels, axis=0).astype(np.uint8)

        if (
            SMART_FILL_MIDTONE_MIN <= median_gray <= SMART_FILL_MIDTONE_MAX
            and ring_std <= SMART_FILL_MIDTONE_STD_MAX
            and context_std <= SMART_FILL_MIDTONE_STD_MAX
            and context_edge_density <= SMART_FILL_BLACK_EDGE_DENSITY_MAX
        ):
            return np.median(ring_pixels, axis=0).astype(np.uint8)

        return None

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

        # Wide/tall free text loses background detail when a dynamic LaMa crop
        # is squeezed to 512px just as it does with the fixed model. Preserve
        # native detail with overlapping tiles for both backends. Small and
        # near-square regions retain the single-call fast path.
        if long_crop or (feather and max_dim > INPAINT_SIZE):
            painted = self._lama_fill_tiled(crop, local_mask)
        else:
            painted = self._lama_fill_single(crop, local_mask)

        original_crop = image[cy1:cy2, cx1:cx2]
        if feather:
            alpha = (local_mask > 127).astype(np.float32)
            k = MANUAL_FEATHER_RADIUS * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
            blended = painted.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1.0 - alpha)
            image[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            mask_3d = (local_mask > 127)[:, :, None]
            image[cy1:cy2, cx1:cx2] = np.where(mask_3d, painted, original_crop)
        return image

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
            mask_resized = cv2.resize(local_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
        mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = mask_resized

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

        mask_resized = cv2.resize(local_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_canvas = np.zeros((INPAINT_SIZE, INPAINT_SIZE), dtype=np.uint8)
        mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = mask_resized

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
            output = self.session.run(None, feed)[0]
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
                output = self.session.run(None, feed)[0]
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

        painted_rgb = output[0].transpose(1, 2, 0)
        if painted_rgb.max() <= 1.0:
            painted_rgb = painted_rgb * 255.0
        painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)
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
        return np.clip(output / weights[:, :, None], 0, 255).astype(np.uint8)

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

    @staticmethod
    def _cluster_boxes(boxes: list[BubbleBox]) -> list[list[BubbleBox]]:
        remaining = list(boxes)
        raw_clusters = []

        while remaining:
            current = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                still_remaining = []
                for b in remaining:
                    if any(Inpainter._boxes_close(b, c) for c in current) and Inpainter._can_add_to_cluster(current, b, INPAINT_CLUSTER_MAX_DIM):
                        current.append(b)
                        changed = True
                    else:
                        still_remaining.append(b)
                remaining = still_remaining
            raw_clusters.append(current)

        final_clusters = []
        for cluster in raw_clusters:
            if len(cluster) > 1:
                avg_h = sum(b.y2 - b.y1 for b in cluster) / len(cluster)
                cluster_h = max(b.y2 for b in cluster) - min(b.y1 for b in cluster)
                if (
                    len(cluster) > INPAINT_CLUSTER_SPLIT_COUNT
                    or cluster_h > INPAINT_CLUSTER_SPLIT_HEIGHT_FACTOR * avg_h
                ):
                    sub_clusters = Inpainter._split_cluster_lines(cluster, avg_h)
                    final_clusters.extend(sub_clusters)
                else:
                    final_clusters.append(cluster)
            else:
                final_clusters.append(cluster)

        return final_clusters

    @staticmethod
    def _split_cluster_lines(cluster: list[BubbleBox], avg_h: float) -> list[list[BubbleBox]]:
        sorted_boxes = sorted(cluster, key=lambda b: (b.y1, b.x1))
        lines = []
        for b in sorted_boxes:
            placed = False
            for line in lines:
                line_y1 = min(x.y1 for x in line)
                line_y2 = max(x.y2 for x in line)
                overlap = min(b.y2, line_y2) - max(b.y1, line_y1)
                min_h = min(b.y2 - b.y1, line_y2 - line_y1)
                if (
                    min_h > 0
                    and overlap / min_h > INPAINT_CLUSTER_LINE_OVERLAP_MIN
                ):
                    line.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])

        lines.sort(key=lambda line: min(b.y1 for b in line))
        sub_clusters = []
        current_group = []
        for line in lines:
            if not current_group:
                current_group = list(line)
            else:
                group_h = max(b.y2 for b in current_group + line) - min(b.y1 for b in current_group + line)
                if group_h > INPAINT_CLUSTER_GROUP_HEIGHT_FACTOR * avg_h:
                    sub_clusters.append(current_group)
                    current_group = list(line)
                else:
                    current_group.extend(line)
        if current_group:
            sub_clusters.append(current_group)

        return sub_clusters

    @staticmethod
    def _boxes_close(a: BubbleBox, b: BubbleBox) -> bool:
        ax1, ay1, ax2, ay2 = (
            a.x1 - INPAINT_CLUSTER_PADDING,
            a.y1 - INPAINT_CLUSTER_PADDING,
            a.x2 + INPAINT_CLUSTER_PADDING,
            a.y2 + INPAINT_CLUSTER_PADDING,
        )
        bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    @staticmethod
    def _can_add_to_cluster(
        cluster: list[BubbleBox],
        b: BubbleBox,
        max_dim: int = INPAINT_CLUSTER_MAX_DIM,
    ) -> bool:
        x1 = min(min(box.x1 for box in cluster), b.x1)
        y1 = min(min(box.y1 for box in cluster), b.y1)
        x2 = max(max(box.x2 for box in cluster), b.x2)
        y2 = max(max(box.y2 for box in cluster), b.y2)
        return (x2 - x1) <= max_dim and (y2 - y1) <= max_dim

    @staticmethod
    def _compute_manual_crop_region(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
        x1 = max(0, x1 - MANUAL_CROP_PADDING)
        y1 = max(0, y1 - MANUAL_CROP_PADDING)
        x2 = min(img_w, x2 + MANUAL_CROP_PADDING)
        y2 = min(img_h, y2 + MANUAL_CROP_PADDING)
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _compute_crop_region(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
        x1 -= INPAINT_CROP_PADDING
        y1 -= INPAINT_CROP_PADDING
        x2 += INPAINT_CROP_PADDING
        y2 += INPAINT_CROP_PADDING

        box_w = x2 - x1
        box_h = y2 - y1

        aspect = max(box_w / max(1, box_h), box_h / max(1, box_w))
        if aspect > INPAINT_CROP_LONG_ASPECT_THRESHOLD:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            return int(x1), int(y1), int(x2), int(y2)

        side = max(box_w, box_h)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        x1 = cx - side / 2
        x2 = cx + side / 2
        y1 = cy - side / 2
        y2 = cy + side / 2

        if x1 < 0:
            x2 = min(img_w, x2 - x1)
            x1 = 0
        if y1 < 0:
            y2 = min(img_h, y2 - y1)
            y1 = 0
        if x2 > img_w:
            x1 = max(0, x1 - (x2 - img_w))
            x2 = img_w
        if y2 > img_h:
            y1 = max(0, y1 - (y2 - img_h))
            y2 = img_h

        return int(x1), int(y1), int(x2), int(y2)
