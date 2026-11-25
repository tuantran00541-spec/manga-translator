import gc
import os
import threading
import time

import numpy as np
import cv2
from app.config import (
    LAMA_DYNAMIC_MODEL,
    LAMA_MODEL,
    MIGAN_MODEL,
    MIGAN_MODEL_URL,
)
from app.detector.boxes import BubbleBox
from app.inpaint.api_inpaint import (
    AI_RESCUE_THRESHOLD,
    composite_api_region,
    inpaint_regions_via_api,
    is_complex_for_ai,
    load_inpaint_settings,
    needs_ai_rescue,
    region_complexity_score,
)
from app.inpaint.clustering import cluster_boxes, compute_crop_region, compute_manual_crop_region, split_oversized_cluster_area
from app.inpaint.linework import continue_linework
from app.inpaint.smart_fill import (
    blend_painted,
    match_boundary_tone,
    quadratic_design,
    restore_ambient_grain,
    restore_inpaint_sharpness,
    ring_fill_color,
    smart_fill_color,
    smart_fill_surface,
)
from app.detector.mask_builder import build_mask
from app.logging_config import logger
from app.model_contracts import decode_lama_output, validate_lama_session
from app.ort_utils import make_session
from app.parameters import (
    DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM,
    DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS,
    DYNAMIC_LAMA_TILE,
    DYNAMIC_LAMA_TILE_OVERLAP,
    FIXED_LAMA_RECYCLE_MEMORY_LIMIT_BYTES,
    FIXED_LAMA_CONCURRENT_INFERENCE,
    FIXED_LAMA_SESSION_MAX_RUNS,
    FIXED_LAMA_TILE_ASPECT,
    INPAINT_NATIVE_TILE_EDGE_DENSITY_MIN,
    INPAINT_NATIVE_TILE_ENABLED,
    INPAINT_NATIVE_TILE_MASK_AREA_MIN,
    INPAINT_SIZE,
    INPAINTER,
    LAMA_MASK_DILATE_MAX,
    LAMA_MASK_DILATE_MIN,
    LAMA_MASK_DILATE_SCALE,
    MANUAL_DILATION_SCALE,
    MANUAL_MAX_DILATION,
    MANUAL_MIN_DILATION,
    MANUAL_TILE_OVERLAP,
    SMART_FILL_CANNY_HIGH,
    SMART_FILL_CANNY_LOW,
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


def _local_box(b: BubbleBox, cx1: int, cy1: int) -> BubbleBox:
    local = BubbleBox(
        b.x1 - cx1, b.y1 - cy1, b.x2 - cx1, b.y2 - cy1, b.confidence, b.mask,
        source_model=b.source_model, class_id=b.class_id, class_name=b.class_name,
        semantic_type=b.semantic_type, mask_source=b.mask_source,
        safe_to_inpaint=bool(b.safe_to_inpaint), ocr_eligible=bool(b.ocr_eligible),
        needs_review=bool(b.needs_review), source_role=b.source_role, deferred_reason=b.deferred_reason,
    )
    if bool(getattr(b, "allow_rectangle_fallback", False)):
        local.allow_rectangle_fallback = True
    return local


class Inpainter:
    def __init__(self):
        self._prefer_dynamic = USE_DYNAMIC_LAMA and LAMA_DYNAMIC_MODEL.is_file()
        self._use_migan = INPAINTER == "migan"
        self._migan_session = None
        self._migan_image_input = None
        self._migan_mask_input = None
        self._migan_output_name = None
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
            "telea_fast_regions": 0,
            "api_inpaint_regions": 0,
            "lama_regions": 0,
            "lama_model_runs": 0,
            "lama_model_ms": 0,
            "migan_regions": 0,
            "migan_model_runs": 0,
            "migan_model_ms": 0,
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
        if self._use_migan:
            self._ensure_migan_session()
            return
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
        self.session = session

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
                        enable_cpu_mem_arena=True if prefer_dynamic else None,
                        enable_mem_pattern=True if prefer_dynamic else None,
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

    def _ensure_migan_session(self) -> None:
        if self._migan_session is not None:
            return
        with self._session_lock:
            if self._migan_session is not None:
                return
            if not MIGAN_MODEL.is_file():
                raise RuntimeError(
                    "MI-GAN model missing: {} (download it from {})".format(
                        MIGAN_MODEL, MIGAN_MODEL_URL
                    )
                )
            load_started_at = time.perf_counter()
            logger.info("Preparing inpaint model {}", MIGAN_MODEL)
            session = make_session(
                MIGAN_MODEL,
                enable_cpu_mem_arena=True,
                enable_mem_pattern=True,
            )
            inputs = {entry.name: entry for entry in session.get_inputs()}
            outputs = session.get_outputs()
            self._migan_image_input = inputs["image"].name
            self._migan_mask_input = inputs["mask"].name
            self._migan_output_name = outputs[0].name
            self._migan_session = session
            logger.info(
                "Inpaint model {} ready in {:.1f} ms (migan)",
                MIGAN_MODEL,
                (time.perf_counter() - load_started_at) * 1000.0,
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
        deferred_ai_items: list[dict] | None = None,
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

        api_cfg = load_inpaint_settings()
        use_hybrid_api = bool(api_cfg.get("enabled") and api_cfg.get("hybrid_auto"))
        rescue_only = bool(api_cfg.get("rescue_only", False))
        ai_candidates: list[dict] = []

        pending = {id(b) for cluster in clusters for b in cluster}
        for cluster in clusters:
            pending.difference_update(id(b) for b in cluster)
            x1 = min(b.x1 for b in cluster)
            y1 = min(b.y1 for b in cluster)
            x2 = max(b.x2 for b in cluster)
            y2 = max(b.y2 for b in cluster)

            crop_box = compute_crop_region(x1, y1, x2, y2, w, h)

            cx1, cy1, cx2, cy2 = crop_box
            crop_img = image[cy1:cy2, cx1:cx2]
            local_mask = build_mask((cy2 - cy1, cx2 - cx1), [_local_box(b, cx1, cy1) for b in cluster], crop_img)
            local_mask = self._subtract_protected_regions(local_mask, crop_box, protected_regions)
            others = [_local_box(b, cx1, cy1) for cl in clusters for b in cl if id(b) in pending
                      and b.x1 < cx2 and b.x2 > cx1 and b.y1 < cy2 and b.y2 > cy1]
            hole = None
            if others:
                hole = build_mask((cy2 - cy1, cx2 - cx1), others, crop_img, repair_ink=False)
                hole = np.maximum(self._subtract_protected_regions(hole, crop_box, protected_regions), local_mask)

            if cluster and all(getattr(b, "mask_source", "") == "border_protected_display" for b in cluster):
                orig_before = result[cy1:cy2, cx1:cx2].copy() if use_hybrid_api else None
                if hole is None:
                    result = self._paint_border_protected_display(result, local_mask, crop_box)
                else:
                    result = self._paint_border_protected_display(result, local_mask, crop_box, hole=hole)
                if use_hybrid_api and orig_before is not None and np.any(local_mask > 127):
                    rescue_score = needs_ai_rescue(orig_before, result[cy1:cy2, cx1:cx2], local_mask)
                    if rescue_score >= AI_RESCUE_THRESHOLD:
                        rescue_mask = local_mask.copy()
                        for b in cluster:
                            bx1 = max(0, b.x1 - cx1)
                            by1 = max(0, b.y1 - cy1)
                            bx2 = min(cx2 - cx1, b.x2 - cx1)
                            by2 = min(cy2 - cy1, b.y2 - cy1)
                            if bx2 > bx1 and by2 > by1:
                                rescue_mask[by1:by2, bx1:bx2] = 255
                        rescue_mask = self._subtract_protected_regions(rescue_mask, crop_box, protected_regions)
                        if np.any(rescue_mask > 127):
                            ai_candidates.append({
                                "score": rescue_score,
                                "crop_box": crop_box,
                                "local_mask": rescue_mask,
                                "hole": hole,
                                "crop": orig_before,
                                "already_local": True,
                            })
                continue

            if use_hybrid_api and not rescue_only and np.any(local_mask > 127):
                cur_crop = result[cy1:cy2, cx1:cx2]
                score = region_complexity_score(cur_crop, local_mask)
                if is_complex_for_ai(cur_crop, local_mask):
                    ai_candidates.append({
                        "score": score,
                        "crop_box": crop_box,
                        "local_mask": local_mask,
                        "hole": hole,
                        "crop": cur_crop.copy(),
                        "already_local": False,
                    })
                    continue

            orig_before = result[cy1:cy2, cx1:cx2].copy() if (use_hybrid_api and rescue_only) else None
            result = self._smart_paint_region(result, local_mask, crop_box, hole=hole)
            if use_hybrid_api and rescue_only and orig_before is not None and np.any(local_mask > 127):
                # Measure rescue artifacts on the tighter inner mask (undoing adaptive dilation padding)
                # so white margins outside letters don't dilute blur/seam detection.
                core_eval = cv2.erode(
                    (local_mask > 127).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                )
                eval_mask = core_eval if int(np.count_nonzero(core_eval > 127)) >= 900 else local_mask
                rescue_score = max(
                    needs_ai_rescue(orig_before, result[cy1:cy2, cx1:cx2], local_mask),
                    needs_ai_rescue(orig_before, result[cy1:cy2, cx1:cx2], eval_mask),
                )
                if rescue_score >= AI_RESCUE_THRESHOLD:
                    ai_candidates.append({
                        "score": rescue_score,
                        "crop_box": crop_box,
                        "local_mask": local_mask,
                        "hole": hole,
                        "crop": orig_before,
                        "already_local": True,
                    })

        if ai_candidates:
            if deferred_ai_items is not None:
                deferred_ai_items.extend(ai_candidates)
                return result

            ai_candidates.sort(key=lambda item: item["score"], reverse=True)
            pack_atlas = bool(api_cfg.get("pack_atlas", True))
            max_calls = max(1, int(api_cfg.get("max_api_calls_per_chapter", 3)))

            api_outputs: list[np.ndarray | None] | None = None
            try:
                regions_payload = [
                    {
                        "crop": item["crop"],
                        "mask": item["local_mask"],
                    }
                    for item in ai_candidates
                ]
                api_outputs = inpaint_regions_via_api(
                    regions_payload,
                    provider_id=api_cfg.get("provider_id") or None,
                    model=api_cfg.get("model") or None,
                    quality=api_cfg.get("quality") or None,
                    pack_atlas=pack_atlas,
                    max_calls=max_calls,
                )
            except Exception as exc:
                logger.warning("Hybrid API inpaint failed; falling back to local LaMa: {}", exc)
                api_outputs = None

            for idx, item in enumerate(ai_candidates):
                ai_crop = api_outputs[idx] if api_outputs is not None and idx < len(api_outputs) else None
                if ai_crop is not None:
                    result = composite_api_region(
                        result,
                        ai_crop,
                        item["local_mask"],
                        item["crop_box"],
                        feather=False,
                    )
                    self._metric_add("api_inpaint_regions")
                elif not item.get("already_local"):
                    result = self._smart_paint_region(
                        result,
                        item["local_mask"],
                        item["crop_box"],
                        hole=item["hole"],
                    )

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

    def inpaint_mask_via_api(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        provider_id: str | None = None,
        model: str | None = None,
        quality: str | None = None,
        pack_atlas: bool = True,
    ) -> np.ndarray:
        """Redraw user-brushed mask components via the configured AI image-edit API."""
        self._begin_metrics()
        if mask is None or not np.any(mask > 127):
            return image.copy()

        binary_mask = (mask > 127).astype(np.uint8) * 255
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        self._metrics_local.value["mask_components"] = max(0, int(num_labels) - 1)

        result = image.copy()
        h, w = image.shape[:2]
        entries: list[dict] = []

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
            entries.append({
                "crop": result[cy1:cy2, cx1:cx2].copy(),
                "mask": local_mask,
                "crop_box": crop_box,
            })

        if not entries:
            return result

        api_outputs = inpaint_regions_via_api(
            entries,
            provider_id=provider_id,
            model=model,
            quality=quality,
            pack_atlas=pack_atlas,
        )
        for entry, ai_crop in zip(entries, api_outputs):
            if ai_crop is None:
                result = self._smart_paint_region(result, entry["mask"], entry["crop_box"], feather=True)
                continue
            result = composite_api_region(
                result,
                ai_crop,
                entry["mask"],
                entry["crop_box"],
                feather=True,
            )
            self._metric_add("api_inpaint_regions")

        return result

    @staticmethod
    def _smart_fill_color(crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray | None:
        return smart_fill_color(crop, local_mask)

    @staticmethod
    def _ring_fill_color(crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray | None:
        return ring_fill_color(crop, local_mask)

    @staticmethod
    def _quadratic_design(
        xs: np.ndarray,
        ys: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        return quadratic_design(xs, ys, width, height)

    @classmethod
    def _smart_fill_surface(
        cls,
        crop: np.ndarray,
        local_mask: np.ndarray,
        fill_color: np.ndarray,
    ) -> np.ndarray | None:
        return smart_fill_surface(crop, local_mask, fill_color)

    def _paint_border_protected_display(
        self,
        image: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple[int, int, int, int],
        hole: np.ndarray | None = None,
    ) -> np.ndarray:
        cx1, cy1, cx2, cy2 = crop_box
        crop = image[cy1:cy2, cx1:cx2]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return image
        mask_bool = local_mask > 127
        if not np.any(mask_bool):
            return image

        hidden = local_mask if hole is None else cv2.bitwise_or(local_mask, hole)
        clean_context = hidden <= 127
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        paper_px = crop[clean_context & (gray >= 235)]
        paper_color = (
            np.median(paper_px, axis=0).astype(np.float32)
            if len(paper_px)
            else np.array([254.0, 254.0, 254.0], dtype=np.float32)
        )
        filled = crop.copy()
        surface = None if hole is not None else self._smart_fill_surface(
            crop,
            local_mask,
            np.clip(np.rint(paper_color), 0, 255).astype(np.uint8),
        )
        if surface is not None:
            self._metric_add("smart_fill_gradient_regions")
            filled[mask_bool] = surface
        else:
            filled[mask_bool] = np.clip(np.rint(paper_color), 0, 255).astype(np.uint8)

        # Repair only tiny notches where giant display letters protruded into colored panel art.
        if crop.ndim == 3 and crop.shape[2] == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            art = clean_context & (gray < 220) & ((hsv[..., 1] >= 45) | (gray < 130))
            near_art = cv2.dilate(
                art.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
            ) > 0
            art_overlap = mask_bool & near_art
            did_lama = False
            if int(np.count_nonzero(art_overlap)) >= 80:
                grouped = cv2.dilate(
                    art_overlap.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                )
                n_comp, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(
                    grouped, connectivity=8
                )
                for cid in range(1, n_comp):
                    comp_mask = art_overlap & (comp_labels == cid)
                    if int(np.count_nonzero(comp_mask)) < 80:
                        continue
                    bx, by, bw, bh, _ = (int(v) for v in comp_stats[cid])
                    ay1 = max(0, by - 48)
                    ay2 = min(crop.shape[0], by + bh + 48)
                    ax1 = max(0, bx - 48)
                    ax2 = min(crop.shape[1], bx + bw + 48)
                    region = np.s_[ay1:ay2, ax1:ax2]
                    source = crop[region]
                    sub_mask = comp_mask[region].astype(np.uint8) * 255
                    sub_hole = hidden[region]
                    painted_sub = self._lama_fill_single(source, sub_hole)
                    painted_sub = match_boundary_tone(source, painted_sub, sub_mask, hole=sub_hole)
                    painted_sub = restore_inpaint_sharpness(source, painted_sub, sub_mask, hole=sub_hole)
                    painted_sub = continue_linework(source, painted_sub, sub_mask, hole=sub_hole)
                    filled[region][sub_mask > 127] = painted_sub[sub_mask > 127]
                    did_lama = True
            if did_lama:
                self._metric_add("lama_regions")
            else:
                self._metric_add("smart_fill_regions")
        else:
            self._metric_add("smart_fill_regions")

        filled = restore_ambient_grain(crop, filled, local_mask, crop_box, hole=hole)
        result = image.copy()
        result_crop = result[cy1:cy2, cx1:cx2]
        result_crop[mask_bool] = filled[mask_bool]
        return result

    def _smart_paint_region(
        self,
        image: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple,
        feather: bool = False,
        force_lama: bool = False,
        hole: np.ndarray | None = None,
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
        if fill_color is None and not force_lama:
            fill_color = self._ring_fill_color(crop, local_mask)
        if fill_color is not None:
            self._metric_add("smart_fill_regions")
            filled = crop.copy()
            surface = self._smart_fill_surface(crop, local_mask, fill_color)
            if surface is not None:
                self._metric_add("smart_fill_gradient_regions")
                filled[mask_bool] = surface
            else:
                filled[mask_bool] = fill_color
            filled = restore_ambient_grain(crop, filled, local_mask, crop_box, hole=hole)
            return self._blend_painted(image, filled, local_mask, crop_box, feather, hole=hole)

        # Fast CPU bypass: tiny isolated specks (<= 180px) on low-edge backgrounds don't need
        # a full 512x512 LaMa ONNX forward pass; OpenCV Telea finishes in <0.5ms cleanly.
        if not force_lama and hole is None:
            mask_area = int(np.count_nonzero(mask_bool))
            if 0 < mask_area <= 180:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                ctx_mask = ~mask_bool
                if np.any(ctx_mask):
                    crop_edges = cv2.Canny(crop_gray, SMART_FILL_CANNY_LOW, SMART_FILL_CANNY_HIGH)
                    ctx_edge_density = float(np.mean(crop_edges[ctx_mask] > 0))
                    if ctx_edge_density < 0.08:
                        self._metric_add("telea_fast_regions")
                        telea_mask = mask_bool.astype(np.uint8) * 255
                        painted = cv2.inpaint(crop, telea_mask, 3, cv2.INPAINT_TELEA)
                        painted = match_boundary_tone(crop, painted, local_mask)
                        painted = restore_ambient_grain(crop, painted, local_mask, crop_box)
                        return self._blend_painted(image, painted, local_mask, crop_box, feather)

        self._metric_add("lama_regions")
        return self._lama_fill(image, crop, local_mask, crop_box, feather=feather, hole=hole)

    def _lama_fill(self, image: np.ndarray, crop: np.ndarray, local_mask: np.ndarray, crop_box: tuple,
                   feather: bool = False, hole: np.ndarray | None = None) -> np.ndarray:
        if self._use_migan:
            return self._migan_fill(image, crop, local_mask, crop_box, feather=feather, hole=hole)
        write_mask = local_mask
        model_mask = local_mask
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        crop_edges = cv2.Canny(crop_gray, SMART_FILL_CANNY_LOW, SMART_FILL_CANNY_HIGH)
        ctx_mask = write_mask <= 127
        ctx_edge_density = float(np.mean(crop_edges[ctx_mask] > 0)) if np.any(ctx_mask) else 0.0
        textured_art = ctx_edge_density >= 0.035
        if hole is None:
            ys, xs = np.where(local_mask > 127)
            if len(ys):
                side = min(int(ys.max()) - int(ys.min()) + 1, int(xs.max()) - int(xs.min()) + 1)
                # Keep model context dilation tight over detailed artwork so nearby strokes stay
                # visible to LaMa instead of being masked out and hallucinated from afar.
                max_ctx_radius = min(LAMA_MASK_DILATE_MAX, 7 if textured_art else 12)
                radius = int(np.clip(round(side * LAMA_MASK_DILATE_SCALE), LAMA_MASK_DILATE_MIN, max_ctx_radius))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                model_mask = cv2.dilate(local_mask, kernel, iterations=1)
        self._ensure_session()
        cx1, cy1, cx2, cy2 = crop_box
        crop_h, crop_w = crop.shape[:2]
        fill_mask = model_mask if hole is None else cv2.bitwise_or(model_mask, hole)

        # Trim dead margin outside the actual mask extent when it meaningfully shrinks LaMa pixels,
        # preserving wider structural context around detailed artwork for FFC line continuation.
        tx1, ty1, tx2, ty2 = 0, 0, crop_w, crop_h
        my, mx = np.where(model_mask > 127)
        if len(my):
            mh = int(my.max()) - int(my.min()) + 1
            mw = int(mx.max()) - int(mx.min()) + 1
            if textured_art:
                pad = min(72, max(36, int(round(0.45 * max(mh, mw)))))
            else:
                pad = min(48, max(24, int(round(0.35 * max(mh, mw)))))
            cand_x1 = max(0, int(mx.min()) - pad)
            cand_y1 = max(0, int(my.min()) - pad)
            cand_x2 = min(crop_w, int(mx.max()) + 1 + pad)
            cand_y2 = min(crop_h, int(my.max()) + 1 + pad)
            if (
                cand_x2 - cand_x1 >= 16
                and cand_y2 - cand_y1 >= 16
                and (cand_x2 - cand_x1) * (cand_y2 - cand_y1) <= 0.80 * crop_h * crop_w
            ):
                tx1, ty1, tx2, ty2 = cand_x1, cand_y1, cand_x2, cand_y2

        work_crop = crop[ty1:ty2, tx1:tx2]
        work_write = write_mask[ty1:ty2, tx1:tx2]
        work_fill = fill_mask[ty1:ty2, tx1:tx2]
        work_h, work_w = work_crop.shape[:2]

        max_dim = max(work_h, work_w)
        min_dim = max(1, min(work_h, work_w))
        aspect = max_dim / min_dim
        long_crop = (
            not self.dynamic_lama
            and max_dim > INPAINT_SIZE
            and aspect >= FIXED_LAMA_TILE_ASPECT
        )
        texture_tiling = False
        if (
            INPAINT_NATIVE_TILE_ENABLED
            and max_dim > DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM
            and int(np.count_nonzero(work_write > 127)) >= INPAINT_NATIVE_TILE_MASK_AREA_MIN
        ):
            work_edges = crop_edges[ty1:ty2, tx1:tx2]
            context = work_write <= 127
            edge_density = float(np.mean(work_edges[context] > 0)) if np.any(context) else 0.0
            texture_tiling = edge_density >= INPAINT_NATIVE_TILE_EDGE_DENSITY_MIN

        crop_pixels = int(work_h * work_w)
        dynamic_native_ok = bool(
            self.dynamic_lama
            and not feather
            and max_dim <= DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM
            and crop_pixels <= DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS
        )
        if dynamic_native_ok:
            self._metric_add("lama_native_single_regions")
            work_painted = self._lama_fill_single(work_crop, work_fill)
        elif long_crop or texture_tiling or (feather and max_dim > INPAINT_SIZE):
            self._metric_add("lama_tiled_regions")
            work_painted = self._lama_fill_tiled(work_crop, work_fill, own=work_write)
        else:
            work_painted = self._lama_fill_single(work_crop, work_fill)

        # Base finishing on real context outside the write mask and any still-unpainted text.
        work_hole = hole[ty1:ty2, tx1:tx2] if hole is not None else None
        work_box = (cx1 + tx1, cy1 + ty1, cx1 + tx2, cy1 + ty2)
        painted = match_boundary_tone(work_crop, work_painted, work_write, hole=work_hole)
        painted = restore_inpaint_sharpness(work_crop, painted, work_write, hole=work_hole)
        painted = continue_linework(work_crop, painted, work_write, hole=work_hole)
        painted = restore_ambient_grain(work_crop, painted, work_write, work_box, hole=work_hole)
        return self._blend_painted(image, painted, work_write, work_box, feather, hole=work_hole)

    def _migan_fill(self, image: np.ndarray, crop: np.ndarray, local_mask: np.ndarray, crop_box: tuple,
                    feather: bool = False, hole: np.ndarray | None = None) -> np.ndarray:
        self._ensure_migan_session()
        fill_mask = local_mask if hole is None else hole
        started = time.perf_counter()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mask_u8 = np.where(fill_mask > 127, 0, 255).astype(np.uint8)
        feed = {
            self._migan_image_input: np.ascontiguousarray(rgb.transpose(2, 0, 1))[None],
            self._migan_mask_input: mask_u8[None, None],
        }
        output = self._migan_session.run([self._migan_output_name], feed)[0]
        painted = cv2.cvtColor(output[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        self._metric_add("migan_regions")
        self._metric_add("migan_model_runs")
        self._metric_add("migan_model_ms", int((time.perf_counter() - started) * 1000))
        painted = match_boundary_tone(crop, painted, local_mask, hole=hole)
        painted = restore_ambient_grain(crop, painted, local_mask, crop_box, hole=hole)
        return self._blend_painted(image, painted, local_mask, crop_box, feather, hole=hole)

    @staticmethod
    def _blend_painted(
        image: np.ndarray,
        painted: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple,
        feather: bool,
        hole: np.ndarray | None = None,
    ) -> np.ndarray:
        return blend_painted(image, painted, local_mask, crop_box, feather, hole=hole)

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

    @staticmethod
    def _dynamic_lama_max_dim(crop: np.ndarray, local_mask: np.ndarray) -> int:
        max_dim = DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM
        if max(crop.shape[:2]) <= 384 or max_dim <= 384:
            return max_dim
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        edges = cv2.Canny(gray, SMART_FILL_CANNY_LOW, SMART_FILL_CANNY_HIGH)
        context = local_mask <= 127
        if np.any(context):
            edge_density = float(np.mean(edges[context] > 0))
            if edge_density < 0.025:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                safe = cv2.erode(context.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                collar = (cv2.dilate(local_mask, kernel) > 127) & safe
                # A single stroke passing the hole can matter more than a smooth crop average.
                if np.count_nonzero(edges[collar]) < max(10, .008 * np.count_nonzero(collar)):
                    return min(max_dim, 384)
        return max_dim

    def _lama_fill_single_dynamic(self, crop: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        crop_h, crop_w = crop.shape[:2]
        max_side = self._dynamic_lama_max_dim(crop, local_mask)

        scale = min(
            1.0,
            max_side / max(crop_h, crop_w),
        )
        new_h = max(1, int(round(crop_h * scale)))
        new_w = max(1, int(round(crop_w * scale)))

        if new_h != crop_h or new_w != crop_w:
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_resized = self._resize_mask_preserve_support(local_mask, new_w, new_h)
        else:
            crop_resized = crop
            mask_resized = local_mask

        canvas_h = max(16, ((new_h + 15) // 16) * 16)
        canvas_w = max(16, ((new_w + 15) // 16) * 16)
        pad_y = (canvas_h - new_h) // 2
        pad_x = (canvas_w - new_w) // 2
        pad_bottom = canvas_h - new_h - pad_y
        pad_right = canvas_w - new_w - pad_x

        border_mode = (
            cv2.BORDER_REFLECT_101
            if min(new_h, new_w) > max(pad_y, pad_bottom, pad_x, pad_right) + 1
            else cv2.BORDER_REPLICATE
        )
        canvas = cv2.copyMakeBorder(
            crop_resized, pad_y, pad_bottom, pad_x, pad_right, border_mode
        )
        mask_canvas = cv2.copyMakeBorder(
            mask_resized, pad_y, pad_bottom, pad_x, pad_right, border_mode
        )

        painted_full = self._run_lama(canvas, mask_canvas)
        painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

        if new_h == crop_h and new_w == crop_w:
            return painted_crop
        return cv2.resize(painted_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

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
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
        )
        pad_bottom = INPAINT_SIZE - new_h - pad_y
        pad_right = INPAINT_SIZE - new_w - pad_x
        border_mode = (
            cv2.BORDER_REFLECT_101
            if min(new_h, new_w) > max(pad_y, pad_bottom, pad_x, pad_right) + 1
            else cv2.BORDER_REPLICATE
        )
        canvas = cv2.copyMakeBorder(
            crop_resized, pad_y, pad_bottom, pad_x, pad_right, border_mode
        )

        mask_resized = self._resize_mask_preserve_support(local_mask, new_w, new_h)
        mask_canvas = cv2.copyMakeBorder(
            mask_resized, pad_y, pad_bottom, pad_x, pad_right, border_mode
        )

        painted_full = self._run_lama(canvas, mask_canvas)
        painted_crop = painted_full[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

        interpolation = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LANCZOS4
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
        img_blob = np.ascontiguousarray(img_blob * (1.0 - mask_blob))
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

    def _lama_fill_tiled(self, crop: np.ndarray, local_mask: np.ndarray, own: np.ndarray | None = None) -> np.ndarray:
        h, w = crop.shape[:2]
        tile = DYNAMIC_LAMA_TILE if self.dynamic_lama else INPAINT_SIZE
        overlap = min(DYNAMIC_LAMA_TILE_OVERLAP if self.dynamic_lama else MANUAL_TILE_OVERLAP, tile // 4)
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

                if not np.any((tile_mask if own is None else own[y0:y1, x0:x1]) > 127):
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







