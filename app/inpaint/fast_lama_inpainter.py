from __future__ import annotations

import os

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.inpaint.lama_inpainter import Inpainter


def _env_bool(name: str, default: bool) -> bool:
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


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    return max(float(minimum), min(float(maximum), float(value)))


# Experimental knobs live with the branch candidate. If this path wins the real
# chapter gate they should move into app.parameters before the production port.
_BUBBLE_FASTPATH_ENABLED = _env_bool("MANGA_BUBBLE_FASTPATH_ENABLED", True)
_BUBBLE_FASTPATH_PAD = _env_int("MANGA_BUBBLE_FASTPATH_PAD", 16, 4, 96)
_BUBBLE_FASTPATH_RING = _env_int("MANGA_BUBBLE_FASTPATH_RING", 12, 4, 64)
_BUBBLE_FASTPATH_GRAY_STD_MAX = _env_float(
    "MANGA_BUBBLE_FASTPATH_GRAY_STD_MAX", 18.0, 0.0, 64.0
)
_BUBBLE_FASTPATH_CHROMA_STD_MAX = _env_float(
    "MANGA_BUBBLE_FASTPATH_CHROMA_STD_MAX", 14.0, 0.0, 64.0
)
_BUBBLE_FASTPATH_EDGE_DENSITY_MAX = _env_float(
    "MANGA_BUBBLE_FASTPATH_EDGE_DENSITY_MAX", 0.020, 0.0, 0.25
)
_BUBBLE_FASTPATH_TELEA_RADIUS = _env_int(
    "MANGA_BUBBLE_FASTPATH_TELEA_RADIUS", 3, 1, 9
)

_ROI_LAMA_ENABLED = _env_bool("MANGA_DYNAMIC_LAMA_TIGHT_ROI_ENABLED", True)
_ROI_CONTEXT_MIN = _env_int("MANGA_DYNAMIC_LAMA_ROI_CONTEXT_MIN", 64, 16, 256)
_ROI_CONTEXT_MAX = _env_int("MANGA_DYNAMIC_LAMA_ROI_CONTEXT_MAX", 192, 32, 512)
_ROI_CONTEXT_SCALE = _env_float(
    "MANGA_DYNAMIC_LAMA_ROI_CONTEXT_SCALE", 0.35, 0.0, 2.0
)
_ROI_MIN_SAVINGS = _env_float(
    "MANGA_DYNAMIC_LAMA_ROI_MIN_SAVINGS", 0.15, 0.0, 0.90
)


class FastInpainter(Inpainter):
    """CPU-oriented inpainting candidate layered on top of production LaMa.

    Two optimizations are deliberately conservative:

    * Verified speech-bubble masks first try a cheap fill path. Existing solid
      smart-fill remains the first choice; mildly varying but low-texture bubble
      interiors may use OpenCV Telea rather than loading/running LaMa.
    * Dynamic LaMa receives a tight mask-derived ROI with bounded context instead
      of the wider detector-cluster crop when doing so removes meaningful pixels.

    Every fallback remains the production Inpainter implementation, and final
    compositing is still restricted to the authorized destructive mask.
    """

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        super()._begin_metrics(boxes=boxes)
        self._metrics_local.value.update(
            {
                "bubble_fast_fill_regions": 0,
                "bubble_fast_fill_solid_regions": 0,
                "bubble_fast_fill_telea_regions": 0,
                "bubble_fast_fill_pixels": 0,
                "roi_lama_regions": 0,
                "roi_lama_source_pixels": 0,
                "roi_lama_input_pixels": 0,
                "roi_lama_saved_pixels": 0,
            }
        )

    @staticmethod
    def _bubble_candidate(box: BubbleBox) -> bool:
        return bool(
            box.semantic_type == "speech_bubble"
            and box.safe_to_inpaint
            and not box.needs_review
            and box.verified_mask
        )

    @staticmethod
    def _local_box(box: BubbleBox, cx1: int, cy1: int) -> BubbleBox:
        local = BubbleBox(
            box.x1 - cx1,
            box.y1 - cy1,
            box.x2 - cx1,
            box.y2 - cy1,
            box.confidence,
            box.mask,
            source_model=box.source_model,
            class_id=box.class_id,
            class_name=box.class_name,
            semantic_type=box.semantic_type,
            mask_source=box.mask_source,
            safe_to_inpaint=bool(box.safe_to_inpaint),
            ocr_eligible=bool(box.ocr_eligible),
            needs_review=bool(box.needs_review),
            source_role=box.source_role,
            deferred_reason=box.deferred_reason,
        )
        if bool(getattr(box, "allow_rectangle_fallback", False)):
            local.allow_rectangle_fallback = True
        return local

    @staticmethod
    def _mask_ring(local_mask: np.ndarray, radius: int) -> np.ndarray:
        mask = local_mask > 127
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        return (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & (~mask)

    def _bubble_telea_safe(self, crop: np.ndarray, local_mask: np.ndarray) -> bool:
        ring = self._mask_ring(local_mask, _BUBBLE_FASTPATH_RING)
        if int(np.count_nonzero(ring)) < 64:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        ring_gray = gray[ring]
        if ring_gray.size < 64 or float(ring_gray.std()) > _BUBBLE_FASTPATH_GRAY_STD_MAX:
            return False

        edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
        if float(edges[ring].mean()) > _BUBBLE_FASTPATH_EDGE_DENSITY_MAX:
            return False

        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
            if ring_ab.size and float(np.max(ring_ab.std(axis=0))) > _BUBBLE_FASTPATH_CHROMA_STD_MAX:
                return False

        return True

    def _allow_bubble_fast_fill(
        self,
        crop: np.ndarray,
        local_mask: np.ndarray,
    ) -> bool:
        """Hook for a candidate to reject cheap fill on risky backgrounds.

        The default keeps the validated FastInpainter behavior unchanged.  A
        subclass may return ``False`` after inspecting the already-authorized
        mask and page-space context; rejection falls through to the normal
        LaMa path and never changes destructive authority.
        """
        return True

    def _try_bubble_fast_fill(
        self,
        image: np.ndarray,
        box: BubbleBox,
        protected_regions: list[dict] | None,
    ) -> bool:
        if not _BUBBLE_FASTPATH_ENABLED or not self._bubble_candidate(box):
            return False

        h, w = image.shape[:2]
        x1 = max(0, int(box.x1) - _BUBBLE_FASTPATH_PAD)
        y1 = max(0, int(box.y1) - _BUBBLE_FASTPATH_PAD)
        x2 = min(w, int(box.x2) + _BUBBLE_FASTPATH_PAD)
        y2 = min(h, int(box.y2) + _BUBBLE_FASTPATH_PAD)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return False

        crop_box = (x1, y1, x2, y2)
        crop = image[y1:y2, x1:x2]
        local_box = self._local_box(box, x1, y1)
        local_mask = build_mask((y2 - y1, x2 - x1), [local_box], crop)
        local_mask = self._subtract_protected_regions(
            local_mask,
            crop_box,
            protected_regions,
        )
        mask_bool = local_mask > 127
        mask_pixels = int(np.count_nonzero(mask_bool))
        if mask_pixels <= 0:
            return False
        if not self._allow_bubble_fast_fill(crop, local_mask):
            return False

        fill_color = self._smart_fill_color(crop, local_mask)
        if fill_color is not None:
            painted = crop.copy()
            painted[mask_bool] = fill_color
            self._metric_add("smart_fill_regions")
            self._metric_add("bubble_fast_fill_solid_regions")
        elif self._bubble_telea_safe(crop, local_mask):
            # Telea is intentionally restricted to low-texture verified bubble
            # interiors. Only authorized mask pixels are copied back, so its
            # neighbourhood sampling cannot alter surrounding artwork.
            painted = cv2.inpaint(
                crop,
                local_mask,
                float(_BUBBLE_FASTPATH_TELEA_RADIUS),
                cv2.INPAINT_TELEA,
            )
            self._metric_add("bubble_fast_fill_telea_regions")
        else:
            return False

        target = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = np.where(mask_bool[:, :, None], painted, target)
        self._metric_add("bubble_fast_fill_regions")
        self._metric_add("bubble_fast_fill_pixels", mask_pixels)
        return True

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
        remaining: list[BubbleBox] = []
        for box in boxes:
            if self._try_bubble_fast_fill(result, box, protected_regions):
                continue
            remaining.append(box)

        if not remaining:
            self._metrics_local.value["clusters"] = 0
            return result

        h, w = result.shape[:2]
        raw_clusters = self._cluster_boxes(remaining)
        clusters: list[list[BubbleBox]] = []
        for cluster in raw_clusters:
            parts = self._split_oversized_cluster_area(cluster, w, h)
            if len(parts) > 1:
                self._metric_add("split_clusters", len(parts) - 1)
            clusters.extend(parts)
        self._metrics_local.value["clusters"] = len(clusters)

        for cluster in clusters:
            x1 = min(b.x1 for b in cluster)
            y1 = min(b.y1 for b in cluster)
            x2 = max(b.x2 for b in cluster)
            y2 = max(b.y2 for b in cluster)
            crop_box = self._compute_crop_region(x1, y1, x2, y2, w, h)
            cx1, cy1, cx2, cy2 = crop_box
            crop_img = result[cy1:cy2, cx1:cx2]
            local_boxes = [self._local_box(box, cx1, cy1) for box in cluster]
            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)
            local_mask = self._subtract_protected_regions(
                local_mask,
                crop_box,
                protected_regions,
            )
            result = self._smart_paint_region(result, local_mask, crop_box)

        return result

    @staticmethod
    def _tight_roi(
        local_mask: np.ndarray,
        crop_w: int,
        crop_h: int,
    ) -> tuple[int, int, int, int] | None:
        ys, xs = np.nonzero(local_mask > 127)
        if xs.size == 0 or ys.size == 0:
            return None

        mx1, my1 = int(xs.min()), int(ys.min())
        mx2, my2 = int(xs.max()) + 1, int(ys.max()) + 1
        mask_w = max(1, mx2 - mx1)
        mask_h = max(1, my2 - my1)
        context = int(
            round(max(mask_w, mask_h) * _ROI_CONTEXT_SCALE)
        )
        context = max(_ROI_CONTEXT_MIN, min(_ROI_CONTEXT_MAX, context))
        return (
            max(0, mx1 - context),
            max(0, my1 - context),
            min(crop_w, mx2 + context),
            min(crop_h, my2 + context),
        )

    def _dynamic_roi_region(
        self,
        crop: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int, int] | None:
        """Return the validated dynamic ROI without running the model.

        E6 uses this helper to prepare several independent ROI inputs for one
        batch.  Keeping the calculation here prevents the batch candidate from
        silently discarding the ROI optimization already measured for E5.
        """
        if not _ROI_LAMA_ENABLED or not self.dynamic_lama:
            return None
        crop_h, crop_w = crop.shape[:2]
        source_pixels = max(1, int(crop_h * crop_w))
        roi = self._tight_roi(local_mask, crop_w, crop_h)
        if roi is None:
            return None
        rx1, ry1, rx2, ry2 = roi
        roi_pixels = max(1, int((rx2 - rx1) * (ry2 - ry1)))
        savings = 1.0 - (roi_pixels / float(source_pixels))
        if savings < _ROI_MIN_SAVINGS:
            return None
        cx1, cy1, _, _ = (int(v) for v in crop_box)
        global_box = (
            cx1 + rx1,
            cy1 + ry1,
            cx1 + rx2,
            cy1 + ry2,
        )
        return (
            crop[ry1:ry2, rx1:rx2],
            local_mask[ry1:ry2, rx1:rx2],
            global_box,
            source_pixels,
            roi_pixels,
        )

    def _lama_fill(
        self,
        image: np.ndarray,
        crop: np.ndarray,
        local_mask: np.ndarray,
        crop_box: tuple,
        feather: bool = False,
    ) -> np.ndarray:
        if not _ROI_LAMA_ENABLED:
            return super()._lama_fill(
                image,
                crop,
                local_mask,
                crop_box,
                feather=feather,
            )

        # Only the dynamic model benefits from arbitrary native ROI dimensions.
        # Fixed LaMa retains its validated 512x512 production path unchanged.
        self._ensure_session()
        if not self.dynamic_lama:
            return super()._lama_fill(
                image,
                crop,
                local_mask,
                crop_box,
                feather=feather,
            )

        region = self._dynamic_roi_region(crop, local_mask, crop_box)
        if region is None:
            return super()._lama_fill(
                image,
                crop,
                local_mask,
                crop_box,
                feather=feather,
            )

        roi_crop, roi_mask, global_box, source_pixels, roi_pixels = region
        self._metric_add("roi_lama_regions")
        self._metric_add("roi_lama_source_pixels", source_pixels)
        self._metric_add("roi_lama_input_pixels", roi_pixels)
        self._metric_add("roi_lama_saved_pixels", source_pixels - roi_pixels)
        return super()._lama_fill(
            image,
            roi_crop,
            roi_mask,
            global_box,
            feather=feather,
        )
