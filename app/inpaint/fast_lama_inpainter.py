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
# Telea is retained only as an opt-in benchmark fallback. Real chapter auditing
# showed large polygon/facet artifacts even when its low-texture ring gate passed.
_BUBBLE_FASTPATH_TELEA_ENABLED = _env_bool(
    "MANGA_BUBBLE_FASTPATH_TELEA_ENABLED", False
)
_BUBBLE_FASTPATH_TELEA_RADIUS = _env_int(
    "MANGA_BUBBLE_FASTPATH_TELEA_RADIUS", 3, 1, 9
)
_BUBBLE_FASTPATH_GRADIENT_ENABLED = _env_bool(
    "MANGA_BUBBLE_FASTPATH_GRADIENT_ENABLED", True
)
_BUBBLE_FASTPATH_GRADIENT_MAX_MASK_FRACTION = _env_float(
    "MANGA_BUBBLE_FASTPATH_GRADIENT_MAX_MASK_FRACTION", 0.76, 0.20, 0.95
)
_BUBBLE_FASTPATH_GRADIENT_RMSE_MAX = _env_float(
    "MANGA_BUBBLE_FASTPATH_GRADIENT_RMSE_MAX", 12.0, 2.0, 40.0
)
_BUBBLE_FASTPATH_OVERLAP_RATIO = _env_float(
    "MANGA_BUBBLE_FASTPATH_OVERLAP_RATIO", 0.80, 0.50, 1.0
)

# Dense segmenter masks are erase authority, not necessarily glyph support.
# On isolated smooth speech regions we derive a tighter stroke mask, but we
# reconstruct its pixels from the *outside of the full authority* rather than
# feeding the stroke silhouette to LaMa. This avoids the white-glyph ghosting
# observed when LaMa is asked to inpaint only the foreground strokes.
_STROKE_REFINE_ENABLED = _env_bool("MANGA_STROKE_REFINE_ENABLED", True)
_STROKE_REFINE_DENSE_FRACTION_MIN = _env_float(
    "MANGA_STROKE_REFINE_DENSE_FRACTION_MIN", 0.55, 0.25, 0.95
)
_STROKE_REFINE_RING_STD_MAX = _env_float(
    "MANGA_STROKE_REFINE_RING_STD_MAX", 18.0, 2.0, 48.0
)
_STROKE_REFINE_MIN_CONTRAST = _env_float(
    "MANGA_STROKE_REFINE_MIN_CONTRAST", 36.0, 8.0, 96.0
)
_STROKE_REFINE_CORE_DELTA_MIN = _env_float(
    "MANGA_STROKE_REFINE_CORE_DELTA_MIN", 30.0, 8.0, 96.0
)
_STROKE_REFINE_WEAK_DELTA_MIN = _env_float(
    "MANGA_STROKE_REFINE_WEAK_DELTA_MIN", 14.0, 4.0, 64.0
)
# Recover outlined/anti-aliased glyph halos from either side of the local
# background luminance. Growth is connectivity-limited, spatially bounded,
# clipped to erase authority, and still subject to the authority-fraction gate.
_STROKE_REFINE_HALO_RADIUS = _env_int(
    "MANGA_STROKE_REFINE_HALO_RADIUS", 2, 1, 4
)
_STROKE_REFINE_HALO_DELTA_MIN = _env_float(
    "MANGA_STROKE_REFINE_HALO_DELTA_MIN", 8.0, 2.0, 48.0
)
_STROKE_REFINE_HALO_STD_SCALE = _env_float(
    "MANGA_STROKE_REFINE_HALO_STD_SCALE", 1.25, 0.5, 4.0
)
# A global median misses pale outlines on gradient bubbles. Fit the local smooth
# background from the authority ring and recover only residual pixels connected
# to the already-confirmed glyph support. This remains bounded and authority-only.
_STROKE_REFINE_SURFACE_HALO_RADIUS = _env_int(
    "MANGA_STROKE_REFINE_SURFACE_HALO_RADIUS", 6, 1, 12
)
_STROKE_REFINE_SURFACE_DELTA_MIN = _env_float(
    "MANGA_STROKE_REFINE_SURFACE_DELTA_MIN", 2.5, 1.0, 16.0
)
_STROKE_REFINE_SURFACE_RMSE_SCALE = _env_float(
    "MANGA_STROKE_REFINE_SURFACE_RMSE_SCALE", 2.5, 1.0, 8.0
)
_STROKE_REFINE_SURFACE_RMSE_MAX = _env_float(
    "MANGA_STROKE_REFINE_SURFACE_RMSE_MAX", 4.0, 1.0, 12.0
)
_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(
    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90
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
    """ CPU-oriented inpainting candidate layered on top of production LaMa. """

    def _begin_metrics(self, *, boxes: int = 0) -> None:
        super()._begin_metrics(boxes=boxes)
        self._metrics_local.value.update(
            {
                "bubble_fast_fill_regions": 0,
                "bubble_fast_fill_solid_regions": 0,
                "bubble_fast_fill_gradient_regions": 0,
                "bubble_fast_fill_telea_regions": 0,
                "bubble_fast_fill_overlap_skips": 0,
                "bubble_fast_fill_pixels": 0,
                "stroke_refined_regions": 0,
                "stroke_refined_authority_pixels": 0,
                "stroke_refined_model_pixels": 0,
                "stroke_refined_residual_rejects": 0,
                "stroke_authority_gradient_regions": 0,
                "stroke_authority_gradient_pixels": 0,
                "roi_lama_regions": 0,
                "roi_lama_source_pixels": 0,
                "roi_lama_input_pixels": 0,
                "roi_lama_saved_pixels": 0,
            }
        )

    @staticmethod
    def _bubble_candidate(box: BubbleBox) -> bool:
        return bool(
            box.semantic_type in {"speech_bubble", "free_text"}
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

    @staticmethod
    def _quadratic_design(
        xs: np.ndarray,
        ys: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Return a numerically stable degree-2 spatial design matrix."""
        x = (xs.astype(np.float32) + 0.5) / max(1.0, float(width))
        y = (ys.astype(np.float32) + 0.5) / max(1.0, float(height))
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        return np.stack(
            (
                np.ones_like(x),
                x,
                y,
                x * x,
                x * y,
                y * y,
            ),
            axis=1,
        )

    @staticmethod
    def _strong_authority_overlap(box: BubbleBox, boxes: list[BubbleBox]) -> bool:
        """ Detect duplicate destructive authorities before per-box fast fill. """
        area = max(0, int(box.x2 - box.x1)) * max(0, int(box.y2 - box.y1))
        if area <= 0:
            return False
        for other in boxes:
            if other is box or not bool(other.safe_to_inpaint):
                continue
            other_area = max(0, int(other.x2 - other.x1)) * max(
                0, int(other.y2 - other.y1)
            )
            if other_area <= 0:
                continue
            ix1 = max(int(box.x1), int(other.x1))
            iy1 = max(int(box.y1), int(other.y1))
            ix2 = min(int(box.x2), int(other.x2))
            iy2 = min(int(box.y2), int(other.y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            intersection = (ix2 - ix1) * (iy2 - iy1)
            if intersection / float(min(area, other_area)) >= _BUBBLE_FASTPATH_OVERLAP_RATIO:
                return True
        return False

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

    def _bubble_gradient_fill(
        self,
        crop: np.ndarray,
        local_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Reconstruct a smooth bubble background from a clean surrounding ring.

        The fit is deliberately conservative: extremely dense masks are likely
        stylized SFX or a bad semantic classification rather than glyph support;
        edge-heavy/chromatically complex rings fall back to LaMa. Dark outlines
        or residual glyph pixels in the ring are rejected by robust residual
        trimming before the quadratic surface is accepted.
        """
        if not _BUBBLE_FASTPATH_GRADIENT_ENABLED:
            return None
        mask = local_mask > 127
        ys, xs = np.nonzero(mask)
        if xs.size < 16:
            return None

        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        tight_area = max(1, (bx2 - bx1) * (by2 - by1))
        occupancy = float(xs.size) / float(tight_area)
        if occupancy > _BUBBLE_FASTPATH_GRADIENT_MAX_MASK_FRACTION:
            return None

        ring = self._mask_ring(local_mask, _BUBBLE_FASTPATH_RING)
        if int(np.count_nonzero(ring)) < 96:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
        # Remove edge-adjacent samples from the fit rather than letting a bubble
        # outline or missed glyph fringe bend the reconstructed background.
        edge_kernel = np.ones((3, 3), dtype=np.uint8)
        edge_margin = cv2.dilate(edges.astype(np.uint8), edge_kernel) > 0
        clean_ring = ring & (~edge_margin)
        if int(np.count_nonzero(clean_ring)) < 96:
            return None
        if float(edges[ring].mean()) > max(0.035, _BUBBLE_FASTPATH_EDGE_DENSITY_MAX * 2.0):
            return None

        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[clean_ring, 1:3].astype(np.float32, copy=False)
            if ring_ab.size and float(np.max(ring_ab.std(axis=0))) > max(
                18.0, _BUBBLE_FASTPATH_CHROMA_STD_MAX * 1.5
            ):
                return None

        sample_y, sample_x = np.nonzero(clean_ring)
        design = self._quadratic_design(sample_x, sample_y, crop.shape[1], crop.shape[0])
        values = crop[sample_y, sample_x].astype(np.float32)
        if values.ndim == 1:
            values = values[:, None]

        keep = np.ones(len(sample_x), dtype=bool)
        coeff = None
        for _ in range(3):
            if int(np.count_nonzero(keep)) < 64:
                return None
            coeff, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
            predicted = design @ coeff
            residual = np.sqrt(np.mean((predicted - values) ** 2, axis=1))
            active = residual[keep]
            median = float(np.median(active))
            mad = float(np.median(np.abs(active - median)))
            limit = median + max(2.0, 3.0 * 1.4826 * mad)
            next_keep = residual <= limit
            if int(np.count_nonzero(next_keep)) < 64:
                break
            if np.array_equal(next_keep, keep):
                keep = next_keep
                break
            keep = next_keep

        if coeff is None or int(np.count_nonzero(keep)) < 64:
            return None
        fitted = design[keep] @ coeff
        rmse = float(np.sqrt(np.mean((fitted - values[keep]) ** 2)))
        if not np.isfinite(rmse) or rmse > _BUBBLE_FASTPATH_GRADIENT_RMSE_MAX:
            return None

        target_y, target_x = np.nonzero(mask)
        target_design = self._quadratic_design(
            target_x, target_y, crop.shape[1], crop.shape[0]
        )
        target_values = np.clip(target_design @ coeff, 0.0, 255.0).astype(np.uint8)
        painted = crop.copy()
        if painted.ndim == 2:
            painted[target_y, target_x] = target_values[:, 0]
        else:
            painted[target_y, target_x] = target_values
        return painted

    def _refine_dense_smooth_stroke_mask(
        self,
        crop: np.ndarray,
        authority_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Derive conservative glyph support from a dense erase authority."""
        if not _STROKE_REFINE_ENABLED:
            return None
        authority = authority_mask > 127
        ys, xs = np.nonzero(authority)
        if xs.size < 64:
            return None

        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        tight_area = max(1, (bx2 - bx1) * (by2 - by1))
        occupancy = float(xs.size) / float(tight_area)
        if occupancy < _STROKE_REFINE_DENSE_FRACTION_MIN:
            return None

        ring = self._mask_ring(authority_mask, _BUBBLE_FASTPATH_RING)
        if int(np.count_nonzero(ring)) < 96:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray_f = gray.astype(np.float32, copy=False)
        ring_gray = gray_f[ring]
        ring_std = float(ring_gray.std())
        if not np.isfinite(ring_std) or ring_std > _STROKE_REFINE_RING_STD_MAX:
            return None

        edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
        if float(edges[ring].mean()) > 0.06:
            return None
        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
            if ring_ab.size and float(np.max(ring_ab.std(axis=0))) > 18.0:
                return None

        background = float(np.median(ring_gray))
        inside = gray_f[authority]
        low = float(np.percentile(inside, 10.0))
        high = float(np.percentile(inside, 90.0))
        dark_contrast = background - low
        light_contrast = high - background
        if max(dark_contrast, light_contrast) < _STROKE_REFINE_MIN_CONTRAST:
            return None

        core_delta = max(_STROKE_REFINE_CORE_DELTA_MIN, ring_std * 3.0)
        weak_delta = max(_STROKE_REFINE_WEAK_DELTA_MIN, ring_std * 1.5)
        if dark_contrast >= light_contrast:
            strong = authority & (gray_f <= background - core_delta)
            weak = authority & (gray_f <= background - weak_delta)
        else:
            strong = authority & (gray_f >= background + core_delta)
            weak = authority & (gray_f >= background + weak_delta)

        if int(np.count_nonzero(strong)) < 24:
            return None
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            weak.astype(np.uint8), 8
        )
        keep = np.zeros_like(authority_mask, dtype=np.uint8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) < 2:
                continue
            component = labels == label
            if np.any(strong & component):
                keep[component] = 255

        authority_pixels = int(np.count_nonzero(authority))
        model_pixels = int(np.count_nonzero(keep))
        if authority_pixels <= 0 or model_pixels < 24:
            return None
        if model_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:
            return None

        # The dominant-polarity component pass above finds the glyph core, but
        # manga/manhwa lettering often has an opposite-polarity outline (e.g.
        # black glyph + white stroke) plus a low-contrast anti-alias fringe.
        # Grow only through contrast-bearing pixels connected to the confirmed
        # core; using absolute contrast recovers both bright and dark halos.
        halo_delta = max(
            _STROKE_REFINE_HALO_DELTA_MIN,
            ring_std * _STROKE_REFINE_HALO_STD_SCALE,
        )
        halo_candidate = authority & (np.abs(gray_f - background) >= halo_delta)
        grown = keep > 127
        halo_kernel = np.ones((3, 3), dtype=np.uint8)
        for _ in range(_STROKE_REFINE_HALO_RADIUS):
            adjacent = cv2.dilate(grown.astype(np.uint8), halo_kernel, iterations=1) > 0
            next_grown = grown | (halo_candidate & adjacent)
            if np.array_equal(next_grown, grown):
                break
            grown = next_grown

        # Refine the low-contrast tail against a local quadratic background, not
        # the global ring median. This is the important gradient-background case:
        # a white anti-alias pixel may be only +3 locally while being almost equal
        # to the median sampled elsewhere. Fit robustly from clean ring samples,
        # then allow bounded connectivity growth only through pixels whose color
        # residual exceeds the ring noise floor.
        edge_margin = cv2.dilate(
            edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
        ) > 0
        surface_ring = ring & (~edge_margin)
        if int(np.count_nonzero(surface_ring)) >= 96:
            sample_y, sample_x = np.nonzero(surface_ring)
            design = self._quadratic_design(
                sample_x, sample_y, crop.shape[1], crop.shape[0]
            )
            values = crop[sample_y, sample_x].astype(np.float32)
            if values.ndim == 1:
                values = values[:, None]

            fit_keep = np.ones(len(sample_x), dtype=bool)
            coeff = None
            for _ in range(3):
                if int(np.count_nonzero(fit_keep)) < 64:
                    coeff = None
                    break
                coeff, *_ = np.linalg.lstsq(
                    design[fit_keep], values[fit_keep], rcond=None
                )
                predicted = design @ coeff
                residual = np.sqrt(np.mean((predicted - values) ** 2, axis=1))
                active = residual[fit_keep]
                median_residual = float(np.median(active))
                mad = float(np.median(np.abs(active - median_residual)))
                limit = median_residual + max(1.5, 3.0 * 1.4826 * mad)
                next_keep = residual <= limit
                if int(np.count_nonzero(next_keep)) < 64:
                    break
                if np.array_equal(next_keep, fit_keep):
                    fit_keep = next_keep
                    break
                fit_keep = next_keep

            if coeff is not None and int(np.count_nonzero(fit_keep)) >= 64:
                fitted = design[fit_keep] @ coeff
                fit_rmse = float(
                    np.sqrt(np.mean((fitted - values[fit_keep]) ** 2))
                )
                if np.isfinite(fit_rmse) and fit_rmse <= _STROKE_REFINE_SURFACE_RMSE_MAX:
                    target_y, target_x = np.nonzero(authority)
                    target_design = self._quadratic_design(
                        target_x, target_y, crop.shape[1], crop.shape[0]
                    )
                    target_expected = target_design @ coeff
                    target_actual = crop[target_y, target_x].astype(np.float32)
                    if target_actual.ndim == 1:
                        target_actual = target_actual[:, None]
                    target_residual = np.sqrt(
                        np.mean((target_actual - target_expected) ** 2, axis=1)
                    )
                    surface_delta = max(
                        _STROKE_REFINE_SURFACE_DELTA_MIN,
                        fit_rmse * _STROKE_REFINE_SURFACE_RMSE_SCALE,
                    )
                    surface_candidate = np.zeros_like(authority, dtype=bool)
                    surface_candidate[target_y, target_x] = (
                        target_residual >= surface_delta
                    )
                    for _ in range(_STROKE_REFINE_SURFACE_HALO_RADIUS):
                        adjacent = (
                            cv2.dilate(
                                grown.astype(np.uint8), halo_kernel, iterations=1
                            )
                            > 0
                        )
                        next_grown = grown | (surface_candidate & adjacent)
                        if np.array_equal(next_grown, grown):
                            break
                        grown = next_grown

        # One final one-pixel support fringe catches outer anti-alias values that
        # are intentionally close to the fitted background. There is no unbounded
        # morphology: the candidate can advance at most HALO_RADIUS pixels from
        # verified support, then one final anti-alias pixel, and never outside the
        # detector authority. Emit a canonical 0/255 mask because downstream mask
        # consumers use the project-wide >127 foreground convention.
        fringe = cv2.dilate(
            grown.astype(np.uint8) * 255,
            halo_kernel,
            iterations=1,
        )
        refined = np.where(authority, fringe, 0).astype(np.uint8)
        refined_pixels = int(np.count_nonzero(refined > 127))
        if refined_pixels <= 0:
            return None
        if refined_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:
            return None
        return refined

    def _bubble_gradient_fill_from_authority_ring(
        self,
        crop: np.ndarray,
        model_mask: np.ndarray,
        authority_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Fit smooth background outside authority; paint only model strokes."""
        if not _BUBBLE_FASTPATH_GRADIENT_ENABLED:
            return None
        target = model_mask > 127
        if int(np.count_nonzero(target)) < 16:
            return None

        ring = self._mask_ring(authority_mask, _BUBBLE_FASTPATH_RING)
        if int(np.count_nonzero(ring)) < 96:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        ring_gray = gray[ring].astype(np.float32, copy=False)
        if ring_gray.size < 96:
            return None
        ring_std = float(ring_gray.std())
        if not np.isfinite(ring_std) or ring_std > _STROKE_REFINE_RING_STD_MAX:
            return None

        edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
        edge_margin = cv2.dilate(
            edges.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
        ) > 0
        clean_ring = ring & (~edge_margin)
        if int(np.count_nonzero(clean_ring)) < 96:
            return None
        if float(edges[ring].mean()) > max(
            0.035, _BUBBLE_FASTPATH_EDGE_DENSITY_MAX * 2.0
        ):
            return None

        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[clean_ring, 1:3].astype(np.float32, copy=False)
            if ring_ab.size and float(np.max(ring_ab.std(axis=0))) > max(
                18.0, _BUBBLE_FASTPATH_CHROMA_STD_MAX * 1.5
            ):
                return None

        sample_y, sample_x = np.nonzero(clean_ring)
        design = self._quadratic_design(
            sample_x, sample_y, crop.shape[1], crop.shape[0]
        )
        values = crop[sample_y, sample_x].astype(np.float32)
        if values.ndim == 1:
            values = values[:, None]

        keep = np.ones(len(sample_x), dtype=bool)
        coeff = None
        for _ in range(3):
            if int(np.count_nonzero(keep)) < 64:
                return None
            coeff, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
            predicted = design @ coeff
            residual = np.sqrt(np.mean((predicted - values) ** 2, axis=1))
            active = residual[keep]
            median = float(np.median(active))
            mad = float(np.median(np.abs(active - median)))
            limit = median + max(2.0, 3.0 * 1.4826 * mad)
            next_keep = residual <= limit
            if int(np.count_nonzero(next_keep)) < 64:
                break
            if np.array_equal(next_keep, keep):
                keep = next_keep
                break
            keep = next_keep

        if coeff is None or int(np.count_nonzero(keep)) < 64:
            return None
        fitted = design[keep] @ coeff
        rmse = float(np.sqrt(np.mean((fitted - values[keep]) ** 2)))
        if not np.isfinite(rmse) or rmse > _BUBBLE_FASTPATH_GRADIENT_RMSE_MAX:
            return None

        target_y, target_x = np.nonzero(target)
        target_design = self._quadratic_design(
            target_x, target_y, crop.shape[1], crop.shape[0]
        )
        target_values = np.clip(target_design @ coeff, 0.0, 255.0).astype(np.uint8)
        painted = crop.copy()
        if painted.ndim == 2:
            painted[target_y, target_x] = target_values[:, 0]
        else:
            painted[target_y, target_x] = target_values
        return painted

    def _stroke_refine_has_unpainted_residue(
        self,
        crop: np.ndarray,
        model_mask: np.ndarray,
        authority_mask: np.ndarray,
    ) -> bool:
        """Reject a stroke-only fast fill when adjacent glyph evidence remains.

        The detector authority is intentionally wider than the refined model mask.
        Most of that difference is clean bubble background and should stay untouched.
        A missed bright/dark outline, however, is both close to the confirmed glyph
        support and meaningfully different from the smooth surface reconstructed from
        outside the full authority. Such a region must fall through to the full
        authority path instead of being declared clean after a partial repaint.
        """
        authority = authority_mask > 127
        model = model_mask > 127
        if not np.any(authority) or not np.any(model):
            return False

        unpainted = authority & (~model)
        if not np.any(unpainted):
            return False

        halo_kernel = np.ones((3, 3), dtype=np.uint8)
        near_model = cv2.dilate(
            model.astype(np.uint8),
            halo_kernel,
            iterations=max(1, int(_STROKE_REFINE_HALO_RADIUS)),
        ) > 0
        probe = unpainted & near_model
        if int(np.count_nonzero(probe)) < 16:
            return False

        probe_mask = probe.astype(np.uint8) * 255
        expected = self._bubble_gradient_fill_from_authority_ring(
            crop,
            probe_mask,
            authority_mask,
        )
        if expected is None:
            return True

        actual_f = crop.astype(np.float32, copy=False)
        expected_f = expected.astype(np.float32, copy=False)
        delta = actual_f - expected_f
        if delta.ndim == 3:
            residual = np.sqrt(np.mean(delta * delta, axis=2))
        else:
            residual = np.abs(delta)

        residual_threshold = max(
            float(_STROKE_REFINE_HALO_DELTA_MIN),
            float(_STROKE_REFINE_SURFACE_DELTA_MIN) * 2.0,
        )
        suspicious = probe & (residual >= residual_threshold)
        if int(np.count_nonzero(suspicious)) < 2:
            return False

        count, _labels, stats, _ = cv2.connectedComponentsWithStats(
            suspicious.astype(np.uint8),
            8,
        )
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 2:
                return True
        return False


    def _try_stroke_authority_fill(
        self,
        image: np.ndarray,
        box: BubbleBox,
        protected_regions: list[dict] | None,
    ) -> bool:
        # Keep this path narrow. Free text and overlapping detector authorities
        # are intentionally left to the established clustered LaMa path.
        if (
            not _STROKE_REFINE_ENABLED
            or not self._bubble_candidate(box)
            or box.semantic_type != "speech_bubble"
            or box.source_role != "text_segmenter"
        ):
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
        authority_mask = build_mask(
            (y2 - y1, x2 - x1), [local_box], crop
        )
        authority_mask = self._subtract_protected_regions(
            authority_mask,
            crop_box,
            protected_regions,
        )
        authority_pixels = int(np.count_nonzero(authority_mask > 127))
        if authority_pixels <= 0:
            return False

        model_mask = self._refine_dense_smooth_stroke_mask(crop, authority_mask)
        if model_mask is None:
            return False
        model_pixels = int(np.count_nonzero(model_mask > 127))
        if model_pixels <= 0:
            return False

        painted = self._bubble_gradient_fill_from_authority_ring(
            crop,
            model_mask,
            authority_mask,
        )
        if painted is None:
            # Critical safety behavior: do not pass the stroke silhouette to
            # LaMa. The caller will continue with the original dense authority.
            return False
        if self._stroke_refine_has_unpainted_residue(
            crop,
            model_mask,
            authority_mask,
        ):
            self._metric_add("stroke_refined_residual_rejects")
            # A nearby high-contrast remainder is likely an outline/halo omitted
            # by the refined model mask. Keep the original authority intact and
            # let the caller use full-authority reconstruction or LaMa.
            return False

        target = image[y1:y2, x1:x2]
        model_bool = model_mask > 127
        image[y1:y2, x1:x2] = np.where(
            model_bool[:, :, None], painted, target
        )
        self._metric_add("stroke_refined_regions")
        self._metric_add("stroke_refined_authority_pixels", authority_pixels)
        self._metric_add("stroke_refined_model_pixels", model_pixels)
        self._metric_add("stroke_authority_gradient_regions")
        self._metric_add("stroke_authority_gradient_pixels", model_pixels)
        self._metric_add("bubble_fast_fill_regions")
        self._metric_add("bubble_fast_fill_gradient_regions")
        self._metric_add("bubble_fast_fill_pixels", model_pixels)
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

        fill_color = self._smart_fill_color(crop, local_mask)
        if fill_color is not None:
            painted = crop.copy()
            painted[mask_bool] = fill_color
            self._metric_add("smart_fill_regions")
            self._metric_add("bubble_fast_fill_solid_regions")
        else:
            painted = self._bubble_gradient_fill(crop, local_mask)
            if painted is not None:
                self._metric_add("bubble_fast_fill_gradient_regions")
            elif _BUBBLE_FASTPATH_TELEA_ENABLED and self._bubble_telea_safe(
                crop, local_mask
            ):
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
            # Strongly overlapping authorities are ambiguous duplicate evidence.
            # Keep their original masks together; never shrink either one.
            if self._bubble_candidate(box) and self._strong_authority_overlap(box, boxes):
                self._metric_add("bubble_fast_fill_overlap_skips")
                remaining.append(box)
                continue
            if self._try_stroke_authority_fill(result, box, protected_regions):
                continue
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

        crop_h, crop_w = crop.shape[:2]
        source_pixels = max(1, int(crop_h * crop_w))
        roi = self._tight_roi(local_mask, crop_w, crop_h)
        if roi is None:
            return image
        rx1, ry1, rx2, ry2 = roi
        roi_pixels = max(1, int((rx2 - rx1) * (ry2 - ry1)))
        savings = 1.0 - (roi_pixels / float(source_pixels))
        if savings < _ROI_MIN_SAVINGS:
            return super()._lama_fill(
                image,
                crop,
                local_mask,
                crop_box,
                feather=feather,
            )

        cx1, cy1, _, _ = (int(v) for v in crop_box)
        global_box = (
            cx1 + rx1,
            cy1 + ry1,
            cx1 + rx2,
            cy1 + ry2,
        )
        roi_crop = image[global_box[1]:global_box[3], global_box[0]:global_box[2]]
        roi_mask = local_mask[ry1:ry2, rx1:rx2]
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
