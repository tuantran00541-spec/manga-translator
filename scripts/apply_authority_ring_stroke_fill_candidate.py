from __future__ import annotations

import subprocess
from pathlib import Path

BASE = "d7132519c11bbb2b4b4695c3ec363d2cbed12ae3"


def restore(path: str) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{BASE}:{path}"],
        text=True,
    )
    Path(path).write_text(content, encoding="utf-8")
    return content


path = Path("app/inpaint/fast_lama_inpainter.py")
text = restore(str(path))

knob_marker = '''_BUBBLE_FASTPATH_OVERLAP_RATIO = _env_float(\n    "MANGA_BUBBLE_FASTPATH_OVERLAP_RATIO", 0.80, 0.50, 1.0\n)\n\n_ROI_LAMA_ENABLED = _env_bool("MANGA_DYNAMIC_LAMA_TIGHT_ROI_ENABLED", True)\n'''
knobs = '''_BUBBLE_FASTPATH_OVERLAP_RATIO = _env_float(\n    "MANGA_BUBBLE_FASTPATH_OVERLAP_RATIO", 0.80, 0.50, 1.0\n)\n\n# Dense segmenter masks are erase authority, not necessarily glyph support.\n# On isolated smooth speech regions we derive a tighter stroke mask, but we\n# reconstruct its pixels from the *outside of the full authority* rather than\n# feeding the stroke silhouette to LaMa. This avoids the white-glyph ghosting\n# observed when LaMa is asked to inpaint only the foreground strokes.\n_STROKE_REFINE_ENABLED = _env_bool("MANGA_STROKE_REFINE_ENABLED", True)\n_STROKE_REFINE_DENSE_FRACTION_MIN = _env_float(\n    "MANGA_STROKE_REFINE_DENSE_FRACTION_MIN", 0.55, 0.25, 0.95\n)\n_STROKE_REFINE_RING_STD_MAX = _env_float(\n    "MANGA_STROKE_REFINE_RING_STD_MAX", 18.0, 2.0, 48.0\n)\n_STROKE_REFINE_MIN_CONTRAST = _env_float(\n    "MANGA_STROKE_REFINE_MIN_CONTRAST", 36.0, 8.0, 96.0\n)\n_STROKE_REFINE_CORE_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_CORE_DELTA_MIN", 30.0, 8.0, 96.0\n)\n_STROKE_REFINE_WEAK_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_WEAK_DELTA_MIN", 14.0, 4.0, 64.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n\n_ROI_LAMA_ENABLED = _env_bool("MANGA_DYNAMIC_LAMA_TIGHT_ROI_ENABLED", True)\n'''
if text.count(knob_marker) != 1:
    raise RuntimeError("authority-ring knob marker mismatch")
text = text.replace(knob_marker, knobs, 1)

metric_marker = '''                "bubble_fast_fill_overlap_skips": 0,\n                "bubble_fast_fill_pixels": 0,\n                "roi_lama_regions": 0,\n'''
metrics = '''                "bubble_fast_fill_overlap_skips": 0,\n                "bubble_fast_fill_pixels": 0,\n                "stroke_refined_regions": 0,\n                "stroke_refined_authority_pixels": 0,\n                "stroke_refined_model_pixels": 0,\n                "stroke_authority_gradient_regions": 0,\n                "stroke_authority_gradient_pixels": 0,\n                "roi_lama_regions": 0,\n'''
if text.count(metric_marker) != 1:
    raise RuntimeError("authority-ring metric marker mismatch")
text = text.replace(metric_marker, metrics, 1)

method_marker = '''    def _try_bubble_fast_fill(\n        self,\n        image: np.ndarray,\n        box: BubbleBox,\n        protected_regions: list[dict] | None,\n    ) -> bool:\n'''
methods = r'''    def _refine_dense_smooth_stroke_mask(
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

        fringe = cv2.dilate(keep, np.ones((3, 3), dtype=np.uint8), iterations=1)
        refined = np.where(authority, fringe, 0).astype(np.uint8)
        refined_pixels = int(np.count_nonzero(refined))
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
'''
if text.count(method_marker) != 1:
    raise RuntimeError("authority-ring method marker mismatch")
text = text.replace(method_marker, methods, 1)

loop_old = '''        result = image.copy()\n        remaining: list[BubbleBox] = []\n        for box in boxes:\n            if self._bubble_candidate(box) and self._strong_authority_overlap(box, boxes):\n                self._metric_add("bubble_fast_fill_overlap_skips")\n                remaining.append(box)\n                continue\n            if self._try_bubble_fast_fill(result, box, protected_regions):\n                continue\n            remaining.append(box)\n'''
loop_new = '''        result = image.copy()\n        remaining: list[BubbleBox] = []\n        for box in boxes:\n            # Strongly overlapping authorities are ambiguous duplicate evidence.\n            # Keep their original masks together; never shrink either one.\n            if self._bubble_candidate(box) and self._strong_authority_overlap(box, boxes):\n                self._metric_add("bubble_fast_fill_overlap_skips")\n                remaining.append(box)\n                continue\n            if self._try_stroke_authority_fill(result, box, protected_regions):\n                continue\n            if self._try_bubble_fast_fill(result, box, protected_regions):\n                continue\n            remaining.append(box)\n'''
if text.count(loop_old) != 1:
    raise RuntimeError("authority-ring loop marker mismatch")
text = text.replace(loop_old, loop_new, 1)
path.write_text(text, encoding="utf-8")


# Restore the focused test file before adding the authority-ring regressions so
# the rejected stroke->LaMa fallback experiment cannot survive accidentally.
test_path = Path("tests/test_fast_inpaint_paths.py")
test_text = restore(str(test_path))
if "test_authority_ring_reconstruction_erases_dense_smooth_text" in test_text:
    raise RuntimeError("authority-ring tests already exist")
test_text += r'''


def test_authority_ring_reconstruction_erases_dense_smooth_text():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(218 + xx * 0.035 + yy * 0.020, 0, 255)
    background[..., 1] = np.clip(222 + xx * 0.032 + yy * 0.018, 0, 255)
    background[..., 2] = np.clip(228 + xx * 0.028 + yy * 0.015, 0, 255)
    image = background.copy()

    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, dense)
    image[75:86, 85:235] = 5
    image[105:117, 100:220] = 8
    image[134:146, 125:200] = 3

    authority = build_mask(image.shape[:2], [box], image) > 127
    inpainter = FastInpainter()
    output = inpainter.inpaint(image.copy(), [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert metrics["stroke_refined_regions"] == 1
    assert metrics["stroke_authority_gradient_regions"] == 1
    assert metrics["stroke_refined_model_pixels"] < metrics["stroke_refined_authority_pixels"] // 2
    assert metrics["lama_model_runs"] == 0
    assert not np.any(changed & (~authority))
    assert int(np.count_nonzero(changed)) < int(np.count_nonzero(authority)) // 2

    glyphs = np.zeros((h, w), dtype=bool)
    glyphs[75:86, 85:235] = True
    glyphs[105:117, 100:220] = True
    glyphs[134:146, 125:200] = True
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[glyphs].mean()
    )
    assert mae < 8.0


def test_authority_ring_reconstruction_rejects_textured_region():
    rng = np.random.default_rng(123)
    image = rng.integers(40, 220, size=(220, 320, 3), dtype=np.uint8)
    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, dense)

    inpainter = FastInpainter()
    inpainter._begin_metrics()
    assert inpainter._try_stroke_authority_fill(image.copy(), box, None) is False
    assert inpainter.last_metrics()["stroke_authority_gradient_regions"] == 0


def test_overlapping_authorities_never_enter_stroke_refinement():
    image = np.full((220, 320, 3), 230, dtype=np.uint8)
    mask_a = np.full((100, 180), 255, dtype=np.uint8)
    mask_b = np.full((80, 160), 255, dtype=np.uint8)
    a = _speech_box(60, 55, 240, 155, mask_a)
    b = _speech_box(70, 65, 230, 145, mask_b)

    inpainter = FastInpainter()
    calls = []
    inpainter._try_stroke_authority_fill = (
        lambda image_arg, box_arg, protected: calls.append(box_arg) or False
    )
    inpainter._smart_paint_region = (
        lambda image_arg, local_mask, crop_box: image_arg
    )
    inpainter.inpaint(image.copy(), [a, b])
    metrics = inpainter.last_metrics()

    assert calls == []
    assert metrics["bubble_fast_fill_overlap_skips"] == 2
    assert metrics["stroke_refined_regions"] == 0
'''
test_path.write_text(test_text, encoding="utf-8")
