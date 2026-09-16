from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one marker, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


path = Path("app/inpaint/fast_lama_inpainter.py")
text = path.read_text(encoding="utf-8")

knob_marker = '''_BUBBLE_FASTPATH_OVERLAP_RATIO = _env_float(\n    "MANGA_BUBBLE_FASTPATH_OVERLAP_RATIO", 0.80, 0.50, 1.0\n)\n\n_ROI_LAMA_ENABLED = _env_bool("MANGA_DYNAMIC_LAMA_TIGHT_ROI_ENABLED", True)\n'''
knobs = '''_BUBBLE_FASTPATH_OVERLAP_RATIO = _env_float(\n    "MANGA_BUBBLE_FASTPATH_OVERLAP_RATIO", 0.80, 0.50, 1.0\n)\n\n# Dense detector masks are erase authority, not necessarily glyph support.\n# Refine them only on smooth, high-contrast text backgrounds and always keep\n# the model mask as a strict subset of the original authority.\n_STROKE_REFINE_ENABLED = _env_bool("MANGA_STROKE_REFINE_ENABLED", True)\n_STROKE_REFINE_DENSE_FRACTION_MIN = _env_float(\n    "MANGA_STROKE_REFINE_DENSE_FRACTION_MIN", 0.55, 0.25, 0.95\n)\n_STROKE_REFINE_RING_STD_MAX = _env_float(\n    "MANGA_STROKE_REFINE_RING_STD_MAX", 18.0, 2.0, 48.0\n)\n_STROKE_REFINE_MIN_CONTRAST = _env_float(\n    "MANGA_STROKE_REFINE_MIN_CONTRAST", 36.0, 8.0, 96.0\n)\n_STROKE_REFINE_CORE_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_CORE_DELTA_MIN", 30.0, 8.0, 96.0\n)\n_STROKE_REFINE_WEAK_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_WEAK_DELTA_MIN", 14.0, 4.0, 64.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n\n_ROI_LAMA_ENABLED = _env_bool("MANGA_DYNAMIC_LAMA_TIGHT_ROI_ENABLED", True)\n'''
if text.count(knob_marker) != 1:
    raise RuntimeError("stroke knob marker mismatch")
text = text.replace(knob_marker, knobs, 1)

metric_marker = '''                "bubble_fast_fill_overlap_skips": 0,\n                "bubble_fast_fill_pixels": 0,\n                "roi_lama_regions": 0,\n'''
metrics = '''                "bubble_fast_fill_overlap_skips": 0,\n                "bubble_fast_fill_pixels": 0,\n                "stroke_refined_regions": 0,\n                "stroke_refined_authority_pixels": 0,\n                "stroke_refined_model_pixels": 0,\n                "roi_lama_regions": 0,\n'''
if text.count(metric_marker) != 1:
    raise RuntimeError("stroke metric marker mismatch")
text = text.replace(metric_marker, metrics, 1)

method_marker = '''    def _try_bubble_fast_fill(\n        self,\n        image: np.ndarray,\n        box: BubbleBox,\n        protected_regions: list[dict] | None,\n    ) -> bool:\n'''
method = r'''    def _refine_dense_smooth_stroke_mask(
        self,
        crop: np.ndarray,
        authority_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Derive conservative glyph support from a dense erase-authority mask."""
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

        # Reject textured or colorful artwork. The refinement is for bubble-like
        # smooth backgrounds only; complex free text stays on the LaMa path.
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

        # Recover a tiny anti-alias fringe, clipped strictly to authority.
        fringe = cv2.dilate(keep, np.ones((3, 3), dtype=np.uint8), iterations=1)
        refined = np.where(authority, fringe, 0).astype(np.uint8)
        refined_pixels = int(np.count_nonzero(refined))
        if refined_pixels <= 0:
            return None
        if refined_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:
            return None
        return refined

    def _try_bubble_fast_fill(
        self,
        image: np.ndarray,
        box: BubbleBox,
        protected_regions: list[dict] | None,
    ) -> bool:
'''
if text.count(method_marker) != 1:
    raise RuntimeError("stroke method marker mismatch")
text = text.replace(method_marker, method, 1)

old_decision = '''        mask_bool = local_mask > 127\n        mask_pixels = int(np.count_nonzero(mask_bool))\n        if mask_pixels <= 0:\n            return False\n\n        fill_color = self._smart_fill_color(crop, local_mask)\n        if fill_color is not None:\n            painted = crop.copy()\n            painted[mask_bool] = fill_color\n            self._metric_add("smart_fill_regions")\n            self._metric_add("bubble_fast_fill_solid_regions")\n        else:\n            painted = self._bubble_gradient_fill(crop, local_mask)\n            if painted is not None:\n                self._metric_add("bubble_fast_fill_gradient_regions")\n            elif _BUBBLE_FASTPATH_TELEA_ENABLED and self._bubble_telea_safe(\n                crop, local_mask\n            ):\n                painted = cv2.inpaint(\n                    crop,\n                    local_mask,\n                    float(_BUBBLE_FASTPATH_TELEA_RADIUS),\n                    cv2.INPAINT_TELEA,\n                )\n                self._metric_add("bubble_fast_fill_telea_regions")\n            else:\n                return False\n'''
new_decision = '''        authority_pixels = int(np.count_nonzero(local_mask > 127))\n        if authority_pixels <= 0:\n            return False\n\n        refined = None\n        if box.source_role == "text_segmenter":\n            refined = self._refine_dense_smooth_stroke_mask(crop, local_mask)\n        stroke_refined = refined is not None\n        if stroke_refined:\n            local_mask = refined\n            self._metric_add("stroke_refined_regions")\n            self._metric_add("stroke_refined_authority_pixels", authority_pixels)\n            self._metric_add(\n                "stroke_refined_model_pixels", int(np.count_nonzero(local_mask > 127))\n            )\n\n        mask_bool = local_mask > 127\n        mask_pixels = int(np.count_nonzero(mask_bool))\n        if mask_pixels <= 0:\n            return False\n\n        painted = None\n        if stroke_refined:\n            # Preserve a smooth spatial gradient instead of flattening the whole\n            # lettering block. This path changes only the refined stroke pixels.\n            painted = self._bubble_gradient_fill(crop, local_mask)\n            if painted is not None:\n                self._metric_add("bubble_fast_fill_gradient_regions")\n\n        if painted is None:\n            fill_color = self._smart_fill_color(crop, local_mask)\n            if fill_color is not None:\n                painted = crop.copy()\n                painted[mask_bool] = fill_color\n                self._metric_add("smart_fill_regions")\n                self._metric_add("bubble_fast_fill_solid_regions")\n            else:\n                painted = self._bubble_gradient_fill(crop, local_mask)\n                if painted is not None:\n                    self._metric_add("bubble_fast_fill_gradient_regions")\n                elif _BUBBLE_FASTPATH_TELEA_ENABLED and self._bubble_telea_safe(\n                    crop, local_mask\n                ):\n                    painted = cv2.inpaint(\n                        crop,\n                        local_mask,\n                        float(_BUBBLE_FASTPATH_TELEA_RADIUS),\n                        cv2.INPAINT_TELEA,\n                    )\n                    self._metric_add("bubble_fast_fill_telea_regions")\n                else:\n                    return False\n'''
if text.count(old_decision) != 1:
    raise RuntimeError("stroke fast-fill decision marker mismatch")
text = text.replace(old_decision, new_decision, 1)
path.write_text(text, encoding="utf-8")


path = Path("tests/test_fast_inpaint_paths.py")
text = path.read_text(encoding="utf-8")
if "test_dense_smooth_authority_is_refined_to_strokes" in text:
    raise RuntimeError("stroke refinement tests already exist")
text += r'''


def test_dense_smooth_authority_is_refined_to_strokes():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(218 + xx * 0.035 + yy * 0.020, 0, 255)
    background[..., 1] = np.clip(222 + xx * 0.032 + yy * 0.018, 0, 255)
    background[..., 2] = np.clip(228 + xx * 0.028 + yy * 0.015, 0, 255)
    image = background.copy()

    mask = np.zeros((120, 220), dtype=np.uint8)
    mask[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, mask)
    image[75:86, 85:235] = 5
    image[105:117, 100:220] = 8
    image[134:146, 125:200] = 3

    authority = build_mask(image.shape[:2], [box], image) > 127
    inpainter = FastInpainter()
    output = inpainter.inpaint(image.copy(), [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert metrics["stroke_refined_regions"] == 1
    assert metrics["stroke_refined_model_pixels"] < metrics["stroke_refined_authority_pixels"] // 2
    assert metrics["bubble_fast_fill_gradient_regions"] == 1
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


def test_dense_textured_authority_rejects_stroke_refinement():
    rng = np.random.default_rng(123)
    crop = rng.integers(40, 220, size=(160, 260, 3), dtype=np.uint8)
    authority = np.zeros((160, 260), dtype=np.uint8)
    authority[20:140, 20:240] = 255
    inpainter = FastInpainter()
    assert inpainter._refine_dense_smooth_stroke_mask(crop, authority) is None
'''
path.write_text(text, encoding="utf-8")
