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
knobs = '''_BUBBLE_FASTPATH_OVERLAP_RATIO = _env_float(\n    "MANGA_BUBBLE_FASTPATH_OVERLAP_RATIO", 0.80, 0.50, 1.0\n)\n\n# Text segmenters sometimes return a dense region around an entire lettering\n# block rather than glyph support. Scene-text erasing literature consistently\n# separates erase authority from a tighter stroke mask; repainting the dense\n# region makes LaMa synthesize a visible polygon even on a smooth background.\n_STROKE_REFINE_ENABLED = _env_bool("MANGA_STROKE_REFINE_ENABLED", True)\n_STROKE_REFINE_DENSE_FRACTION_MIN = _env_float(\n    "MANGA_STROKE_REFINE_DENSE_FRACTION_MIN", 0.55, 0.25, 0.95\n)\n_STROKE_REFINE_RING_STD_MAX = _env_float(\n    "MANGA_STROKE_REFINE_RING_STD_MAX", 18.0, 2.0, 48.0\n)\n_STROKE_REFINE_MIN_CONTRAST = _env_float(\n    "MANGA_STROKE_REFINE_MIN_CONTRAST", 36.0, 8.0, 96.0\n)\n_STROKE_REFINE_CORE_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_CORE_DELTA_MIN", 30.0, 8.0, 96.0\n)\n_STROKE_REFINE_WEAK_DELTA_MIN = _env_float(\n    "MANGA_STROKE_REFINE_WEAK_DELTA_MIN", 14.0, 4.0, 64.0\n)\n_STROKE_REFINE_MAX_AUTHORITY_FRACTION = _env_float(\n    "MANGA_STROKE_REFINE_MAX_AUTHORITY_FRACTION", 0.60, 0.10, 0.90\n)\n\n_ROI_LAMA_ENABLED = _env_bool("MANGA_DYNAMIC_LAMA_TIGHT_ROI_ENABLED", True)\n'''
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
        """Shrink a dense region mask to high-confidence glyph strokes.

        This is intentionally conservative. It activates only when the supplied
        mask is suspiciously dense, the clean exterior ring is smooth, and one
        foreground polarity has strong contrast. The returned mask is always a
        strict subset of the original authority (plus a one-pixel fringe still
        clipped inside that authority), so preserve/authority semantics cannot
        expand.
        """
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
        ring_gray = gray[ring].astype(np.float32, copy=False)
        ring_std = float(ring_gray.std())
        if not np.isfinite(ring_std) or ring_std > _STROKE_REFINE_RING_STD_MAX:
            return None

        # Reject colorful/edge-heavy artwork even when luminance alone looks flat.
        edges = cv2.Canny(gray, 64, 128, L2gradient=True) > 0
        if float(edges[ring].mean()) > 0.06:
            return None
        if crop.ndim == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            ring_ab = lab[ring, 1:3].astype(np.float32, copy=False)
            if ring_ab.size and float(np.max(ring_ab.std(axis=0))) > 18.0:
                return None

        background = float(np.median(ring_gray))
        inside = gray[authority].astype(np.float32, copy=False)
        low = float(np.percentile(inside, 10.0))
        high = float(np.percentile(inside, 90.0))
        dark_contrast = background - low
        light_contrast = high - background
        if max(dark_contrast, light_contrast) < _STROKE_REFINE_MIN_CONTRAST:
            return None

        core_delta = max(_STROKE_REFINE_CORE_DELTA_MIN, ring_std * 3.0)
        weak_delta = max(_STROKE_REFINE_WEAK_DELTA_MIN, ring_std * 1.5)
        if dark_contrast >= light_contrast:
            strong = authority & (gray.astype(np.float32) <= background - core_delta)
            weak = authority & (gray.astype(np.float32) <= background - weak_delta)
        else:
            strong = authority & (gray.astype(np.float32) >= background + core_delta)
            weak = authority & (gray.astype(np.float32) >= background + weak_delta)

        if int(np.count_nonzero(strong)) < 24:
            return None
        weak_u8 = weak.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(weak_u8, 8)
        keep = np.zeros_like(weak_u8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2:
                continue
            component = labels == label
            if np.any(strong & component):
                keep[component] = 255

        model_pixels = int(np.count_nonzero(keep))
        authority_pixels = int(np.count_nonzero(authority))
        if model_pixels < 24 or authority_pixels <= 0:
            return None
        if model_pixels / float(authority_pixels) > _STROKE_REFINE_MAX_AUTHORITY_FRACTION:
            return None

        # Recover antialiased fringe without ever leaving the original authority.
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

flow_old = '''        mask_bool = local_mask > 127\n        mask_pixels = int(np.count_nonzero(mask_bool))\n        if mask_pixels <= 0:\n            return False\n\n        fill_color = self._smart_fill_color(crop, local_mask)\n        if fill_color is not None:\n            painted = crop.copy()\n            painted[mask_bool] = fill_color\n            self._metric_add("smart_fill_regions")\n            self._metric_add("bubble_fast_fill_solid_regions")\n        else:\n            painted = self._bubble_gradient_fill(crop, local_mask)\n            if painted is not None:\n                self._metric_add("bubble_fast_fill_gradient_regions")\n            elif _BUBBLE_FASTPATH_TELEA_ENABLED and self._bubble_telea_safe(\n                crop, local_mask\n            ):\n'''
flow_new = '''        authority_pixels = int(np.count_nonzero(local_mask > 127))\n        if authority_pixels <= 0:\n            return False\n\n        refined = None\n        if box.source_role == "text_segmenter":\n            refined = self._refine_dense_smooth_stroke_mask(crop, local_mask)\n        stroke_refined = refined is not None\n        if stroke_refined:\n            local_mask = refined\n            self._metric_add("stroke_refined_regions")\n            self._metric_add("stroke_refined_authority_pixels", authority_pixels)\n            self._metric_add(\n                "stroke_refined_model_pixels", int(np.count_nonzero(local_mask > 127))\n            )\n\n        mask_bool = local_mask > 127\n        mask_pixels = int(np.count_nonzero(mask_bool))\n        if mask_pixels <= 0:\n            return False\n\n        # A refined stroke mask should preserve the smooth spatial background,\n        # not flatten it to one color. Prefer the robust quadratic reconstruction\n        # first; ordinary masks retain the existing solid-first fast path.\n        painted = self._bubble_gradient_fill(crop, local_mask) if stroke_refined else None\n        if painted is not None:\n            self._metric_add("bubble_fast_fill_gradient_regions")\n        else:\n            fill_color = self._smart_fill_color(crop, local_mask)\n            if fill_color is not None:\n                painted = crop.copy()\n                painted[mask_bool] = fill_color\n                self._metric_add("smart_fill_regions")\n                self._metric_add("bubble_fast_fill_solid_regions")\n            else:\n                painted = self._bubble_gradient_fill(crop, local_mask)\n        if painted is not None:\n            if not stroke_refined:\n                self._metric_add("bubble_fast_fill_gradient_regions")\n        elif _BUBBLE_FASTPATH_TELEA_ENABLED and self._bubble_telea_safe(\n                crop, local_mask\n            ):\n'''
if text.count(flow_old) != 1:
    raise RuntimeError("stroke fast-fill flow marker mismatch")
text = text.replace(flow_old, flow_new, 1)
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

    # Simulate a text-segmenter mask that authorizes almost an entire lettering
    # block rather than the glyph strokes themselves.
    mask = np.zeros((120, 220), dtype=np.uint8)
    mask[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, mask)
    # Dark glyph-like bars inside the dense authority.
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
    mae = float(np.abs(output.astype(np.int16) - background.astype(np.int16))[glyphs].mean())
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
