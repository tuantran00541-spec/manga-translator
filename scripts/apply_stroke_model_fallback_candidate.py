from pathlib import Path


path = Path("app/inpaint/fast_lama_inpainter.py")
text = path.read_text(encoding="utf-8")

metric_old = '''                "stroke_refined_authority_pixels": 0,\n                "stroke_refined_model_pixels": 0,\n                "roi_lama_regions": 0,\n'''
metric_new = '''                "stroke_refined_authority_pixels": 0,\n                "stroke_refined_model_pixels": 0,\n                "stroke_refined_lama_fallback_regions": 0,\n                "roi_lama_regions": 0,\n'''
if text.count(metric_old) != 1:
    raise RuntimeError("stroke fallback metric marker mismatch")
text = text.replace(metric_old, metric_new, 1)

local_old = '''        if bool(getattr(box, "allow_rectangle_fallback", False)):\n            local.allow_rectangle_fallback = True\n        return local\n\n    @staticmethod\n    def _mask_ring(local_mask: np.ndarray, radius: int) -> np.ndarray:\n'''
local_new = '''        if bool(getattr(box, "allow_rectangle_fallback", False)):\n            local.allow_rectangle_fallback = True\n        if bool(getattr(box, "stroke_refined_model_mask", False)):\n            local.stroke_refined_model_mask = True\n        return local\n\n    @staticmethod\n    def _build_model_mask(\n        image_shape: tuple[int, int],\n        boxes: list[BubbleBox],\n        crop_img: np.ndarray | None = None,\n    ) -> np.ndarray:\n        """Build the model mask while keeping refined strokes exact.\n\n        Ordinary detector masks retain the production adaptive dilation. A mask\n        already refined from dense erase authority to glyph support must not be\n        dilated a second time because that can escape the authority we just\n        proved safe.\n        """\n        h, w = image_shape\n        refined = [\n            box for box in boxes\n            if bool(getattr(box, "stroke_refined_model_mask", False))\n        ]\n        ordinary = [\n            box for box in boxes\n            if not bool(getattr(box, "stroke_refined_model_mask", False))\n        ]\n        if ordinary:\n            mask = build_mask(image_shape, ordinary, crop_img)\n        else:\n            mask = np.zeros((h, w), dtype=np.uint8)\n\n        for box in refined:\n            if not bool(box.safe_to_inpaint) or box.mask is None:\n                continue\n            box_w = int(box.x2 - box.x1)\n            box_h = int(box.y2 - box.y1)\n            if box_w <= 0 or box_h <= 0 or box.mask.shape != (box_h, box_w):\n                continue\n            x1 = max(0, int(box.x1))\n            y1 = max(0, int(box.y1))\n            x2 = min(w, int(box.x2))\n            y2 = min(h, int(box.y2))\n            if x2 <= x1 or y2 <= y1:\n                continue\n            src = box.mask[\n                y1 - int(box.y1):y2 - int(box.y1),\n                x1 - int(box.x1):x2 - int(box.x1),\n            ]\n            dest = mask[y1:y2, x1:x2]\n            mask[y1:y2, x1:x2] = np.maximum(dest, src)\n        return mask\n\n    @staticmethod\n    def _mask_ring(local_mask: np.ndarray, radius: int) -> np.ndarray:\n'''
if text.count(local_old) != 1:
    raise RuntimeError("model-mask builder marker mismatch")
text = text.replace(local_old, local_new, 1)

# Existing fast-fill and clustered LaMa paths must both honor the refined model
# mask without re-dilating it.
fast_build_old = '''        local_mask = build_mask((y2 - y1, x2 - x1), [local_box], crop)\n'''
fast_build_new = '''        local_mask = self._build_model_mask(\n            (y2 - y1, x2 - x1), [local_box], crop\n        )\n'''
if text.count(fast_build_old) != 1:
    raise RuntimeError("fast model-mask marker mismatch")
text = text.replace(fast_build_old, fast_build_new, 1)

cluster_build_old = '''            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)\n'''
cluster_build_new = '''            local_mask = self._build_model_mask(\n                (cy2 - cy1, cx2 - cx1), local_boxes, crop_img\n            )\n'''
if text.count(cluster_build_old) != 1:
    raise RuntimeError("cluster model-mask marker mismatch")
text = text.replace(cluster_build_old, cluster_build_new, 1)

method_marker = '''    def _try_bubble_fast_fill(\n        self,\n        image: np.ndarray,\n        box: BubbleBox,\n        protected_regions: list[dict] | None,\n    ) -> bool:\n'''
method_insert = r'''    def _stroke_model_box(
        self,
        image: np.ndarray,
        box: BubbleBox,
        protected_regions: list[dict] | None,
    ) -> BubbleBox:
        """Return a box whose model mask may be tighter than erase authority.

        The original detector mask remains the authority source. Refinement is
        derived against a padded context crop, clipped by preserve regions, then
        stored as an exact bbox-local model mask. If any safety gate fails the
        original box is returned unchanged.
        """
        if (
            not self._bubble_candidate(box)
            or box.source_role != "text_segmenter"
        ):
            return box

        h, w = image.shape[:2]
        x1 = max(0, int(box.x1) - _BUBBLE_FASTPATH_PAD)
        y1 = max(0, int(box.y1) - _BUBBLE_FASTPATH_PAD)
        x2 = min(w, int(box.x2) + _BUBBLE_FASTPATH_PAD)
        y2 = min(h, int(box.y2) + _BUBBLE_FASTPATH_PAD)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return box

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
            return box

        refined = self._refine_dense_smooth_stroke_mask(crop, authority_mask)
        if refined is None:
            return box

        bx1 = int(box.x1) - x1
        by1 = int(box.y1) - y1
        bx2 = bx1 + int(box.x2 - box.x1)
        by2 = by1 + int(box.y2 - box.y1)
        model_mask = refined[by1:by2, bx1:bx2].copy()
        expected_shape = (int(box.y2 - box.y1), int(box.x2 - box.x1))
        model_pixels = int(np.count_nonzero(model_mask > 127))
        if model_mask.shape != expected_shape or model_pixels <= 0:
            return box

        model_box = self._local_box(box, 0, 0)
        model_box.mask = model_mask
        model_box.stroke_refined_model_mask = True
        self._metric_add("stroke_refined_regions")
        self._metric_add("stroke_refined_authority_pixels", authority_pixels)
        self._metric_add("stroke_refined_model_pixels", model_pixels)
        return model_box

    def _try_bubble_fast_fill(
        self,
        image: np.ndarray,
        box: BubbleBox,
        protected_regions: list[dict] | None,
    ) -> bool:
'''
if text.count(method_marker) != 1:
    raise RuntimeError("stroke model-box insertion marker mismatch")
text = text.replace(method_marker, method_insert, 1)

old_refine = '''        authority_pixels = int(np.count_nonzero(local_mask > 127))\n        if authority_pixels <= 0:\n            return False\n\n        refined = None\n        if box.source_role == "text_segmenter":\n            refined = self._refine_dense_smooth_stroke_mask(crop, local_mask)\n        stroke_refined = refined is not None\n        if stroke_refined:\n            local_mask = refined\n            self._metric_add("stroke_refined_regions")\n            self._metric_add("stroke_refined_authority_pixels", authority_pixels)\n            self._metric_add(\n                "stroke_refined_model_pixels", int(np.count_nonzero(local_mask > 127))\n            )\n\n        mask_bool = local_mask > 127\n'''
new_refine = '''        authority_pixels = int(np.count_nonzero(local_mask > 127))\n        if authority_pixels <= 0:\n            return False\n\n        stroke_refined = bool(getattr(box, "stroke_refined_model_mask", False))\n        mask_bool = local_mask > 127\n'''
if text.count(old_refine) != 1:
    raise RuntimeError("old in-fast-fill refinement marker mismatch")
text = text.replace(old_refine, new_refine, 1)

loop_old = '''        result = image.copy()\n        remaining: list[BubbleBox] = []\n        for box in boxes:\n            if self._bubble_candidate(box) and self._strong_authority_overlap(box, boxes):\n                self._metric_add("bubble_fast_fill_overlap_skips")\n                remaining.append(box)\n                continue\n            if self._try_bubble_fast_fill(result, box, protected_regions):\n                continue\n            remaining.append(box)\n'''
loop_new = '''        result = image.copy()\n        remaining: list[BubbleBox] = []\n        for box in boxes:\n            model_box = self._stroke_model_box(result, box, protected_regions)\n            stroke_refined = bool(\n                getattr(model_box, "stroke_refined_model_mask", False)\n            )\n            if self._bubble_candidate(box) and self._strong_authority_overlap(box, boxes):\n                self._metric_add("bubble_fast_fill_overlap_skips")\n                if stroke_refined:\n                    self._metric_add("stroke_refined_lama_fallback_regions")\n                remaining.append(model_box)\n                continue\n            if self._try_bubble_fast_fill(result, model_box, protected_regions):\n                continue\n            if stroke_refined:\n                self._metric_add("stroke_refined_lama_fallback_regions")\n            remaining.append(model_box)\n'''
if text.count(loop_old) != 1:
    raise RuntimeError("stroke fallback loop marker mismatch")
text = text.replace(loop_old, loop_new, 1)
path.write_text(text, encoding="utf-8")


path = Path("tests/test_fast_inpaint_paths.py")
text = path.read_text(encoding="utf-8")
if "test_refined_stroke_mask_survives_fast_fill_fallback" in text:
    raise RuntimeError("stroke fallback test already exists")
text += r'''


def test_refined_stroke_mask_survives_fast_fill_fallback():
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
    # Force the refined candidate past the cheap renderer so the clustered LaMa
    # path is exercised without loading a real model.
    inpainter._try_bubble_fast_fill = lambda image, box, protected: False
    captured = {}

    def fake_smart_paint(image_arg, local_mask, crop_box):
        captured["mask"] = local_mask.copy()
        captured["crop_box"] = tuple(int(v) for v in crop_box)
        return image_arg

    inpainter._smart_paint_region = fake_smart_paint
    inpainter.inpaint(image.copy(), [box])
    metrics = inpainter.last_metrics()

    assert metrics["stroke_refined_regions"] == 1
    assert metrics["stroke_refined_lama_fallback_regions"] == 1
    assert metrics["stroke_refined_model_pixels"] < metrics["stroke_refined_authority_pixels"] // 2
    assert "mask" in captured
    assert int(np.count_nonzero(captured["mask"] > 127)) == metrics["stroke_refined_model_pixels"]

    global_model = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = captured["crop_box"]
    global_model[y1:y2, x1:x2] = captured["mask"]
    assert not np.any((global_model > 127) & (~authority))
'''
path.write_text(text, encoding="utf-8")
