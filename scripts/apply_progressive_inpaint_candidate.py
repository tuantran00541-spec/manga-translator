from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one marker, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Keep medium text/artwork crops native instead of forcing four 512px tiles.
replace_once(
    "app/parameters.py",
    '    "MANGA_DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 768, minimum=128, maximum=4096\n',
    '    "MANGA_DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM", 1024, minimum=128, maximum=4096\n',
)
# Progressive verification is cheap for flat negatives and grouped ROIs; allow a
# normal manga slice with >4 authorized regions to verify every uncertain source.
replace_once(
    "app/parameters.py",
    '    "MANGA_DETECTOR_RESIDUE_VERIFY_MAX_ROIS", 4, minimum=1, maximum=24\n',
    '    "MANGA_DETECTOR_RESIDUE_VERIFY_MAX_ROIS", 8, minimum=1, maximum=24\n',
)


# Residue verification must follow the verified stroke mask rather than a large
# detector bbox. This is the scene-text-erasing pattern used by stroke-aware
# methods: localize the actual glyph support first, then spend neural verification
# only on the tight region that can contain residue.
path = Path("app/detector/fast_residue_detector.py")
text = path.read_text(encoding="utf-8")
marker = "    def verify_post_inpaint_residue(\n"
if text.count(marker) != 1:
    raise RuntimeError("fast_residue_detector verify marker mismatch")
helper = '''    @staticmethod\n    def _tight_verified_mask_roi(\n        image_shape: tuple[int, ...],\n        source: BubbleBox,\n    ) -> tuple[int, int, int, int] | None:\n        \"\"\"Return the smallest padded page ROI containing verified mask support.\n\n        Detector bboxes can be much larger than the glyphs they authorize. Using\n        the whole bbox for residue verification wastes the source-side budget and\n        can defer exactly the large free-text cases that need a second look.\n        \"\"\"\n        h, w = int(image_shape[0]), int(image_shape[1])\n        box_w = max(0, int(source.x2) - int(source.x1))\n        box_h = max(0, int(source.y2) - int(source.y1))\n        if box_w <= 0 or box_h <= 0:\n            return None\n\n        mask = source.mask\n        if mask is not None and getattr(mask, \"ndim\", 0) >= 2:\n            support = mask\n            if support.shape[:2] != (box_h, box_w):\n                support = cv2.resize(\n                    support,\n                    (box_w, box_h),\n                    interpolation=cv2.INTER_NEAREST,\n                )\n            ys, xs = np.nonzero(support > 127)\n            if xs.size and ys.size:\n                pad = int(DETECTOR_RESIDUE_VERIFY_PAD)\n                x1 = max(0, int(source.x1) + int(xs.min()) - pad)\n                y1 = max(0, int(source.y1) + int(ys.min()) - pad)\n                x2 = min(w, int(source.x1) + int(xs.max()) + 1 + pad)\n                y2 = min(h, int(source.y1) + int(ys.max()) + 1 + pad)\n                if x2 > x1 and y2 > y1:\n                    return x1, y1, x2, y2\n\n        pad = int(DETECTOR_RESIDUE_VERIFY_PAD)\n        x1 = max(0, int(source.x1) - pad)\n        y1 = max(0, int(source.y1) - pad)\n        x2 = min(w, int(source.x2) + pad)\n        y2 = min(h, int(source.y2) + pad)\n        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None\n\n'''
text = text.replace(marker, helper + marker, 1)
old_loop = '''        for index, source in enumerate(candidates):\n            # Preserve the historical source-budget/order semantics. A cheap\n            # negative inside the first N candidates does not silently make a\n            # later source eligible when it used to be review-deferred.\n            if index >= int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS):\n                deferred_budget += 1\n                residue.append(\n                    replace(\n                        source,\n                        safe_to_inpaint=False,\n                        ocr_eligible=True,\n                        needs_review=True,\n                        deferred_reason=\"post_inpaint_verification_budget\",\n                    )\n                )\n                continue\n\n            x1 = max(0, int(source.x1) - int(DETECTOR_RESIDUE_VERIFY_PAD))\n            y1 = max(0, int(source.y1) - int(DETECTOR_RESIDUE_VERIFY_PAD))\n            x2 = min(w, int(source.x2) + int(DETECTOR_RESIDUE_VERIFY_PAD))\n            y2 = min(h, int(source.y2) + int(DETECTOR_RESIDUE_VERIFY_PAD))\n            if x2 <= x1 or y2 <= y1:\n                continue\n            if max(x2 - x1, y2 - y1) > int(\n                DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE\n            ):\n                deferred_size += 1\n                residue.append(\n                    replace(\n                        source,\n                        safe_to_inpaint=False,\n                        ocr_eligible=True,\n                        needs_review=True,\n                        deferred_reason=\"post_inpaint_verification_size\",\n                    )\n                )\n                continue\n\n            if self._is_flat_negative_residue_source(image, source):\n                flat_negative_sources += 1\n                continue\n\n            roi = (x1, y1, x2, y2)\n            source_pixels += (x2 - x1) * (y2 - y1)\n            scheduled.append((source, roi))\n'''
new_loop = '''        neural_budget_used = 0\n        for source in candidates:\n            roi = self._tight_verified_mask_roi(image.shape, source)\n            if roi is None:\n                continue\n            x1, y1, x2, y2 = roi\n            if max(x2 - x1, y2 - y1) > int(\n                DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE\n            ):\n                deferred_size += 1\n                residue.append(\n                    replace(\n                        source,\n                        safe_to_inpaint=False,\n                        ocr_eligible=True,\n                        needs_review=True,\n                        deferred_reason=\"post_inpaint_verification_size\",\n                    )\n                )\n                continue\n\n            # Cheap clean negatives should not consume the neural verification\n            # budget. This preserves CPU while letting later uncertain glyphs be\n            # verified instead of becoming false \"text residue\" review flags.\n            if self._is_flat_negative_residue_source(image, source):\n                flat_negative_sources += 1\n                continue\n\n            if neural_budget_used >= int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS):\n                deferred_budget += 1\n                residue.append(\n                    replace(\n                        source,\n                        safe_to_inpaint=False,\n                        ocr_eligible=True,\n                        needs_review=True,\n                        deferred_reason=\"post_inpaint_verification_budget\",\n                    )\n                )\n                continue\n\n            neural_budget_used += 1\n            source_pixels += (x2 - x1) * (y2 - y1)\n            scheduled.append((source, roi))\n'''
if text.count(old_loop) != 1:
    raise RuntimeError("fast_residue_detector loop marker mismatch")
path.write_text(text.replace(old_loop, new_loop, 1), encoding="utf-8")


# Preserve the verified residue stroke mask in the transient result so the repair
# pass can erase only those strokes instead of repainting a padded rectangle.
replace_once(
    "app/page_processing.py",
    '''        residue_regions = [\n            {k: getattr(box, k) for k in decision_fields}\n            for box in residue_boxes\n        ]\n''',
    '''        residue_regions = [\n            {\n                **{k: getattr(box, k) for k in decision_fields},\n                \"mask\": encode_mask(box.mask),\n            }\n            for box in residue_boxes\n        ]\n''',
)


# Progressive second pass: prefer the verifier's stroke mask; keep the old padded
# rectangle only as a compatibility fallback for historical records without a
# mask. Final writes are still intersected with original erase authority.
replace_once(
    "app/optimized_pipeline.py",
    '''        h, w = clean_before.shape[:2]\n        scope = np.zeros((h, w), dtype=np.uint8)\n        pad = 6\n        for region in actual_hits:\n            try:\n                x1 = max(0, int(region[\"x1\"]) - pad)\n                y1 = max(0, int(region[\"y1\"]) - pad)\n                x2 = min(w, int(region[\"x2\"]) + pad)\n                y2 = min(h, int(region[\"y2\"]) + pad)\n            except (KeyError, TypeError, ValueError):\n                continue\n            if x2 > x1 and y2 > y1:\n                scope[y1:y2, x1:x2] = 255\n\n        repair_mask = cv2.bitwise_and(full_authority, scope)\n''',
    '''        h, w = clean_before.shape[:2]\n        scope = np.zeros((h, w), dtype=np.uint8)\n        fallback_pad = 6\n        for region in actual_hits:\n            try:\n                raw_x1, raw_y1 = int(region[\"x1\"]), int(region[\"y1\"])\n                raw_x2, raw_y2 = int(region[\"x2\"]), int(region[\"y2\"])\n            except (KeyError, TypeError, ValueError):\n                continue\n            x1, y1 = max(0, raw_x1), max(0, raw_y1)\n            x2, y2 = min(w, raw_x2), min(h, raw_y2)\n            if x2 <= x1 or y2 <= y1:\n                continue\n\n            residue_mask = decode_mask_value(region.get(\"mask\"))\n            if residue_mask is not None:\n                target_w, target_h = x2 - x1, y2 - y1\n                if residue_mask.shape[:2] != (target_h, target_w):\n                    residue_mask = cv2.resize(\n                        residue_mask,\n                        (target_w, target_h),\n                        interpolation=cv2.INTER_NEAREST,\n                    )\n                local_scope = scope[y1:y2, x1:x2]\n                local_scope[residue_mask > MANUAL_MASK_THRESHOLD] = 255\n                continue\n\n            # Backward-compatible fallback for old manifests that did not retain\n            # the verifier mask. New processing results should take the branch\n            # above and stay stroke-shaped.\n            px1 = max(0, raw_x1 - fallback_pad)\n            py1 = max(0, raw_y1 - fallback_pad)\n            px2 = min(w, raw_x2 + fallback_pad)\n            py2 = min(h, raw_y2 + fallback_pad)\n            if px2 > px1 and py2 > py1:\n                scope[py1:py2, px1:px2] = 255\n\n        repair_mask = cv2.bitwise_and(full_authority, scope)\n''',
)


# Unit regressions for the two paper-driven behaviors.
path = Path("tests/test_fast_residue_detector.py")
text = path.read_text(encoding="utf-8")
if "test_large_detector_box_uses_tight_verified_mask_roi" in text:
    raise RuntimeError("residue candidate tests already exist")
text += r'''


def test_large_detector_box_uses_tight_verified_mask_roi():
    detector = _detector_with_fake_text()
    max_side = int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE)
    width = max_side + 300
    height = 120
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[35:55, 100:160] = 255
    source = BubbleBox(
        0,
        0,
        width,
        height,
        0.9,
        mask,
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
    )
    image = np.full((height + 20, width + 20, 3), 248, dtype=np.uint8)
    image[40:48, 110:135] = 0

    detector.verify_post_inpaint_residue(image, [source])
    metrics = detector.last_residue_metrics()

    assert metrics["deferred_size"] == 0
    assert metrics["neural_sources"] == 1
    assert metrics["model_calls"] == 1
    assert detector.text_detector.calls == 1


def test_flat_negative_sources_do_not_consume_neural_budget():
    detector = _detector_with_fake_text()
    image = np.full((160, 500, 3), 248, dtype=np.uint8)
    sources = [_box(i * 80 + 5, 20, i * 80 + 55, 55) for i in range(5)]

    residue = detector.verify_post_inpaint_residue(image, sources)
    metrics = detector.last_residue_metrics()

    assert residue == []
    assert metrics["flat_negative_sources"] == 5
    assert metrics["deferred_budget"] == 0
    assert metrics["model_calls"] == 0
'''
path.write_text(text, encoding="utf-8")


path = Path("tests/test_fast_inpaint_paths.py")
text = path.read_text(encoding="utf-8")
if "test_textured_medium_dynamic_lama_stays_one_native_call" in text:
    raise RuntimeError("native 1024 test already exists")
text += r'''


def test_textured_medium_dynamic_lama_stays_one_native_call():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(900, 920, 3), dtype=np.uint8)
    mask = np.zeros((900, 920), dtype=np.uint8)
    mask[70:830, 70:850] = 255

    inpainter = FastInpainter()
    inpainter._begin_metrics()
    inpainter.dynamic_lama = True
    inpainter._ensure_session = lambda: None
    model_shapes = []

    def fake_run_lama(canvas, mask_canvas):
        model_shapes.append(canvas.shape[:2])
        return canvas.copy()

    inpainter._run_lama = fake_run_lama
    output = inpainter._lama_fill(
        image.copy(),
        image.copy(),
        mask,
        (0, 0, 920, 900),
    )

    assert output.shape == image.shape
    assert len(model_shapes) == 1
    assert model_shapes[0][0] >= 900
    assert model_shapes[0][1] >= 920
'''
path.write_text(text, encoding="utf-8")
