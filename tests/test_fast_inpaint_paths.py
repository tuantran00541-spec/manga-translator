import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.image_io import encode_mask, read_image, write_image
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.optimized_pipeline import OptimizedChapterPipeline


def _speech_box(x1, y1, x2, y2, mask):
    return BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.95,
        mask,
        source_model="text_segmenter.onnx",
        semantic_type="speech_bubble",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )


def test_verified_flat_bubble_skips_lama():
    image = np.full((160, 220, 3), 245, dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[15:22, 18:82] = 255
    mask[29:36, 28:72] = 255
    image[50:100, 60:160][mask > 127] = 20

    box = _speech_box(60, 50, 160, 100, mask)

    inpainter = FastInpainter()
    output = inpainter.inpaint(image, [box])
    metrics = inpainter.last_metrics()

    assert metrics["bubble_fast_fill_regions"] == 1
    assert metrics["lama_model_runs"] == 0
    assert float(output[65:86, 78:142].mean()) > float(image[65:86, 78:142].mean())


def test_smooth_gradient_bubble_uses_reconstruction_without_telea():
    h, w = 180, 240
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(226 + xx * 0.035 + yy * 0.025, 0, 255)
    background[..., 1] = np.clip(231 + xx * 0.030 + yy * 0.020, 0, 255)
    background[..., 2] = np.clip(238 + xx * 0.020 + yy * 0.015, 0, 255)
    image = background.copy()

    # Keep synthetic glyph support below the real safety occupancy gate even
    # after automatic mask dilation. Dense masks intentionally fall back to LaMa.
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[14:18, 18:102] = 255
    mask[34:38, 26:94] = 255
    box = _speech_box(60, 55, 180, 115, mask)
    image[55:115, 60:180][mask > 127] = 18

    expected_authority = build_mask(image.shape[:2], [box], image)
    inpainter = FastInpainter()
    inpainter._smart_fill_color = lambda crop, local_mask: None
    output = inpainter.inpaint(image, [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert not np.any(changed & (expected_authority <= 127))
    assert metrics["bubble_fast_fill_gradient_regions"] == 1
    assert metrics["bubble_fast_fill_telea_regions"] == 0
    assert metrics["lama_model_runs"] == 0

    authority = expected_authority > 127
    mae = float(np.abs(output.astype(np.int16) - background.astype(np.int16))[authority].mean())
    assert mae < 5.0


def test_dense_pseudo_bubble_rejects_gradient_and_telea_by_default():
    image = np.full((140, 180, 3), 230, dtype=np.uint8)
    mask = np.full((80, 120), 255, dtype=np.uint8)
    box = _speech_box(30, 30, 150, 110, mask)

    inpainter = FastInpainter()
    inpainter._begin_metrics()
    inpainter._smart_fill_color = lambda crop, local_mask: None
    used = inpainter._try_bubble_fast_fill(image.copy(), box, None)
    metrics = inpainter.last_metrics()

    assert used is False
    assert metrics["bubble_fast_fill_gradient_regions"] == 0
    assert metrics["bubble_fast_fill_telea_regions"] == 0


def test_overlapping_authorities_skip_per_box_bubble_fast_fill():
    image = np.full((180, 220, 3), 220, dtype=np.uint8)
    bubble_mask = np.zeros((80, 120), dtype=np.uint8)
    bubble_mask[20:60, 20:100] = 255
    bubble = _speech_box(50, 40, 170, 120, bubble_mask)

    free_mask = np.zeros((50, 90), dtype=np.uint8)
    free_mask[8:42, 8:82] = 255
    free = BubbleBox(
        65,
        65,
        155,
        115,
        0.94,
        free_mask,
        source_model="text_segmenter.onnx",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )

    inpainter = FastInpainter()
    fast_attempts = []
    paint_calls = []

    original_try = inpainter._try_bubble_fast_fill

    def record_fast(image_arg, box_arg, protected):
        fast_attempts.append(box_arg.semantic_type)
        return original_try(image_arg, box_arg, protected)

    inpainter._try_bubble_fast_fill = record_fast
    inpainter._cluster_boxes = lambda boxes: [boxes]
    inpainter._split_oversized_cluster_area = lambda cluster, width, height: [cluster]

    def fake_paint(image_arg, local_mask, crop_box, **kwargs):
        paint_calls.append((int(np.count_nonzero(local_mask)), crop_box))
        return image_arg

    inpainter._smart_paint_region = fake_paint
    output = inpainter.inpaint(image.copy(), [bubble, free])
    metrics = inpainter.last_metrics()

    assert output.shape == image.shape
    assert metrics["bubble_fast_fill_overlap_skips"] == 2
    assert "speech_bubble" not in fast_attempts
    assert len(paint_calls) == 1


def test_dynamic_lama_tightens_mask_roi_before_model_call():
    image = np.full((800, 800, 3), 127, dtype=np.uint8)
    mask = np.zeros((800, 800), dtype=np.uint8)
    mask[380:420, 360:440] = 255

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
        (0, 0, 800, 800),
    )
    metrics = inpainter.last_metrics()

    assert output.shape == image.shape
    assert metrics["roi_lama_regions"] == 1
    assert metrics["roi_lama_saved_pixels"] > 0
    assert metrics["roi_lama_input_pixels"] < metrics["roi_lama_source_pixels"]
    assert model_shapes
    assert max(model_shapes[0]) < 800


def test_optimized_pipeline_uses_fast_inpainter():
    pipeline = OptimizedChapterPipeline()
    assert isinstance(pipeline.inpainter, FastInpainter)



def test_medium_dynamic_lama_keeps_native_resolution_single_call():
    image = np.full((600, 700, 3), 127, dtype=np.uint8)
    mask = np.zeros((600, 700), dtype=np.uint8)
    mask[80:520, 80:620] = 255

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
        (0, 0, 700, 600),
    )

    assert output.shape == image.shape
    assert len(model_shapes) == 1
    assert model_shapes[0][0] >= 600
    assert model_shapes[0][1] >= 700


def test_residue_second_pass_is_clipped_to_existing_authority(tmp_path):
    original = np.full((120, 160, 3), 230, dtype=np.uint8)
    clean = original.copy()
    local_mask = np.zeros((50, 80), dtype=np.uint8)
    local_mask[18:24, 14:66] = 255
    local_mask[31:37, 24:58] = 255
    clean[35:85, 40:120][local_mask > 127] = 120

    raw_path = tmp_path / "raw.png"
    clean_path = tmp_path / "clean.png"
    write_image(raw_path, original)
    write_image(clean_path, clean)

    record = {
        "x1": 40,
        "y1": 35,
        "x2": 120,
        "y2": 85,
        "confidence": 0.95,
        "mask": encode_mask(local_mask),
        "source_model": "text_segmenter.onnx",
        "class_id": 0,
        "class_name": "text_comic",
        "semantic_type": "free_text",
        "mask_source": "text_segmenter",
        "safe_to_inpaint": True,
        "ocr_eligible": True,
        "needs_review": False,
        "source_role": "text_segmenter",
    }
    result = {
        "tmp_clean": clean_path.as_posix(),
        "boxes": [record],
        "residue_regions": [
            {
                "x1": 52,
                "y1": 47,
                "x2": 108,
                "y2": 75,
                "confidence": 0.9,
                "source_model": "text_segmenter.onnx",
                "source_role": "text_segmenter",
                "class_name": "text_comic",
                "semantic_type": "free_text",
                "deferred_reason": "post_inpaint_text_residue",
            }
        ],
        "detection_issues": ["post_inpaint_text_residue"],
        "processing_metrics": {
            "timing_ms": {"total": 10.0},
            "detector": {"post_inpaint_residue": 1},
        },
    }

    class FakeInpainter:
        def __init__(self):
            self.mask = None

        def inpaint_mask(self, image, mask, force_lama=False):
            assert force_lama is True
            self.mask = mask.copy()
            candidate = np.zeros_like(image)
            candidate[:] = 17
            return candidate

        def last_metrics(self):
            return {"lama_model_runs": 1, "lama_model_ms": 3}

    class FakeDetector:
        def verify_post_inpaint_residue(self, image, boxes):
            return []

        def last_residue_metrics(self):
            return {"residue_hits": 0, "model_calls": 1}

    pipeline = OptimizedChapterPipeline()
    fake_inpainter = FakeInpainter()
    pipeline._inpainter = fake_inpainter
    pipeline._detector = FakeDetector()
    updated = pipeline._repair_post_inpaint_result(
        raw_path,
        result,
        None,
    )
    repaired = read_image(clean_path)

    authority = fake_inpainter.mask > 10
    assert np.any(authority)
    assert np.array_equal(repaired[~authority], clean[~authority])
    assert np.all(repaired[authority] == 17)
    assert updated["residue_regions"] == []
    assert "post_inpaint_text_residue" not in updated["detection_issues"]
    assert updated["processing_metrics"]["residue_repair"]["attempted"] == 1
    assert (
        updated["processing_metrics"]["residue_repair"]
        ["outside_authority_changed_channel_values"]
        == 0
    )



def test_smooth_gradient_free_text_uses_reconstruction_without_lama():
    h, w = 180, 240
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(225 + xx * 0.025 + yy * 0.018, 0, 255)
    background[..., 1] = np.clip(229 + xx * 0.022 + yy * 0.016, 0, 255)
    background[..., 2] = np.clip(234 + xx * 0.018 + yy * 0.013, 0, 255)
    image = background.copy()

    mask = np.zeros((70, 130), dtype=np.uint8)
    mask[16:20, 18:112] = 255
    mask[36:40, 30:100] = 255
    box = _speech_box(55, 50, 185, 120, mask)
    box.semantic_type = "free_text"
    image[50:120, 55:185][mask > 127] = 15

    expected_authority = build_mask(image.shape[:2], [box], image)
    inpainter = FastInpainter()
    inpainter._smart_fill_color = lambda crop, local_mask: None
    output = inpainter.inpaint(image, [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert not np.any(changed & (expected_authority <= 127))
    assert metrics["bubble_fast_fill_gradient_regions"] == 1
    assert metrics["bubble_fast_fill_telea_regions"] == 0
    assert metrics["lama_model_runs"] == 0

    authority = expected_authority > 127
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[authority].mean()
    )
    assert mae < 5.0



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



def test_dynamic_long_crop_uses_native_single_call_within_pixel_budget():
    # Old policy tiled this crop solely because 900/400 >= 2 and max_dim > 512.
    # Dynamic LaMa should preserve one global-context call when the area is safe.
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(400, 900, 3), dtype=np.uint8)
    mask = np.zeros((400, 900), dtype=np.uint8)
    mask[30:370, 40:860] = 255

    inpainter = FastInpainter()
    inpainter._begin_metrics()
    inpainter.dynamic_lama = True
    inpainter._ensure_session = lambda: None
    shapes = []

    def fake_run_lama(canvas, mask_canvas):
        shapes.append(canvas.shape[:2])
        return canvas.copy()

    inpainter._run_lama = fake_run_lama
    output = inpainter._lama_fill(
        image.copy(),
        image.copy(),
        mask,
        (0, 0, 900, 400),
    )
    metrics = inpainter.last_metrics()

    assert output.shape == image.shape
    assert len(shapes) == 1
    assert shapes[0][0] >= 400
    assert shapes[0][1] >= 900
    assert metrics["lama_native_single_regions"] == 1
    assert metrics["lama_tiled_regions"] == 0



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


def test_dual_polarity_halo_recovers_bright_outline_around_dark_strokes():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(210 + xx * 0.040 + yy * 0.022, 0, 255)
    background[..., 1] = np.clip(216 + xx * 0.036 + yy * 0.020, 0, 255)
    background[..., 2] = np.clip(224 + xx * 0.030 + yy * 0.017, 0, 255)
    image = background.copy()

    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, dense)

    core = np.zeros((h, w), dtype=np.uint8)
    core[76:84, 84:236] = 255
    core[106:114, 102:220] = 255
    core[136:144, 126:202] = 255
    outline = cv2.dilate(
        core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    outline_only = (outline > 0) & (core == 0)
    image[outline_only] = 252
    image[core > 0] = 4

    authority = build_mask(image.shape[:2], [box], image) > 127
    inpainter = FastInpainter()
    output = inpainter.inpaint(image.copy(), [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert metrics["stroke_refined_regions"] == 1
    assert metrics["stroke_authority_gradient_regions"] == 1
    assert metrics["lama_model_runs"] == 0
    assert not np.any(changed & (~authority))
    # Regression for the real p045 failure: a one-pixel same-polarity fringe
    # leaves the outer white outline visible. The dual-polarity halo must cover
    # almost all of the 2px bright outline before background reconstruction.
    assert float(changed[outline_only].mean()) > 0.95

    text_support = outline > 0
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[text_support].mean()
    )
    assert mae < 6.0


def test_surface_residual_halo_recovers_subtle_outline_on_gradient():
    h, w = 220, 320
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(188 + xx * 0.16 + yy * 0.025, 0, 255)
    background[..., 1] = np.clip(194 + xx * 0.15 + yy * 0.022, 0, 255)
    background[..., 2] = np.clip(202 + xx * 0.13 + yy * 0.018, 0, 255)
    image = background.copy()

    dense = np.zeros((120, 220), dtype=np.uint8)
    dense[8:112, 8:212] = 255
    box = _speech_box(50, 45, 270, 165, dense)

    core = np.zeros((h, w), dtype=np.uint8)
    core[77:84, 86:234] = 255
    core[107:114, 104:218] = 255
    core[137:144, 128:200] = 255
    outline = cv2.dilate(
        core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    outline_only = (outline > 0) & (core == 0)
    # Deliberately subtle: only +4 over the *local* gradient. A single global
    # median plus the old 2px halo cannot reliably recover the full 4px outline.
    lifted = np.clip(background.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    image[outline_only] = lifted[outline_only]
    image[core > 0] = 4

    authority = build_mask(image.shape[:2], [box], image) > 127
    inpainter = FastInpainter()
    output = inpainter.inpaint(image.copy(), [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert metrics["stroke_refined_regions"] == 1
    assert metrics["stroke_authority_gradient_regions"] == 1
    assert metrics["lama_model_runs"] == 0
    assert not np.any(changed & (~authority))
    assert float(changed[outline_only].mean()) > 0.97

    text_support = outline > 0
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[text_support].mean()
    )
    assert mae < 4.0


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
