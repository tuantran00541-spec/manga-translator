import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter


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


def test_dense_smooth_authority_uses_surface_without_lama():
    h, w = 210, 300
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(226 + xx * 0.035 + yy * 0.020, 0, 255)
    background[..., 1] = np.clip(231 + xx * 0.030 + yy * 0.018, 0, 255)
    background[..., 2] = np.clip(238 + xx * 0.022 + yy * 0.014, 0, 255)
    image = background.copy()

    mask = np.full((92, 166), 255, dtype=np.uint8)
    box = _speech_box(67, 58, 233, 150, mask)
    image[78:88, 96:204] = 8
    image[104:114, 86:214] = 12
    image[130:140, 105:195] = 10

    expected_authority = build_mask(image.shape[:2], [box], image)
    inpainter = AdaptiveFastInpainter()
    inpainter._begin_metrics()
    # Force the conservative stroke-refine stage to reject so this test targets
    # exactly the dense full-authority surface fallback.
    inpainter._refine_dense_smooth_stroke_mask = lambda crop, authority: None

    before = image.copy()
    used = inpainter._try_stroke_authority_fill(image, box, None)
    metrics = inpainter.last_metrics()

    assert used is True
    assert metrics["dense_authority_gradient_regions"] == 1
    assert metrics["dense_authority_gradient_pixels"] > 0
    assert metrics["lama_model_runs"] == 0

    authority = expected_authority > 127
    changed = np.any(image != before, axis=2)
    assert not np.any(changed & (~authority))
    mae = float(
        np.abs(image.astype(np.int16) - background.astype(np.int16))[authority].mean()
    )
    assert mae < 5.0


def test_dense_textured_authority_rejects_surface_fallback():
    h, w = 210, 300
    yy, xx = np.mgrid[:h, :w]
    checker = (((xx // 5) + (yy // 5)) % 2) * 120 + 70
    image = np.stack(
        (
            checker,
            np.roll(checker, 2, axis=1),
            np.roll(checker, 3, axis=0),
        ),
        axis=2,
    ).astype(np.uint8)

    mask = np.full((92, 166), 255, dtype=np.uint8)
    box = _speech_box(67, 58, 233, 150, mask)
    before = image.copy()

    inpainter = AdaptiveFastInpainter()
    inpainter._begin_metrics()
    inpainter._refine_dense_smooth_stroke_mask = lambda crop, authority: None
    used = inpainter._try_stroke_authority_fill(image, box, None)
    metrics = inpainter.last_metrics()

    assert used is False
    assert metrics.get("dense_authority_gradient_regions", 0) == 0
    assert np.array_equal(image, before)
