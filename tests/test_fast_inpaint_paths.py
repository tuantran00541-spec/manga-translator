import numpy as np
from unittest.mock import patch

from app.detector.bubble_detector import BubbleBox
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.optimized_pipeline import OptimizedChapterPipeline
from app.pipeline import ChapterPipeline


def test_verified_flat_bubble_skips_lama():
    image = np.full((160, 220, 3), 245, dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[15:22, 18:82] = 255
    mask[29:36, 28:72] = 255
    image[50:100, 60:160][mask > 127] = 20

    box = BubbleBox(
        60,
        50,
        160,
        100,
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

    inpainter = FastInpainter()
    output = inpainter.inpaint(image, [box])
    metrics = inpainter.last_metrics()

    assert metrics["bubble_fast_fill_regions"] == 1
    assert metrics["lama_model_runs"] == 0
    assert float(output[65:86, 78:142].mean()) > float(image[65:86, 78:142].mean())


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


def test_optimized_pipeline_preloads_after_selecting_worker_profile():
    events = []

    class FakeInpainter:
        def prepare_for_page_workers(self, workers):
            events.append(("prepare", workers))

        def preload(self):
            events.append(("preload",))

    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    pipeline._inpainter = FakeInpainter()

    def fake_process(self, chapter_id, page_indices, workers):
        events.append(("process", chapter_id, page_indices, workers))
        return {"ok": True}

    with patch(
        "app.optimized_pipeline.responsive_process_workers",
        return_value=2,
    ), patch.object(ChapterPipeline, "process_pages", fake_process):
        result = pipeline.process_pages("deadbeef", [0, 1], workers=4)

    assert result == {"ok": True}
    assert events == [
        ("prepare", 2),
        ("preload",),
        ("process", "deadbeef", [0, 1], 2),
    ]
