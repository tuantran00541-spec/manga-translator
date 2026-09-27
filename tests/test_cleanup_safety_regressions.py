import numpy as np
import pytest
import cv2
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.boxes import BubbleBox
from app.detector.mask_builder import build_mask
from app.render.text_renderer import _fit_text, render_text_in_box
from app.pipeline import ChapterPipeline


def _letter_mask() -> np.ndarray:
    mask = np.zeros((21, 29), dtype=np.uint8)
    mask[5:16, 8:21] = 255
    return mask


def test_padded_free_text_mask_is_not_a_rectangle_fallback():
    local_mask = _letter_mask()
    box = BubbleBox(
        20, 15, 49, 36, 0.9, local_mask,
        source_role="text_segmenter",
        source_model="kiuyha_text_1280.onnx",
        semantic_type="free_text",
        safe_to_inpaint=True,
    )
    page_mask = build_mask((60, 80), [box])

    assert page_mask[15:36, 20:49].sum() > 0
    assert page_mask[15:18, 20:23].sum() == 0
    assert page_mask[0:10, :].sum() == 0


def test_validation_max_dilation_covers_both_runtime_dilation_branches():
    local_mask = np.zeros((21, 29), dtype=np.uint8)
    local_mask[10, 14] = 255
    box = BubbleBox(
        20, 15, 49, 36, 0.9, local_mask,
        source_role="text_segmenter",
        source_model="kiuyha_text_1280.onnx",
        semantic_type="free_text",
        safe_to_inpaint=True,
    )

    smooth = np.full((60, 80, 3), 230, dtype=np.uint8)
    checker = smooth.copy()
    yy, xx = np.indices(checker.shape[:2])
    checker[(xx + yy) % 2 == 0] = 20

    normal = build_mask((60, 80), [box], smooth)
    adaptive = build_mask((60, 80), [box], checker)
    validation = build_mask(
        (60, 80),
        [box],
        smooth,
        force_max_dilation=True,
    )

    assert np.all(validation[normal > 127] > 127)
    assert np.all(validation[adaptive > 127] > 127)
    assert np.count_nonzero(validation) > np.count_nonzero(normal)
    assert np.array_equal(validation, adaptive)


class _PipelineDetector:
    def __init__(self, source, leftover):
        self._source = source
        self._leftover = leftover

    def detect(self, image, **_kwargs):
        return [self._source] if self._source is not None else []

    @staticmethod
    def last_metrics():
        return {}

    def leftover_boxes(self, image, first_pass):
        assert first_pass
        leftover, self._leftover = self._leftover, None  # the next pass finds it erased
        return [leftover] if leftover is not None else []


class _PipelineInpainter:
    lama_model_path = None
    session_loaded = False
    _prefer_dynamic = False
    serialized_inference = False

    @staticmethod
    def inpaint(image, boxes, **_kwargs):
        return image.copy()

    @staticmethod
    def inpaint_mask(image, _mask, **_kwargs):
        return image.copy()

    @staticmethod
    def last_metrics():
        return {}


def _process(tmp_path, leftover, text=True):
    source = BubbleBox(
        10, 10, 40, 30, 0.9, np.full((20, 30), 255, np.uint8),
        source_role="text_segmenter", source_model="kiuyha_text_1280.onnx",
        semantic_type="free_text", safe_to_inpaint=True,
    ) if text else None
    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline._detector = _PipelineDetector(source, leftover)
    pipeline._inpainter = _PipelineInpainter()
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), np.full((60, 80, 3), 200, np.uint8))
    return pipeline._process_page(image_path, tmp_path)


def test_second_pass_leftover_is_saved_with_the_first_pass_box(tmp_path):
    leftover = BubbleBox(20, 15, 60, 35, 0.8, np.full((20, 40), 255, np.uint8), source_role="text_segmenter")
    result = _process(tmp_path, leftover)

    (box,) = result["boxes"]
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (10, 10, 60, 35)
    assert result["processing_metrics"]["detector"]["second_pass_boxes"] == 1
    assert result["cleanup_verified"] is True


def test_clean_page_without_leftovers_is_verified(tmp_path):
    result = _process(tmp_path, None)

    assert result["residue_checked"] is True
    assert result["cleanup_verified"] is True
    assert result["detection_issues"] == []


@pytest.mark.parametrize("background", [(255, 255, 255), (0, 0, 0)])
def test_white_and_black_caption_boxes_keep_a_readable_auto_font(background):
    image = Image.new("RGB", (220, 120), background)
    draw = ImageDraw.Draw(image)
    size, lines, fits = _fit_text(
        draw,
        "A concise caption that wraps naturally.",
        190,
        80,
        str(DEFAULT_FONT),
        minimum_size=12,
    )
    assert fits
    assert size >= 12
    assert lines
    rendered = render_text_in_box(
        image, "A concise caption that wraps naturally.", (15, 20, 205, 100),
        bg_color="#ffffff" if background == (0, 0, 0) else "#000000",
    )
    assert rendered is image


def test_auto_fit_refuses_unreadable_transparent_dialogue():
    image = Image.new("RGB", (100, 60), "white")
    with pytest.raises(ValueError, match="minimum readable"):
        render_text_in_box(
            image,
            "This deliberately long translation cannot fit in this tiny box.",
            (45, 25, 55, 35),
            bg_color="transparent",
        )


def test_a_page_without_text_is_processed(tmp_path):
    result = _process(tmp_path, None, text=False)

    assert result["boxes"] == []
    assert result["processing_metrics"]["detector"]["second_pass_boxes"] == 0
