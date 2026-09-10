import numpy as np
import pytest
import cv2
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.detector.mask_builder import build_mask
from app.render.text_renderer import _fit_text, render_text_in_box
from app.pipeline import ChapterPipeline


def _probability_canvas() -> np.ndarray:
    """Core glyph with anti-aliased outline, coloured edge and glow support."""
    probabilities = np.zeros((21, 29), dtype=np.float32)
    probabilities[7:14, 10:19] = 0.92  # confident glyph core
    probabilities[6:15, 9:20] = np.maximum(probabilities[6:15, 9:20], 0.38)
    probabilities[5:16, 8:21] = np.maximum(probabilities[5:16, 8:21], 0.33)
    probabilities[0:3, 0:3] = 0.40  # disconnected artwork must not be kept
    return probabilities


@pytest.mark.parametrize("style", ["outlined", "coloured", "glow_shadow"])
def test_text_mask_hysteresis_keeps_connected_stylized_edges_not_artwork(style):
    mask = YoloDetector._decode_text_mask_hysteresis(_probability_canvas())

    # The support reaches beyond the high-confidence core, proving the decode
    # does not drop outlines/shadow merely because their probability < 0.50.
    assert mask[5, 8] == 255
    assert mask[15, 20] == 255
    # It must still fail closed for unrelated low-confidence artwork.
    assert not np.any(mask[0:3, 0:3])


def test_padded_free_text_mask_is_not_a_rectangle_fallback():
    local_mask = YoloDetector._decode_text_mask_hysteresis(_probability_canvas())
    box = BubbleBox(
        20, 15, 49, 36, 0.9, local_mask,
        source_role="text_segmenter",
        source_model="text_segmenter.onnx",
        semantic_type="free_text",
        safe_to_inpaint=True,
    )
    page_mask = build_mask((60, 80), [box])

    assert page_mask[15:36, 20:49].sum() > 0
    # The disconnected low-probability corner and every outside page pixel
    # remain protected; no raw detection rectangle was introduced.
    assert page_mask[15:18, 20:23].sum() == 0
    assert page_mask[0:10, :].sum() == 0


def test_mser_proposal_requires_independent_segmenter_mask_for_promotion():
    proposal = BubbleBox(
        20, 20, 100, 55, 0.4, None,
        source_model="opencv_mser", semantic_type="free_text",
        safe_to_inpaint=False, needs_review=True,
    )
    raw_mser = BubbleBox(
        24, 24, 96, 51, 0.5, None,
        source_model="opencv_mser", semantic_type="free_text",
        safe_to_inpaint=False,
    )
    verified_segmenter = BubbleBox(
        24, 24, 96, 51, 0.9, np.full((27, 72), 255, np.uint8),
        source_role="text_segmenter", semantic_type="free_text",
        safe_to_inpaint=True,
    )

    assert not CombinedTextDetector._recovery_has_segmenter_evidence(proposal, [raw_mser])
    assert CombinedTextDetector._recovery_has_segmenter_evidence(proposal, [verified_segmenter])


class _ResidueTextDetector:
    def _detect_single_plain(self, image, offset_x, offset_y):
        return [BubbleBox(
            offset_x + 8, offset_y + 8, offset_x + 20, offset_y + 18,
            0.9, np.full((10, 12), 255, np.uint8), source_role="text_segmenter",
        )]

    @staticmethod
    def _with_semantics(box):
        return BubbleBox(
            box.x1, box.y1, box.x2, box.y2, box.confidence, box.mask,
            source_role="text_segmenter", source_model="text_segmenter.onnx",
            semantic_type="free_text", safe_to_inpaint=True,
        )


def test_post_inpaint_verifier_marks_partial_mask_residue_for_review():
    detector = CombinedTextDetector.__new__(CombinedTextDetector)
    detector.text_detector = _ResidueTextDetector()
    source = BubbleBox(
        0, 0, 40, 35, 0.9, np.full((35, 40), 255, np.uint8),
        source_role="text_segmenter", safe_to_inpaint=True,
    )
    residue = detector.verify_post_inpaint_residue(
        np.full((60, 80, 3), 180, np.uint8), [source]
    )

    assert len(residue) == 1
    assert residue[0].safe_to_inpaint is False
    assert residue[0].needs_review is True
    assert residue[0].deferred_reason == "post_inpaint_text_residue"


class _PipelineDetector:
    def __init__(self, source, residue):
        self._source = source
        self._residue = residue

    def detect(self, image, **_kwargs):
        return [self._source]

    @staticmethod
    def last_metrics():
        return {}

    def verify_post_inpaint_residue(self, image, authorized):
        assert authorized
        return [self._residue]


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


def test_pipeline_never_marks_post_inpaint_residue_verified(tmp_path):
    source = BubbleBox(
        10, 10, 40, 30, 0.9, np.full((20, 30), 255, np.uint8),
        source_role="text_segmenter", source_model="text_segmenter.onnx",
        semantic_type="free_text", safe_to_inpaint=True,
    )
    residue = BubbleBox(
        15, 14, 28, 25, 0.8, np.full((11, 13), 255, np.uint8),
        source_role="text_segmenter", source_model="text_segmenter.onnx",
        semantic_type="free_text", needs_review=True,
        deferred_reason="post_inpaint_text_residue",
    )
    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline._detector = _PipelineDetector(source, residue)
    pipeline._inpainter = _PipelineInpainter()
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), np.full((60, 80, 3), 200, np.uint8))

    result = pipeline._process_page(image_path, tmp_path)

    assert result["detection_state"] == "needs_review"
    assert result["cleanup_verified"] is False
    assert result["detection_issues"] == ["post_inpaint_text_residue"]
    assert result["residue_regions"][0]["deferred_reason"] == "post_inpaint_text_residue"


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
