import threading

import numpy as np
import pytest
import cv2
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.mask_builder import build_mask
from app.render.text_renderer import _fit_text, render_text_in_box
from app.pipeline import ChapterPipeline
from app.optimized_pipeline import OptimizedChapterPipeline
from app.one_shot_cleanup import OneShotProductionDetector, OneShotTextMaskDetector
from app.parameters import TEXT_CONF_THRESHOLD
from app.image_io import encode_mask, read_image


def test_one_shot_uses_only_normal_threshold_without_low_conf_rescue():
    detector = OneShotTextMaskDetector.__new__(OneShotTextMaskDetector)
    detector.RESCUE_CONF_THRESHOLD = 0.12

    normal = BubbleBox(
        10, 10, 30, 20, 0.82, np.full((10, 20), 255, np.uint8),
        source_role="text_segmenter",
        source_model="text_segmenter.onnx",
    )
    weak = BubbleBox(
        60, 12, 84, 24, 0.15, np.full((12, 24), 255, np.uint8),
        source_role="text_segmenter",
        source_model="text_segmenter.onnx",
    )

    detector._single_forward_outputs = lambda _image: (object(), object())
    thresholds = []

    def fake_postprocess(_outputs, _transform, threshold):
        thresholds.append(float(threshold))
        if float(threshold) >= float(TEXT_CONF_THRESHOLD):
            return [normal]
        return [normal, weak]

    detector._postprocess_at_threshold = fake_postprocess

    boxes, metrics = detector.detect(np.zeros((40, 100, 3), dtype=np.uint8))

    assert [round(float(box.confidence), 2) for box in boxes] == [0.82]
    assert boxes[0].safe_to_inpaint is True
    assert thresholds == [float(TEXT_CONF_THRESHOLD)]
    assert metrics["detector_forward_calls"] == 1
    assert metrics["normal_conf_boxes"] == 1
    assert metrics["low_conf_rescue"] == 0
    assert metrics["low_conf_rescue_boxes"] == 0

def _probability_canvas() -> np.ndarray:
    probabilities = np.zeros((21, 29), dtype=np.float32)
    probabilities[7:14, 10:19] = 0.92
    probabilities[6:15, 9:20] = np.maximum(probabilities[6:15, 9:20], 0.38)
    probabilities[5:16, 8:21] = np.maximum(probabilities[5:16, 8:21], 0.33)
    probabilities[0:3, 0:3] = 0.40
    return probabilities


@pytest.mark.parametrize("style", ["outlined", "coloured", "glow_shadow"])
def test_text_mask_hysteresis_keeps_connected_stylized_edges_not_artwork(style):
    mask = YoloDetector._decode_text_mask_hysteresis(_probability_canvas())

    assert mask[5, 8] == 255
    assert mask[15, 20] == 255
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
    assert page_mask[15:18, 20:23].sum() == 0
    assert page_mask[0:10, :].sum() == 0


def test_validation_max_dilation_covers_both_runtime_dilation_branches():
    local_mask = np.zeros((21, 29), dtype=np.uint8)
    local_mask[10, 14] = 255
    box = BubbleBox(
        20, 15, 49, 36, 0.9, local_mask,
        source_role="text_segmenter",
        source_model="text_segmenter.onnx",
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
    detector = OneShotProductionDetector.__new__(OneShotProductionDetector)
    detector.text_detector = _ResidueTextDetector()
    detector._residue_metrics_local = threading.local()
    detector._residue_metrics_lock = threading.Lock()
    detector._residue_totals = {}
    detector._residue_flat_gate_enabled = False
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
        return [self._residue] if self._residue is not None else []


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


def _residue_check_page(tmp_path, residue):
    source = BubbleBox(
        10, 10, 40, 30, 0.9, np.full((20, 30), 255, np.uint8),
        source_role="text_segmenter", source_model="text_segmenter.onnx",
        semantic_type="free_text", safe_to_inpaint=True,
    )
    pipeline = ChapterPipeline.__new__(ChapterPipeline)
    pipeline._detector = _PipelineDetector(source, residue)
    pipeline._inpainter = _PipelineInpainter()
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), np.full((60, 80, 3), 200, np.uint8))
    return pipeline._process_page(image_path, tmp_path)


_RESIDUE = BubbleBox(
    15, 14, 28, 25, 0.8, np.full((11, 13), 255, np.uint8),
    source_role="text_segmenter", source_model="text_segmenter.onnx",
    semantic_type="free_text", needs_review=True,
    deferred_reason="post_inpaint_text_residue",
)


def test_unchecked_cleanup_is_not_reported_as_verified(tmp_path):
    result = _residue_check_page(tmp_path, _RESIDUE)

    assert result["detection_state"] == "verified"
    assert result["detection_issues"] == []
    assert result["residue_regions"] == []
    assert result["residue_checked"] is False
    assert result["cleanup_verified"] is False
    assert result["processing_metrics"]["detector"]["residue_checked"] is False


def test_residue_check_flags_leftover_text(tmp_path, monkeypatch):
    monkeypatch.setattr("app.page_processing.DETECTOR_RESIDUE_VERIFY_ENABLED", True)

    result = _residue_check_page(tmp_path, _RESIDUE)

    assert result["residue_checked"] is True
    assert result["cleanup_verified"] is False
    assert result["detection_issues"] == ["post_inpaint_text_residue"]
    assert len(result["residue_regions"]) == 1


def test_residue_check_without_leftover_text_verifies_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr("app.page_processing.DETECTOR_RESIDUE_VERIFY_ENABLED", True)

    result = _residue_check_page(tmp_path, None)

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


class _VerifiedResidueRepairDetector:
    @staticmethod
    def verify_post_inpaint_residue(_image, _authorized):
        return []

    @staticmethod
    def last_residue_metrics():
        return {}


class _VerifiedResidueRepairInpainter:
    lama_model_path = None
    session_loaded = False
    _prefer_dynamic = True
    serialized_inference = False

    def __init__(self):
        self.last_mask = None

    def inpaint_mask(self, image, mask, *, force_lama=False):
        assert force_lama is True
        self.last_mask = mask.copy()
        result = image.copy()
        result[mask > 10] = (17, 29, 43)
        return result

    @staticmethod
    def last_metrics():
        return {"lama_model_runs": 1, "lama_model_ms": 1}


def test_verified_residue_mask_can_extend_repair_without_old_authority(tmp_path):
    original = np.full((80, 120, 3), 210, dtype=np.uint8)
    image_path = tmp_path / "original.png"
    clean_path = tmp_path / "clean.png"
    assert cv2.imwrite(str(image_path), original)
    assert cv2.imwrite(str(clean_path), original)

    residue_local = np.zeros((18, 24), dtype=np.uint8)
    residue_local[4:14, 5:19] = 255
    weak_mask = np.zeros((18, 24), dtype=np.uint8)
    weak_mask[7:11, 9:15] = 255

    result = {
        "tmp_clean": clean_path.as_posix(),
        "boxes": [
            {
                "x1": 58,
                "y1": 24,
                "x2": 82,
                "y2": 42,
                "confidence": 0.15,
                "mask": encode_mask(weak_mask),
                "source_model": "text_segmenter.onnx",
                "source_role": "text_segmenter",
                "class_name": "text",
                "semantic_type": "free_text",
                "mask_source": "text_segmenter",
                "safe_to_inpaint": False,
                "ocr_eligible": True,
                "needs_review": True,
                "deferred_reason": "low_confidence_unconfirmed",
            }
        ],
        "residue_regions": [
            {
                "x1": 58,
                "y1": 24,
                "x2": 82,
                "y2": 42,
                "confidence": 0.81,
                "mask": encode_mask(residue_local),
                "source_model": "text_segmenter.onnx",
                "source_role": "text_segmenter",
                "class_name": "text",
                "semantic_type": "free_text",
                "deferred_reason": "post_inpaint_text_residue",
            }
        ],
        "processing_metrics": {},
        "detection_issues": ["post_inpaint_text_residue"],
        "detection_state": "needs_review",
        "cleanup_verified": False,
        "needs_review": True,
    }

    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    pipeline._detector = _VerifiedResidueRepairDetector()
    pipeline._inpainter = _VerifiedResidueRepairInpainter()

    updated = pipeline._repair_post_inpaint_result(image_path, result, None)

    metrics = updated["processing_metrics"]["residue_repair"]
    assert metrics["attempted"] == 1
    assert metrics["verified_extension_pixels"] == int(np.count_nonzero(residue_local))
    assert metrics["repair_mask_pixels"] == int(np.count_nonzero(residue_local))
    assert metrics["outside_authority_changed_channel_values"] == 0
    assert pipeline._inpainter.last_mask is not None

    repaired = read_image(clean_path)
    authority = np.zeros(repaired.shape[:2], dtype=bool)
    authority[24:42, 58:82] = residue_local > 10
    assert np.all(repaired[authority] == np.array([17, 29, 43], dtype=np.uint8))
    assert np.array_equal(repaired[~authority], original[~authority])
    assert updated["residue_regions"] == []
    assert updated["cleanup_verified"] is True
