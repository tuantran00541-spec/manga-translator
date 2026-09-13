import numpy as np

from app.ocr.consistency import (
    ConsistencyOCRService,
    expanded_context_crop_bounds,
    recognition_safe_crop_bounds,
)
from app.ocr.paddle_v6 import OCRReadResult


def _box(**overrides):
    box = {
        "id": "box_1",
        "x1": 30,
        "y1": 30,
        "x2": 130,
        "y2": 110,
        "semantic_type": "speech_bubble",
        "source_role": "text_segmenter",
        "source_model": "text_segmenter.onnx",
        "ocr_eligible": True,
    }
    box.update(overrides)
    return box


def test_recognition_safe_crop_never_shrinks_to_inner_mask():
    image = np.zeros((180, 240, 3), np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[25:45, 35:65] = 255

    bounds = recognition_safe_crop_bounds(image.shape, _box(mask=mask))

    # Production detector-context baseline is bbox +/- OCR_BOX_CROP_PADDING (20).
    assert bounds[0] <= 10
    assert bounds[1] <= 10
    assert bounds[2] >= 150
    assert bounds[3] >= 130


def test_mask_support_may_expand_but_never_contract_detector_context():
    image = np.zeros((180, 240, 3), np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[20:45, 0:30] = 255

    bounds = recognition_safe_crop_bounds(image.shape, _box(mask=mask))

    assert bounds[0] == 0  # edge context expands left beyond detector baseline
    assert bounds[1] <= 10
    assert bounds[2] >= 150
    assert bounds[3] >= 130


def test_expanded_context_retry_is_strictly_larger_when_page_space_allows():
    image = np.zeros((240, 320, 3), np.uint8)
    base = recognition_safe_crop_bounds(image.shape, _box())
    retry = expanded_context_crop_bounds(image.shape, _box())

    assert retry[0] < base[0]
    assert retry[1] < base[1]
    assert retry[2] > base[2]
    assert retry[3] > base[3]


class _SequenceOCR:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def read_detailed(self, _rgb, _lang, *, target_mode="all"):
        result = self.results[self.calls]
        self.calls += 1
        return result


class _NoopPipeline:
    pass


def _service_with_image(results):
    service = ConsistencyOCRService(_SequenceOCR(results), _NoopPipeline())
    image = np.full((240, 320, 3), 255, np.uint8)
    service._cached_source_image = lambda _path: image
    return service


def test_crop_edge_result_gets_one_expanded_context_retry_and_uses_better_text():
    base = OCRReadResult(
        "IS TO SAY,\nGIVE UP, EVEN",
        0.98,
        "fake",
        "horizontal",
        2,
        quality="review",
        quality_reason="crop-edge-text",
        text_bounds=(20, 20, 160, 80),
        input_shape=(120, 180),
    )
    recovered = OCRReadResult(
        "IS TO SAY,\nGIVE UP, EVEN NOW.",
        0.97,
        "fake",
        "horizontal",
        3,
        quality="good",
        quality_reason=None,
        text_bounds=(40, 35, 180, 105),
        input_shape=(160, 220),
    )
    service = _service_with_image([base, recovered])

    text = service._read_box_text("unused.png", _box(), "en")
    metadata = service._result_local.metadata

    assert service.ocr.calls == 2
    assert text.endswith("NOW.")
    assert metadata["quality"] == "good"
    assert metadata["context_retry_applied"] is True
    assert metadata["retry_applied"] is True


def test_good_complete_result_does_not_pay_for_service_level_context_retry():
    good = OCRReadResult(
        "I UNDERSTAND.",
        0.99,
        "fake",
        "horizontal",
        1,
        quality="good",
        quality_reason=None,
        text_bounds=(30, 25, 130, 55),
        input_shape=(120, 180),
    )
    service = _service_with_image([good])

    text = service._read_box_text("unused.png", _box(), "en")

    assert service.ocr.calls == 1
    assert text == "I UNDERSTAND."
    assert service._result_local.metadata["context_retry_applied"] is False


def test_empty_story_candidate_can_recover_only_when_retry_text_is_target_anchored():
    empty = OCRReadResult(
        "",
        None,
        "fake",
        "horizontal",
        0,
        quality="reject",
        quality_reason="empty",
        text_bounds=None,
        input_shape=(120, 180),
    )
    recovered = OCRReadResult(
        "H-HE'S USING THAT?",
        0.94,
        "fake",
        "horizontal",
        1,
        quality="good",
        quality_reason=None,
        # Centre maps back inside the original detector box.
        text_bounds=(55, 45, 150, 85),
        input_shape=(160, 220),
    )
    service = _service_with_image([empty, recovered])

    text = service._read_box_text("unused.png", _box(), "en")

    assert service.ocr.calls == 2
    assert text == "H-HE'S USING THAT?"
    assert service._result_local.metadata["quality"] == "good"


def test_expanded_retry_does_not_replace_target_with_neighbour_text():
    base = OCRReadResult(
        "STRIKE!",
        0.96,
        "fake",
        "horizontal",
        1,
        quality="review",
        quality_reason="incomplete-coverage",
        text_bounds=(30, 25, 100, 55),
        input_shape=(120, 180),
    )
    neighbour = OCRReadResult(
        "UNRELATED BUBBLE",
        0.99,
        "fake",
        "horizontal",
        1,
        quality="good",
        quality_reason=None,
        # Centre maps well outside the original detector target.
        text_bounds=(190, 120, 218, 150),
        input_shape=(160, 220),
    )
    service = _service_with_image([base, neighbour])

    text = service._read_box_text("unused.png", _box(), "en")

    assert service.ocr.calls == 2
    assert text == "STRIKE!"
    assert service._result_local.metadata["quality_reason"] == "incomplete-coverage"


def test_punctuation_only_story_text_is_reviewable_instead_of_hard_rejected():
    punctuation = OCRReadResult(
        "……?!",
        0.99,
        "fake",
        "horizontal",
        1,
        quality="reject",
        quality_reason="no-content",
        text_bounds=(50, 45, 100, 65),
        input_shape=(120, 180),
    )
    service = _service_with_image([punctuation])

    text = service._read_box_text("unused.png", _box(), "en")
    metadata = service._result_local.metadata

    assert text == "……?!"
    assert metadata["quality"] == "review"
    assert metadata["quality_reason"] == "punctuation-only"


def test_punctuation_only_non_story_target_stays_rejected():
    punctuation = OCRReadResult(
        "……?!",
        0.99,
        "fake",
        "horizontal",
        1,
        quality="reject",
        quality_reason="no-content",
        text_bounds=(50, 45, 100, 65),
        input_shape=(120, 180),
    )
    service = _service_with_image([punctuation])

    service._read_box_text(
        "unused.png",
        _box(class_name="sfx", semantic_type="sfx", source_role="bubble_detector"),
        "en",
    )

    assert service._result_local.metadata["quality"] == "reject"
    assert service._result_local.metadata["quality_reason"] == "no-content"
