import numpy as np

from app.ocr.paddle_v6 import OCRReadResult, PaddleV6OCR, _should_selective_retry
from app.ocr.quality import classify_ocr_quality
from app.ocr.service import (
    OCRService,
    _ocr_crop_bounds,
    ocr_target_mode_for_box,
    ocr_target_skip_reason,
)


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


def test_grouped_dialogue_and_free_text_never_use_centered_selection():
    assert ocr_target_mode_for_box(_box(semantic_type="speech_bubble")) == "all"
    assert ocr_target_mode_for_box(_box(semantic_type="narration")) == "all"
    assert ocr_target_mode_for_box(_box(semantic_type="free_text")) == "all"
    assert ocr_target_mode_for_box(_box(grouped=True, semantic_type="text")) == "all"


def test_centered_selection_is_reserved_for_explicit_horizontal_single_line_text():
    line = _box(
        semantic_type="text",
        source_role="bubble_detector",
        x2=230,
        y2=70,
    )
    assert ocr_target_mode_for_box(line) == "centered"
    assert ocr_target_mode_for_box(_box(semantic_type="text", source_role="bubble_detector")) == "all"


def test_mask_crop_gets_bounded_page_context_only_at_detector_edge():
    image = np.zeros((180, 240, 3), np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[20:45, 0:30] = 255  # text support meets detector's left edge
    x1, y1, x2, y2 = _ocr_crop_bounds(image.shape, _box(mask=mask))

    assert x1 < 30  # former implementation clamped this to detector x1
    assert y1 > 30  # untouched sides still use the normal local padding
    assert x2 < 130
    assert y2 < 110


def test_high_confidence_partial_mask_coverage_is_review_not_good():
    quality = classify_ocr_quality(
        "Perfect-looking middle line",
        "en",
        confidence=0.98,
        coverage=0.42,
    )
    assert (quality.status, quality.reason) == ("review", "incomplete-coverage")


def test_mask_coverage_compares_detected_span_to_segmented_multiline_support():
    mask = np.zeros((80, 100), np.uint8)
    mask[0:70, 10:90] = 255
    result = OCRReadResult(
        "middle", 0.99, "fake", "horizontal", 1,
        text_bounds=(15, 28, 85, 42), input_shape=(80, 100),
    )
    coverage = OCRService._mask_text_coverage(
        _box(mask=mask), (30, 30, 130, 110), result
    )
    assert coverage is not None and coverage < 0.30


def test_raw_recovery_and_explicit_non_dialogue_targets_are_not_planned_for_batch_ocr():
    assert ocr_target_skip_reason(_box(source_model="opencv_mser", source_role="unknown"))
    assert ocr_target_skip_reason(_box(class_name="text_recovery"))
    assert ocr_target_skip_reason(_box(class_name="sfx"))
    assert ocr_target_skip_reason(_box(class_name="credits"))
    assert ocr_target_skip_reason(_box()) is None


class _FakePipeline:
    def predict(self, *, input):
        return [{
            "rec_texts": ["First line", "Middle line", "Last line"],
            "rec_scores": [0.98, 0.98, 0.98],
            "rec_polys": [
                [[20, 18], [190, 18], [190, 34], [20, 34]],
                [[20, 52], [190, 52], [190, 68], [20, 68]],
                [[20, 86], [190, 86], [190, 102], [20, 102]],
            ],
        }]


def test_all_mode_transcribes_all_detected_dialogue_lines_without_global_retry():
    ocr = PaddleV6OCR()
    pipeline = _FakePipeline()
    ocr._get_pipeline = lambda _key: pipeline
    image = np.full((120, 220, 3), 255, np.uint8)

    result = ocr.read(image, "en", target_mode="all")

    assert result.text.splitlines() == ["First line", "Middle line", "Last line"]
    assert result.region_count == 3
    assert result.target_mode == "all"
    assert not result.retry_applied


def test_selective_retry_only_runs_for_suspicious_results_and_is_bounded_by_crop_size():
    good = OCRReadResult("clear", 0.99, "fake", "horizontal", 1, "good")
    review = OCRReadResult("maybe", 0.99, "fake", "horizontal", 1, "review")
    assert not _should_selective_retry(good, np.zeros((80, 160, 3), np.uint8))
    assert _should_selective_retry(review, np.zeros((80, 160, 3), np.uint8))
    assert not _should_selective_retry(review, np.zeros((2000, 2000, 3), np.uint8))
