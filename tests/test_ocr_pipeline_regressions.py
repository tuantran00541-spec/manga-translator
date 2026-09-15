import numpy as np

from app.ocr.paddle_v6 import (
    OCRReadResult,
    PaddleV6OCR,
    _result_rank,
    _should_selective_retry,
)
from app.ocr.quality import classify_ocr_quality
from app.ocr.service import (
    OCRService,
    _expand_ocr_crop_bounds,
    _prefer_edge_recrop,
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


def test_punctuation_only_dialogue_is_preserved_when_confident_and_complete():
    for text in ("?!", "!!!!", "…….", "…!", "..."):
        quality = classify_ocr_quality(
            text,
            "en",
            confidence=0.90,
            coverage=1.0,
        )
        assert (quality.status, quality.reason) == ("good", None)


def test_punctuation_only_still_respects_confidence_and_completeness_gates():
    low_conf = classify_ocr_quality("?!", "en", confidence=0.20, coverage=1.0)
    clipped = classify_ocr_quality(
        "!!!!", "en", confidence=0.99, coverage=1.0, may_be_truncated=True
    )
    unknown_conf = classify_ocr_quality("……", "en", confidence=None, coverage=1.0)

    assert (low_conf.status, low_conf.reason) == ("reject", "very-low-confidence")
    assert (clipped.status, clipped.reason) == ("review", "crop-edge-text")
    assert (unknown_conf.status, unknown_conf.reason) == ("review", "punctuation-only")


def test_lone_decorative_punctuation_remains_noise():
    assert classify_ocr_quality("”", "en", confidence=0.99).status == "reject"
    assert classify_ocr_quality(".", "en", confidence=0.99).status == "reject"


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


def test_retry_rank_prefers_complete_review_over_crop_edge_high_confidence():
    clipped = OCRReadResult(
        "CHILD\nSAVETHIS",
        0.9995,
        "fake",
        "horizontal",
        2,
        "review",
        "crop-edge-text",
        1.0,
    )
    complete = OCRReadResult(
        "SAVE THIS CHILD",
        0.61,
        "fake",
        "horizontal",
        1,
        "review",
        "low-confidence",
        1.0,
    )
    assert _result_rank(complete) > _result_rank(clipped)


class _SequencePipeline:
    def __init__(self):
        self.calls = 0

    def predict(self, *, input):
        scores = [0.40, 0.45, 0.99]
        score = scores[min(self.calls, len(scores) - 1)]
        self.calls += 1
        return [{
            "rec_texts": ["retry target"],
            "rec_scores": [score],
            "rec_polys": [
                [[25, 45], [190, 45], [190, 70], [25, 70]],
            ],
        }]


def test_grayscale_retry_runs_only_when_first_retry_remains_suspicious():
    ocr = PaddleV6OCR()
    pipeline = _SequencePipeline()
    ocr._get_pipeline = lambda _key: pipeline
    image = np.full((120, 220, 3), 255, np.uint8)

    result = ocr.read(image, "en", target_mode="all")

    assert pipeline.calls == 3
    assert result.quality == "good"
    assert result.confidence == 0.99
    assert result.retry_applied


def test_edge_recrop_bounds_add_only_bounded_context():
    assert _expand_ocr_crop_bounds((100, 120, 3), (10, 20, 80, 90)) == (0, 0, 104, 100)


def test_edge_recrop_prefers_quality_upgrade_without_extra_regions():
    assert _prefer_edge_recrop(
        base_text="(..H?",
        base_quality="review",
        base_reason="crop-edge-text",
        base_confidence=0.84,
        base_region_count=1,
        expanded_text="...HUH?",
        expanded_quality="good",
        expanded_reason=None,
        expanded_confidence=0.95,
        expanded_region_count=1,
    )
    assert not _prefer_edge_recrop(
        base_text="(..H?",
        base_quality="review",
        base_reason="crop-edge-text",
        base_confidence=0.84,
        base_region_count=1,
        expanded_text="NEIGHBOUR TEXT ...HUH?",
        expanded_quality="good",
        expanded_reason=None,
        expanded_confidence=0.99,
        expanded_region_count=2,
    )


def test_edge_recrop_can_repair_order_but_not_add_lines():
    assert _prefer_edge_recrop(
        base_text="CHILD\nSAVETHIS",
        base_quality="review",
        base_reason="crop-edge-text",
        base_confidence=0.9994,
        base_region_count=2,
        expanded_text="SAVETHIS\nCHILD",
        expanded_quality="review",
        expanded_reason="crop-edge-text",
        expanded_confidence=0.9992,
        expanded_region_count=2,
    )
    assert not _prefer_edge_recrop(
        base_text="CHILD\nSAVETHIS",
        base_quality="review",
        base_reason="crop-edge-text",
        base_confidence=0.9994,
        base_region_count=2,
        expanded_text="SAVETHIS\nCHILD\nNEXT",
        expanded_quality="review",
        expanded_reason="crop-edge-text",
        expanded_confidence=0.9999,
        expanded_region_count=3,
    )
