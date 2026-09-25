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
    _edge_recrop_bounds_sequence,
    _expand_ocr_crop_bounds,
    _expanded_context_crop_bounds,
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


def test_mask_crop_may_expand_but_never_contract_detector_context():
    image = np.zeros((180, 240, 3), np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[20:45, 0:30] = 255
    x1, y1, x2, y2 = _ocr_crop_bounds(image.shape, _box(mask=mask))

    assert x1 == 0
    assert y1 <= 10
    assert x2 >= 150
    assert y2 >= 130


def test_inner_stroke_mask_never_shrinks_recognition_below_detector_context():
    image = np.zeros((180, 240, 3), np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[25:45, 35:65] = 255

    bounds = _ocr_crop_bounds(image.shape, _box(mask=mask))

    assert bounds[0] <= 10
    assert bounds[1] <= 10
    assert bounds[2] >= 150
    assert bounds[3] >= 130


def test_expanded_context_retry_is_strictly_larger_when_page_space_allows():
    image = np.zeros((240, 320, 3), np.uint8)
    base = _ocr_crop_bounds(image.shape, _box())
    retry = _expanded_context_crop_bounds(image.shape, _box())

    assert retry[0] < base[0]
    assert retry[1] < base[1]
    assert retry[2] > base[2]
    assert retry[3] > base[3]


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


def test_geometry_deferred_text_segmenter_still_runs_batch_ocr():
    wide = _box(
        id="box_5232e98aed3a4e1b",
        x1=0,
        y1=4193,
        x2=900,
        y2=4537,
        confidence=0.9258392453193665,
        semantic_type="free_text",
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
        safe_to_inpaint=False,
        needs_review=True,
        deferred_reason="box_width_limit",
    )
    assert ocr_target_skip_reason(wide) is None

    scheduler = _box(
        source_role="scheduler",
        deferred_reason="focus_budget_exhausted",
    )
    assert ocr_target_skip_reason(scheduler) == "deferred-review-region"


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


def test_edge_recrop_bounds_sequence_grows_from_original_crop_without_cumulative_drift():
    assert _edge_recrop_bounds_sequence((400, 500, 3), (100, 100, 200, 200)) == (
        (76, 76, 224, 224),
        (52, 52, 248, 248),
        (4, 4, 296, 296),
    )


def test_edge_recrop_bounds_sequence_deduplicates_page_clamped_retries():
    assert _edge_recrop_bounds_sequence((100, 120, 3), (0, 0, 120, 100)) == ()


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


class _DetailedSequenceOCR:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def read_detailed(self, _rgb, _lang, *, target_mode="all"):
        result = self.results[self.calls]
        self.calls += 1
        return result


class _NoopOCRPipeline:
    pass


def _ocr_service_with_image(results):
    service = OCRService(_DetailedSequenceOCR(results), _NoopOCRPipeline())
    image = np.full((240, 320, 3), 255, np.uint8)
    service._cached_source_image = lambda _path: image
    return service


def test_empty_story_candidate_recovers_only_when_expanded_text_is_target_anchored():
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
        text_bounds=(55, 45, 150, 85),
        input_shape=(160, 220),
    )
    service = _ocr_service_with_image([empty, recovered])

    text = service._read_box_text("unused.png", _box(), "en")

    assert service.ocr.calls == 2
    assert text == "H-HE'S USING THAT?"
    assert service._result_local.metadata["context_retry_applied"] is True


def test_incomplete_coverage_retry_cannot_replace_target_with_neighbour_text():
    base = OCRReadResult(
        "STRIKE!",
        0.96,
        "fake",
        "horizontal",
        1,
        quality="review",
        quality_reason="incomplete-coverage",
        coverage=0.40,
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
        coverage=1.0,
        text_bounds=(190, 120, 218, 150),
        input_shape=(160, 220),
    )
    service = _ocr_service_with_image([base, neighbour])

    text = service._read_box_text("unused.png", _box(), "en")

    assert service.ocr.calls == 2
    assert text == "STRIKE!"
    assert service._result_local.metadata["quality_reason"] == "incomplete-coverage"
