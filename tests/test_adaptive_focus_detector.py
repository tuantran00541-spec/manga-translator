import numpy as np

from app.detector.adaptive_focus_detector import (
    plan_adaptive_windows,
    plan_focus_chips,
)
from app.detector.bubble_detector import BubbleBox, YoloDetector


def test_adaptive_window_plan_matches_validated_v4_shapes():
    assert plan_adaptive_windows(1537) == [(0, 1024), (513, 1537)]
    assert plan_adaptive_windows(2400) == [(0, 1300), (1100, 2400)]
    assert plan_adaptive_windows(4096) == [
        (0, 1174),
        (974, 2148),
        (1948, 3122),
        (2922, 4096),
    ]


def test_focus_chips_are_bounded_and_surface_budget_deferred_regions():
    proposals = [
        BubbleBox(0, 0, 900, 4096, 0.9, semantic_type="free_text"),
        BubbleBox(30, 300, 180, 440, 0.9, semantic_type="speech_bubble"),
    ]

    chips, deferred = plan_focus_chips(4096, 900, proposals, max_chips=2)

    assert 1 <= len(chips) <= 2
    assert deferred
    assert all(0 <= x1 < x2 <= 900 and 0 <= y1 < y2 <= 4096 for x1, y1, x2, y2 in chips)
    assert all(max(x2 - x1, y2 - y1) <= 1344 for x1, y1, x2, y2 in chips)


def test_full_width_verified_segmenter_mask_keeps_stroke_authority():
    mask = np.zeros((80, 900), dtype=np.uint8)
    mask[20:30, 160:740] = 255
    box = BubbleBox(
        0, 10, 900, 90, 0.93, mask,
        source_model="text_segmenter.onnx",
        class_name="text_comic",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        source_role="text_segmenter",
    )

    result = YoloDetector._filter_invalid([box], 900, 1000)

    assert len(result) == 1
    assert result[0].safe_to_inpaint is True
    assert result[0].needs_review is False
    assert result[0].deferred_reason is None


def test_full_width_segmenter_without_verified_mask_stays_review_only():
    box = BubbleBox(
        0, 10, 900, 90, 0.93, None,
        source_model="text_segmenter.onnx",
        class_name="text_comic",
        semantic_type="free_text",
        mask_source="none",
        safe_to_inpaint=False,
        ocr_eligible=True,
        needs_review=True,
        source_role="text_segmenter",
    )

    result = YoloDetector._filter_invalid([box], 900, 1000)

    assert len(result) == 1
    assert result[0].safe_to_inpaint is False
    assert result[0].ocr_eligible is True
    assert result[0].needs_review is True
    assert "box_width_limit" in str(result[0].deferred_reason)
