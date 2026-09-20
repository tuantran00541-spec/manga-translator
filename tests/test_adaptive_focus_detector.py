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


def test_roi_first_focus_skips_full_page_when_verified_roi_resolves_proposal():
    from dataclasses import replace

    from app.detector.adaptive_focus_detector import _focus_text_detect_roi_first

    image = np.zeros((1800, 900, 3), dtype=np.uint8)
    proposal = BubbleBox(
        200, 500, 400, 760, 0.9, None,
        semantic_type="speech_bubble",
        source_model="bubble_yolo.onnx",
        source_role="bubble_detector",
    )

    class Detector:
        model_role = "text_segmenter"

        def __init__(self):
            self.calls = []

        def _detect_single_plain(self, crop, offset_x, offset_y):
            self.calls.append((crop.shape[:2], offset_x, offset_y))
            mask = np.full((80, 100), 255, dtype=np.uint8)
            return [
                BubbleBox(
                    250, 580, 350, 660, 0.95, mask,
                    source_model="text_segmenter.onnx",
                    class_name="text_comic",
                    semantic_type="text",
                    source_role="text_segmenter",
                )
            ]

        @staticmethod
        def _nms_boxes(boxes):
            return list(boxes)

        @staticmethod
        def _filter_invalid(boxes, _w, _h):
            return list(boxes)

        @staticmethod
        def _with_semantics(box):
            return replace(
                box,
                mask_source="text_segmenter",
                safe_to_inpaint=box.verified_mask,
                ocr_eligible=box.verified_mask,
                needs_review=not box.verified_mask,
            )

    detector = Detector()
    _boxes, metrics, deferred = _focus_text_detect_roi_first(
        detector,
        image,
        [proposal],
    )

    assert deferred == []
    assert metrics["focus_roi_first_page"] == 1
    assert metrics["focus_full_page_calls"] == 0
    assert metrics["focus_full_page_skipped"] == 1
    assert len(detector.calls) == metrics["focus_chip_calls"]
    assert all(shape != image.shape[:2] for shape, _x, _y in detector.calls)


def test_roi_first_focus_falls_back_when_proposal_is_unresolved():
    from app.detector.adaptive_focus_detector import _focus_text_detect_roi_first

    image = np.zeros((1800, 900, 3), dtype=np.uint8)
    proposal = BubbleBox(
        200, 500, 400, 760, 0.9, None,
        semantic_type="free_text",
        source_model="opencv_mser",
        source_role="recovery",
    )

    class Detector:
        model_role = "text_segmenter"

        def __init__(self):
            self.calls = []

        def _detect_single_plain(self, crop, offset_x, offset_y):
            self.calls.append((crop.shape[:2], offset_x, offset_y))
            return []

        @staticmethod
        def _nms_boxes(boxes):
            return list(boxes)

        @staticmethod
        def _filter_invalid(boxes, _w, _h):
            return list(boxes)

        @staticmethod
        def _with_semantics(box):
            return box

    detector = Detector()
    _boxes, metrics, _deferred = _focus_text_detect_roi_first(
        detector,
        image,
        [proposal],
    )

    assert metrics["focus_full_page_calls"] == 1
    assert metrics["focus_fallback_unresolved"] == 1
    assert any(
        shape == image.shape[:2] and x == 0 and y == 0
        for shape, x, y in detector.calls
    )


def test_roi_first_clusters_nearby_proposals_into_one_authority_crop():
    from dataclasses import replace

    from app.detector.adaptive_focus_detector import _focus_text_detect_roi_first

    image = np.zeros((1800, 900, 3), dtype=np.uint8)
    proposals = [
        BubbleBox(
            200, 500, 400, 760, 0.90, None,
            semantic_type="speech_bubble",
            source_model="bubble_yolo.onnx",
            source_role="bubble_detector",
        ),
        BubbleBox(
            430, 520, 630, 780, 0.88, None,
            semantic_type="speech_bubble",
            source_model="bubble_yolo.onnx",
            source_role="bubble_detector",
        ),
    ]

    class Detector:
        model_role = "text_segmenter"

        def __init__(self):
            self.calls = []

        def _detect_single_plain(self, crop, offset_x, offset_y):
            self.calls.append((crop.shape[:2], offset_x, offset_y))
            first_mask = np.full((80, 100), 255, dtype=np.uint8)
            second_mask = np.full((80, 100), 255, dtype=np.uint8)
            return [
                BubbleBox(
                    250, 580, 350, 660, 0.95, first_mask,
                    source_model="text_segmenter.onnx",
                    class_name="text_comic",
                    semantic_type="text",
                    source_role="text_segmenter",
                ),
                BubbleBox(
                    480, 600, 580, 680, 0.94, second_mask,
                    source_model="text_segmenter.onnx",
                    class_name="text_comic",
                    semantic_type="text",
                    source_role="text_segmenter",
                ),
            ]

        @staticmethod
        def _nms_boxes(boxes):
            return list(boxes)

        @staticmethod
        def _filter_invalid(boxes, _w, _h):
            return list(boxes)

        @staticmethod
        def _with_semantics(box):
            return replace(
                box,
                mask_source="text_segmenter",
                safe_to_inpaint=box.verified_mask,
                ocr_eligible=box.verified_mask,
                needs_review=not box.verified_mask,
            )

    detector = Detector()
    _boxes, metrics, deferred = _focus_text_detect_roi_first(
        detector,
        image,
        proposals,
    )

    assert deferred == []
    assert metrics["focus_chip_calls"] == 1
    assert metrics["focus_clustered_groups"] == 1
    assert metrics["focus_clustered_covered_proposals"] == 2
    assert metrics["focus_full_page_calls"] == 0
    assert len(detector.calls) == 1
