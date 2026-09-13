import threading

import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.fast_residue_detector import (
    FastResidueAdaptiveFocusCombinedTextDetector,
)
from app.parameters import DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE


def _box(x1, y1, x2, y2):
    return BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.9,
        np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8),
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
        semantic_type="speech_bubble",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
    )


def test_residue_groups_coalesce_nearby_windows_but_not_distant_ones():
    a = _box(10, 10, 50, 35)
    b = _box(48, 12, 88, 37)
    c = _box(300, 300, 340, 330)
    groups = FastResidueAdaptiveFocusCombinedTextDetector._plan_residue_groups(
        [
            (a, (0, 0, 62, 47)),
            (b, (36, 0, 100, 49)),
            (c, (288, 288, 352, 342)),
        ]
    )
    assert len(groups) == 2
    assert len(groups[0]["sources"]) == 2
    assert len(groups[1]["sources"]) == 1


def test_residue_groups_refuse_union_that_exceeds_source_side_limit():
    max_side = int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE)
    method = FastResidueAdaptiveFocusCombinedTextDetector._can_merge_roi
    ok, union = method(
        (0, 0, max_side - 10, 30),
        (max_side - 10, 0, max_side + 20, 30),
    )
    assert union[2] - union[0] > max_side
    assert ok is False


class _FakeTextDetector:
    def __init__(self):
        self.calls = 0

    def _detect_single_plain(self, crop, x1, y1):
        self.calls += 1
        return [
            BubbleBox(
                x1 + 20,
                y1 + 15,
                x1 + 35,
                y1 + 28,
                0.8,
                np.full((13, 15), 255, dtype=np.uint8),
                source_model="text_segmenter.onnx",
                source_role="text_segmenter",
                semantic_type="text",
                mask_source="text_segmenter",
            )
        ]

    @staticmethod
    def _with_semantics(box):
        return box


def test_coalesced_verifier_uses_one_model_call_and_keeps_residue_evidence():
    detector = FastResidueAdaptiveFocusCombinedTextDetector.__new__(
        FastResidueAdaptiveFocusCombinedTextDetector
    )
    detector._residue_metrics_local = threading.local()
    detector._residue_metrics_lock = threading.Lock()
    detector._residue_totals = {}
    detector.text_detector = _FakeTextDetector()

    first = _box(10, 10, 50, 35)
    second = _box(48, 12, 88, 37)
    image = np.full((120, 140, 3), 255, dtype=np.uint8)

    residue = detector.verify_post_inpaint_residue(image, [first, second])

    assert detector.text_detector.calls == 1
    assert len(residue) == 1
    assert residue[0].deferred_reason == "post_inpaint_text_residue"
    metrics = detector.last_residue_metrics()
    assert metrics["scheduled_sources"] == 2
    assert metrics["groups"] == 1
    assert metrics["model_calls"] == 1
    assert metrics["merged_sources"] == 1

    totals = detector.residue_metrics_snapshot()
    assert totals["verification_calls"] == 1
    assert totals["scheduled_sources"] == 2
    assert totals["groups"] == 1
    assert totals["model_calls"] == 1
    assert totals["merged_sources"] == 1

    reset = detector.residue_metrics_snapshot(reset=True)
    assert reset == totals
    assert detector.residue_metrics_snapshot() == {}
