import threading

import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.fast_residue_detector import (
    FastResidueAdaptiveFocusCombinedTextDetector,
)
from app.detector.parallel_focus_detector import (
    ParallelAdaptiveFocusCombinedTextDetector,
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


def _detector_with_fake_text(*, flat_gate=True, coalescing=None):
    detector = FastResidueAdaptiveFocusCombinedTextDetector.__new__(
        FastResidueAdaptiveFocusCombinedTextDetector
    )
    detector._residue_metrics_local = threading.local()
    detector._residue_metrics_lock = threading.Lock()
    detector._residue_totals = {}
    if flat_gate is not None:
        detector._residue_flat_gate_enabled = flat_gate
    if coalescing is not None:
        detector._residue_coalescing_enabled = coalescing
    detector.text_detector = _FakeTextDetector()
    return detector


def test_fast_residue_detector_inherits_parallel_focus_path():
    assert issubclass(
        FastResidueAdaptiveFocusCombinedTextDetector,
        ParallelAdaptiveFocusCombinedTextDetector,
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


def test_flat_clean_source_skips_neural_residue_verifier():
    detector = _detector_with_fake_text()
    source = _box(10, 10, 60, 40)
    image = np.full((100, 120, 3), 248, dtype=np.uint8)

    residue = detector.verify_post_inpaint_residue(image, [source])

    assert residue == []
    assert detector.text_detector.calls == 0
    metrics = detector.last_residue_metrics()
    assert metrics["flat_negative_sources"] == 1
    assert metrics["neural_sources"] == 0
    assert metrics["model_calls"] == 0


def test_flat_negative_skip_requires_explicit_opt_in():
    detector = _detector_with_fake_text(flat_gate=None, coalescing=False)
    source = _box(10, 10, 60, 40)
    image = np.full((100, 120, 3), 248, dtype=np.uint8)

    detector.verify_post_inpaint_residue(image, [source])

    assert FastResidueAdaptiveFocusCombinedTextDetector._FLAT_NEGATIVE_GATE_DEFAULT is False
    assert detector.text_detector.calls == 1
    metrics = detector.last_residue_metrics()
    assert metrics["flat_negative_sources"] == 0
    assert metrics["neural_sources"] == 1
    assert metrics["model_calls"] == 1


def test_single_contrasting_residual_forces_neural_verifier():
    detector = _detector_with_fake_text()
    source = _box(10, 10, 60, 40)
    image = np.full((100, 120, 3), 248, dtype=np.uint8)
    image[20, 30] = 0

    detector.verify_post_inpaint_residue(image, [source])

    assert detector.text_detector.calls == 1
    metrics = detector.last_residue_metrics()
    assert metrics["flat_negative_sources"] == 0
    assert metrics["neural_sources"] == 1
    assert metrics["model_calls"] == 1


def test_coalesced_verifier_uses_one_model_call_and_keeps_residue_evidence():
    detector = _detector_with_fake_text(coalescing=True)

    first = _box(10, 10, 50, 35)
    second = _box(48, 12, 88, 37)
    image = np.full((120, 140, 3), 255, dtype=np.uint8)
    # Force both sources through the neural path so the coalescing contract is
    # exercised independently of the flat-negative gate.
    image[18:26, 22:34] = 0
    image[18:28, 58:72] = 0

    residue = detector.verify_post_inpaint_residue(image, [first, second])

    assert detector.text_detector.calls == 1
    assert len(residue) == 1
    assert residue[0].deferred_reason == "post_inpaint_text_residue"
    metrics = detector.last_residue_metrics()
    assert metrics["scheduled_sources"] == 2
    assert metrics["flat_negative_sources"] == 0
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



def test_residue_coalescing_requires_explicit_opt_in():
    detector = _detector_with_fake_text(flat_gate=False, coalescing=None)
    first = _box(10, 10, 50, 35)
    second = _box(48, 12, 88, 37)
    image = np.full((120, 140, 3), 255, dtype=np.uint8)
    image[18:26, 22:34] = 0
    image[18:28, 58:72] = 0

    residue = detector.verify_post_inpaint_residue(image, [first, second])

    assert FastResidueAdaptiveFocusCombinedTextDetector._RESIDUE_COALESCING_DEFAULT is False
    assert detector.text_detector.calls == 2
    assert len(residue) == 2
    metrics = detector.last_residue_metrics()
    assert metrics["groups"] == 2
    assert metrics["model_calls"] == 2
    assert metrics["merged_sources"] == 0


def test_large_detector_box_uses_tight_verified_mask_roi():
    detector = _detector_with_fake_text()
    max_side = int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE)
    width = max_side + 300
    height = 120
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[35:55, 100:160] = 255
    source = BubbleBox(
        0,
        0,
        width,
        height,
        0.9,
        mask,
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
        semantic_type="free_text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
    )
    image = np.full((height + 20, width + 20, 3), 248, dtype=np.uint8)
    image[40:48, 110:135] = 0

    detector.verify_post_inpaint_residue(image, [source])
    metrics = detector.last_residue_metrics()

    assert metrics["deferred_size"] == 0
    assert metrics["neural_sources"] == 1
    assert metrics["model_calls"] == 1
    assert detector.text_detector.calls == 1


def test_flat_negative_sources_do_not_consume_neural_budget():
    detector = _detector_with_fake_text()
    image = np.full((160, 500, 3), 248, dtype=np.uint8)
    sources = [_box(i * 80 + 5, 20, i * 80 + 55, 55) for i in range(5)]

    residue = detector.verify_post_inpaint_residue(image, sources)
    metrics = detector.last_residue_metrics()

    assert residue == []
    assert metrics["flat_negative_sources"] == 5
    assert metrics["deferred_budget"] == 0
    assert metrics["model_calls"] == 0


def test_review_only_maskless_segmenter_region_gets_non_destructive_verification():
    detector = _detector_with_fake_text()
    source = BubbleBox(
        10, 10, 80, 50, 0.93, None,
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
        semantic_type="free_text",
        mask_source="none",
        safe_to_inpaint=False,
        ocr_eligible=True,
        needs_review=True,
        deferred_reason="box_width_limit",
        verify_region_only=True,
    )
    image = np.full((100, 120, 3), 248, dtype=np.uint8)
    image[20:28, 30:42] = 0

    residue = detector.verify_post_inpaint_residue(image, [source])

    assert detector.text_detector.calls == 1
    assert residue
    assert residue[0].deferred_reason == "post_inpaint_text_residue"
    assert residue[0].safe_to_inpaint is False
