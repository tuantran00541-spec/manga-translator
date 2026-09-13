import threading
from contextlib import contextmanager

import numpy as np

import app.detector.parallel_focus_detector as parallel_module
from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.detector.bubble_detector import BubbleBox
from app.detector.combined_detector import CombinedTextDetector
from app.detector.parallel_focus_detector import (
    ParallelAdaptiveFocusCombinedTextDetector,
    _finish_focus_from_full,
)


def _box(x1, y1, x2, y2, *, semantic_type="speech_bubble"):
    return BubbleBox(
        x1,
        y1,
        x2,
        y2,
        0.9,
        np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8),
        source_model="text_segmenter.onnx",
        source_role="text_segmenter",
        semantic_type=semantic_type,
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
    )


class _FinishFakeDetector:
    def __init__(self):
        self.chip_calls = 0

    def _detect_single_plain(self, crop, x1, y1):
        self.chip_calls += 1
        return []

    @staticmethod
    def _nms_boxes(boxes):
        return list(boxes)

    @staticmethod
    def _with_semantics(box):
        return box

    @staticmethod
    def _filter_invalid(boxes, width, height):
        return list(boxes)


def test_finish_focus_reuses_complete_full_pass_without_extra_chip_calls():
    detector = _FinishFakeDetector()
    image = np.zeros((1800, 800, 3), dtype=np.uint8)
    proposal = _box(100, 100, 300, 250)
    full_text = _box(140, 130, 260, 200)

    result, metrics, deferred, chip_ms = _finish_focus_from_full(
        detector,
        image,
        [proposal],
        [full_text],
    )

    assert result == [full_text]
    assert detector.chip_calls == 0
    assert metrics["focus_uncovered_proposals"] == 0
    assert metrics["focus_chip_calls"] == 0
    assert deferred == []
    assert chip_ms >= 0.0


def test_parallel_false_delegates_to_validated_adaptive_detector(monkeypatch):
    sentinel = [_box(1, 1, 10, 10)]
    calls = []

    def fake_detect(self, image, *, parallel=False):
        calls.append(parallel)
        return sentinel

    monkeypatch.setattr(AdaptiveFocusCombinedTextDetector, "detect", fake_detect)
    detector = ParallelAdaptiveFocusCombinedTextDetector.__new__(
        ParallelAdaptiveFocusCombinedTextDetector
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    assert detector.detect(image, parallel=False) is sentinel
    assert calls == [False]


class _TextModel:
    def __init__(self, bubble_started, text_started):
        self.bubble_started = bubble_started
        self.text_started = text_started

    def _detect_single(self, image, x1, y1):
        self.text_started.set()
        assert self.bubble_started.wait(2.0), "bubble prefetch did not overlap text prefetch"
        return [_box(20, 20, 50, 45)]

    @staticmethod
    def _filter_invalid(boxes, width, height):
        return list(boxes)

    @staticmethod
    def _with_semantics(box):
        return box


class _Recovery:
    @staticmethod
    def detect(image, existing=None):
        return []


class _Proxy:
    @contextmanager
    def prefetched(self, image, boxes):
        yield


class _MetricLocal:
    value = {}


def test_parallel_true_overlaps_independent_prefetch_and_reuses_downstream_path(monkeypatch):
    bubble_started = threading.Event()
    text_started = threading.Event()
    bubble_box = _box(10, 10, 70, 60)
    downstream = [_box(12, 12, 68, 58)]

    def fake_adaptive_detect(model, image):
        bubble_started.set()
        assert text_started.wait(2.0), "text prefetch did not overlap bubble prefetch"
        return [bubble_box]

    downstream_calls = []

    def fake_combined_detect(self, image, *, parallel=False):
        downstream_calls.append(parallel)
        self._metrics_local.value = {"mser_ms": 3.0}
        return downstream

    monkeypatch.setattr(parallel_module, "_adaptive_detect", fake_adaptive_detect)
    monkeypatch.setattr(CombinedTextDetector, "detect", fake_combined_detect)

    detector = ParallelAdaptiveFocusCombinedTextDetector.__new__(
        ParallelAdaptiveFocusCombinedTextDetector
    )
    detector._bubble_model = object()
    detector._text_model = _TextModel(bubble_started, text_started)
    detector.recovery = _Recovery()
    detector.bubble_detector = _Proxy()
    detector.text_detector = _Proxy()
    detector._metrics_local = _MetricLocal()

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    result = detector.detect(image, parallel=True)

    assert result == downstream
    assert downstream_calls == [False]
    metrics = detector._metrics_local.value
    assert metrics["parallel_prefetch_enabled"] == 1
    assert metrics["parallel_prefetch_wall_ms"] >= 0.0
    assert metrics["parallel_prefetch_overlap_ms"] >= 0.0
    assert metrics["focus_chip_calls"] == 0
