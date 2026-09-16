from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.detector.inference_cache import DetectorInferenceCache
from app.detector.bubble_detector import YoloDetector


@dataclass(frozen=True)
class _Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    mask: np.ndarray | None = None


def test_reuse_is_content_addressed_and_returns_defensive_masks():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    cache = DetectorInferenceCache(mode="reuse", max_entries=4, namespace="text:v1")
    key = cache.key(image, offset_x=3, offset_y=7)
    source = [_Box(1, 2, 5, 6, 0.8, mask=np.ones((4, 4), dtype=np.uint8))]
    cache.store(key, source)

    hit = cache.lookup(key)
    assert hit is not None
    assert hit[0] is not source[0]
    hit[0].mask[0, 0] = 99
    assert source[0].mask[0, 0] == 1
    assert cache.snapshot()["hits"] == 1


def test_mutating_image_changes_key_and_shadow_detects_exact_duplicates():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cache = DetectorInferenceCache(mode="shadow", max_entries=4, namespace="bubble:v1")
    key = cache.key(image, offset_x=0, offset_y=0)
    cache.store(key, [])
    assert cache.lookup(key) == []
    image[0, 0, 0] = 1
    changed_key = cache.key(image, offset_x=0, offset_y=0)
    assert changed_key != key
    snapshot = cache.snapshot()
    assert snapshot["shadow_hits"] == 1
    assert snapshot["shadow_mismatches"] == 0


def test_off_mode_does_not_hash_or_store():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cache = DetectorInferenceCache(mode="off")
    key = cache.key(image, offset_x=0, offset_y=0)
    cache.store(key, [])
    assert cache.lookup(key) is None
    assert cache.snapshot()["entries"] == 0


def test_yolo_plain_path_reuses_cached_postprocess_result(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    detector = YoloDetector.__new__(YoloDetector)
    detector._inference_cache = DetectorInferenceCache(
        mode="reuse", max_entries=4, namespace="fake:v1"
    )
    detector.input_name = "input"
    detector.session = type("Session", (), {})()
    calls = {"run": 0, "post": 0}

    def run(_outputs, _inputs):
        calls["run"] += 1
        return [np.zeros((1, 1), dtype=np.float32)]

    def preprocess(_image, *, offset_x=0, offset_y=0):
        return np.zeros((1, 3, 2, 2), dtype=np.float32), object()

    def postprocess(_outputs, _transform):
        calls["post"] += 1
        return [_Box(2, 3, 8, 9, 0.9, mask=np.ones((6, 6), dtype=np.uint8))]

    detector.session.run = run
    detector._preprocess = preprocess
    detector._postprocess = postprocess

    first = detector._detect_single_plain(image, 4, 5)
    second = detector._detect_single_plain(image.copy(), 4, 5)
    assert calls == {"run": 1, "post": 1}
    assert second[0] is not first[0]
    second[0].mask[0, 0] = 77
    assert first[0].mask[0, 0] == 1
