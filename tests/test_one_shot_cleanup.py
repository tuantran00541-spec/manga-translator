import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.one_shot_cleanup import OneShotCleanupPipeline, OneShotTextMaskDetector


def _box(x1, y1, x2, y2, mask):
    return BubbleBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=0.9,
        mask=mask,
        source_model="text_segmenter.onnx",
        class_id=0,
        class_name="text_comic",
        semantic_type="text",
        source_role="text_segmenter",
    )


class FakeUnderlyingDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def _detect_single(self, image, offset_x, offset_y):
        assert offset_x == 0
        assert offset_y == 0
        self.calls += 1
        return list(self.boxes)


class FakeMaskDetector:
    def __init__(self, mask):
        self.mask = mask
        self.calls = 0

    def detect_mask(self, image):
        self.calls += 1
        return self.mask.copy(), {
            "detector_ms": 1.0,
            "detector_forward_calls": 1,
            "detector_boxes": 1 if np.any(self.mask) else 0,
            "accepted_mask_boxes": 1 if np.any(self.mask) else 0,
            "mask_pixels": int(np.count_nonzero(self.mask)),
        }


class FakeInpainter:
    def __init__(self, value=17):
        self.value = value
        self.calls = 0
        self.last_shape = None
        self.last_mask = None
        self._metrics = {}

    def _begin_metrics(self):
        self._metrics = {"lama_model_runs": 0}

    def _lama_fill_single(self, crop, mask):
        self.calls += 1
        self.last_shape = crop.shape
        self.last_mask = mask.copy()
        self._metrics["lama_model_runs"] = self._metrics.get("lama_model_runs", 0) + 1
        painted = np.full_like(crop, self.value)
        return painted

    def last_metrics(self):
        return dict(self._metrics)


def test_one_shot_detector_unions_only_verified_text_masks():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    mask_a = np.zeros((12, 20), dtype=np.uint8)
    mask_a[2:6, 3:12] = 255
    mask_b = np.zeros((10, 14), dtype=np.uint8)
    mask_b[1:8, 2:9] = 255

    ignored = BubbleBox(
        2,
        2,
        12,
        12,
        0.9,
        None,
        source_model="bubble_yolo.onnx",
        class_name="text_bubble",
        semantic_type="speech_bubble",
        source_role="bubble_detector",
    )
    underlying = FakeUnderlyingDetector(
        [
            _box(10, 10, 30, 22, mask_a),
            _box(45, 35, 59, 45, mask_b),
            ignored,
        ]
    )
    detector = OneShotTextMaskDetector(detector=underlying)

    mask, metrics = detector.detect_mask(image)

    assert underlying.calls == 1
    assert metrics["detector_forward_calls"] == 1
    assert metrics["detector_boxes"] == 3
    assert metrics["accepted_mask_boxes"] == 2
    assert np.array_equal(mask[10:22, 10:30], mask_a)
    assert np.array_equal(mask[35:45, 45:59], mask_b)
    assert int(np.count_nonzero(mask)) == (
        int(np.count_nonzero(mask_a)) + int(np.count_nonzero(mask_b))
    )


def test_one_shot_cleanup_runs_lama_once_and_writes_only_authority_pixels():
    image = np.full((120, 160, 3), 220, dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[40:50, 60:90] = 255
    mask[70:76, 100:120] = 255

    inpainter = FakeInpainter(value=11)
    pipeline = OneShotCleanupPipeline(
        detector=FakeMaskDetector(mask),
        inpainter=inpainter,
        padding=8,
    )

    result = pipeline.clean(image)

    authority = mask > 127
    changed = np.any(result.image != image, axis=2)
    assert inpainter.calls == 1
    assert result.metrics["lama_model_runs"] == 1
    assert result.metrics["detector_forward_calls"] == 1
    assert np.array_equal(changed, authority)
    assert np.all(result.image[authority] == 11)
    assert np.all(result.image[~authority] == 220)


def test_one_shot_cleanup_skips_lama_when_detector_finds_no_text():
    image = np.full((48, 64, 3), 123, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    inpainter = FakeInpainter()

    pipeline = OneShotCleanupPipeline(
        detector=FakeMaskDetector(mask),
        inpainter=inpainter,
    )
    result = pipeline.clean(image)

    assert inpainter.calls == 0
    assert result.roi is None
    assert result.metrics["lama_model_runs"] == 0
    assert np.array_equal(result.image, image)
