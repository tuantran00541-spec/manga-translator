import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
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


class FakeAcceptedDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self.boxes), {
            "detector_ms": 1.0,
            "detector_forward_calls": 1,
            "detector_boxes": len(self.boxes),
            "accepted_mask_boxes": len(self.boxes),
        }


class FakeNormalInpainter:
    def __init__(self, value=17):
        self.value = value
        self.calls = 0
        self._metrics = {}

    def inpaint(self, image, boxes):
        self.calls += 1
        authority = build_mask(image.shape[:2], boxes, image)
        result = image.copy()
        result[authority > 127] = self.value
        self._metrics = {
            "lama_model_runs": 0,
            "smart_fill_regions": len(boxes),
            "bubble_fast_fill_regions": len(boxes),
            "clusters": 0,
        }
        return result

    def last_metrics(self):
        return dict(self._metrics)


def test_one_shot_detector_runs_once_and_accepts_only_verified_text_masks():
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

    boxes, metrics = detector.detect(image)

    assert underlying.calls == 1
    assert metrics["detector_forward_calls"] == 1
    assert metrics["detector_boxes"] == 3
    assert metrics["accepted_mask_boxes"] == 2
    assert len(boxes) == 2
    assert all(box.safe_to_inpaint for box in boxes)
    assert all(not box.needs_review for box in boxes)
    assert all(box.source_role == "text_segmenter" for box in boxes)
    assert all(box.semantic_type == "free_text" for box in boxes)


def test_hybrid_cleanup_uses_normal_inpainter_and_preserves_authority_boundary():
    image = np.full((120, 160, 3), 220, dtype=np.uint8)
    mask_a = np.zeros((10, 30), dtype=np.uint8)
    mask_a[:, :] = 255
    mask_b = np.zeros((6, 20), dtype=np.uint8)
    mask_b[:, :] = 255
    boxes = [
        _box(60, 40, 90, 50, mask_a),
        _box(100, 70, 120, 76, mask_b),
    ]
    boxes = [
        OneShotTextMaskDetector._accept(box)
        for box in boxes
    ]
    boxes = [box for box in boxes if box is not None]

    inpainter = FakeNormalInpainter(value=11)
    detector = FakeAcceptedDetector(boxes)
    pipeline = OneShotCleanupPipeline(detector=detector, inpainter=inpainter)

    result = pipeline.clean(image)

    authority = result.mask > 127
    changed = np.any(result.image != image, axis=2)
    assert detector.calls == 1
    assert inpainter.calls == 1
    assert result.metrics["detector_forward_calls"] == 1
    assert result.metrics["lama_model_runs"] == 0
    assert result.metrics["smart_fill_regions"] == 2
    assert np.array_equal(changed, authority)
    assert np.all(result.image[authority] == 11)
    assert np.all(result.image[~authority] == 220)


def test_hybrid_cleanup_passes_empty_detection_to_normal_inpainter():
    image = np.full((48, 64, 3), 123, dtype=np.uint8)
    detector = FakeAcceptedDetector([])
    inpainter = FakeNormalInpainter()
    pipeline = OneShotCleanupPipeline(detector=detector, inpainter=inpainter)

    result = pipeline.clean(image)

    assert detector.calls == 1
    assert inpainter.calls == 1
    assert result.metrics["lama_model_runs"] == 0
    assert not np.any(result.mask)
    assert np.array_equal(result.image, image)
