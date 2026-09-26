from types import SimpleNamespace

import cv2
import numpy as np

from app.detector.bubble_detector import YoloDetector
from app.one_shot_cleanup import OneShotTextMaskDetector as Detector


class _DarkBlobModel:
    """Stands in for the segmenter: every dark blob on the canvas is a text box."""

    def __init__(self):
        self.canvases = []

    def run(self, _names, feeds):
        blob = feeds["images"][0]
        self.canvases.append(blob)
        gray = (blob.mean(axis=0) * 255).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats((gray < 60).astype(np.uint8))
        rows = []
        for x, y, w, h, _area in stats[1:count]:
            coeffs = np.zeros(32, np.float32)
            coeffs[0] = 1.0
            rows.append(np.concatenate([[x + w / 2, y + h / 2, w, h, 0.9], coeffs]))
        rows += [np.zeros(37, np.float32)] * 64  # real models emit thousands of anchors
        preds = np.array(rows, np.float32).T[None]
        protos = np.zeros((1, 32, 256, 256), np.float32)
        protos[0, 0] = 10.0
        return [preds, protos]


def _detector():
    yolo = object.__new__(YoloDetector)
    yolo.model_path = yolo.source_model = "text_segmenter.onnx"
    yolo.model_role = "text_segmenter"
    yolo.conf_threshold = 0.2
    yolo.input_name = "images"
    yolo.contract = SimpleNamespace(class_names=("text_comic",))
    yolo.session = _DarkBlobModel()
    detector = Detector(yolo)
    detector.collage = True
    return detector


def test_plan_puts_two_overlapping_halves_side_by_side_only_for_tall_slices():
    scale, halves = Detector.collage_plan(2400, 800)
    assert halves[0][0] == 0 and halves[1][1] == 2400
    assert halves[0][1] - halves[1][0] >= 256, "the halves overlap"
    assert scale > 1.4 * (1024 / 2400), "text reaches the model much bigger than one pass"
    assert Detector.collage_plan(1100, 800) is None, "a short slice keeps one pass"


def test_boxes_from_both_halves_come_back_at_page_coordinates_in_one_pass():
    image = np.full((2400, 800, 3), 255, np.uint8)
    blocks = [(100, 200, 400, 300), (300, 1200, 600, 1260), (200, 2100, 500, 2200)]  # top, overlap, bottom
    for x1, y1, x2, y2 in blocks:
        image[y1:y2, x1:x2] = 0
    detector = _detector()
    boxes, metrics = detector.detect(image)
    assert metrics["detector_forward_calls"] == 1 and len(detector.detector.session.canvases) == 1
    found = sorted((b.x1, b.y1, b.x2, b.y2) for b in boxes)
    assert len(found) == len(blocks), found
    for (x1, y1, x2, y2), want in zip(found, sorted(blocks)):
        assert all(abs(a - b) <= 20 for a, b in zip((x1, y1, x2, y2), want)), (found, blocks)
    assert all(b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1) for b in boxes)
