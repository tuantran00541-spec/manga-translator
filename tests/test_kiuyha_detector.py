from types import SimpleNamespace

import numpy as np

from app.detector.kiuyha_detector import KiuyhaTextDetector


class _Session:
    """Returns one detection where the input has dark pixels, like an end-to-end YOLO26 head."""

    def __init__(self, shape):
        self.shape = shape
        self.blobs = []

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.shape)]

    def run(self, _names, feeds):
        blob = feeds["images"]
        self.blobs.append(blob)
        ys, xs = np.nonzero(blob[0].mean(axis=0) < 0.1)
        rows = np.zeros((1, 300, 6), np.float32)
        rows[0, 0] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 0]
        rows[0, 1] = [0, 0, 50, 50, 0.1, 0]  # below the threshold
        return [rows]


def _image():
    image = np.full((2400, 800, 3), 255, np.uint8)
    image[1000:1100, 200:600] = 0
    return image


def test_static_square_model_letterboxes_and_maps_back():
    session = _Session([1, 3, 1280, 1280])
    boxes = KiuyhaTextDetector("unused", session=session).detect(_image())
    assert session.blobs[0].shape == (1, 3, 1280, 1280)
    assert len(boxes) == 1
    x1, y1, x2, y2, score = boxes[0]
    assert abs(x1 - 200) <= 3 and abs(y1 - 1000) <= 3 and abs(x2 - 600) <= 3 and abs(y2 - 1100) <= 3
    assert score == np.float32(0.9)


def test_dynamic_model_runs_at_the_slice_size():
    session = _Session([1, 3, "height", "width"])
    boxes = KiuyhaTextDetector("unused", session=session).detect(_image())
    assert session.blobs[0].shape == (1, 3, 2400, 800)
    assert boxes[0][:4] == (200, 1000, 600, 1100)
