from types import SimpleNamespace

import numpy as np

from app.detector.kiuyha_detector import KiuyhaTextDetector


class _Session:
    """One detection where the input has dark pixels, as an end-to-end head
    (``[1, 300, 6]``) or as a raw head (``[1, 5, anchors]``, duplicates included)."""

    def __init__(self, shape, raw=False):
        self.shape = shape
        self.raw = raw
        self.blobs = []

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.shape)]

    def run(self, _names, feeds):
        blob = feeds["images"]
        self.blobs.append(blob)
        ys, xs = np.nonzero(blob[0].mean(axis=0) < 0.1)
        if self.raw:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            cols = np.zeros((1, 5, 400), np.float32)
            cols[0, :, 0] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.9]
            cols[0, :, 1] = [(x1 + x2) / 2 + 1, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.8]  # NMS drops it
            cols[0, :, 2] = [20, 20, 10, 10, 0.1]  # below the threshold
            return [cols]
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


def test_raw_head_is_decoded_and_deduplicated():
    session = _Session([1, 3, 1280, 1280], raw=True)
    boxes = KiuyhaTextDetector("unused", session=session).detect(_image())
    assert len(boxes) == 1
    x1, y1, x2, y2, _ = boxes[0]
    assert abs(x1 - 200) <= 3 and abs(y1 - 1000) <= 3 and abs(x2 - 600) <= 3 and abs(y2 - 1100) <= 3
