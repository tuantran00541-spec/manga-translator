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


def _text_slice():
    import cv2

    image = np.full((2400, 800, 3), (250, 235, 220), np.uint8)
    for y in (300, 1250, 2150):  # top half, the overlap, bottom half
        cv2.putText(image, "HELLO", (180, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 16)  # outline
        cv2.putText(image, "HELLO", (180, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (40, 60, 110), 6)
    return image


def test_tall_slice_runs_as_two_halves_in_one_pass_and_masks_letters_with_outline():
    image = _text_slice()
    session = _BlobSession()
    detector = KiuyhaTextDetector("unused", session=session)
    boxes = detector.text_boxes(image)
    assert len(session.blobs) == 1, "one forward pass"
    assert len(boxes) == 3, "the line in the overlap is merged, not doubled"
    for baseline in (300, 1250, 2150):
        assert any(b.y1 < baseline - 20 and b.y2 > baseline for b in boxes), baseline
    paper = np.array((250, 235, 220))
    ink = np.abs(image.astype(int) - paper).sum(axis=2) > 30  # letters and outline
    covered = np.zeros(image.shape[:2], bool)
    for b in boxes:
        covered[b.y1:b.y2, b.x1:b.x2] |= b.mask > 0
        assert b.safe_to_inpaint and b.semantic_type == "free_text"
    assert (covered & ink).sum() >= 0.97 * ink.sum()
    assert covered.sum() < 3 * ink.sum(), "masks stay near the text"


class _BlobSession(_Session):
    """Raw head: one box per ink blob on the (static 1280) input."""

    def __init__(self):
        super().__init__([1, 3, 1280, 1280], raw=True)

    def run(self, _names, feeds):
        import cv2

        blob = feeds["images"]
        self.blobs.append(blob)
        gray = (blob[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        ink = (np.abs(gray.astype(int) - gray[0, 0].astype(int)).sum(axis=2) > 30).astype(np.uint8)
        ink[:, :] &= (np.abs(gray.astype(int) - 114).sum(axis=2) > 10).astype(np.uint8)  # not the letterbox
        count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((25, 25), np.uint8)))
        cols = np.zeros((1, 5, 400), np.float32)
        for i, (x, y, w, h, _a) in enumerate(stats[1:count]):
            cols[0, :, i] = [x + w / 2, y + h / 2, w, h, 0.9]
        return [cols]


def test_second_pass_only_keeps_leftovers_inside_first_pass_boxes():
    image = _text_slice()
    detector = KiuyhaTextDetector("unused", session=_BlobSession())
    first = detector.text_boxes(image)
    kept = [b for b in first if b.y1 > 1000]
    leftovers = detector.leftover_boxes(image, kept)
    assert len(leftovers) == 2 and all(b.y1 > 1000 for b in leftovers)
    assert detector.leftover_boxes(image, []) == []


def test_leftover_mask_is_folded_into_the_saved_first_pass_box():
    from app.detector.bubble_detector import BubbleBox
    from app.page_processing import _fold_leftover

    record = {"x1": 10, "y1": 10, "x2": 50, "y2": 30, "_mask_array": np.full((20, 40), 255, np.uint8)}
    leftover = BubbleBox(25, 20, 65, 40, 0.9, np.full((20, 40), 255, np.uint8))
    _fold_leftover([record], leftover)
    assert (record["x1"], record["y1"], record["x2"], record["y2"]) == (10, 10, 65, 40)
    assert record["_mask_array"].shape == (30, 55)
    assert record["_mask_array"][25, 50] == 255 and record["_mask_array"][25, 5] == 0
