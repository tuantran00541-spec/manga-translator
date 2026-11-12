from types import SimpleNamespace

import cv2
import numpy as np

import pytest
from ctd_fake import InkModel

from app.detector import ctd_mask
from app.detector.kiuyha_detector import KiuyhaTextDetector, _split_blocks, _text_box


@pytest.fixture(autouse=True)
def ink_model(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=100))


class _Session:

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
            cols[0, :, 1] = [(x1 + x2) / 2 + 1, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.8]
            cols[0, :, 2] = [20, 20, 10, 10, 0.1]
            return [cols]
        rows = np.zeros((1, 300, 6), np.float32)
        rows[0, 0] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 0]
        rows[0, 1] = [0, 0, 50, 50, 0.1, 0]
        return [rows]


def _image():
    image = np.full((2400, 800, 3), 255, np.uint8)
    image[1000:1100, 200:600] = 0
    return image


def test_static_square_model_letterboxes_and_maps_back():
    session = _Session([1, 3, 1280, 1280])
    boxes = KiuyhaTextDetector("unused", session=session).raw_boxes(_image())
    assert session.blobs[0].shape == (1, 3, 1280, 1280)
    assert len(boxes) == 1
    x1, y1, x2, y2, score = boxes[0]
    assert abs(x1 - 200) <= 3 and abs(y1 - 1000) <= 3 and abs(x2 - 600) <= 3 and abs(y2 - 1100) <= 3
    assert score == np.float32(0.9)


def test_dynamic_model_runs_at_the_slice_size():
    session = _Session([1, 3, "height", "width"])
    boxes = KiuyhaTextDetector("unused", session=session).raw_boxes(_image())
    assert session.blobs[0].shape == (1, 3, 2400, 800)
    assert boxes[0][:4] == (200, 1000, 600, 1100)


def test_raw_head_is_decoded_and_deduplicated():
    session = _Session([1, 3, 1280, 1280], raw=True)
    boxes = KiuyhaTextDetector("unused", session=session).raw_boxes(_image())
    assert len(boxes) == 1
    x1, y1, x2, y2, _ = boxes[0]
    assert abs(x1 - 200) <= 3 and abs(y1 - 1000) <= 3 and abs(x2 - 600) <= 3 and abs(y2 - 1100) <= 3


def _text_slice():
    import cv2

    image = np.full((2400, 800, 3), (250, 235, 220), np.uint8)
    for y in (300, 1250, 2150):
        cv2.putText(image, "HELLO", (180, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 16)
        cv2.putText(image, "HELLO", (180, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (40, 60, 110), 6)
    return image


def test_tall_slice_runs_coarse_and_page_scale_passes_and_masks_letters_with_outline():
    image = _text_slice()
    session = _BlobSession()
    detector = KiuyhaTextDetector("unused", session=session)
    boxes = detector.text_boxes(image)
    assert len(session.blobs) == 1 + 1 + 1, "the whole slice coarse, once at page scale, once low for the rescue"
    assert len(boxes) == 3, "a line seen by several passes is one box"
    for baseline in (300, 1250, 2150):
        assert any(b.y1 < baseline - 20 and b.y2 > baseline for b in boxes), baseline
    paper = np.array((250, 235, 220))
    ink = np.abs(image.astype(int) - paper).sum(axis=2) > 30
    covered = np.zeros(image.shape[:2], bool)
    for b in boxes:
        covered[b.y1:b.y2, b.x1:b.x2] |= b.mask > 0
        assert b.safe_to_inpaint and b.semantic_type == "free_text"
    assert (covered & ink).sum() >= 0.97 * ink.sum()
    assert covered.sum() < 3 * ink.sum(), "masks stay near the text"


class _BlobSession(_Session):

    def __init__(self):
        super().__init__([1, 3, 1280, 1280], raw=True)

    def run(self, _names, feeds):
        import cv2

        blob = feeds["images"]
        self.blobs.append(blob)
        gray = (blob[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        ink = (np.abs(gray.astype(int) - gray[0, 0].astype(int)).sum(axis=2) > 30).astype(np.uint8)
        ink[:, :] &= (np.abs(gray.astype(int) - 114).sum(axis=2) > 10).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((25, 25), np.uint8)))
        cols = np.zeros((1, 5, 400), np.float32)
        for i, (x, y, w, h, _a) in enumerate(stats[1:count]):
            cols[0, :, i] = [x + w / 2, y + h / 2, w, h, 0.9]
        return [cols]


def test_second_pass_takes_only_first_pass_boxes_the_letter_model_still_reads():
    image = _text_slice()
    session = _BlobSession()
    detector = KiuyhaTextDetector("unused", session=session)
    first = detector.text_boxes(image)
    runs = len(session.blobs)
    clean = image.copy()
    clean[1100:1400] = (250, 235, 220)
    leftovers = detector.leftover_boxes(clean, first)
    assert len(leftovers) == 2 and not any(1100 <= b.y1 < 1400 for b in leftovers)
    assert len(session.blobs) == runs, "the box model is not run again"
    assert detector.leftover_boxes(image, []) == []


def test_leftover_mask_is_folded_into_the_saved_first_pass_box():
    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_leftover

    record = {"x1": 10, "y1": 10, "x2": 50, "y2": 30, "_mask_array": np.full((20, 40), 255, np.uint8)}
    leftover = BubbleBox(25, 20, 65, 40, 0.9, np.full((20, 40), 255, np.uint8))
    _fold_leftover([record], leftover)
    assert (record["x1"], record["y1"], record["x2"], record["y2"]) == (10, 10, 65, 40)
    assert record["_mask_array"].shape == (30, 55)
    assert record["_mask_array"][25, 50] == 255 and record["_mask_array"][25, 5] == 0




def test_pipeline_detect_names_the_model_and_reports_timing():
    detector = KiuyhaTextDetector("models/kiuyha-tuned-1023.onnx", session=_BlobSession())
    boxes = detector.detect(_text_slice())
    assert boxes and all(b.source_model == "kiuyha-tuned-1023.onnx" for b in boxes)
    assert detector.last_metrics()["result_boxes"] == len(boxes)








def test_pieces_of_one_text_from_different_passes_become_one_box():
    from app.detector.kiuyha_detector import _union_coherent

    image = np.full((600, 800, 3), 255, np.uint8)
    cv2.putText(image, "HELLO WORLD", (110, 145), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "SECOND LINE", (110, 435), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    pieces = [(100, 100, 400, 160, 0.9), (350, 100, 700, 160, 0.8), (100, 400, 300, 450, 0.7)]
    assert _union_coherent(image, pieces) == [(100, 100, 700, 160, 0.9), (100, 400, 300, 450, 0.7)]
    assert len(_union_coherent(image, [(0, 0, 100, 100, 0.9), (95, 95, 200, 200, 0.9)])) == 2, "a touching corner is two texts"


def test_two_texts_sharing_a_line_are_not_merged_before_letters():
    from app.detector.kiuyha_detector import _union_coherent

    image = np.full((300, 800, 3), 255, np.uint8)
    cv2.putText(image, "HELLO", (120, 130), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "WORLD", (500, 190), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    pair = [(100, 100, 400, 200, 0.9), (350, 100, 700, 200, 0.9)]
    assert len(_union_coherent(image, pair)) == 2


def test_a_rescued_word_needs_a_letter_standing_inside_its_box():
    from app.detector.kiuyha_detector import _has_letter_piece

    word = np.full((200, 300, 3), 255, np.uint8)
    cv2.putText(word, "HELLO", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    assert _has_letter_piece(word, (20, 60, 280, 160))
    strokes = np.full((200, 300, 3), 255, np.uint8)
    for x in (60, 120, 180):
        cv2.rectangle(strokes, (x, 0), (x + 25, 200), (0, 0, 0), -1)
    assert not _has_letter_piece(strokes, (40, 60, 260, 160))
    assert not _has_letter_piece(np.full((100, 100, 3), 255, np.uint8), (0, 0, 100, 100))


class _NestedSession(_Session):

    def __init__(self):
        super().__init__([1, 3, 1280, 1280])

    def run(self, _names, feeds):
        self.blobs.append(feeds["images"])
        rows = np.zeros((1, 300, 6), np.float32)
        rows[0, 0] = [300, 300, 700, 500, 0.9, 0]
        rows[0, 1] = [320, 320, 560, 360, 0.6, 0]
        return [rows]


def test_a_line_box_inside_a_block_box_is_not_a_second_text():
    image = np.full((1280, 1280, 3), 255, np.uint8)
    cv2.putText(image, "HELLO", (330, 355), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "THERE", (330, 455), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    boxes = KiuyhaTextDetector("unused", session=_NestedSession()).text_boxes(image)
    assert len(boxes) == 1 and boxes[0].x1 <= 300 and boxes[0].y2 >= 500















class _WholeSession(_Session):

    def __init__(self, rect):
        super().__init__([1, 3, 1280, 1280])
        self.rect = rect

    def run(self, _names, feeds):
        self.blobs.append(feeds["images"])
        rows = np.zeros((1, 300, 6), np.float32)
        rows[0, 0] = [*self.rect, 0.9, 0]
        return [rows]


def test_two_captions_in_a_staircase_of_boxes_become_two_texts():
    image = np.full((1280, 1280, 3), 150, np.uint8)
    image[100:560, 40:760] = 255
    image[520:1000, 380:1240] = 255
    for i, line in enumerate(("IT LOOKED LIKE AN", "ILLNESS THAT CAUSED", "CONSTANT FATIGUE")):
        cv2.putText(image, line, (120, 220 + 70 * i), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 3)
    for i, line in enumerate(("BUT WHEN ITS VICTIMS", "BEGAN FALLING INTO", "AN ENDLESS SLUMBER")):
        cv2.putText(image, line, (470, 680 + 70 * i), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 3)
    boxes = KiuyhaTextDetector("unused", session=_WholeSession((40, 100, 1240, 1000))).text_boxes(image)
    boxes.sort(key=lambda b: b.y1)
    assert len(boxes) == 2
    assert boxes[0].x2 <= 760 and boxes[0].y2 <= 560, "the first caption stays in its own box"
    assert boxes[1].x1 >= 380 and boxes[1].y1 >= 520, "the second caption stays in its own box"
    for box in boxes:
        assert box.mask.shape == (box.y2 - box.y1, box.x2 - box.x1) and box.mask.any()


def test_split_texts_keep_plain_int_coordinates():
    letters = np.zeros((200, 300), bool)
    for top in (10, 40):
        letters[top:top + 20, 10:150] = True
    for top in (92, 122):
        letters[top:top + 20, 140:290] = True
    box = _text_box(5, 7, 305, 207, 0.9, letters.copy(), "kiuyha", letters)
    pieces = _split_blocks(box)
    assert len(pieces) == 2
    for piece in pieces:
        assert all(type(v) is int for v in (piece.x1, piece.y1, piece.x2, piece.y2)), "coordinates stay JSON-safe"



def test_a_stray_mark_by_a_bubble_text_stays_in_its_box():
    letters = np.zeros((300, 400), bool)
    letters[5:25, 330:380] = True
    for top in (80, 110, 140):
        letters[top:top + 20, 40:360] = True
    letters[230:250, 50:110] = True
    box = _text_box(0, 0, 400, 300, 0.9, letters.copy(), "kiuyha", letters)
    assert len(_split_blocks(box)) == 1


def test_drawn_letters_under_a_text_are_not_a_second_text():
    letters = np.zeros((400, 400), bool)
    for top in (10, 40, 70):
        letters[top:top + 20, 20:320] = True
    for top in (170, 280):
        letters[top:top + 70, 150:400] = True
    box = _text_box(0, 0, 400, 400, 0.9, letters.copy(), "kiuyha", letters)
    assert len(_split_blocks(box)) == 1


def test_two_touching_lines_are_never_cut_through():
    letters = np.zeros((420, 700), bool)
    for top, bottom in ((98, 149), (154, 197), (318, 363)):
        letters[top:bottom, 150:520] = True
    letters[208:309, 100:600] = True
    letters[311:312, 300:340] = True
    box = _text_box(0, 338, 700, 758, 0.9, letters.copy(), "kiuyha", letters)
    assert len(_split_blocks(box)) == 1


def test_two_bubbles_side_by_side_are_not_joined():
    from app.detector.kiuyha_detector import _union_coherent

    image = np.full((300, 700, 3), 255, np.uint8)
    cv2.putText(image, "HELLO", (20, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "WORLD", (420, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    pair = [(0, 0, 300, 200, 0.9), (285, 10, 585, 210, 0.9)]
    assert len(_union_coherent(image, pair)) == 2, "an edge sliver is a neighbour bubble, not a fragment"


def test_two_texts_side_by_side_split_into_their_own_boxes():
    from app.detector.kiuyha_detector import _split_sideways

    letters = np.zeros((140, 700), bool)
    for top in (10, 40, 70):
        letters[top:top + 20, 20:300] = True
        letters[top:top + 20, 420:680] = True
    box = _text_box(0, 0, 700, 140, 0.9, letters.copy(), "kiuyha", letters)
    pieces = _split_sideways(box)
    assert len(pieces) == 2
    pieces.sort(key=lambda b: b.x1)
    assert pieces[0].x2 <= 420 and pieces[1].x1 >= 300, "each text keeps to its own side of the gutter"


def _union_mask(bbox=(60, 60, 560, 340)):
    import math

    x1, y1, x2, y2 = bbox
    h = y2 - y1
    rows = []
    for i in range(24):
        y = y1 + (i + 0.5) * h / 24
        spans = []
        for (cx, cy) in ((200, 150), (420, 250)):
            dy = y - cy
            if abs(dy) < 140:
                dx = math.sqrt(140 * 140 - dy * dy)
                spans.append((int(cx - dx), int(cx + dx)))
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        rows.append([[a - x1, b - x1] for a, b in merged])
    return {"bbox": list(bbox), "cx": 310, "cy": 200, "rows": rows, "kind": "bubble"}


def _box(x1, y1, x2, y2, core, lines):
    from app.detector.boxes import BubbleBox

    letters = np.zeros((y2 - y1, x2 - x1), bool)
    for (r1, r2, c1, c2) in lines:
        letters[r1:r2, c1:c2] = True
    return BubbleBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9,
                     letters=letters, bubble_mask=_union_mask(), core=core)


def test_two_boxes_sharing_one_union_mask_get_their_own_lobes():
    from app.detector.kiuyha_detector import _split_shared_union_masks

    left = _box(100, 100, 300, 220, (110, 110, 290, 210),
                [(20, 34, 10, 150), (60, 74, 10, 150)])
    right = _box(300, 180, 520, 300, (310, 190, 510, 290),
                 [(20, 34, 70, 200), (60, 74, 70, 200),
                  (20, 34, 2, 8)])
    _split_shared_union_masks([left, right])
    assert left.bubble_mask["bbox"][2] <= 330
    assert right.bubble_mask["bbox"][0] >= 290
    assert left.bubble_mask["bbox"] != right.bubble_mask["bbox"]
    assert not np.asarray(right.letters)[:, :10].any()
    assert np.asarray(right.letters)[:, 70:200].any()
    assert np.asarray(left.letters).sum() == 2 * 14 * 140


def test_trim_rewrites_letters_to_the_kept_lobe():
    from app.detector.kiuyha_detector import _trim_bubble_mask

    box = _box(100, 100, 520, 300, (110, 110, 350, 290),
               [(20, 34, 10, 150), (60, 74, 10, 150),
                (20, 34, 260, 400), (60, 74, 260, 400)])
    _trim_bubble_mask(box)
    assert box.bubble_mask["bbox"][2] <= 400
    assert not np.asarray(box.letters)[:, 260:].any()
    assert np.asarray(box.letters)[:, :150].any()


def test_one_wide_text_with_word_gaps_stays_whole():
    from app.detector.kiuyha_detector import _split_sideways

    letters = np.zeros((140, 700), bool)
    for top in (10, 40, 70):
        letters[top:top + 20, 20:330] = True
        letters[top:top + 20, 345:680] = True
    box = _text_box(0, 0, 700, 140, 0.9, letters.copy(), "kiuyha", letters)
    assert len(_split_sideways(box)) == 1, "a word gap is not a gutter between texts"


class _FixedSession(_Session):

    def __init__(self, boxes):
        super().__init__([1, 3, 1280, 1280])
        self.boxes = boxes

    def run(self, _names, feeds):
        self.blobs.append(feeds["images"])
        rows = np.zeros((1, 300, 6), np.float32)
        for i, (x1, y1, x2, y2, score) in enumerate(self.boxes):
            rows[0, i] = [x1, y1, x2, y2, score, 0]
        return [rows]


def _blind_ctd(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=0))


def test_a_confident_box_the_letter_model_cannot_read_is_kept_whole(monkeypatch):
    _blind_ctd(monkeypatch)
    image = np.full((1280, 1280, 3), (200, 180, 160), np.uint8)
    cv2.putText(image, "SHANDONG", (100, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (60, 50, 40), 4)
    boxes = KiuyhaTextDetector("unused", session=_BlobSession()).text_boxes(image)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.mask.any() and (box.mask > 0).all(), "a kept box is masked whole, so it is erased and read"
    assert box.letters is None, "no letters were read; downstream falls back to the box extent"


def test_one_connected_shape_the_letter_model_cannot_read_is_still_dropped(monkeypatch):
    _blind_ctd(monkeypatch)
    image = np.full((1280, 1280, 3), (230, 220, 210), np.uint8)
    for x in range(100, 301, 40):
        cv2.line(image, (x, 100), (x, 260), (60, 50, 40), 6)
    for y in range(100, 261, 40):
        cv2.line(image, (100, y), (300, y), (60, 50, 40), 6)
    boxes = KiuyhaTextDetector("unused", session=_BlobSession()).text_boxes(image)
    assert boxes == []


def test_a_faint_box_the_letter_model_cannot_read_is_still_dropped(monkeypatch):
    _blind_ctd(monkeypatch)
    image = np.full((1280, 1280, 3), (200, 180, 160), np.uint8)
    image[100:160, 100:300] = (60, 50, 40)
    boxes = KiuyhaTextDetector("unused", session=_FixedSession([(100, 100, 300, 160, 0.4)])).text_boxes(image)
    assert boxes == []


def test_a_vertical_sign_is_found_when_the_upright_passes_see_nothing(monkeypatch):
    _blind_ctd(monkeypatch)
    image = np.full((900, 300, 3), (200, 180, 160), np.uint8)
    detector = KiuyhaTextDetector("unused", session=_FixedSession([]))
    sideways = iter([[(100, 50, 200, 150, 0.8)], [(699, 149, 799, 249, 0.7)]])

    def raw_boxes(blob_image):
        if blob_image.shape[1] > blob_image.shape[0]:
            return next(sideways)
        return []

    monkeypatch.setattr(detector, "raw_boxes", raw_boxes)
    found = detector.detect_slice(image)
    assert len(found) == 1, "both sideways passes map onto the one upright box"
    from app.knobs import knob

    pad = knob("detect.box_pad")
    assert found[0][:4] == (50 - pad, 699 - pad, 150 + pad, 799 + pad)
    assert found[0][4] == 0.8


def test_sideways_boxes_below_the_page_bar_are_not_kept(monkeypatch):
    _blind_ctd(monkeypatch)
    image = np.full((900, 300, 3), (200, 180, 160), np.uint8)
    detector = KiuyhaTextDetector("unused", session=_FixedSession([]))
    monkeypatch.setattr(detector, "raw_boxes",
                        lambda blob_image: [(100, 50, 200, 150, 0.4)] if blob_image.shape[1] > blob_image.shape[0] else [])
    assert detector.detect_slice(image) == []


class _GivenSession(_Session):

    def __init__(self, boxes):
        super().__init__([1, 3, "height", "width"])
        self.boxes = boxes

    def run(self, _names, feeds):
        self.blobs.append(feeds["images"])
        rows = np.zeros((1, 300, 6), np.float32)
        for i, (x1, y1, x2, y2, score) in enumerate(self.boxes):
            rows[0, i] = [x1, y1, x2, y2, score, 0]
        return [rows]


def _two_words():
    image = np.full((1280, 1280, 3), (250, 235, 220), np.uint8)
    cv2.putText(image, "HELLO", (100, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (40, 30, 25), 4)
    cv2.putText(image, "WORLD", (600, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (40, 30, 25), 4)
    return image


def test_a_word_in_a_low_box_is_rescued_as_its_own_box():
    boxes = KiuyhaTextDetector("unused", session=_GivenSession([
        (80, 100, 400, 180, 0.9), (580, 100, 900, 180, 0.15)])).text_boxes(_two_words())
    assert len(boxes) == 2
    rescued = [b for b in boxes if b.confidence < 0.25]
    assert len(rescued) == 1, "only the low box is rescued; the kept word is untouched"
    box = rescued[0]
    assert box.letters is None, "a rescued word is masked whole, like the maskless keep"
    assert box.mask.any() and (box.mask > 0).all()
    assert box.x1 <= 610 and box.x2 >= 740 and box.y1 <= 135 and box.y2 >= 155, "the box holds the whole word"


def test_a_low_box_over_a_kept_word_rescues_nothing():
    boxes = KiuyhaTextDetector("unused", session=_GivenSession([
        (80, 100, 400, 180, 0.9), (100, 110, 380, 170, 0.15)])).text_boxes(_two_words())
    assert len(boxes) == 1, "the low box's letters are already erased by the kept box"


def test_a_low_box_with_no_text_rescues_nothing():
    image = np.full((1280, 1280, 3), (200, 180, 160), np.uint8)
    image[100:160, 100:300] = (60, 50, 40)
    boxes = KiuyhaTextDetector("unused", session=_GivenSession([(100, 100, 300, 160, 0.15)])).text_boxes(image)
    assert boxes == []


def test_welded_words_takes_a_fat_word_and_not_a_wide_panel():
    from app.detector.kiuyha_detector import _welded_words

    image = np.full((400, 900, 3), (210, 190, 170), np.uint8)
    cv2.putText(image, "ANHUI", (100, 120), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 30, 25), 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    welded = cv2.dilate((gray < 150).astype(np.uint8), np.ones((5, 5), np.uint8))
    image[welded > 0] = (40, 30, 25)
    words = _welded_words(image, (80, 80, 300, 160))
    assert len(words) == 1
    (x1, y1, x2, y2), _ = words[0]
    assert x1 <= 100 and x2 >= 160 and 100 <= y1 <= 120 and 115 <= y2 <= 140
    panel = np.full((400, 900, 3), (210, 190, 170), np.uint8)
    for x in range(500, 591, 30):
        cv2.line(panel, (x, 100), (x, 141), (60, 50, 40), 4)
    for y in (100, 120, 141):
        cv2.line(panel, (500, y), (590, y), (60, 50, 40), 4)
    assert _welded_words(panel, (480, 80, 610, 160)) == []


def test_a_welded_word_passes_the_rescue_gate():
    from app.detector.kiuyha_detector import _has_letter_piece

    word = np.full((100, 200, 3), 255, np.uint8)
    cv2.rectangle(word, (60, 40), (139, 59), (0, 0, 0), -1)
    assert not _has_letter_piece(word, (40, 20, 160, 80)), "the old gate kills an 80x20 word"
    assert _has_letter_piece(word, (40, 20, 160, 80), allow_wide=True)


def test_a_welded_word_in_a_low_box_is_rescued(monkeypatch):
    _blind_ctd(monkeypatch)
    image = np.full((1280, 1280, 3), (250, 235, 220), np.uint8)
    cv2.putText(image, "HELLO", (100, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (40, 30, 25), 4)
    cv2.putText(image, "JIANGXI", (100, 360), cv2.FONT_HERSHEY_DUPLEX, 0.5, (40, 30, 25), 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    welded = cv2.dilate((gray < 150).astype(np.uint8), np.ones((5, 5), np.uint8))
    image[welded > 0] = (40, 30, 25)
    cv2.putText(image, "H", (400, 360), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 30, 25), 3)
    cv2.putText(image, "K", (500, 360), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 30, 25), 3)
    boxes = KiuyhaTextDetector("unused", session=_GivenSession([
        (80, 100, 400, 180, 0.9), (80, 300, 600, 420, 0.15)])).text_boxes(image)
    rescued = [b for b in boxes if b.confidence < 0.25]
    assert any(b.x1 <= 100 and b.x2 >= 160 and b.y1 <= 350 and b.y2 >= 360 for b in rescued), \
        "the welded word comes back as its own box"


def test_missing_letter_geometry_is_unknown_not_coherent():
    from app.detector.kiuyha_detector import _letters_coherent

    geometry = (set(range(10, 30)), (10, 10, 190, 30))
    assert _letters_coherent(None, geometry) is False
    assert _letters_coherent(geometry, None) is False
    assert _letters_coherent(None, None) is False
    assert _letters_coherent(geometry, geometry) is True


def test_join_grown_does_not_fold_a_maskless_box_into_a_lettered_one():
    from app.detector.kiuyha_detector import _join_grown

    letters = np.zeros((100, 200), bool)
    letters[10:30, 10:190] = True
    lettered = _text_box(0, 0, 200, 100, 0.9, np.full((100, 200), 255, np.uint8), "kiuyha",
                         letters, core=(0, 0, 200, 100))
    maskless = _text_box(50, 20, 250, 120, 0.9, np.full((100, 200), 255, np.uint8), "kiuyha",
                         None, core=(50, 20, 250, 120))
    assert len(_join_grown([lettered, maskless], [(0, 0, 200, 100), (50, 20, 250, 120)])) == 2


def test_join_grown_still_dedupes_a_maskless_box_seen_twice():
    from app.detector.kiuyha_detector import _join_grown

    first = _text_box(0, 0, 200, 100, 0.9, np.full((100, 200), 255, np.uint8), "kiuyha",
                      None, core=(0, 0, 200, 100))
    second = _text_box(2, 2, 202, 102, 0.8, np.full((100, 200), 255, np.uint8), "kiuyha",
                       None, core=(2, 2, 202, 102))
    joined = _join_grown([first, second], [(0, 0, 200, 100), (2, 2, 202, 102)])
    assert len(joined) == 1
    assert (joined[0].x1, joined[0].y1, joined[0].x2, joined[0].y2) == (0, 0, 202, 102)
    assert joined[0].core == (0, 0, 202, 102)


def test_join_grown_keeps_the_core_of_the_boxes_it_joins():
    from app.detector.kiuyha_detector import _join_grown, _letter_geometry

    letters = np.zeros((100, 400), bool)
    letters[10:30, 10:390] = True
    left = _text_box(0, 0, 200, 100, 0.9, np.full((100, 200), 255, np.uint8), "kiuyha",
                     letters[:, :200].copy(), core=(0, 0, 200, 100))
    right = _text_box(150, 0, 400, 100, 0.8, np.full((100, 250), 255, np.uint8), "kiuyha",
                      letters[:, 150:].copy(), core=(150, 0, 400, 100))
    joined = _join_grown([left, right], [(0, 0, 200, 100), (150, 0, 400, 100)])
    assert len(joined) == 1
    assert joined[0].core == (0, 0, 400, 100)
    assert _letter_geometry(joined[0], joined[0].core) is not None


def test_split_blocks_keeps_a_core_clipped_to_each_piece():
    from app.detector.kiuyha_detector import _letter_geometry

    letters = np.zeros((200, 300), bool)
    for top in (10, 40):
        letters[top:top + 20, 10:150] = True
    for top in (92, 122):
        letters[top:top + 20, 140:290] = True
    box = _text_box(5, 7, 305, 207, 0.9, letters.copy(), "kiuyha", letters, core=(5, 7, 305, 207))
    pieces = _split_blocks(box)
    assert len(pieces) == 2
    for piece in pieces:
        assert piece.core is not None
        assert piece.core[0] >= piece.x1 and piece.core[1] >= piece.y1
        assert piece.core[2] <= piece.x2 and piece.core[3] <= piece.y2
        assert _letter_geometry(piece, piece.core) is not None


def test_split_sideways_keeps_a_core_clipped_to_each_piece():
    from app.detector.kiuyha_detector import _letter_geometry, _split_sideways

    letters = np.zeros((140, 700), bool)
    for top in (10, 40, 70):
        letters[top:top + 20, 20:300] = True
        letters[top:top + 20, 420:680] = True
    box = _text_box(0, 0, 700, 140, 0.9, letters.copy(), "kiuyha", letters, core=(0, 0, 700, 140))
    pieces = _split_sideways(box)
    assert len(pieces) == 2
    for piece in pieces:
        assert piece.core is not None
        assert piece.core[0] >= piece.x1 and piece.core[1] >= piece.y1
        assert piece.core[2] <= piece.x2 and piece.core[3] <= piece.y2
        assert _letter_geometry(piece, piece.core) is not None


def test_a_box_cut_by_an_inner_window_edge_is_kept_in_page_boxes():
    image = np.full((1200, 400, 3), 255, np.uint8)
    image[400:500, 100:300] = 0
    image[900:1000, 100:300] = 0
    session = _Session([1, 3, 416, 416])
    boxes = KiuyhaTextDetector("unused", session=session).page_boxes(image)
    rects = [tuple(box[:4]) for box in boxes]
    assert (100, 400, 300, 416) in rects, "the window-0 partial rect is kept"
    assert (100, 400, 300, 500) in rects, "the whole rect is still seen in a later window"


def test_a_small_box_with_no_letter_mask_is_kept_not_dropped(monkeypatch):
    from app.detector import kiuyha_detector as kd

    image = np.full((600, 800, 3), 255, np.uint8)

    def no_mask(_image, core, with_letters=False):
        x1, y1, x2, y2 = core
        empty = np.zeros((y2 - y1, x2 - x1), bool)
        return ((x1, y1, x2, y2), empty, empty)

    monkeypatch.setattr(kd, "letter_mask", no_mask)
    detector = KiuyhaTextDetector("unused", session=_Session([1, 3, 1280, 1280]))
    monkeypatch.setattr(detector, "_raw_slice_boxes",
                        lambda _image: [(100, 100, 220, 180, 0.9), (100, 300, 700, 600, 0.9)])
    monkeypatch.setattr(detector, "_rescue_boxes", lambda _image, kept: [])
    boxes = detector.text_boxes(image)
    assert [(b.x1, b.y1, b.x2, b.y2) for b in boxes] == [(100, 100, 220, 180)]


def test_a_leftover_without_a_usable_mask_leaves_its_record_unfolded():
    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_leftover

    record = {"x1": 10, "y1": 10, "x2": 50, "y2": 30, "_mask_array": np.full((20, 40), 255, np.uint8)}
    _fold_leftover([record], BubbleBox(25, 20, 65, 40, 0.9, None))
    assert (record["x1"], record["y1"], record["x2"], record["y2"]) == (10, 10, 50, 30)
    assert record["_mask_array"].shape == (20, 40)


def test_a_leftover_matching_no_record_is_left_as_stored():
    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_leftover

    record = {"x1": 10, "y1": 10, "x2": 50, "y2": 30, "_mask_array": np.full((20, 40), 255, np.uint8)}
    _fold_leftover([record], BubbleBox(200, 200, 240, 220, 0.9, np.full((20, 40), 255, np.uint8)))
    assert (record["x1"], record["y1"], record["x2"], record["y2"]) == (10, 10, 50, 30)
