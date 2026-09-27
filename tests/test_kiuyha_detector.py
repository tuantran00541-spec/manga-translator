from types import SimpleNamespace

import cv2
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
    for y in (300, 1250, 2150):  # top half, the overlap, bottom half
        cv2.putText(image, "HELLO", (180, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 16)  # outline
        cv2.putText(image, "HELLO", (180, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (40, 60, 110), 6)
    return image


def test_tall_slice_runs_a_coarse_pass_and_near_native_bands_and_masks_letters_with_outline():
    image = _text_slice()
    session = _BlobSession()
    detector = KiuyhaTextDetector("unused", session=session)
    boxes = detector.text_boxes(image)
    assert len(session.blobs) == 1 + 3, "the whole slice once, then three overlapping bands"
    assert len(boxes) == 3, "a line seen by several passes is one box"
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
    from app.detector.boxes import BubbleBox
    from app.page_processing import _fold_leftover

    record = {"x1": 10, "y1": 10, "x2": 50, "y2": 30, "_mask_array": np.full((20, 40), 255, np.uint8)}
    leftover = BubbleBox(25, 20, 65, 40, 0.9, np.full((20, 40), 255, np.uint8))
    _fold_leftover([record], leftover)
    assert (record["x1"], record["y1"], record["x2"], record["y2"]) == (10, 10, 65, 40)
    assert record["_mask_array"].shape == (30, 55)
    assert record["_mask_array"][25, 50] == 255 and record["_mask_array"][25, 5] == 0


def test_letters_cut_by_the_slice_edge_are_masked_and_leftovers_take_the_whole_box():
    from app.detector.kiuyha_detector import letter_mask

    page = np.full((300, 400, 3), 230, np.uint8)
    page[0:20, 140:260] = 30  # the bottom of a line cut by the slice's top edge
    page[150:170, 20:380] = 30  # a bar of art running out of any crop around the box
    box, mask = letter_mask(page, (130, 0, 270, 60))
    assert box[1] == 0 and mask[10, 200 - box[0]], "letters cut by the slice edge are masked"
    box, mask = letter_mask(page, (100, 140, 300, 180))
    assert not mask.any(), "inside the image, art running past the box is kept"
    detector = KiuyhaTextDetector("unused", session=_BlobSession())
    image = _text_slice()
    (left,) = detector.leftover_boxes(image, [b for b in detector.text_boxes(image) if b.y1 > 2000])
    ink = np.abs(image.astype(int) - (250, 235, 220)).sum(axis=2) > 30
    region = ink[left.y1:left.y2, left.x1:left.x2]
    assert (left.mask[region] == 255).all(), "every stroke left is taken"
    assert (left.mask == 255).mean() < 0.9, "the art round it is not painted over as a rectangle"


def test_pipeline_detect_names_the_model_and_reports_timing():
    detector = KiuyhaTextDetector("models/kiuyha_text_1280.onnx", session=_BlobSession())
    boxes = detector.detect(_text_slice())
    assert boxes and all(b.source_model == "kiuyha_text_1280.onnx" for b in boxes)
    assert detector.last_metrics()["result_boxes"] == len(boxes)


def _letters(image, xs, top=100, size=40):
    for x in xs:
        image[top:top + size, x:x + 24] = 20  # a solid letter
    return image


def test_first_and_last_letters_cut_by_the_box_edge_are_erased():
    from app.detector.kiuyha_detector import letter_mask

    image = _letters(np.full((260, 600, 3), 245, np.uint8), range(100, 461, 40))
    box, mask = letter_mask(image, (112, 90, 470, 150))  # the box stops inside the first and last letters
    assert box[0] <= 100 and box[2] >= 484
    assert mask[120 - box[1], 105 - box[0]] and mask[120 - box[1], 480 - box[0]]


def test_a_bubble_outline_past_the_box_is_not_erased():
    from app.detector.kiuyha_detector import letter_mask

    image = np.full((300, 600, 3), 245, np.uint8)
    cv2.ellipse(image, (300, 150), (230, 110), 0, 0, 360, (20, 20, 20), 3)
    _letters(image, range(180, 420, 40), top=130)
    box, mask = letter_mask(image, (160, 110, 440, 190))
    grown = np.zeros(image.shape[:2], bool)
    grown[box[1]:box[3], box[0]:box[2]] = mask
    outline = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(outline, (300, 150), (230, 110), 0, 0, 360, 255, 3)
    assert not (grown & (outline > 0)).any()
    assert grown[150, 190]


def test_pieces_of_one_text_from_different_passes_become_one_box():
    from app.detector.kiuyha_detector import _union_overlapping

    pieces = [(100, 100, 400, 160, 0.9), (350, 100, 700, 160, 0.8), (100, 400, 300, 450, 0.7)]
    assert _union_overlapping(pieces) == [(100, 100, 700, 160, 0.9), (100, 400, 300, 450, 0.7)]
    assert len(_union_overlapping([(0, 0, 100, 100, 0.9), (95, 95, 200, 200, 0.9)])) == 2, "a touching corner is two texts"


class _NestedSession(_Session):
    """A block box with a line box inside it, as the model returns for one bubble."""

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


def test_a_line_running_far_past_its_box_is_erased_to_its_last_letter():
    from app.detector.kiuyha_detector import letter_mask

    image = _letters(np.full((260, 900, 3), 245, np.uint8), range(100, 701, 40))  # letters up to x=724
    box, mask = letter_mask(image, (90, 90, 560, 150))  # the box stops four letters short
    assert box[2] >= 724
    assert mask[120 - box[1], 710 - box[0]], "the last letter of the line is erased"


def test_big_lettering_takes_its_outline_with_it():
    from app.detector.kiuyha_detector import letter_mask

    image = np.full((400, 900, 3), (120, 140, 150), np.uint8)
    cv2.putText(image, "WORTH", (120, 260), cv2.FONT_HERSHEY_DUPLEX, 4.0, (255, 255, 255), 40)  # thick white outline
    cv2.putText(image, "WORTH", (120, 260), cv2.FONT_HERSHEY_DUPLEX, 4.0, (20, 20, 20), 12)
    box, mask = letter_mask(image, (100, 120, 800, 300))
    full = np.zeros(image.shape[:2], bool)
    full[box[1]:box[3], box[0]:box[2]] = mask
    outline = (np.abs(image.astype(int) - (255, 255, 255)).sum(axis=2) < 30)
    assert (full & outline).sum() >= 0.9 * outline.sum()


def test_glow_round_letters_is_erased_with_them():
    from app.detector.kiuyha_detector import letter_mask

    rng = np.random.default_rng(1)
    background = np.clip(rng.normal(30, 6, (400, 1000, 3)), 0, 255).astype(np.uint8)
    glow = np.zeros((400, 1000), np.uint8)
    cv2.putText(glow, "DARE TO DREAM", (80, 240), cv2.FONT_HERSHEY_DUPLEX, 3.0, 255, 22)
    glow = cv2.GaussianBlur(glow, (0, 0), 14).astype(np.float32)[..., None] / 255
    image = background + glow * np.array([200, 190, 60], np.float32)
    core = np.zeros((400, 1000), np.uint8)
    cv2.putText(core, "DARE TO DREAM", (80, 240), cv2.FONT_HERSHEY_DUPLEX, 3.0, 255, 8)
    image[core > 0] = (255, 240, 170)
    image = np.clip(image, 0, 255).astype(np.uint8)
    box, mask = letter_mask(image, (70, 150, 900, 270))
    full = np.zeros(image.shape[:2], bool)
    full[box[1]:box[3], box[0]:box[2]] = mask
    changed = np.abs(image.astype(int) - background.astype(int)).max(axis=2) > 18
    assert (full & changed).sum() >= 0.99 * changed.sum(), "a glow left round the hole paints the letters back"


def test_outlined_letters_over_sky_and_trees_are_erased():
    from app.detector.kiuyha_detector import letter_mask

    rng = np.random.default_rng(4)
    image = np.empty((500, 1200, 3), np.uint8)
    image[:170] = (225, 205, 185)  # light sky over dark trees: the box border holds both
    image[170:] = np.clip(rng.normal(60, 18, (330, 1200, 3)), 0, 255).astype(np.uint8)
    for y in (200, 330):
        cv2.putText(image, "THE SLUMS", (160, y), cv2.FONT_HERSHEY_DUPLEX, 3.2, (255, 255, 255), 26)
        cv2.putText(image, "THE SLUMS", (160, y), cv2.FONT_HERSHEY_DUPLEX, 3.2, (10, 10, 10), 10)
    box, mask = letter_mask(image, (130, 90, 1000, 370))
    full = np.zeros(image.shape[:2], bool)
    full[box[1]:box[3], box[0]:box[2]] = mask
    ink = image.max(axis=2) < 20
    assert (full & ink).sum() >= 0.95 * ink.sum(), "both lines are erased"
    assert full[450:, :].mean() < 0.05, "the trees below stay"
