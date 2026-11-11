"""Bubble masks: derivation, waist split, and persistence.

The synthetic images are drawn with thin (2px) outlines and text-line
bars, like real scanlation pages -- a solid filled shape would flood
as its own outline and prove nothing about the edge flood.
"""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from ctd_fake import InkModel

from app.detector import ctd_mask
from app.detector.bubble_mask import (
    derive_bubble_mask,
    find_waist,
    mask_inscribed_rect,
    mask_row_spans,
    merge_bubble_masks,
    shift_bubble_mask,
    split_bubble_mask,
)
from app.detector.boxes import BubbleBox, _merge_mask_evidence
from app.detector.kiuyha_detector import KiuyhaTextDetector, split_waist_boxes

FIXTURES = Path(__file__).parent / "fixtures" / "bubble_mask"


def _paper(w=800, h=600):
    return np.full((h, w, 3), 255, np.uint8)


def _text_lines(img, lines, thickness=2):
    """Dark bars, one per text line (x1, y1, x2, y2) in image coords.

    Dashed, like real lettering: a solid bar would seal the bubble
    interior into bands and the flood would fill only one of them,
    while real glyphs leave gaps the flood flows through.
    """
    for (x1, y1, x2, y2) in lines:
        mid = (y1 + y2) // 2
        x = x1
        while x < x2:
            seg = min(x + 60, x2)
            cv2.line(img, (x, mid), (seg, mid), (20, 20, 20), thickness)
            x = seg + 15


def _circle(w=800, h=600, centre=(400, 300), radius=140, lines=((260, 250, 540, 271), (260, 290, 540, 311))):
    img = _paper(w, h)
    cv2.circle(img, centre, radius, (20, 20, 20), 2)
    _text_lines(img, lines)
    return img


def _double_side():
    """Two overlapping circles, text in each lobe."""
    img = _paper()
    cv2.circle(img, (200, 200), 140, (20, 20, 20), 2)
    cv2.circle(img, (420, 200), 140, (20, 20, 20), 2)
    _text_lines(img, [(100, 180, 280, 201), (100, 220, 280, 241),
                      (340, 180, 520, 201), (340, 220, 520, 241)])
    return img


def _double_stacked(gap):
    """Two circles stacked, centres `gap` apart; text in each lobe."""
    img = _paper()
    cy = 300 - gap // 2
    cv2.circle(img, (400, cy), 140, (20, 20, 20), 2)
    cv2.circle(img, (400, cy + gap), 140, (20, 20, 20), 2)
    _text_lines(img, [(300, cy - 20, 500, cy + 1), (300, cy + 20, 500, cy + 41),
                      (300, cy + gap - 20, 500, cy + gap + 1), (300, cy + gap + 20, 500, cy + gap + 41)])
    return img


def _spiky():
    img = _paper()
    pts = np.array([
        (400, 100), (430, 180), (520, 160), (470, 230), (560, 260),
        (480, 300), (540, 380), (440, 370), (420, 460), (380, 380),
        (280, 420), (300, 320), (200, 300), (290, 250), (260, 160), (350, 200),
    ])
    cv2.polylines(img, [pts], True, (20, 20, 20), 2)
    _text_lines(img, [(300, 250, 500, 271), (300, 290, 500, 311)])
    return img


def _open_arc():
    img = _paper()
    cv2.ellipse(img, (400, 300), (180, 140), 0, 30, 300, (20, 20, 20), 2)
    _text_lines(img, [(280, 250, 520, 271), (280, 290, 520, 311)])
    return img


def _free_text():
    img = _paper()
    _text_lines(img, [(200, 200, 600, 221), (200, 250, 600, 271), (200, 300, 500, 321)])
    return img


def _letters_for(img, lines):
    """A letters array for a box, one ink bar per text line, dashed like the drawn lines."""
    letters = np.zeros(img.shape[:2], bool)
    for (x1, y1, x2, y2) in lines:
        x = x1
        while x < x2:
            seg = min(x + 60, x2)
            letters[y1 + 4:y2 - 4, x + 4:seg - 4] = True
            x = seg + 15
    return letters


def _mask_from_widths(widths, bbox=(100, 100, 500, 340)):
    """A mask dict with one centred interval per band (bbox-local)."""
    x1, y1, x2, y2 = bbox
    centre = (x2 - x1) // 2
    rows = []
    for w in widths:
        rows.append([[centre - w // 2, centre + w // 2]] if w > 0 else [])
    return {"bbox": list(bbox), "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2, "rows": rows, "kind": "bubble"}


def _mask_from_circles(c1, c2, r, bbox):
    """The union of two circles, one interval per band (for the letter-gap tests)."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    rows = []
    for i in range(24):
        y = y1 + (i + 0.5) * h / 24
        spans = []
        for (cx, cy) in (c1, c2):
            dy = y - cy
            if abs(dy) < r:
                dx = (r * r - dy * dy) ** 0.5
                spans.append((int(cx - dx), int(cx + dx)))
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        rows.append([[a - x1, b - x1] for a, b in merged])
    return {"bbox": list(bbox), "cx": (c1[0] + c2[0]) // 2, "cy": (c1[1] + c2[1]) // 2,
            "rows": rows, "kind": "bubble"}


# ---------------------------------------------------------------- derive

def test_circle_derives_bubble_with_tapered_rows():
    img = _circle()
    mask = derive_bubble_mask(img, (250, 150, 550, 450))
    assert mask["kind"] == "bubble"
    assert 240 <= mask["bbox"][0] <= 270 and 140 <= mask["bbox"][1] <= 170
    assert 530 <= mask["bbox"][2] <= 560 and 430 <= mask["bbox"][3] <= 460
    widths = [sum(b - a for a, b in row) if row else 0 for row in mask["rows"]]
    assert widths[0] < widths[11] > widths[-1], "a circle tapers at both caps"
    assert widths[11] > 250
    assert find_waist(mask) is None


def test_spiky_bubble_is_not_split_when_its_text_is_all_in_one_lobe():
    # The star's outline pinches between spikes, so the shape alone
    # shows a waist -- but both text lines sit in the lower lobe, so
    # the split (which needs letter rows on both sides) must not fire.
    # The mask instead trims to the lettered lobe.
    from app.detector.kiuyha_detector import _trim_bubble_mask

    img = _spiky()
    mask = derive_bubble_mask(img, (200, 100, 560, 460))
    assert mask["kind"] == "bubble"
    assert find_waist(mask) is not None, "the shape alone does show a pinch"
    letters = np.zeros((360, 360), bool)  # box-local: the box stores letters in its own frame
    letters[150:171, 100:300] = True
    letters[190:211, 100:300] = True
    box = BubbleBox(200, 100, 560, 460, 0.9, mask=np.zeros((360, 360), np.uint8),
                    letters=letters, bubble_mask=mask, core=(200, 100, 560, 460))
    assert len(split_waist_boxes([box])) == 1, "one-sided lettering never splits"
    _trim_bubble_mask(box)
    assert box.bubble_mask["bbox"][1] >= 200, "the empty top spikes are trimmed away"
    assert box.bubble_mask["bbox"][3] == mask["bbox"][3]


def test_open_arc_derives_rect():
    img = _open_arc()
    mask = derive_bubble_mask(img, (220, 160, 580, 440))
    assert mask["kind"] == "rect"
    assert mask["bbox"] == [220, 160, 580, 440]
    assert find_waist(mask) is None


def test_free_text_derives_rect():
    img = _free_text()
    mask = derive_bubble_mask(img, (200, 200, 600, 321))
    assert mask["kind"] == "rect"
    assert mask["bbox"] == [200, 200, 600, 321]


def test_degenerate_box_and_missing_image_derive_none():
    img = _circle()
    for box in ((100, 100, 100, 200), (100, 100, 200, 100), (0, 0, 0, 0)):
        mask = derive_bubble_mask(img, box)
        assert mask["kind"] == "none" and mask["rows"] == [[] for _ in range(24)]
    assert derive_bubble_mask(None, (100, 100, 200, 200))["kind"] == "none"


def test_mask_inscribed_rect_and_row_spans():
    mask = derive_bubble_mask(_circle(), (250, 150, 550, 450))
    x1, y1, x2, y2 = mask_inscribed_rect(mask)
    assert 0 < x2 - x1 and 0 < y2 - y1
    # Every corner of the inscribed rectangle sits inside the mask rows
    # (on the boundary counts: the rectangle is maximal, so its edges
    # touch the outline). The spans are slice coords; the bottom edge
    # is the start of the next band, so the corner reads one pixel up.
    for (x, y) in ((x1, y1), (x2, y2 - 1)):
        assert any(a <= x <= b for a, b in mask_row_spans(mask, y))
    # The rectangle covers a real share of the bubble, not a sliver.
    assert (x2 - x1) * (y2 - y1) >= 0.35 * (mask["bbox"][2] - mask["bbox"][0]) * (mask["bbox"][3] - mask["bbox"][1])
    assert mask_row_spans(mask, mask["cy"]) != []
    assert mask_row_spans(mask, mask["bbox"][1] - 50) == []
    assert mask_row_spans(mask, mask["bbox"][3] + 50) == []


# ---------------------------------------------------------------- waist

def test_double_side_by_side_waist_fires_on_the_merged_union():
    img = _double_side()
    box = (60, 60, 560, 340)
    left = derive_bubble_mask(img, box, seed=(200, 200))
    right = derive_bubble_mask(img, box, seed=(420, 200))
    assert left["kind"] == "bubble" and right["kind"] == "bubble"
    union = merge_bubble_masks([left, right])
    waist = find_waist(union)
    assert waist is not None and waist["orientation"] == "vertical"
    assert 290 <= waist["cut_x"] <= 330
    pieces = split_bubble_mask(union, waist)
    assert pieces is not None and len(pieces) == 2
    assert pieces[0]["bbox"][2] <= waist["cut_x"] <= pieces[1]["bbox"][0]
    # The pieces tile the union: each keeps its lobe, and together they
    # lose only the re-quantization at the band edges (a few percent).
    union_width = sum(sum(b - a for a, b in row) for row in union["rows"])
    pieces_width = sum(sum(b - a for a, b in row)
                       for piece in pieces for row in piece["rows"])
    assert pieces_width >= 0.9 * union_width


def test_double_stacked_deep_pinch_fires_shallow_does_not():
    img = _double_stacked(260)
    box = (260, 40, 540, 560)
    top = derive_bubble_mask(img, box, seed=(400, 170))
    bottom = derive_bubble_mask(img, box, seed=(400, 430))
    union = merge_bubble_masks([top, bottom])
    waist = find_waist(union)
    assert waist is not None and waist["orientation"] == "horizontal"
    assert 280 <= waist["cut_y"] <= 320
    pieces = split_bubble_mask(union, waist)
    assert pieces is not None and len(pieces) == 2
    assert pieces[0]["bbox"][3] <= waist["cut_y"] <= pieces[1]["bbox"][1]

    img = _double_stacked(140)
    box = (260, 90, 540, 510)
    top = derive_bubble_mask(img, box, seed=(400, 230))
    bottom = derive_bubble_mask(img, box, seed=(400, 370))
    union = merge_bubble_masks([top, bottom])
    assert find_waist(union) is None, "a barely pinched union is one bubble"


def test_profile_waist_needs_a_real_dip():
    deep = [100, 150, 200, 240, 260, 265, 260, 200, 150, 140, 130, 120,
            130, 150, 200, 250, 265, 270, 260, 240, 200, 150, 100, 60]
    waist = find_waist(_mask_from_widths(deep))
    assert waist is not None and waist["orientation"] == "horizontal"
    shallow = [100, 150, 200, 240, 260, 265, 260, 250, 245, 240, 240, 245,
               250, 255, 260, 265, 270, 265, 260, 240, 200, 150, 100, 60]
    assert find_waist(_mask_from_widths(shallow)) is None


def test_letter_gap_waist_fires_and_a_single_oval_does_not():
    # Two lobes offset diagonally: every row is one interval (the
    # lobes overlap in x), so only the letter gap sees the waist.
    mask = _mask_from_circles((200, 150), (420, 250), 140, (60, 60, 560, 340))
    letters = np.zeros((280, 500), bool)
    letters[80:94, 100 - 60:280 - 60] = True    # upper lobe line
    letters[120:134, 100 - 60:280 - 60] = True
    letters[180:194, 340 - 60:520 - 60] = True  # lower lobe line
    letters[220:234, 340 - 60:520 - 60] = True
    waist = find_waist(mask, letters=letters, box=(60, 60, 560, 340))
    assert waist is not None and waist["orientation"] == "vertical"
    assert 290 <= waist["cut_x"] <= 330
    pieces = split_bubble_mask(mask, waist)
    assert pieces is not None and len(pieces) == 2
    assert pieces[0]["bbox"][2] <= waist["cut_x"] <= pieces[1]["bbox"][0]

    # The same letters, but one line crosses the gap: no waist.
    letters[80:94, 280 - 60:340 - 60] = True
    assert find_waist(mask, letters=letters, box=(60, 60, 560, 340)) is None

    # One oval, two staggered lines with a column gap between them: the
    # profile is smooth in the middle, so the gap alone must not split it.
    oval = _mask_from_circles((310, 200), (310, 200), 140, (170, 60, 450, 340))
    letters = np.zeros((280, 280), bool)
    letters[120:134, 190 - 170:260 - 170] = True   # upper line, left
    letters[160:174, 280 - 170:350 - 170] = True   # lower line, right
    assert find_waist(oval, letters=letters, box=(170, 60, 450, 340)) is None


# ---------------------------------------------------------------- real crops
# Fixtures are the bubble regions cut from the source slices (the rerun-5
# QA instances); the letter bars are the line positions the detector
# read on those slices, so the waist logic runs on real ink.

def _fixture_lines(shape, lines):
    """A letters array with one ink bar per text line (fixture coords)."""
    letters = np.zeros(shape, bool)
    for (x1, y1, x2, y2) in lines:
        h = y2 - y1
        letters[y1 + h // 4:y1 + 3 * h // 4, x1 + 4:x2 - 4] = True
    return letters


def test_ch42_p038_double_bubble_waist():
    img = cv2.imread(str(FIXTURES / "p038-double.png"))
    box = (184, 104, 800, 500)
    letters = _fixture_lines((396, 616), [
        (11, 6, 163, 27), (11, 31, 163, 52),                    # left lobe, 2 lines
        (177, 101, 528, 122), (177, 146, 528, 167),
        (177, 191, 528, 212), (177, 236, 528, 257),             # right lobe, 4 lines
    ])
    mask = derive_bubble_mask(img, box, letters=letters)
    assert mask["kind"] == "bubble"
    assert mask["bbox"] == [129, 58, 771, 422]
    waist = find_waist(mask, letters=letters, box=box)
    assert waist is not None and waist["orientation"] == "vertical"
    assert 345 <= waist["cut_x"] <= 365
    assert waist["letter_shares"][0] >= 0.10 and waist["letter_shares"][1] >= 0.10


def test_ch111_p005_double_bubble_waist():
    img = cv2.imread(str(FIXTURES / "ch111-double.png"))
    box = (45, 83, 560, 333)
    letters = _fixture_lines((250, 515), [
        (41, 40, 264, 61), (41, 84, 264, 105), (41, 128, 264, 149), (41, 172, 264, 193),
        (282, 164, 496, 185), (282, 187, 496, 208), (282, 210, 496, 231),
    ])
    mask = derive_bubble_mask(img, box, letters=letters)
    assert mask["kind"] == "bubble"
    assert mask["bbox"][0] <= 50 and mask["bbox"][2] >= 560
    waist = find_waist(mask, letters=letters, box=box)
    assert waist is not None and waist["orientation"] == "vertical"
    assert 305 <= waist["cut_x"] <= 330
    pieces = split_bubble_mask(mask, waist)
    assert pieces is not None and len(pieces) == 2
    # Each piece holds its own lobe's letters and nothing of the other.
    left_letters = letters[:, :waist["cut_x"] - box[0]]
    right_letters = letters[:, waist["cut_x"] - box[0]:]
    assert left_letters.any() and right_letters.any()


def test_mh_p068_stacked_double_bubble_waist():
    img = cv2.imread(str(FIXTURES / "mh068-stacked.png"))
    box = (106, 64, 436, 263)
    letters = _fixture_lines((199, 330), [
        (14, 6, 314, 27), (14, 46, 314, 67), (14, 86, 314, 107),
    ])
    mask = derive_bubble_mask(img, box, letters=letters)
    assert mask["kind"] == "bubble"
    waist = find_waist(mask, letters=letters, box=box)
    assert waist is not None and waist["orientation"] == "horizontal"
    assert 240 <= waist["cut_y"] <= 265
    pieces = split_bubble_mask(mask, waist)
    assert pieces is not None and len(pieces) == 2
    assert pieces[0]["bbox"][3] <= waist["cut_y"] <= pieces[1]["bbox"][1]


def test_split_waist_boxes_splits_the_ch111_fixture_box():
    img = cv2.imread(str(FIXTURES / "ch111-double.png"))
    box = (45, 83, 560, 333)
    letters = _fixture_lines((250, 515), [
        (41, 40, 264, 61), (41, 84, 264, 105), (41, 128, 264, 149), (41, 172, 264, 193),
        (282, 164, 496, 185), (282, 187, 496, 208), (282, 210, 496, 231),
    ])
    # Letters in box-local coords, as the detector stores them.
    mask = derive_bubble_mask(img, box, letters=letters)
    bubble = BubbleBox(box[0], box[1], box[2], box[3], 0.9,
                       mask=np.zeros((250, 515), np.uint8), letters=letters,
                       bubble_mask=mask, core=box)
    split = split_waist_boxes([bubble])
    assert len(split) == 2, "the double bubble becomes one box per lobe"
    assert split[0].x2 <= split[1].x1
    assert split[0].bubble_mask["bbox"][2] <= split[1].bubble_mask["bbox"][0]
    assert split[0].letters.sum() > 0 and split[1].letters.sum() > 0


# ---------------------------------------------------------------- wiring

class _Session:
    """One detection where the input has dark pixels (end-to-end head)."""

    def __init__(self, shape=(1, 3, 1280, 1280)):
        self.shape = shape

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.shape)]

    def run(self, _names, feeds):
        blob = feeds["images"]
        ys, xs = np.nonzero(blob[0].mean(axis=0) < 0.1)
        rows = np.zeros((1, 300, 6), np.float32)
        rows[0, 0] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 0]
        return [rows]


@pytest.fixture(autouse=True)
def ink_model(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=100))


def test_text_boxes_attaches_a_bubble_mask_to_every_box():
    # The outline is light grey: the fake letter model reads only dark
    # pixels, and a real letter model reads letter bodies, not the
    # outline -- an outline read as letters would fail the coverage
    # test and is not what the detector sees in production.
    img = _paper()
    cv2.circle(img, (400, 300), 140, (110, 110, 110), 2)
    _text_lines(img, [(260, 250, 540, 271), (260, 290, 540, 311)])
    boxes = KiuyhaTextDetector("unused", session=_Session()).text_boxes(img)
    # The dashed lines split into several boxes; the contract is that
    # every final box carries a mask (the bubble kind is proven by the
    # direct derive tests and the real-crop fixtures below).
    assert boxes
    for box in boxes:
        assert isinstance(box.bubble_mask, dict)
        assert box.bubble_mask["kind"] in ("bubble", "rect", "none")
        assert len(box.bubble_mask["rows"]) == 24
    # Free text gets a rect mask, not a missing one.
    boxes = KiuyhaTextDetector("unused", session=_Session()).text_boxes(_free_text())
    assert boxes and all(b.bubble_mask is not None and b.bubble_mask["kind"] == "rect" for b in boxes)


def _box_with_mask(x1, y1, x2, y2, mask_bbox):
    # The rows are narrower than the mask bbox (an interior, not the outline).
    width = max(1, (mask_bbox[2] - mask_bbox[0]) - 10)
    mask = _mask_from_widths([width] * 24, bbox=mask_bbox)
    return BubbleBox(x1, y1, x2, y2, 0.9, mask=np.zeros((y2 - y1, x2 - x1), np.uint8),
                     letters=np.zeros((y2 - y1, x2 - x1), bool), bubble_mask=mask, core=(x1, y1, x2, y2))


def test_merge_mask_evidence_unions_bubble_masks():
    a = _box_with_mask(10, 10, 50, 50, (10, 10, 50, 50))
    b = _box_with_mask(40, 10, 80, 50, (40, 10, 80, 50))
    merged = _merge_mask_evidence([a, b])
    assert merged is not None
    assert merged.bubble_mask is not None
    assert merged.bubble_mask["kind"] == "bubble"
    # The union spans both members' interiors (their 30px-wide rows).
    assert merged.bubble_mask["bbox"] == [15, 10, 75, 50]


def test_shift_detection_y_moves_the_mask_with_the_box():
    from app.pipeline import ChapterPipeline

    box = _box_with_mask(10, 10, 50, 50, (14, 10, 46, 50))
    shifted = ChapterPipeline._shift_detection_y(box, 100)
    assert (shifted.y1, shifted.y2) == (110, 150)
    assert shifted.bubble_mask["bbox"] == [14, 110, 46, 150]
    assert shifted.bubble_mask["cy"] == box.bubble_mask["cy"] + 100
    assert shifted.bubble_mask["rows"] == box.bubble_mask["rows"]


def test_mask_json_round_trip_through_the_manifest(tmp_path, monkeypatch):
    import app.manifest_utils as manifests

    monkeypatch.setattr(manifests, "PROCESSED_DIR", tmp_path)
    mask = _mask_from_widths([100, 200, 250, 200, 100] + [0] * 19, bbox=(100, 100, 500, 340))
    manifest = {
        "chapter_id": "a1b2c3d5", "schema_version": 3,
        "pages": [{
            "original": "orig.png", "clean": None, "width": 800, "height": 600,
            "boxes": [{"id": "b1", "x1": 100, "y1": 100, "x2": 500, "y2": 340,
                       "bubble_mask": mask}],
            "text_objects": [{"id": "o1", "source_boxes": ["b1"], "bubble_mask": mask}],
        }],
    }
    manifests.save_manifest_raw("a1b2c3d5", manifest)
    loaded = manifests.load_manifest_raw("a1b2c3d5")
    assert loaded["pages"][0]["boxes"][0]["bubble_mask"] == mask
    assert loaded["pages"][0]["text_objects"][0]["bubble_mask"] == mask
    assert loaded["pages"][0]["boxes"][0]["bubble_mask"]["rows"] == mask["rows"]


def test_normalize_manifest_schema_keeps_bubble_mask():
    from app.manifest_utils import normalize_manifest_schema

    mask = _mask_from_widths([100] * 24)
    manifest = {"chapter_id": "t", "pages": [{
        "boxes": [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "bubble_mask": mask}],
        "text_objects": [{"id": "o", "source_boxes": [], "bubble_mask": mask}],
    }]}
    normalize_manifest_schema(manifest)
    assert manifest["pages"][0]["boxes"][0]["bubble_mask"] is mask
    assert manifest["pages"][0]["text_objects"][0]["bubble_mask"] is mask


def test_text_object_mirror_and_seam_mirror_carry_the_mask():
    from app.ai_mode.seams import set_letter_bounds, sync_seam_mirrors
    from app.text_objects import ensure_page_text_objects

    mask = _mask_from_widths([200] * 24, bbox=(100, 500, 500, 740))
    owner_page = {
        "source_page": "p1",
        "stitch_core": {"source_y1": 0, "core_source_y1": 0, "core_source_y2": 1000},
        "boxes": [{"id": "b1", "x1": 100, "y1": 500, "x2": 500, "y2": 740,
                   "bubble_mask": mask, "letters_mask": None, "mask": None}],
        "text_objects": [],
    }
    ensure_page_text_objects(owner_page)
    assert len(owner_page["text_objects"]) == 1
    owner = owner_page["text_objects"][0]
    assert owner["bubble_mask"] == mask
    owner["translation"] = "Chào"
    set_letter_bounds(owner_page)
    assert owner["bubble_mask"] == mask
    # The mirror is the same text seen on the next slice, whose source
    # offset is 300px lower: its region maps to the owner's source rect.
    mirror_page = {
        "source_page": "p1",
        "stitch_core": {"source_y1": 300, "core_source_y1": 300, "core_source_y2": 1300},
        "boxes": [],
        "text_objects": [{"id": "text_b1m", "region": {"x1": 100, "y1": 200, "x2": 500, "y2": 440}}],
    }
    sync_seam_mirrors({"pages": [owner_page, mirror_page]})
    mirror = mirror_page["text_objects"][0]
    assert mirror["bubble_mask"]["bbox"] == [100, 200, 500, 440]
    assert mirror["bubble_mask"]["rows"] == mask["rows"]
    assert mirror["bubble_mask"]["cy"] == mask["cy"] - 300
