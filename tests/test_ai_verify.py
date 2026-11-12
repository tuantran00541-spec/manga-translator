import cv2
import numpy as np
import pytest

from scripts import scorecard
from scripts.verify import leftovers
from app.detector import ctd_mask
from ctd_fake import InkModel


@pytest.fixture(autouse=True)
def ink_model(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel())


def _page():
    image = np.full((400, 900, 3), 240, np.uint8)
    cv2.putText(image, "HELLO WORLD", (60, 200), cv2.FONT_HERSHEY_DUPLEX, 2.4, (10, 10, 10), 6)
    return image


def test_letters_still_on_the_clean_slice_are_reported():
    original = _page()
    clean = original.copy()
    clean[:, :400] = 240
    found = leftovers(original, clean, original.max(axis=2) < 60)
    assert [item.kind for item in found] == ["text"]


def test_a_clean_slice_kept_art_and_balloon_outlines_are_not_reported():
    original = _page()
    cv2.ellipse(original, (450, 330), (300, 40), 0, 200, 340, (10, 10, 10), 4)
    clean = np.full_like(original, 240)
    cv2.ellipse(clean, (450, 330), (300, 40), 0, 200, 340, (10, 10, 10), 4)
    letters = original.max(axis=2) < 60
    assert leftovers(original, clean, letters) == []
    assert leftovers(original, original, letters, keep=[(0, 0, 900, 260)]) == []


def test_the_scorecard_counts_objects_and_marks_better_and_worse():
    pages = [{"text_objects": [{"translation": "Xin chào"}, {"translation": ""},
                               {"translation": "", "joined_into": "a"}]},
             {"skipped": True, "text_objects": [{"translation": ""}]}]
    assert scorecard.text_score(pages) == {
        "objects": 2, "translated": 1, "untranslated": 1, "needs_review": 0, "kept_original": 0}
    base = {"clean": {"text_left": 3}, "text": {"translated": 10}, "spend": {"cost_usd": 0.2}}
    new = {"clean": {"text_left": 1}, "text": {"translated": 9}, "spend": {"cost_usd": 0.2}}
    assert scorecard.compare(base, new) == [
        "clean.text_left: 3 -> 1 (better)", "text.translated: 10 -> 9 (worse)"]


def _label_page():
    """A dark label on a textured map: the letter model cannot read it, the box model can."""
    original = np.full((400, 900, 3), (210, 190, 170), np.uint8)
    cv2.putText(original, "SHANDONG", (100, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (40, 30, 25), 4)
    return original


def test_a_detector_box_the_cleanup_never_touched_is_reported():
    original = _label_page()
    found = leftovers(original, original.copy(), np.zeros(original.shape[:2], bool),
                      raw_boxes=[(90, 150, 330, 210)])
    assert [(item.kind, item.box) for item in found] == [("text", (90, 150, 330, 210))]


def test_a_detector_box_the_cleanup_repainted_is_not_reported():
    original = _label_page()
    clean = original.copy()
    clean[150:210, 90:330] = (210, 190, 170)
    assert leftovers(original, clean, np.zeros(original.shape[:2], bool),
                     raw_boxes=[(90, 150, 330, 210)]) == []


def test_a_flat_detector_box_is_not_a_leftover():
    original = np.full((400, 900, 3), (210, 190, 170), np.uint8)
    assert leftovers(original, original.copy(), np.zeros(original.shape[:2], bool),
                     raw_boxes=[(90, 150, 330, 210)]) == []


def test_a_raw_box_covered_by_a_letter_block_is_not_counted_twice():
    original = _page()
    clean = original.copy()
    clean[:, :400] = 240
    letters = original.max(axis=2) < 60
    found = leftovers(original, clean, letters, raw_boxes=[(60, 150, 400, 210)])
    assert len(found) == 1 and found[0].kind == "text"


def test_clean_score_flags_a_box_the_manifest_never_held(tmp_path):
    import cv2 as _cv2

    from scripts import scorecard as _scorecard

    original = _label_page()
    orig_path, clean_path = tmp_path / "orig.png", tmp_path / "clean.png"
    _cv2.imwrite(str(orig_path), original)
    _cv2.imwrite(str(clean_path), original.copy())

    class _Detector:
        def text_boxes(self, image):
            return []

        def detect_slice(self, image):
            return [(90, 150, 330, 210, 0.8)]

    pages = [{"original": str(orig_path), "clean": str(clean_path), "boxes": [],
              "stitch_core": {"core_y1": 0, "core_y2": 400}}]
    card = _scorecard.clean_score(pages, _Detector())
    assert card["text_left"] == 1
    assert card["items"][0]["slice"] == 1 and card["items"][0]["kind"] == "text"


def test_clean_score_flags_a_fallback_box_with_no_letters(tmp_path):
    """A box the detector kept whole because the segmenter was blind there must be
    judged by the raw-box path, not hidden by its own whole-box mask: the mask would
    cover the raw box and let the text escape both paths."""
    from scripts.scorecard import clean_score
    import cv2 as _cv2
    import numpy as np

    original = np.full((200, 300, 3), 220, np.uint8)
    original[60:80, 90:190] = (70, 60, 50)
    clean = original.copy()
    orig_path, clean_path = tmp_path / "orig.png", tmp_path / "clean.png"
    _cv2.imwrite(str(orig_path), original)
    _cv2.imwrite(str(clean_path), clean)

    class Box:
        def __init__(self, x1, y1, x2, y2, letters):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
            self.letters = letters
            self.mask = np.full((y2 - y1, x2 - x1), 255, np.uint8)

    class FakeDetector:
        def text_boxes(self, image):
            return [Box(60, 50, 220, 90, None)]

        def detect_slice(self, image):
            return [(60, 50, 220, 90, 0.85)]

    pages = [{"original": str(orig_path), "clean": str(clean_path), "boxes": [],
              "stitch_core": {"core_y1": 0, "core_y2": 200}}]
    card = clean_score(pages, FakeDetector())
    assert card["text_left"] == 1, card["items"]


def test_clean_score_skips_a_raw_box_the_manifest_already_holds(tmp_path):
    import cv2 as _cv2

    from scripts import scorecard as _scorecard

    original = _label_page()
    orig_path, clean_path = tmp_path / "orig.png", tmp_path / "clean.png"
    _cv2.imwrite(str(orig_path), original)
    _cv2.imwrite(str(clean_path), original.copy())

    class _Detector:
        def text_boxes(self, image):
            return []

        def detect_slice(self, image):
            return [(90, 150, 330, 210, 0.8)]

    pages = [{"original": str(orig_path), "clean": str(clean_path),
              "boxes": [{"x1": 90, "y1": 150, "x2": 330, "y2": 210}],
              "stitch_core": {"core_y1": 0, "core_y2": 400}}]
    assert _scorecard.clean_score(pages, _Detector())["text_left"] == 0
