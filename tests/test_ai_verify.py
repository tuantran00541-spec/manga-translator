import cv2
import numpy as np
import pytest

from app.ai_mode import scorecard
from app.ai_mode.verify import leftovers
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
    clean[:, :400] = 240  # only the first half of the line was erased
    found = leftovers(original, clean, original.max(axis=2) < 60)
    assert [item.kind for item in found] == ["text"]


def test_a_clean_slice_kept_art_and_balloon_outlines_are_not_reported():
    original = _page()
    cv2.ellipse(original, (450, 330), (300, 40), 0, 200, 340, (10, 10, 10), 4)  # one stroke of a balloon outline
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
