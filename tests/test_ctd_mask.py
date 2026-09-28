import cv2
import numpy as np
import pytest

from app.detector import ctd_mask
from ctd_fake import InkModel


@pytest.fixture(autouse=True)
def ink_model(monkeypatch):
    model = InkModel()
    monkeypatch.setattr(ctd_mask, "_session", model)
    return model


def _full(image, box, mask):
    full = np.zeros(image.shape[:2], bool)
    full[box[1]:box[3], box[0]:box[2]] = mask
    return full


def test_the_model_sees_the_padded_box_at_full_and_half_size(ink_model):
    image = np.full((300, 700, 3), 235, np.uint8)
    cv2.putText(image, "HELLO", (120, 180), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 20), 5)
    box, mask = ctd_mask.letter_mask(image, (110, 120, 420, 200))
    assert box[0] < 110 and box[1] < 120 and box[2] > 420 and box[3] > 200, "the box is padded for cut letters"
    assert len(ink_model.shapes) == 2 and all(s[2] % 64 == 0 and s[3] % 64 == 0 for s in ink_model.shapes)
    ink = image.max(axis=2) < 60
    assert (_full(image, box, mask) & ink).sum() >= 0.98 * ink.sum()


def test_an_outline_the_model_misses_is_erased_with_the_letters():
    image = np.full((400, 900, 3), (120, 140, 150), np.uint8)
    cv2.putText(image, "WORTH", (120, 260), cv2.FONT_HERSHEY_DUPLEX, 4.0, (255, 255, 255), 40)  # thick white outline
    cv2.putText(image, "WORTH", (120, 260), cv2.FONT_HERSHEY_DUPLEX, 4.0, (20, 20, 20), 12)
    box, mask = ctd_mask.letter_mask(image, (100, 120, 800, 300))
    outline = np.abs(image.astype(int) - (255, 255, 255)).sum(axis=2) < 30
    assert (_full(image, box, mask) & outline).sum() >= 0.9 * outline.sum()


def test_glow_round_letters_is_erased_to_the_background():
    ctd_mask._session = InkModel(light_above=225)
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
    box, mask = ctd_mask.letter_mask(image, (70, 150, 900, 270))
    changed = np.abs(image.astype(int) - background.astype(int)).max(axis=2) > 40
    assert (_full(image, box, mask) & changed).sum() >= 0.95 * changed.sum(), "a glow left round the hole paints the letters back"


def test_textured_art_below_the_letters_stays():
    rng = np.random.default_rng(4)
    image = np.empty((500, 1200, 3), np.uint8)
    image[:170] = (225, 205, 185)  # light sky over dark trees
    image[170:] = np.clip(rng.normal(60, 18, (330, 1200, 3)), 0, 255).astype(np.uint8)
    for y in (200, 330):
        cv2.putText(image, "THE SLUMS", (160, y), cv2.FONT_HERSHEY_DUPLEX, 3.2, (255, 255, 255), 26)
        cv2.putText(image, "THE SLUMS", (160, y), cv2.FONT_HERSHEY_DUPLEX, 3.2, (10, 10, 10), 10)
    box, mask = ctd_mask.letter_mask(image, (130, 90, 1000, 370))
    full = _full(image, box, mask)
    ink = image.max(axis=2) < 20
    assert (full & ink).sum() >= 0.95 * ink.sum(), "both lines are erased"
    assert full[450:, :].mean() < 0.05, "the trees below stay"


def test_art_running_off_the_box_is_not_taken_as_glow():
    ctd_mask._session = InkModel(light_above=235)
    image = np.full((300, 800, 3), 40, np.uint8)
    image[146:154, :] = 150  # a smooth bar of art crossing the whole panel
    cv2.putText(image, "GULP", (250, 190), cv2.FONT_HERSHEY_DUPLEX, 3.0, (250, 250, 250), 10)
    box, mask = ctd_mask.letter_mask(image, (230, 90, 560, 210))
    full = _full(image, box, mask)
    assert not full[146:154, box[0]:box[0] + 20].any(), "the bar far from the letters stays"


def test_an_erased_block_is_passed_again_only_while_letters_remain():
    clean = np.full((200, 400, 3), 230, np.uint8)
    assert not ctd_mask.still_reads(clean, (50, 50, 350, 150))
    cv2.putText(clean, "LEFT", (120, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (20, 20, 20), 4)
    assert ctd_mask.still_reads(clean, (50, 50, 350, 150))
