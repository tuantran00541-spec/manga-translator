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


def test_letters_the_box_cuts_are_erased_up_to_the_end_of_the_line():
    # Shadow Slave 1, slice 70: the detector box stopped short of "TRIALS" and "NIGHTMARE...".
    image = np.full((420, 1600, 3), 245, np.uint8)
    cv2.putText(image, "BY OVERCOMING TRIALS", (330, 180), cv2.FONT_HERSHEY_DUPLEX, 2.2, (15, 15, 15), 6)
    cv2.putText(image, "INSIDE THE NIGHTMARE...", (330, 290), cv2.FONT_HERSHEY_DUPLEX, 2.2, (15, 15, 15), 6)
    cv2.putText(image, "OTHER BUBBLE", (1450, 390), cv2.FONT_HERSHEY_DUPLEX, 1.0, (15, 15, 15), 3)
    box, mask = ctd_mask.letter_mask(image, (330, 120, 1000, 310))
    full = _full(image, box, mask)
    ink = image.max(axis=2) < 60
    line = ink.copy()
    line[330:, :] = False  # the far bubble
    assert (full & line).sum() >= 0.98 * line.sum(), "the whole caption is erased, dots included"
    assert not (full & ink)[340:, 1440:].any(), "a separate text farther away is not taken"
    assert box[2] < 1440, "the box grows over the cut letters only"


def test_every_dot_of_a_wide_ellipsis_stays_with_its_text():
    # Shadow Slave 1, slice 70 cleaned from its core: the crop held all of "...", yet only the dot beside the E was kept.
    seed = np.zeros((120, 400), bool)
    for x in range(40, 200, 32):
        seed[30:80, x:x + 24] = True  # letters the first box holds
    for x in (212, 240, 268):
        seed[70:80, x:x + 10] = True  # dots 18 px apart, the last two past the reach of the letters
    seed[70:80, 380:390] = True  # a stray speck far from the text
    inside = np.zeros_like(seed)
    inside[:, :205] = True
    kept = ctd_mask._chained(seed, inside)
    assert all(kept[70:80, x:x + 10].all() for x in (212, 240, 268)), "all three dots stay"
    assert not kept[70:80, 380:390].any(), "a speck far from the text does not"

def test_flecks_and_outline_scraps_beside_a_run_of_dots_do_not_join_it():
    # 392 real text crops: a 1-4 px fleck or a bubble-outline scrap chained to the last dot grew a bite out of the outline.
    seed = np.zeros((120, 400), bool)
    for x in range(40, 200, 32):
        seed[30:80, x:x + 24] = True
    for x in (212, 240):
        seed[70:80, x:x + 10] = True  # the dots
    seed[72:74, 268:270] = True  # a 2 px fleck one step on
    seed[50:80, 262:266] = True  # a thin outline scrap
    inside = np.zeros_like(seed)
    inside[:, :205] = True
    kept = ctd_mask._chained(seed, inside)
    assert kept[70:80, 212:222].all() and kept[70:80, 240:250].all(), "the dots stay"
    assert not kept[72:74, 268:270].any() and not kept[50:80, 262:266].any(), "the fleck and the scrap do not"

def test_one_image_always_gives_one_mask():
    # The World After the End 254: a second call on the same crop grew a different mask, so LaMa painted it differently.
    ctd_mask._session = InkModel(light_above=225)
    rng = np.random.default_rng(7)
    patches = rng.integers(20, 120, (8, 20, 3)).astype(np.uint8)
    image = cv2.resize(patches, (1000, 400), interpolation=cv2.INTER_NEAREST)
    image = np.clip(image + rng.normal(0, 14, image.shape), 0, 255).astype(np.uint8)
    cv2.putText(image, "DARE TO DREAM", (80, 240), cv2.FONT_HERSHEY_DUPLEX, 3.0, (200, 190, 60), 22)
    cv2.putText(image, "DARE TO DREAM", (80, 240), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 240, 170), 8)
    masks = [ctd_mask.letter_mask(image, (70, 150, 900, 270))[1] for _ in range(4)]
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])


def test_a_soft_drop_shadow_on_a_flat_balloon_is_erased_but_the_outline_stays():
    # Solo Swordmaster 1, slice 105: a grey blurred shadow the mask left behind came back as ghost letters.
    image = np.full((420, 1000, 3), 255, np.uint8)
    image[:60] = (230, 200, 150)  # sky past the balloon, which makes the ring many-coloured
    cv2.rectangle(image, (60, 90), (940, 390), (20, 20, 20), 5)  # the balloon outline
    shadow = np.zeros((420, 1000), np.uint8)
    cv2.putText(shadow, "STOP!!", (250, 290), cv2.FONT_HERSHEY_DUPLEX, 3.5, 255, 14)
    shadow = cv2.GaussianBlur(np.roll(shadow, (8, 8), (0, 1)), (0, 0), 4).astype(np.float32)[..., None] / 255
    image = np.clip(image - shadow * 110, 0, 255).astype(np.uint8)
    cv2.putText(image, "STOP!!", (250, 290), cv2.FONT_HERSHEY_DUPLEX, 3.5, (10, 10, 10), 14)
    box, mask = ctd_mask.letter_mask(image, (230, 170, 800, 320))
    full = _full(image, box, mask)
    grey = (image.min(axis=2) < 235) & (image.min(axis=2) > 40)
    grey[:100] = grey[380:] = False
    grey[:, :100] = grey[:, 900:] = False
    assert (full & grey).sum() >= 0.95 * grey.sum(), "the shadow goes with the letters"
    assert not full[88:93, 60:940].any(), "the outline stays"


def test_letters_on_a_dark_badge_are_erased_with_the_badge():
    ctd_mask._session = InkModel(light_above=200)
    image = np.full((300, 900, 3), (25, 15, 20), np.uint8)
    cv2.rectangle(image, (60, 128), (420, 176), (150, 40, 120), -1)  # a purple scanlator badge
    cv2.putText(image, "SCANS.COM", (75, 165), cv2.FONT_HERSHEY_DUPLEX, 1.4, (230, 230, 230), 3)
    box, mask = ctd_mask.letter_mask(image, (70, 125, 410, 175))
    badge = np.zeros(image.shape[:2], bool)
    badge[128:177, 60:421] = True
    assert (_full(image, box, mask) & badge).sum() >= 0.95 * badge.sum(), "the badge goes with its letters"


def test_a_white_bubble_round_the_letters_stays():
    image = np.full((400, 900, 3), (60, 70, 90), np.uint8)
    cv2.ellipse(image, (450, 200), (300, 120), 0, 0, 360, (255, 255, 255), -1)
    cv2.putText(image, "HELLO", (300, 225), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 20), 5)
    box, mask = ctd_mask.letter_mask(image, (290, 160, 610, 240))
    paper = (image.min(axis=2) > 250)
    assert (_full(image, box, mask) & paper).sum() < 0.2 * paper.sum(), "the bubble keeps its paper"


def test_welded_glowing_letters_still_get_a_letter_tall_reach():
    seed = np.zeros((300, 600), bool)
    seed[40:260, 50:550] = True  # glow welds three lines into one blob
    prob = np.zeros(seed.shape, np.float32)
    for top in (50, 130, 210):
        prob[top:top + 40, 60:540] = 1.0  # the surest pixels show the lines
    assert ctd_mask.text_size(seed, prob) == 40
