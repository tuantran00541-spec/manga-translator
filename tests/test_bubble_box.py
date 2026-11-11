"""Bubble masks: the detector's outline, not the detection region.

The old band/ray/divider machinery measured the outline from the
cleaned image on every render; the new core lays out on the mask the
detector derived once. These tests pin the mask behaviour the render
relies on: a shifted region still finds the bubble, art claims no
bubble, a dark line splits the flood, a waist trims to the lettering's
lobe, and shading does not collapse the interior.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.detector.bubble_mask import derive_bubble_mask, find_waist
from app.render.text_renderer import _mask_bands, _trim_mask_to_letters

FIXTURES = Path(__file__).parent / "fixtures" / "bubble_mask"


def _bubble_image():
    """White oval on black, plus a white panel strip below a divider."""
    image = Image.new("RGB", (400, 500), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 350, 300], fill="white", outline=(0, 0, 0), width=3)
    draw.rectangle([0, 400, 400, 500], fill="white")
    draw.line([0, 400, 400, 400], fill=(0, 0, 0), width=3)
    return image


def _bgr(image):
    return np.asarray(image)[:, :, ::-1].copy()


def test_mask_derives_the_bubble_from_a_shifted_region():
    """A region shifted off the bubble still finds the bubble itself."""
    mask = derive_bubble_mask(_bgr(_bubble_image()), (100, 80, 320, 280))
    assert mask is not None and mask["kind"] == "bubble"
    assert 40 <= mask["bbox"][0] <= 60 and 40 <= mask["bbox"][1] <= 60
    assert 340 <= mask["bbox"][2] <= 360 and 290 <= mask["bbox"][3] <= 310


def test_mask_rejects_a_box_on_art():
    """A box on black art claims no bubble: the render falls back to a rectangle."""
    mask = derive_bubble_mask(_bgr(_bubble_image()), (10, 320, 90, 390))
    assert mask is None or mask["kind"] != "bubble"


def test_mask_splits_a_bubble_on_a_dark_line():
    """A dark line across the oval floods one piece: the mask is that piece, inside the oval."""
    image = Image.new("RGB", (400, 500), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 350, 300], fill="white", outline=(0, 0, 0), width=3)
    draw.line([50, 175, 350, 175], fill=(0, 0, 0), width=2)
    mask = derive_bubble_mask(_bgr(image), (50, 50, 350, 300))
    assert mask is not None and mask["kind"] == "bubble"
    assert 50 <= mask["bbox"][0] and mask["bbox"][2] <= 350
    assert 50 <= mask["bbox"][1] and mask["bbox"][3] <= 300
    assert len(mask["rows"]) == 24


def test_mask_straddling_a_divider_claims_no_bubble():
    """A box across the panel divider is no closed bubble, so no bubble is claimed."""
    mask = derive_bubble_mask(_bgr(_bubble_image()), (50, 300, 350, 450))
    assert mask is None or mask["kind"] != "bubble"


def test_trim_mask_keeps_the_lobe_holding_the_lettering():
    """MH p068's stacked double bubble trims to the lettering's lobe."""
    img = cv2.imread(str(FIXTURES / "mh068-stacked.png"))
    box = (106, 64, 436, 263)
    mask = derive_bubble_mask(img, box)
    assert mask["kind"] == "bubble"
    waist = find_waist(mask)
    assert waist is not None and waist["orientation"] == "horizontal"
    top = _trim_mask_to_letters(mask, box, (120, 70, 420, 130))
    assert top is not None and top["bbox"][3] <= waist["cut_y"]
    assert top["bbox"][3] < mask["bbox"][3]
    bottom = _trim_mask_to_letters(mask, box, (120, 300, 420, 360))
    assert bottom is not None and bottom["bbox"][1] >= waist["cut_y"]
    assert bottom["bbox"][1] > mask["bbox"][1]
    both = _trim_mask_to_letters(mask, box, (120, 70, 420, 360))
    assert both is mask, "lettering spanning both lobes keeps the whole mask"


def test_mask_rows_keep_the_interior_under_diagonal_shading():
    """A grey bubble with diagonal shading keeps wide rows, like slice 45's bubble.

    The shading lines are interior art, not the outline loop: the mask's
    widest row still spans most of the bubble, so the fit does not
    collapse onto a shading strip.
    """
    image = Image.new("RGB", (700, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 650, 350], fill=(150, 150, 150), outline=(0, 0, 0), width=4)
    for k in range(6):
        draw.line([(120 + k * 90, 50), (200 + k * 90, 350)], fill=(110, 110, 110), width=12)
    mask = derive_bubble_mask(_bgr(image), (60, 60, 640, 340))
    assert mask is not None and mask["kind"] == "bubble"
    bands = _mask_bands(mask, (60, 60, 640, 340))
    assert bands is not None
    widest = max(b - a for band in bands for a, b in band)
    assert widest >= 0.5 * (640 - 60), f"widest mask interval is {widest}"
