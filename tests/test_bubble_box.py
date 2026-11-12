from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.detector.bubble_mask import derive_bubble_mask, find_waist
from app.render.text_renderer import _mask_bands, _trim_mask_to_letters

FIXTURES = Path(__file__).parent / "fixtures" / "bubble_mask"


def _bubble_image():
    image = Image.new("RGB", (400, 500), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 350, 300], fill="white", outline=(0, 0, 0), width=3)
    draw.rectangle([0, 400, 400, 500], fill="white")
    draw.line([0, 400, 400, 400], fill=(0, 0, 0), width=3)
    return image


def _bgr(image):
    return np.asarray(image)[:, :, ::-1].copy()


def test_mask_derives_the_bubble_from_a_shifted_region():
    mask = derive_bubble_mask(_bgr(_bubble_image()), (100, 80, 320, 280))
    assert mask is not None and mask["kind"] == "bubble"
    assert 40 <= mask["bbox"][0] <= 60 and 40 <= mask["bbox"][1] <= 60
    assert 340 <= mask["bbox"][2] <= 360 and 290 <= mask["bbox"][3] <= 310


def test_mask_rejects_a_box_on_art():
    mask = derive_bubble_mask(_bgr(_bubble_image()), (10, 320, 90, 390))
    assert mask is None or mask["kind"] != "bubble"


def test_mask_splits_a_bubble_on_a_dark_line():
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
    mask = derive_bubble_mask(_bgr(_bubble_image()), (50, 300, 350, 450))
    assert mask is None or mask["kind"] != "bubble"


def test_trim_mask_keeps_the_lobe_holding_the_lettering():
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


def _union_mask_from_circles(bbox=(60, 60, 560, 340)):
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


def test_trim_with_letters_cuts_a_diagonal_waist():
    from app.render.text_renderer import _mask_centroid

    mask = _union_mask_from_circles()
    assert find_waist(mask) is None
    letters = np.zeros((80, 420), bool)
    letters[10:30, 0:180] = True
    letters[40:60, 0:180] = True
    letters[10:30, 240:420] = True
    letters[40:60, 240:420] = True
    piece = _trim_mask_to_letters(mask, (60, 60, 560, 340), (100, 180, 280, 240),
                                  letters=letters, letters_box=(100, 170, 520, 250))
    assert piece is not mask
    assert piece["bbox"][2] <= 340
    assert piece["bbox"][0] == mask["bbox"][0] and piece["cx"] < mask["cx"]
    assert (piece["cx"], piece["cy"]) == _mask_centroid(piece)
    assert piece["cy"] != mask["cy"]
    again = _trim_mask_to_letters(piece, (60, 60, 560, 340), (100, 180, 280, 240),
                                  letters=letters, letters_box=(100, 170, 520, 250))
    assert again is piece


def test_mask_rows_keep_the_interior_under_diagonal_shading():
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
