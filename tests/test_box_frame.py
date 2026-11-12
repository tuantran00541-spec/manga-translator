import numpy as np
from PIL import Image, ImageDraw

from app.render.text_renderer import _box_frame


def _box_image(w, h, rules, bg="white"):
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, wd in rules:
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=wd)
    return img


def test_a_rule_flush_with_the_region_edge_is_measured():
    img = _box_image(900, 400, [
        (70, 50, 70, 350, 4), (466, 50, 466, 350, 4),
        (70, 50, 466, 50, 4), (70, 350, 466, 350, 4),
    ])
    assert _box_frame(img, (63, 40, 467, 360)) == (73, 53, 465, 349)


def test_a_rule_one_pixel_from_the_region_edge_is_measured():
    img = _box_image(900, 400, [
        (70, 50, 70, 350, 4), (466, 50, 466, 350, 4),
        (70, 50, 466, 50, 4), (70, 350, 466, 350, 4),
    ])
    assert _box_frame(img, (63, 40, 468, 360)) == (73, 53, 465, 349)


def test_a_stepped_frame_straddling_the_region_edge_is_measured():
    img = _box_image(900, 600, [
        (236, 100, 236, 500, 4), (841, 100, 841, 500, 4),
        (841, 500, 236, 500, 4),
    ])
    frame = _box_frame(img, (187, 90, 833, 480))
    assert frame is not None
    assert frame[0] == 239 and frame[2] == 840, frame


def test_a_neighbours_rule_across_a_dark_gap_is_not_adopted():
    img = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((500, 0, 900, 400), fill=(40, 40, 40))
    for x1, y1, x2, y2 in [(236, 50, 236, 350), (700, 50, 700, 350),
                           (236, 50, 700, 50), (236, 350, 700, 350)]:
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=4)
    frame = _box_frame(img, (187, 40, 660, 360))
    assert frame is None or frame[2] == 660, frame


def test_a_plain_centred_box_still_measures_exactly():
    img = _box_image(900, 400, [
        (100, 50, 100, 350, 4), (500, 50, 500, 350, 4),
        (100, 50, 500, 50, 4), (100, 350, 500, 350, 4),
    ])
    assert _box_frame(img, (120, 70, 480, 330)) == (103, 53, 499, 349)


def test_a_box_with_no_rules_is_unmeasured_not_guessed():
    img = Image.new("RGB", (900, 400), "white")
    assert _box_frame(img, (100, 50, 500, 350)) is None
