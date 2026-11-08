"""Bubble boxes measured from outlines, not detection regions (Wimp's
Strategy Guide ch.27)."""
import numpy as np
from PIL import Image, ImageDraw

from app.render.text_renderer import (
    _bubble_bands,
    _divider_cut,
    _edge_magnitude,
    _ray_boxes,
    _seed_points,
    _trim_bubble_box,
)
import cv2


def _bubble_image():
    """White oval on black, plus a white panel strip below a divider."""
    image = Image.new("RGB", (400, 500), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 350, 300], fill="white", outline=(0, 0, 0), width=3)
    draw.rectangle([0, 400, 400, 500], fill="white")
    draw.line([0, 400, 400, 400], fill=(0, 0, 0), width=3)
    return image


def _mag_and_dist(image):
    mag = _edge_magnitude(image)
    dist = cv2.distanceTransform((mag <= 80).astype(np.uint8), cv2.DIST_L2, 5)
    return mag, dist


def test_bubble_bands_union_split_bubble():
    """A bubble split by a dark line floods in pieces; the union is the interior."""
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.line([0, 100, 200, 100], fill=(0, 0, 0), width=2)
    bands = _bubble_bands(image, (10, 10, 190, 190))
    assert bands is not None
    assert len(bands) == 24


def test_ray_boxes_recover_a_shifted_region():
    """A region shifted off the bubble still letters inside the bubble."""
    image = _bubble_image()
    mag, dist = _mag_and_dist(image)
    grey = np.asarray(image.convert("L"))
    box = (100, 80, 320, 280)  # shifted down-right from the oval
    seeds = _seed_points(mag, dist, box)
    assert seeds
    ray = _ray_boxes(mag, grey, box, seeds)
    assert ray is not None
    assert 40 <= ray[0] <= 60 and 40 <= ray[1] <= 60
    assert 340 <= ray[2] <= 360 and 290 <= ray[3] <= 310


def test_ray_boxes_reject_a_box_on_art():
    """A box on black art has no flat deep point and no bubble."""
    image = _bubble_image()
    mag, dist = _mag_and_dist(image)
    grey = np.asarray(image.convert("L"))
    box = (10, 320, 90, 390)  # black strip between oval and panel
    seeds = _seed_points(mag, dist, box)
    assert _ray_boxes(mag, grey, box, seeds) is None


def test_divider_cut_keeps_the_centre_side():
    """A box straddling the panel divider is cut to the side holding its centre."""
    image = _bubble_image()
    mag, _ = _mag_and_dist(image)
    box = (50, 300, 350, 450)  # centre above the divider at y=400
    cut = _divider_cut(mag, box)
    assert cut[3] <= 402
    assert cut[1] == 300


def test_trim_bubble_box_keeps_a_single_lobe():
    """Two lobes joined by a narrow neck trim to the lobe holding the widest row."""
    image = Image.new("RGB", (400, 500), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 350, 250], fill="white", outline=(0, 0, 0), width=3)
    draw.ellipse([150, 240, 250, 450], fill="white", outline=(0, 0, 0), width=3)
    draw.rectangle([150, 240, 250, 260], fill="white")
    trimmed = _trim_bubble_box(image, (50, 50, 350, 450))
    assert 50 < trimmed[1] < 100  # the top cap rows are not text rows
    assert trimmed[3] < 300  # the neck cut keeps the lower lobe out
