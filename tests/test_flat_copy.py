import numpy as np

from app.inpaint.lama_inpainter import Inpainter


def test_flat_colour_is_copied_and_a_straight_box_edge_stays_straight():
    # A white caption box under a grey band; the hole crosses the band's lower edge, as on Hero Cannot Rest 1.
    crop = np.full((200, 300, 3), 250, np.uint8)
    crop[:60] = (120, 120, 120)
    hole = np.zeros((200, 300), bool)
    hole[40:150, 80:220] = True
    filled, rest = Inpainter._flat_copy(crop, hole)
    assert (filled[hole & ~rest] == crop[hole & ~rest]).all(), "copied pixels are the true flat colours"
    assert not rest[80:150].any(), "the white inside is copied, not left to LaMa"
    assert rest[55:66].any(), "the band edge is left to LaMa"


def test_busy_art_is_left_to_lama():
    rng = np.random.default_rng(3)
    crop = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    hole = np.zeros((120, 160), bool)
    hole[40:80, 50:110] = True
    assert Inpainter._flat_copy(crop, hole) is None
