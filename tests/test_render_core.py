"""The new render core: mask layout, centroid anchor, single cap, floor, graceful fallback."""
import numpy as np
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.detector.bubble_mask import derive_bubble_mask
from app.render.source_size import matching_font_px
from app.render.text_renderer import auto_fit, render_text_in_box


def _ellipse_bubble():
    image = Image.new("RGB", (700, 400), "white")
    ImageDraw.Draw(image).ellipse([50, 50, 650, 350], fill="white",
                                  outline=(0, 0, 0), width=4)
    return image


def _ink(image, base):
    return ((np.abs(np.asarray(image).astype(int)
                    - np.asarray(base).astype(int)).max(axis=2)) > 0)


def test_mask_layout_uses_row_spans_not_the_rectangle():
    """Text laid out on the mask stays inside the ellipse, not just its bbox."""
    image = _ellipse_bubble()
    base = image.copy()
    stats: dict = {}
    render_text_in_box(image, "Ta đã bảo ngươi đừng gọi ta là Các Chủ. "
                       "Ngươi có nghe rõ không?", (60, 60, 640, 340),
                       font_name="dialogue.mac-dinh-3", source_cap_px=40,
                       container="bubble", source_letter_box=(200, 150, 400, 250),
                       stats=stats)
    assert stats["frame_source"] == "mask"
    ink = _ink(image, base)
    ys, xs = np.nonzero(ink)
    assert len(xs) > 0
    assert bool((((xs - 350) / 300) ** 2 + ((ys - 200) / 150) ** 2 <= 1.05).all()), \
        "ink crossed the ellipse outline"


def test_block_centres_on_the_mask_centroid_not_the_box():
    """A bubble low in its region centres the block on the mask, not the region."""
    image = Image.new("RGB", (700, 400), "white")
    ImageDraw.Draw(image).ellipse([100, 200, 600, 340], fill="white",
                                  outline=(0, 0, 0), width=4)
    mask = derive_bubble_mask(np.asarray(image)[:, :, ::-1].copy(),
                              (60, 60, 640, 340), seed=(350, 270))
    assert mask is not None and mask["kind"] == "bubble"
    assert mask["cy"] == 270 and mask["cy"] != 200
    stats: dict = {}
    size, lines, fits, _box = auto_fit(
        ImageDraw.Draw(image), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
        (60, 60, 640, 340), image.size, str(DEFAULT_FONT), source_cap_px=40,
        image=image, container="bubble", centered=True, bubble_mask=mask,
        stats_out=stats)
    assert fits
    from app.render.text_renderer import _calc_line_height, get_font_object
    line_height = _calc_line_height(ImageDraw.Draw(image),
                                    get_font_object(str(DEFAULT_FONT), size), stroke_w=2)
    block_centre = stats["block_top"] + line_height * len(lines) / 2
    assert abs(block_centre - mask["cy"]) <= line_height, \
        f"block centred at {block_centre}, mask centroid is {mask['cy']}"


def test_single_cap_source_anchors_the_size():
    """One cap: the source-matched size; the old ceiling and share parameters change nothing."""
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    font = str(DEFAULT_FONT)
    expected = matching_font_px(font, 90)
    size, _lines, fits, _box = auto_fit(draw, "Khoan.", (0, 0, 800, 600), (900, 1000),
                                       font, source_cap_px=90)
    assert fits and size == expected
    size_capped, _lines, _fits, _box = auto_fit(
        draw, "Khoan.", (0, 0, 800, 600), (900, 1000), font,
        source_cap_px=90, size_ceiling_px=65, min_source_share=0.85)
    assert size_capped == size, "the dead ceiling/share parameters must not move the size"


def test_floor_is_six_tenths_of_what_the_box_holds():
    """A misread tiny cap does not shrink the text far below what its box holds."""
    def ink_height(cap):
        image = render_text_in_box(Image.new("RGB", (500, 300), "white"), "Thả ra ngay!",
                                   (0, 0, 500, 300), fill="#000000",
                                   font_name="dialogue.mac-dinh-3", source_cap_px=cap)
        rows = np.where(np.asarray(image.convert("L")) < 128)[0]
        return rows.max() - rows.min()

    assert ink_height(5) >= 0.55 * ink_height(None)
    stats: dict = {}
    render_text_in_box(Image.new("RGB", (500, 300), "white"), "Thả ra ngay!",
                       (0, 0, 500, 300), font_name="dialogue.mac-dinh-3",
                       source_cap_px=5, stats=stats)
    assert stats["size"] == 28 and stats["floor_px"] == 28


def test_graceful_degradation_draws_at_the_floor_in_the_inscribed_rectangle():
    """A band fit that fails at the floor still draws at the floor inside
    the inscribed rectangle instead of refusing the block.

    The hourglass mask is wide in both lobes and narrow at its centroid
    waist, so the block's lines land on the waist and the band fit fails
    — while the inscribed rectangle of one lobe holds them at the floor.
    """
    rows = ([[(110, 290)] for _ in range(9)] + [[(190, 210)] for _ in range(6)]
            + [[(110, 290)] for _ in range(9)])
    mask = {"bbox": [100, 100, 300, 300], "cx": 200, "cy": 200,
            "rows": rows, "kind": "bubble"}
    image = Image.new("RGB", (500, 400), "white")
    stats: dict = {}
    size, lines, fits, _box = auto_fit(
        ImageDraw.Draw(image), "Một hai ba bốn năm sáu bảy tám chín mười",
        (100, 100, 300, 300), image.size, str(DEFAULT_FONT), source_cap_px=40,
        image=image, container="bubble", centered=True, bubble_mask=mask,
        stats_out=stats)
    assert fits, "the floor-in-rect fallback must draw the block"
    assert size == stats["floor_px"]
    assert "outline_fit_failed_floor_rect" in stats["fallbacks"]


def test_refusal_when_even_the_floor_holds_nowhere():
    """A block the floor size cannot hold anywhere is refused, not drawn tiny."""
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    _size, _lines, fits, _box = auto_fit(
        draw, "Một hai ba bốn năm sáu bảy tám chín mười", (0, 0, 20, 10),
        (900, 1000), str(DEFAULT_FONT), source_cap_px=40)
    assert not fits


def test_bubble_without_a_mask_derives_one_on_the_fly():
    """A bubble container with no persisted mask derives the mask from the image."""
    image = _ellipse_bubble()
    stats: dict = {}
    render_text_in_box(image.copy(), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
                       (60, 60, 640, 340), font_name="dialogue.mac-dinh-3",
                       source_cap_px=40, container="bubble",
                       source_letter_box=(200, 150, 400, 250), stats=stats)
    assert stats["mask_source"] == "derived"
    assert stats["frame_source"] == "mask"


def test_free_text_falls_back_to_the_box_rectangle():
    """Free text has no bubble to find: the fit is the box rectangle."""
    image = Image.new("RGB", (700, 400), "white")
    stats: dict = {}
    render_text_in_box(image.copy(), "Ta đã bảo ngươi đừng gọi ta là Các Chủ.",
                       (60, 60, 640, 340), font_name="dialogue.mac-dinh-3",
                       source_cap_px=40, container="free", stats=stats)
    assert stats["frame_source"] == "region_no_frame"
    assert stats["mask_source"] == "none"
