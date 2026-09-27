import cv2
import numpy as np
from PIL import Image

from app.render.font_catalog import resolve_font_id
from app.render.source_size import char_budget, matching_font_px, source_cap_px
from app.render.text_renderer import render_text_in_box


def _lettered(height: int) -> np.ndarray:
    image = np.full((300, 600, 3), 255, np.uint8)
    cv2.putText(image, "HELLO THERE", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, height / 22, (0, 0, 0), 2)
    return image


def test_source_letter_height_follows_the_lettering():
    small, large = source_cap_px(_lettered(20), (0, 0, 600, 300)), source_cap_px(_lettered(40), (0, 0, 600, 300))
    assert small and large and 1.7 < large / small < 2.3
    assert source_cap_px(np.full((100, 100, 3), 255, np.uint8), (0, 0, 100, 100)) is None


def test_budget_grows_with_the_box_and_shrinks_with_the_letters():
    font = resolve_font_id("dialogue.mac-dinh-3")
    assert char_budget(font, (0, 0, 400, 200), 20) > char_budget(font, (0, 0, 200, 200), 20)
    assert char_budget(font, (0, 0, 400, 200), 20) > char_budget(font, (0, 0, 400, 200), 40)


def test_short_text_in_a_big_box_is_not_lettered_bigger_than_the_source():
    def ink_height(cap):
        image = render_text_in_box(Image.new("RGB", (800, 600), "white"), "Ừ", (0, 0, 800, 600),
                                   fill="#000000", font_name="dialogue.mac-dinh-3", source_cap_px=cap)
        rows = np.where(np.asarray(image.convert("L")) < 128)[0]
        return rows.max() - rows.min()

    assert ink_height(14) < ink_height(None) * 0.7
    assert matching_font_px(resolve_font_id("dialogue.mac-dinh-3"), 30) > 30


def test_large_source_lettering_is_matched_past_the_default_maximum():
    def ink_height(cap):
        image = render_text_in_box(Image.new("RGB", (900, 400), "white"), "Rầm", (0, 0, 900, 400),
                                   fill="#000000", font_name="dialogue.mac-dinh-3", source_cap_px=cap)
        rows = np.where(np.asarray(image.convert("L")) < 128)[0]
        return rows.max() - rows.min()

    assert ink_height(90) > 1.5 * ink_height(None)
