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


def test_enlarge_letters_small_source_text_at_a_readable_size():
    def ink_height(enlarge):
        image = render_text_in_box(Image.new("RGB", (900, 600), "white"), "Chậm chạp~", (300, 250, 520, 330),
                                   fill="#000000", font_name="dialogue.mac-dinh-3", source_cap_px=8, enlarge=enlarge)
        rows = np.where(np.asarray(image.convert("L")) < 128)[0]
        return rows.max() - rows.min()

    assert ink_height(True) > 1.5 * ink_height(False)


def test_source_ink_colour_is_measured_and_low_contrast_text_gets_an_outline():
    from app.render.source_size import source_ink_hex

    raw = np.full((200, 400, 3), 255, np.uint8)
    cv2.putText(raw, "HELLO", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, (30, 30, 220), 12)  # BGR red
    ink = source_ink_hex(raw, (0, 0, 400, 200))
    assert ink and int(ink[1:3], 16) > 180 and int(ink[5:7], 16) < 80, "red letters read as red"

    dark = Image.new("RGB", (400, 200), (20, 20, 20))
    image = render_text_in_box(dark, "Chào", (0, 0, 400, 200), fill="#202020", font_name="dialogue.mac-dinh-3")
    assert np.asarray(image).max() > 200, "dark text on a dark background gets a light outline"


def test_a_misread_tiny_source_size_does_not_shrink_a_shout():
    def ink_height(cap):
        image = render_text_in_box(Image.new("RGB", (500, 300), "white"), "Thả ra ngay!", (0, 0, 500, 300),
                                   fill="#000000", font_name="dialogue.mac-dinh-3", source_cap_px=cap)
        rows = np.where(np.asarray(image.convert("L")) < 128)[0]
        return rows.max() - rows.min()

    assert ink_height(5) >= 0.55 * ink_height(None)


def test_wide_pages_letter_as_large_as_narrow_ones_relative_to_the_page():
    def ink_height(width):
        image = render_text_in_box(Image.new("RGB", (width, 1200), "white"), "Cái gì?!", (0, 0, width // 2, 600),
                                   fill="#000000", font_name="dialogue.mac-dinh-3")
        rows = np.where(np.asarray(image.convert("L")) < 128)[0]
        return rows.max() - rows.min()

    assert ink_height(1600) > 1.6 * ink_height(800)


def test_a_one_line_shout_filling_its_box_is_measured_at_its_own_height():
    image = np.full((90, 400, 3), 255, np.uint8)
    cv2.putText(image, "WHAT?", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 6)
    assert source_cap_px(image, (0, 0, 400, 90)) >= 45


def test_specks_in_the_box_do_not_shrink_the_source_size():
    import cv2
    import numpy as np

    from app.render.source_size import source_cap_px

    rng = np.random.default_rng(6)
    image = np.full((400, 1200, 3), 200, np.uint8)
    for x, y in rng.integers((20, 20), (1180, 380), (80, 2)):
        cv2.circle(image, (int(x), int(y)), 4, (40, 40, 40), -1)  # grit in the art
    cv2.putText(image, "WORTH IT", (60, 260), cv2.FONT_HERSHEY_DUPLEX, 4.0, (20, 20, 20), 14)
    cap = source_cap_px(image, (0, 0, 1200, 400))
    assert cap is not None and cap >= 80, cap
