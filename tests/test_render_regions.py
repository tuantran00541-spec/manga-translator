import numpy as np
from PIL import Image

from app.render.page_renderer import letter_box, _render_in_region

STYLE = dict(fill="#202020", font_size=40)


def test_a_text_crossing_the_slice_edge_is_laid_out_as_on_the_slice_that_holds_it_whole():
    # Shadow Slave 1, slices 14 and 15: one caption, lettered by both slices across their cut.
    page = Image.new("RGB", (800, 1200), (180, 200, 230))
    whole = _render_in_region(page.copy(), "Không phải thứ đồ tổng hợp rẻ tiền tôi thường uống", (100, 700, 700, 1100), **STYLE)
    upper = _render_in_region(page.crop((0, 0, 800, 900)), "Không phải thứ đồ tổng hợp rẻ tiền tôi thường uống",
                              (100, 700, 700, 1100), **STYLE)
    assert np.array_equal(np.asarray(upper), np.asarray(whole)[:900]), "the half each slice draws matches"


def test_lettering_goes_where_the_source_lettering_was_inside_its_region():
    obj = {"letter_bounds": {"x1": 337, "y1": 900, "x2": 1366, "y2": 1380}}
    box = letter_box(obj, (165, 827, 1336, 1441))
    assert box == (265, 828, 1336, 1441), "72 px of room round the source, cut to the region"
    assert letter_box({}, (0, 0, 10, 10)) is None


def test_a_box_that_shrinks_a_short_caption_well_below_the_source_size_gives_way():
    # Shadow Slave 1, page 16: a one-line source caption box made its longer translation small.
    import pytest

    from app.render.text_renderer import render_text_in_box

    page = Image.new("RGB", (1600, 900), "white")
    text = "Nó xuất hiện lần đầu hàng chục năm trước"
    style = dict(fill="#202020", source_cap_px=40)
    with pytest.raises(ValueError):
        render_text_in_box(page.copy(), text, (500, 400, 1100, 470), min_source_share=0.85, **style)
    render_text_in_box(page.copy(), text, (500, 400, 1100, 470), **style)  # the last box always letters
    render_text_in_box(page.copy(), text, (400, 330, 1200, 540), min_source_share=0.85, **style)


def test_a_wrapped_line_leaves_no_word_alone():
    # Shadow Slave 1, slice 70: "Với năng lực kỳ diệu có được" wrapped as "... CÓ" over a lone "ĐƯỢC".
    from PIL import ImageDraw
    from app.config import DEFAULT_FONT
    from app.render.text_renderer import _wrap_text, get_font_object
    draw, font = ImageDraw.Draw(Image.new("RGB", (10, 10))), get_font_object(str(DEFAULT_FONT), 40)
    text = "VỚI NĂNG LỰC KỲ DIỆU CÓ ĐƯỢC\nSAU KHI VƯỢT QUA THỬ THÁCH"
    width = draw.textbbox((0, 0), "VỚI NĂNG LỰC KỲ DIỆU CÓ", font=font)[2] + 4
    assert _wrap_text(draw, text, font, width)[1] == "ĐƯỢC", "the greedy fill the size search uses"
    lines = _wrap_text(draw, text, font, width, balance=True)
    assert len(lines) == len(_wrap_text(draw, text, font, width)), "as many lines, so the size still fits"
    assert all(len(line.split()) > 1 for line in lines)
    assert all(draw.textbbox((0, 0), line, font=font)[2] <= width for line in lines)


def test_a_vietnamese_word_is_not_split_across_lines():
    # Shadow Slave 1: "ĐƯỢC HUẤN / LUYỆN ĐẶC BIỆT" split "huấn luyện" between two lines.
    from PIL import ImageDraw
    from app.config import DEFAULT_FONT
    from app.render.text_renderer import _wrap_text, get_font_object
    draw, font = ImageDraw.Draw(Image.new("RGB", (10, 10))), get_font_object(str(DEFAULT_FONT), 40)
    text = "ĐƯỢC HUẤN LUYỆN ĐẶC BIỆT"
    width = draw.textbbox((0, 0), "ĐƯỢC HUẤN LUYỆN ĐẶC", font=font)[2] + 4
    lines = _wrap_text(draw, text, font, width, balance=True)
    assert len(lines) == len(_wrap_text(draw, text, font, width)), "as many lines, so the size still fits"
    assert any("HUẤN LUYỆN" in line for line in lines) and any("ĐẶC BIỆT" in line for line in lines), lines


def test_a_narrow_box_shrinks_the_letters_before_it_breaks_a_word():
    # Shadow Slave 1, slice 114: "Không..." in a narrow box was lettered "Khôn / g...".
    from PIL import ImageDraw

    from app.config import DEFAULT_FONT
    from app.render.text_renderer import _fit_text

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    size, lines, fits = _fit_text(draw, "Không...", 120, 400, str(DEFAULT_FONT), maximum_size=120)
    assert lines == ["Không..."] and fits, f"kept whole at {size}px"
