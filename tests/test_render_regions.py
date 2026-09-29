import numpy as np
from PIL import Image

from app.render.page_renderer import _letter_box, _render_in_region

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
    box = _letter_box(obj, (165, 827, 1336, 1441))
    assert box == (265, 828, 1336, 1441), "72 px of room round the source, cut to the region"
    assert _letter_box({}, (0, 0, 10, 10)) is None
