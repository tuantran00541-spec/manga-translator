"""Chapter size bands: one chapter's body lettering reads as one size."""
import pytest
from PIL import Image, ImageDraw

from app.config import DEFAULT_FONT
from app.render.page_renderer import render_text_objects
from app.render.size_bands import (
    BAND_HI_FACTOR,
    BAND_LO_FACTOR,
    chapter_size_bands,
)
from app.render.source_size import SIZE_SLACK, matching_font_px
from app.render.text_renderer import auto_fit, get_font_object, render_text_in_box
from app.schemas import RenderRequest


def _page(*objs):
    return {"text_objects": list(objs)}


def _obj(cap, role="dialogue", translation="Lời thoại", oid="t"):
    return {"id": oid, "translation": translation, "source_cap_px": cap,
            "typography_role": role}


def test_bands_come_from_the_role_median_not_the_loudest_bubble():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page(_obj(90, oid="big"), _obj(8, oid="dots"))]
    bands = chapter_size_bands(pages)
    assert bands["dialogue"] == pytest.approx(
        (BAND_LO_FACTOR * 38, BAND_HI_FACTOR * 38))


def test_bands_cover_only_body_roles_and_need_enough_caps():
    pages = [_page(*[_obj(40, role="sfx", oid=f"s{i}") for i in range(8)]),
             _page(*[_obj(40, role="title", oid=f"t{i}") for i in range(8)]),
             _page(*[_obj(40, role="free_text", oid=f"f{i}") for i in range(8)]),
             _page(*[_obj(40, role="shout", oid=f"h{i}") for i in range(8)]),
             _page(*[_obj(40, oid=f"d{i}") for i in range(4)])]
    assert chapter_size_bands(pages) == {}


def test_bands_skip_unrendered_and_capless_objects():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page({"id": "x", "translation": "", "source_cap_px": 200,
                    "typography_role": "dialogue"},
                   {"id": "y", "translation": "Lời", "source_cap_px": None,
                    "typography_role": "dialogue"},
                   {"id": "z", "translation": "Lời", "source_cap_px": 200,
                    "typography_role": "dialogue", "source_missing": True})]
    bands = chapter_size_bands(pages)
    assert bands["dialogue"] == pytest.approx(
        (BAND_LO_FACTOR * 38, BAND_HI_FACTOR * 38))


def test_a_ceiling_param_no_longer_sizes_anything():
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    font = str(DEFAULT_FONT)
    ceiling = int(matching_font_px(font, 57) * SIZE_SLACK)
    size, _lines, fits, _box = auto_fit(draw, "Khoan.", (0, 0, 800, 600), (900, 1000),
                                       font, source_cap_px=90, size_ceiling_px=ceiling)
    assert fits and size == matching_font_px(font, 90), f"size is source-matched, got {size}"
    size_free, _lines, _fits, _box = auto_fit(draw, "Khoan.", (0, 0, 800, 600),
                                             (900, 1000), font, source_cap_px=90)
    assert size_free == size, "the ceiling parameter must not move the size"


def test_a_band_hi_cap_no_longer_sizes_anything():
    page = Image.new("RGB", (900, 1000), "white")
    stats: dict = {}
    render_text_in_box(page.copy(), "Khoan.", (50, 50, 850, 650),
                       source_cap_px=90, size_band_hi_cap=57, stats=stats)
    assert stats["size"] == matching_font_px(str(DEFAULT_FONT), 90)


def test_a_cap_inside_its_band_renders_exactly_as_before():
    page = Image.new("RGB", (900, 1000), "white")
    text = "Ta đã bảo ngươi đừng gọi ta là Các Chủ."
    box = (100, 100, 700, 400)
    plain: dict = {}
    render_text_in_box(page.copy(), text, box, source_cap_px=39, stats=plain)
    banded: dict = {}
    render_text_in_box(page.copy(), text, box, source_cap_px=39,
                       size_band_hi_cap=57, stats=banded)
    assert banded["size"] == plain["size"] and banded["lines"] == plain["lines"]


def _render_one(obj, bands):
    req = RenderRequest(chapter_id="chapter-test", page_index=0,
                        translations={obj["id"]: obj["translation"]})
    image = Image.new("RGB", (900, 1000), "white")
    verdicts: list = []
    render_text_objects(image, req, [obj], *({} for _ in range(10)),
                        verdicts=verdicts, size_bands=bands)
    return verdicts[0]


def test_the_chapter_band_moves_nothing_any_more():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page(_obj(90, oid="big"), _obj(8, oid="dots"))]
    bands = chapter_size_bands(pages)
    base = {"translation": "Chín lần sao?", "region": {"x1": 100, "y1": 100, "x2": 700, "y2": 500},
            "container": "bubble", "typography_role": "dialogue",
            "style": {"font": "dialogue.mac-dinh-3"}}
    big = _render_one({**base, "id": "big", "source_cap_px": 90}, bands)
    big_plain = _render_one({**base, "id": "big", "source_cap_px": 90}, None)
    assert big["size"] == big_plain["size"], "the 90 px cap is no longer pulled toward the chapter"
    assert "size_band" not in big
    dots = _render_one({**base, "id": "dots", "translation": ".......",
                        "region": {"x1": 100, "y1": 100, "x2": 282, "y2": 205},
                        "source_cap_px": 8}, bands)
    dots_plain = _render_one({**base, "id": "dots", "translation": ".......",
                              "region": {"x1": 100, "y1": 100, "x2": 282, "y2": 205},
                              "source_cap_px": 8}, None)
    assert dots["size"] == dots_plain["size"], "the 8 px dots cap is no longer lifted toward the chapter"
    assert dots["size"] >= 24, "still readable, still inside its bubble"
    mid = _render_one({**base, "id": "mid", "source_cap_px": 38}, bands)
    mid_plain = _render_one({**base, "id": "mid", "source_cap_px": 38}, None)
    assert mid["size"] == mid_plain["size"], "an in-band cap renders exactly as before"


def test_no_bands_means_no_change():
    pages = [_page(_obj(90, oid="big"))]
    assert chapter_size_bands(pages) == {}
