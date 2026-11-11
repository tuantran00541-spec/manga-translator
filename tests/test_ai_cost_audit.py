"""The cost-run audit reproduces renderer sizing, so it must not apply the chapter band.

The renderer no longer sizes from the chapter band (the band is dead
after the render rewrite), so an audit that band-clamps the cap
describes a render that never happened. Banded and raw caps must agree.
"""
from scripts.ai_cost_run import _fit_metrics
from app.render.size_bands import chapter_size_bands


def _page(*objs):
    return {"text_objects": list(objs)}


def _obj(cap, role="dialogue", translation="Chín lần sao?", oid="t"):
    return {"id": oid, "translation": translation, "source_cap_px": cap,
            "typography_role": role, "region": {"x1": 100, "y1": 100, "x2": 700, "y2": 500},
            "container": "bubble", "style": {"font": "dialogue.mac-dinh-3"}}


def test_fit_metrics_sizes_from_the_raw_cap_ignoring_the_band():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page(_obj(90, oid="big"), _obj(8, oid="dots", translation="......."))]
    bands = chapter_size_bands(pages)
    big = next(o for p in pages for o in p["text_objects"] if o["id"] == "big")
    dots = next(o for p in pages for o in p["text_objects"] if o["id"] == "dots")
    assert _fit_metrics(big, 900, bands) == _fit_metrics(big, 900), "the band must not move the 90 px cap"
    assert _fit_metrics(big, 900)["font_px"] == 143
    assert _fit_metrics(dots, 900, bands) == _fit_metrics(dots, 900), "the band must not lift the 8 px dots cap"


def test_fit_metrics_ignores_the_band_for_every_role():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page(_obj(90, role="sfx", oid="sfx", translation="BÙM"))]
    bands = chapter_size_bands(pages)
    sfx = next(o for p in pages for o in p["text_objects"] if o["id"] == "sfx")
    assert _fit_metrics(sfx, 900, bands) == _fit_metrics(sfx, 900)
    assert _fit_metrics(sfx, 900, None) == _fit_metrics(sfx, 900)
