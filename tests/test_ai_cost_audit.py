"""The cost-run audit reproduces renderer sizing, so it must apply the chapter band.

Run 42505427459's report showed dialogue 28.4-104.9 px, identical to run 3,
because _fit_metrics sized every object from its raw cap: the audit's numbers
described a render that never happened, and no live fix could ever move them.
"""
from scripts.ai_cost_run import _fit_metrics
from app.render.size_bands import chapter_size_bands


def _page(*objs):
    return {"text_objects": list(objs)}


def _obj(cap, role="dialogue", translation="Chín lần sao?", oid="t"):
    return {"id": oid, "translation": translation, "source_cap_px": cap,
            "typography_role": role, "region": {"x1": 100, "y1": 100, "x2": 700, "y2": 500},
            "container": "bubble", "style": {"font": "dialogue.mac-dinh-3"}}


def test_fit_metrics_sizes_from_the_banded_cap_not_the_raw_cap():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page(_obj(90, oid="big"), _obj(8, oid="dots", translation="......."))]
    bands = chapter_size_bands(pages)
    big = next(o for p in pages for o in p["text_objects"] if o["id"] == "big")
    dots = next(o for p in pages for o in p["text_objects"] if o["id"] == "dots")
    assert (_fit_metrics(big, 900, bands)["font_px_at_800"]
            < _fit_metrics(big, 900)["font_px_at_800"]), "the 90 px cap is pulled toward the chapter"
    assert (_fit_metrics(dots, 900, bands)["font_px_at_800"]
            > _fit_metrics(dots, 900)["font_px_at_800"]), "the 8 px dots cap is lifted toward the chapter"


def test_fit_metrics_leaves_unbanded_roles_and_missing_bands_alone():
    pages = [_page(*[_obj(38, oid=f"d{i}") for i in range(8)]),
             _page(_obj(90, role="sfx", oid="sfx", translation="BÙM"))]
    bands = chapter_size_bands(pages)
    sfx = next(o for p in pages for o in p["text_objects"] if o["id"] == "sfx")
    assert _fit_metrics(sfx, 900, bands) == _fit_metrics(sfx, 900)
    assert _fit_metrics(sfx, 900, None) == _fit_metrics(sfx, 900)
