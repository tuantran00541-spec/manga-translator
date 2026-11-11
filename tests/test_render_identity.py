"""Render identity covers every per-object render input, including the letter bitmap."""
from pathlib import Path

from PIL import Image

from app.render.identity import render_input_signature
from app.routers.export import _render_request_from_page
from app.routers.render_commit import _request_style_maps, _resolve_snapshot_fonts


def _manifest(tmp_path: Path, pages: list[dict]) -> dict:
    base = tmp_path / "base.png"
    Image.new("RGB", (20, 20), "white").save(base)
    for page in pages:
        page.setdefault("original", str(base))
        page.setdefault("clean", None)
    return {"pages": pages}


def test_signature_covers_the_objects_render_inputs(tmp_path):
    obj = {"id": "t1", "region": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "translation": "Chào"}
    manifest = _manifest(tmp_path, [{"text_objects": [obj]}])
    before = render_input_signature(manifest, 0)
    for field, value in (
        ("font_ai_id", "emphasis.bangers"),
        ("font_selection_mode", "user"),
        ("letter_bounds", {"x1": 1, "y1": 1, "x2": 8, "y2": 8}),
        ("bubble_mask", {"bbox": [0, 0, 10, 10], "cx": 5, "cy": 5,
                        "rows": [[[0, 10]] for _ in range(24)], "kind": "bubble"}),
        ("letters_mask", "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAK"),
        ("source_cap_px", 38),
        ("typography_role", "shout"),
        ("role", "narration"),
        ("container", "spiky"),
        ("ocr_text_region", {"x1": 0, "y1": 0, "x2": 9, "y2": 9}),
    ):
        obj[field] = value
        after = render_input_signature(manifest, 0)
        assert after != before, f"{field} must invalidate the render"
        before = after


def _two_page_chapter() -> list[dict]:
    return [
        {"text_objects": [
            {"id": f"d{i}", "translation": "Lời thoại.", "source_cap_px": 38,
             "typography_role": "dialogue"}
            for i in range(8)
        ]},
        {"text_objects": [
            {"id": "dots", "translation": ".......", "source_cap_px": 8,
             "typography_role": "dialogue"},
        ]},
    ]


def test_finishing_the_chapter_keeps_a_sibling_slice_current(tmp_path):
    """Finishing the chapter does not stale a sibling slice's render.

    The render signature covers only the page's own objects -- the
    chapter band is dead -- so filling in the rest of the chapter
    leaves a rendered slice current.
    """
    pages = _two_page_chapter()
    for obj in pages[0]["text_objects"]:
        obj["translation"] = ""
    manifest = _manifest(tmp_path, pages)
    before = render_input_signature(manifest, 1)
    for obj in pages[0]["text_objects"]:
        obj["translation"] = "Lời thoại."
    assert render_input_signature(manifest, 1) == before, (
        "a sibling's translation is not this page's render input")


def test_sibling_edits_on_another_page_keep_the_render_current(tmp_path):
    manifest = _manifest(tmp_path, _two_page_chapter())
    before = render_input_signature(manifest, 1)
    sibling = manifest["pages"][0]["text_objects"][0]
    sibling["style"] = {"color": "#ffffff"}
    sibling["source_cap_px"] = 40  # a sibling's own inputs do not enter this page's signature
    assert render_input_signature(manifest, 1) == before


def _font_page() -> dict:
    return {"text_objects": [
        {"id": "auto", "translation": "A", "style": {"font": "dialogue.mac-dinh-3"},
         "font_selection_mode": "auto"},
        {"id": "ai", "translation": "B", "style": {"font": "emphasis.bangers"},
         "font_selection_mode": "ai", "font_ai_id": "emphasis.bangers"},
        {"id": "user", "translation": "C", "style": {"font": "dialogue.roboto"},
         "font_selection_mode": "user"},
        {"id": "plain", "translation": "D", "style": {"font": "default"}},
    ]}


def test_export_request_sends_only_explicit_user_fonts():
    req = _render_request_from_page("abcd1234", 0, _font_page())
    assert req.fonts["auto"] == "auto"
    assert "ai" not in req.fonts, "an AI pick must stay re-resolvable"
    assert req.fonts["user"] == "dialogue.roboto"
    assert "plain" not in req.fonts, "a stored default is not a user pin"


def test_chapter_rerender_does_not_turn_auto_or_ai_into_user_pins():
    page = _font_page()
    req = _render_request_from_page("abcd1234", 0, page)
    styles = _request_style_maps(req)
    _resolve_snapshot_fonts(page, styles)
    by_id = {obj["id"]: obj for obj in page["text_objects"]}
    resolved = styles["resolved_fonts"]
    assert by_id["ai"]["font_selection_mode"] == "ai"
    assert resolved["ai"]["font_selection_mode"] == "ai"
    assert resolved["ai"]["font_match"]["reason"] == "ai_selection"
    assert resolved["ai"]["font"] == "emphasis.bangers"
    assert by_id["auto"]["font_selection_mode"] == "auto"
    assert resolved["auto"]["font_match"]["reason"] == "auto_guide"
    assert by_id["user"]["font_selection_mode"] == "user"
    assert resolved["plain"]["font_selection_mode"] == "default"