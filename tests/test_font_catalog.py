import json
from pathlib import Path

import pytest


def test_catalog_contains_comic_categories_and_real_font_files():
    from app.render.font_catalog import load_font_catalog

    catalog = load_font_catalog()
    assert 60 <= len(catalog.records) <= 80
    categories = {record.category for record in catalog.records}
    assert {"dialogue", "emphasis", "thought", "narration", "skill", "sfx", "horror", "romance"} <= categories
    for record in catalog.records:
        assert record.path.is_file(), record.path
        assert record.license_file
        assert (catalog.root / record.license_file).is_file()


def test_font_ids_are_stable_and_legacy_default_resolves():
    from app.render.font_catalog import load_font_catalog, resolve_font_id

    catalog = load_font_catalog()
    ids = [record.id for record in catalog.records]
    assert len(ids) == len(set(ids))
    assert resolve_font_id("default").is_file()
    assert not str(catalog.records[0].as_dict(catalog.root)["path"]).startswith("/")


def test_unknown_font_id_is_rejected():
    from app.render.font_catalog import FontNotFoundError, resolve_font_id

    with pytest.raises(FontNotFoundError):
        resolve_font_id("does-not-exist")


def test_vietnamese_flag_matches_each_fonts_character_map():
    """A font marked Vietnamese must map every Vietnamese letter, or the AI may pick it and print boxes."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_font_catalog import validate

    from app.render.font_catalog import load_font_catalog

    root = load_font_catalog().root
    errors = [error for error in validate(root, root / "font_catalog.json") if "vietnamese" in error]
    assert errors == []


def test_text_the_font_cannot_draw_falls_back_to_the_default_font():
    from PIL import Image

    from app.config import DEFAULT_FONT
    from app.render.text_renderer import font_draws_text, get_font_path, render_text_in_box

    display = get_font_path("thought.comic-neue")
    text = "Không thể kết thúc được"
    assert not font_draws_text(display, text)
    assert font_draws_text(display, "Game over")
    assert font_draws_text(DEFAULT_FONT, text)

    box = (0, 0, 320, 120)
    drawn = render_text_in_box(Image.new("RGB", (320, 120), "white"), text, box, font_path=display)
    expected = render_text_in_box(Image.new("RGB", (320, 120), "white"), text, box, font_path=DEFAULT_FONT)
    assert drawn.tobytes() == expected.tobytes()
