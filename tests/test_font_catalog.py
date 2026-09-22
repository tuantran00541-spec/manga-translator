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
