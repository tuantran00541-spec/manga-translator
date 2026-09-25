"""Dependency-light acceptance checks for the bundled font catalog."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.render.font_catalog import load_font_catalog, list_font_records, resolve_font_id


def main() -> int:
    catalog = load_font_catalog()
    assert 60 <= len(catalog.records) <= 80
    assert len({record.id for record in catalog.records}) == len(catalog.records)
    assert {record.category for record in catalog.records} >= {
        "dialogue", "emphasis", "thought", "narration", "skill", "sfx", "horror", "romance"
    }
    for record in catalog.records:
        assert record.path.is_file(), record.path
        assert (catalog.root / record.license_file).is_file(), record.license_file
        assert resolve_font_id(record.id) == record.path
    assert resolve_font_id("default").is_file()
    assert all(not str(item["path"]).startswith("/") for item in list_font_records())

    print(f"font library sanity: {len(catalog.records)} fonts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
