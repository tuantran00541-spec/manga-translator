"""Dependency-light acceptance checks for the bundled font catalog."""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.render.font_catalog import load_font_catalog, list_font_records, resolve_font_id
from app.render.font_matcher import match_fonts


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

    image = Image.new("RGB", (320, 100), "white")
    ImageDraw.Draw(image).text((20, 28), "font smoke test", fill="black")
    matches = match_fonts(image, (0, 0, 320, 100), "font smoke test", top_k=3)
    assert len(matches) == 3
    print(f"font library sanity: {len(catalog.records)} fonts, matcher returned {len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
