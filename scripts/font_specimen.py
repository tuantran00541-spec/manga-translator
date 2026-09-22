"""Render a compact specimen sheet for the bundled comic font catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.render.font_catalog import load_font_catalog


def render_specimen(output: Path, sample_text: str = "Xin chào · Manga") -> Path:
    catalog = load_font_catalog()
    columns = 2
    row_height = 92
    width = 1100
    height = 80 + ((len(catalog.records) + columns - 1) // columns) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(str(catalog.root / "default.ttf"), 28)
    label_font = ImageFont.truetype(str(catalog.root / "default.ttf"), 16)
    draw.text((24, 20), "Comic font library specimen · 70 families", fill="black", font=title_font)
    for index, record in enumerate(catalog.records):
        col, row = index % columns, index // columns
        x = 24 + col * 535
        y = 72 + row * row_height
        draw.text((x, y), f"{record.category} · {record.name}", fill="#555", font=label_font)
        try:
            font = ImageFont.truetype(str(record.path), 30)
            draw.text((x, y + 24), sample_text, fill="black", font=font)
        except OSError:
            draw.text((x, y + 24), "(font unavailable)", fill="#a00", font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/output/font-specimen.png"))
    parser.add_argument("--text", default="Xin chào · Manga")
    args = parser.parse_args()
    print(render_specimen(args.output, args.text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
