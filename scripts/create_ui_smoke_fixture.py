"""Create a tiny, model-free chapter for live browser smoke tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CHAPTER_ID = "f00d0001"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / CHAPTER_ID
PROCESSED_DIR = ROOT / "data" / "processed" / CHAPTER_ID


def _page(index: int) -> Image.Image:
    image = Image.new("RGB", (1200, 1600), (231, 232, 236))
    draw = ImageDraw.Draw(image)
    heading_font = ImageFont.load_default(size=42)
    dialogue_font = ImageFont.load_default(size=34)
    draw.rectangle((0, 0, 1200, 190), fill=(42, 50, 69))
    draw.rectangle((80, 260, 1120, 1510), outline=(63, 70, 86), width=10)
    draw.ellipse((155, 350, 850, 780), fill=(252, 252, 250), outline=(33, 38, 48), width=8)
    draw.ellipse((430, 940, 1080, 1320), fill=(252, 252, 250), outline=(33, 38, 48), width=8)
    draw.polygon(((770, 740), (865, 850), (715, 790)), fill=(252, 252, 250), outline=(33, 38, 48))
    draw.text((100, 70), f"UI smoke fixture — page {index + 1}", fill=(245, 247, 251), font=heading_font)
    draw.text((280, 505), "Sample dialogue", fill=(25, 28, 35), font=dialogue_font)
    draw.text((560, 1090), "Second bubble", fill=(25, 28, 35), font=dialogue_font)
    return image


def main() -> None:
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    RAW_DIR.mkdir(parents=True)
    PROCESSED_DIR.mkdir(parents=True)

    pages = []
    for index in range(3):
        image = _page(index)
        original = f"page_{index:03d}.png"
        clean = f"clean_{index:03d}.png"
        image.save(RAW_DIR / original)
        image.save(PROCESSED_DIR / clean)
        pages.append(
            {
                "source_page": index,
                "slice_index": 0,
                "width": image.width,
                "height": image.height,
                # Production manifests retain managed absolute paths; the API
                # turns them into stable /api/image URLs for the browser.  Keep
                # the fixture on that same contract instead of using filenames
                # that only make sense relative to this script's working dir.
                "original": str((RAW_DIR / original).resolve()),
                "clean": str((PROCESSED_DIR / clean).resolve()),
                "boxes": [],
                "text_objects": [],
                "excluded_regions": [],
            }
        )

    pages[0]["text_objects"] = [
        {
            "id": "bubble-a",
            "shape": "ellipse",
            "region": {"x1": 155, "y1": 350, "x2": 850, "y2": 780},
            "ocr_text": "Sample dialogue",
            "translation": "Đoạn thoại mẫu",
            "style": {
                "font": "default",
                "fontSize": "42",
                "bold": True,
                "color": "#ffffff",
                "strokeWidth": "3",
                "strokeColor": "#000000",
                "bgColor": "transparent",
                "cornerRadius": "0",
                "horizontalAlign": "center",
                "verticalAlign": "middle",
            },
        }
    ]
    pages[1]["text_objects"] = [
        {
            "id": "bubble-b",
            "shape": "rectangle",
            "region": {"x1": 430, "y1": 940, "x2": 1080, "y2": 1320},
            "ocr_text": "Second bubble",
            "translation": "Bong bóng thứ hai",
            "style": {},
        }
    ]

    manifest = {
        "chapter_id": CHAPTER_ID,
        "chapter_name": "UI smoke fixture",
        "source_url": "ui-smoke://fixture",
        "source_lang": "ja",
        "workflow": {"stage": "editor", "page_index": 0},
        "pages": pages,
    }
    (PROCESSED_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    print(CHAPTER_ID)


if __name__ == "__main__":
    main()
