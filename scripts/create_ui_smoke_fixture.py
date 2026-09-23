"""Create a tiny, model-free chapter for live browser smoke tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CHAPTER_ID = "f00d0001"
LONG_CHAPTER_ID = "f00d0042"
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
    draw.text((100, 70), f"UI smoke fixture - page {index + 1}", fill=(245, 247, 251), font=heading_font)
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
        if index == 2:
            image = image.crop((0, 0, 1000, 1600))
        original = f"page_{index:03d}.png"
        clean = f"clean_{index:03d}.png"
        image.save(RAW_DIR / original)
        image.save(PROCESSED_DIR / clean)
        pages.append(
            {
                "source_page": index,
                **({"width": image.width} if index == 2 else {}),
                "slice_index": 0,
                "stitch_core": {
                    "source_y1": 0,
                    "source_y2": image.height,
                    "core_y1": 0,
                    "core_y2": image.height,
                    "core_source_y1": 0,
                    "core_source_y2": image.height,
                    "unsafe_before": False,
                    "unsafe_after": False,
                    "source_height": image.height,
                },
                # Deliberately omit width/height here. Production manifests made
                # before slice-dimension persistence relied on stitch metadata,
                # and the stitched Review must render those chapters immediately.
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
    long_raw = ROOT / "data" / "raw" / LONG_CHAPTER_ID
    long_processed = ROOT / "data" / "processed" / LONG_CHAPTER_ID
    shutil.rmtree(long_raw, ignore_errors=True)
    shutil.rmtree(long_processed, ignore_errors=True)
    long_raw.mkdir(parents=True)
    long_processed.mkdir(parents=True)
    long_image = Image.new("RGB", (1200, 42000), "white")
    long_draw = ImageDraw.Draw(long_image)
    long_draw.rectangle((0, 0, 1199, 119), fill=(220, 35, 35))
    long_draw.rectangle((0, 41880, 1199, 41999), fill=(35, 65, 220))
    raw_path = long_raw / "page_000.png"
    clean_path = long_processed / "clean_000.png"
    long_image.save(raw_path)
    long_image.save(clean_path)
    long_manifest = {
        "chapter_id": LONG_CHAPTER_ID,
        "chapter_name": "Long image UI smoke fixture",
        "source_url": "ui-smoke://long-image",
        "source_lang": "ja",
        "workflow": {"stage": "review", "page_index": 0},
        "pages": [{
            "source_page": 0,
            "slice_index": 0,
            "width": 1200,
            "height": 42000,
            "stitch_core": {
                "source_y1": 0, "source_y2": 42000,
                "core_y1": 0, "core_y2": 42000,
                "core_source_y1": 0, "core_source_y2": 42000,
                "source_height": 42000,
            },
            "original": str(raw_path.resolve()),
            "clean": str(clean_path.resolve()),
            "boxes": [],
            "text_objects": [],
            "excluded_regions": [],
        }],
    }
    (long_processed / "manifest.json").write_text(
        json.dumps(long_manifest, ensure_ascii=False), encoding="utf-8"
    )
    print(CHAPTER_ID)
    print(LONG_CHAPTER_ID)


if __name__ == "__main__":
    main()
