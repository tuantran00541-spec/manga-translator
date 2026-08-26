from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image
from playwright.sync_api import sync_playwright


def unwrap_rsc(value):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], int):
        return unwrap_rsc(value[1])
    if isinstance(value, list):
        return [unwrap_rsc(v) for v in value]
    if isinstance(value, dict):
        return {k: unwrap_rsc(v) for k, v in value.items()}
    return value


def find_pages(value):
    if isinstance(value, dict):
        pages = value.get("pages")
        if isinstance(pages, list) and pages and all(isinstance(p, dict) for p in pages):
            if any(p.get("url") for p in pages):
                return pages
        for child in value.values():
            found = find_pages(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_pages(child)
            if found:
                return found
    return None


def extract_pages(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(attrs={"props": True}):
        raw = tag.get("props")
        if not raw or "pages" not in raw:
            continue
        try:
            obj = unwrap_rsc(json.loads(unescape(raw)))
        except Exception:
            continue
        pages = find_pages(obj)
        if pages:
            return pages

    decoded = unescape(html)
    urls = []
    seen = set()
    for url in re.findall(r'https://cdn\.asurascans\.com/asura-images/chapters/[^"&\\s<]+', decoded):
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return [{"url": url} for url in urls]


def unscramble(image: Image.Image, tiles, cols: int, rows: int) -> Image.Image:
    tile_w = image.width // cols
    tile_h = image.height // rows
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("invalid tile geometry")
    out = Image.new("RGB", (tile_w * cols, tile_h * rows), "white")
    src = image.convert("RGB")
    for source_index, destination_index in enumerate(tiles):
        src_col = source_index % cols
        src_row = source_index // cols
        dst_col = int(destination_index) % cols
        dst_row = int(destination_index) // cols
        box = (
            src_col * tile_w,
            src_row * tile_h,
            (src_col + 1) * tile_w,
            (src_row + 1) * tile_h,
        )
        out.paste(src.crop(box), (dst_col * tile_w, dst_row * tile_h))
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: fetch_asura_chapter.py URL OUTPUT_DIR", file=sys.stderr)
        return 2

    chapter_url = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=ua, viewport={"width": 1280, "height": 900})
        page = context.new_page()
        response = page.goto(chapter_url, wait_until="domcontentloaded", timeout=120_000)
        if response is None:
            raise RuntimeError("chapter navigation returned no response")
        print("chapter_status", response.status)
        page.wait_for_timeout(8_000)
        html = page.content()
        (out_dir / "chapter.html").write_text(html, encoding="utf-8")

        pages = extract_pages(html)
        if not pages:
            raise RuntimeError("no chapter page assets discovered")

        manifest = {
            "chapter_url": chapter_url,
            "page_count": len(pages),
            "pages": [],
        }

        for index, page_dto in enumerate(pages):
            image_url = str(page_dto.get("url") or "")
            if not image_url:
                raise RuntimeError(f"page {index}: missing image url")
            img_response = context.request.get(
                image_url,
                headers={"Referer": chapter_url, "User-Agent": ua},
                timeout=120_000,
            )
            if not img_response.ok:
                raise RuntimeError(f"page {index}: image HTTP {img_response.status}")
            raw = img_response.body()
            image = Image.open(io.BytesIO(raw))
            tiles = page_dto.get("tiles")
            cols = int(page_dto.get("tileCols") or 4)
            rows = int(page_dto.get("tileRows") or 5)
            scrambled = bool(isinstance(tiles, list) and tiles)
            if scrambled:
                final_image = unscramble(image, tiles, cols, rows)
            else:
                final_image = image.convert("RGB")

            output_path = out_dir / f"{index:03d}.png"
            final_image.save(output_path, format="PNG", optimize=False)
            payload = output_path.read_bytes()
            manifest["pages"].append(
                {
                    "index": index,
                    "source_url": image_url,
                    "width": final_image.width,
                    "height": final_image.height,
                    "scrambled": scrambled,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
            print(
                f"page {index + 1}/{len(pages)} {final_image.width}x{final_image.height} "
                f"scrambled={scrambled} bytes={len(payload)}"
            )
            if final_image is not image:
                final_image.close()
            image.close()

        browser.close()

    (out_dir / "fetch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FETCH_OK", len(manifest["pages"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
