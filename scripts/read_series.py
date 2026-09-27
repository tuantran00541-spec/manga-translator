"""One-off: fetch a run of chapters of one random series and tile them for reading."""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

OUT = Path("audit-results/reading")
WIDTH, COLUMN_H, COLUMNS = 640, 2600, 4


def chapter_number(url: str, text: str) -> float:
    match = re.search(r"(?:chuong|chap|chapter)[-\s]*(\d+(?:[.-]\d+)?)", f"{url} {text}", re.I)
    return float(match.group(1).replace("-", ".")) if match else -1.0


def links(page, base: str) -> list[tuple[str, str]]:
    found = page.eval_on_selector_all("a[href]", "els => els.map(e => [e.href, (e.innerText || e.title || '').trim()])")
    host = urlparse(base).hostname
    return [(urljoin(base, h), t) for h, t in found if urlparse(urljoin(base, h)).hostname == host]


def tile(images: list[np.ndarray]) -> list[np.ndarray]:
    strip = np.vstack([cv2.resize(im, (WIDTH, max(1, round(im.shape[0] * WIDTH / im.shape[1])))) for im in images])
    columns = [strip[y:y + COLUMN_H] for y in range(0, strip.shape[0], COLUMN_H)]
    columns = [np.vstack([c, np.full((COLUMN_H - c.shape[0], WIDTH, 3), 255, np.uint8)]) for c in columns]
    gap = np.full((COLUMN_H, 12, 3), 128, np.uint8)
    tiles = []
    for i in range(0, len(columns), COLUMNS):
        row = []
        for column in columns[i:i + COLUMNS]:
            row += [column, gap]
        tiles.append(np.hstack(row[:-1]))
    return tiles


def main() -> int:
    conf = dict(line.split(":", 1) for line in Path(".github/reading.txt").read_text().splitlines() if ":" in line)
    home = conf.get("site", "").strip()
    series = conf.get("series", "").strip()
    count = int(conf.get("chapters", "15"))
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"site": home}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0 Safari/537.36"), viewport={"width": 1280, "height": 2000})
        if not series:
            page.goto(home, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
            (OUT / "home.html").write_text(page.content(), encoding="utf-8")
            candidates = sorted({h for h, _ in links(page, home)
                                 if re.search(r"/truyen-tranh/[^/]+/?$", urlparse(h).path)})
            log["series_found"] = len(candidates)
            if not candidates:
                print("no series links found")
                (OUT / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
                return 1
            series = random.Random(os.environ.get("GITHUB_RUN_ID", "0")).choice(candidates)
        log["series"] = series
        page.goto(series, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        (OUT / "series.html").write_text(page.content(), encoding="utf-8")
        log["title"] = page.title()
        chapters = {}
        for href, text in links(page, series):
            number = chapter_number(href, text)
            if number >= 0 and urlparse(series).path.rstrip("/") in href:
                chapters.setdefault(number, href)
        order = sorted(chapters)[:count]
        log["chapters_found"] = len(chapters)
        log["read"] = []
        for index, number in enumerate(order, start=1):
            url = chapters[number]
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            for _ in range(40):
                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(250)
            srcs = page.eval_on_selector_all(
                "img", "els => els.map(e => [e.getAttribute('data-src') || e.getAttribute('data-original') || e.src,"
                " e.naturalHeight || 0, e.naturalWidth || 0])")
            srcs = [urljoin(url, s) for s, h, w in srcs if s and not s.startswith("data:") and (h == 0 or h > 300)
                    and not re.search(r"logo|avatar|icon|banner|ads?/", s, re.I)]
            images = []
            for src in dict.fromkeys(srcs):
                try:
                    body = page.request.get(src, headers={"Referer": url}, timeout=30000).body()
                except Exception as exc:  # noqa: BLE001
                    print("skip", src, exc)
                    continue
                image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if image is not None and image.shape[1] >= 300 and image.shape[0] >= 200:
                    images.append(image)
            entry = {"number": number, "url": url, "images": len(images)}
            if images:
                for t, sheet in enumerate(tile(images), start=1):
                    name = f"ch{index:02d}-{t}.jpg"
                    cv2.imwrite(str(OUT / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 72])
                entry["sheets"] = t
            log["read"].append(entry)
            print(entry)
        browser.close()
    (OUT / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
