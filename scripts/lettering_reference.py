"""Collect a few recently published translated chapters as lettering references.

The chapters come from the MangaDex API: different series and different
translation groups, so their typesetting (fonts per text type, size, bubble
fill, system windows) can be compared with ours. Pages are scaled to phone
width and stacked three columns per image, the same view used to read our own
rendered chapters.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.mangadex.org"
HEADERS = {"User-Agent": "manga-translator-lettering-reference/1.0"}
PHONE_WIDTH = 400
COLUMN_HEIGHT = 1900


def _get(url: str, **params) -> dict:
    for attempt in range(4):
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"rate limited: {url}")


def _title(manga: dict) -> str:
    titles = manga.get("attributes", {}).get("title") or {}
    return titles.get("en") or next(iter(titles.values()), "untitled")


def _pick(language: str, count: int, long_strip: bool) -> list[dict]:
    params = {
        "translatedLanguage[]": [language], "limit": 100, "order[readableAt]": "desc",
        "includes[]": ["manga", "scanlation_group"], "contentRating[]": ["safe", "suggestive"],
        "includeExternalUrl": 0,
    }
    chapters = _get(f"{API}/chapter", **params)["data"]
    picked, groups, series = [], set(), set()
    for chapter in chapters:
        rel = {r["type"]: r for r in chapter.get("relationships", [])}
        manga, group = rel.get("manga"), rel.get("scanlation_group")
        if not manga or not group or (chapter["attributes"].get("pages") or 0) < 8:
            continue
        tags = {t["attributes"]["name"].get("en") for t in manga.get("attributes", {}).get("tags", [])}
        if long_strip != ("Long Strip" in tags):
            continue
        if group["id"] in groups or manga["id"] in series:
            continue
        groups.add(group["id"])
        series.add(manga["id"])
        picked.append({
            "chapter_id": chapter["id"], "language": language, "manga": _title(manga),
            "chapter": chapter["attributes"].get("chapter"),
            "group": (group.get("attributes") or {}).get("name"),
            "url": f"https://mangadex.org/chapter/{chapter['id']}", "long_strip": long_strip,
        })
        if len(picked) >= count:
            break
    return picked


def _pick_popular(language: str, count: int, tags: list[str]) -> list[dict]:
    """Latest chapter of the most followed long-strip series with these tags, one per group."""
    by_name = {t["attributes"]["name"]["en"].lower(): t["id"] for t in _get(f"{API}/manga/tag")["data"]}
    tag_ids = [by_name[name.lower()] for name in tags if name.lower() in by_name]
    mangas = _get(f"{API}/manga", **{
        "availableTranslatedLanguage[]": [language], "includedTags[]": tag_ids, "order[followedCount]": "desc",
        "limit": 40, "contentRating[]": ["safe", "suggestive"],
    })["data"]
    picked, groups = [], set()
    for manga in mangas:
        chapters = _get(f"{API}/chapter", **{
            "manga": manga["id"], "translatedLanguage[]": [language], "order[chapter]": "desc", "limit": 20,
            "includes[]": ["scanlation_group"], "includeExternalUrl": 0,
        })["data"]
        for chapter in chapters:
            group = next((r for r in chapter.get("relationships", []) if r["type"] == "scanlation_group"), None)
            if not group or group["id"] in groups or (chapter["attributes"].get("pages") or 0) < 8:
                continue
            groups.add(group["id"])
            picked.append({
                "chapter_id": chapter["id"], "language": language, "manga": _title(manga),
                "chapter": chapter["attributes"].get("chapter"),
                "group": (group.get("attributes") or {}).get("name"),
                "url": f"https://mangadex.org/chapter/{chapter['id']}", "long_strip": True, "tags": tags,
            })
            break
        if len(picked) >= count:
            break
    return picked


def _pages(chapter_id: str, max_pages: int) -> list[np.ndarray]:
    home = _get(f"{API}/at-home/server/{chapter_id}")
    base, data = home["baseUrl"], home["chapter"]
    images = []
    for name in data["dataSaver"][:max_pages]:
        response = requests.get(f"{base}/data-saver/{data['hash']}/{name}", headers=HEADERS, timeout=60)
        if not response.ok:
            continue
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    return images


def _phone_view(images: list[np.ndarray], out: Path, max_images: int) -> int:
    strip = np.concatenate([
        cv2.resize(image, (PHONE_WIDTH, max(1, round(image.shape[0] * PHONE_WIDTH / image.shape[1]))),
                   interpolation=cv2.INTER_AREA)
        for image in images
    ])
    columns = [strip[y:y + COLUMN_HEIGHT] for y in range(0, strip.shape[0], COLUMN_HEIGHT)]
    columns = [np.pad(c, ((0, COLUMN_HEIGHT - c.shape[0]), (0, 0), (0, 0)), constant_values=255) for c in columns]
    gap = np.full((COLUMN_HEIGHT, 12, 3), 128, np.uint8)
    written = 0
    for start in range(0, len(columns), 3):
        if written >= max_images:
            break
        group = columns[start:start + 3]
        group += [np.full((COLUMN_HEIGHT, PHONE_WIDTH, 3), 255, np.uint8)] * (3 - len(group))
        sheet = np.concatenate([group[0], gap, group[1], gap, group[2]], axis=1)
        written += 1
        cv2.imwrite(str(out / f"{written:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vi", type=int, default=5, help="Vietnamese chapters")
    parser.add_argument("--en", type=int, default=2, help="English chapters")
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--max-images", type=int, default=4, help="phone-view sheets per chapter")
    parser.add_argument("--popular-tags", default="",
                        help="comma-separated MangaDex tags; picks popular long-strip series instead of recent chapters")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit(f"{args.out} must be inside {ROOT}")
    out.mkdir(parents=True, exist_ok=True)

    if args.popular_tags:
        tags = ["Long Strip"] + [t.strip() for t in args.popular_tags.split(",") if t.strip()]
        wanted = _pick_popular("vi", args.vi, tags) + (_pick_popular("en", args.en, tags) if args.en else [])
    else:
        wanted = (_pick("vi", args.vi - args.vi // 2, True) + _pick("vi", args.vi // 2, False)
                  + _pick("en", args.en, True))
    report = {"source": "MangaDex API", "chapters": []}
    for index, chapter in enumerate(wanted, start=1):
        folder = out / f"{index:02d}-{chapter['language']}"
        folder.mkdir(exist_ok=True)
        try:
            chapter["sheets"] = _phone_view(_pages(chapter["chapter_id"], args.max_pages), folder, args.max_images)
        except Exception as exc:
            chapter["error"] = f"{type(exc).__name__}: {exc}"[:300]
        report["chapters"].append(chapter)
        print(json.dumps(chapter, ensure_ascii=False))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
