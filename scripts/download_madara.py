from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
CHAPTER_URL_RE = re.compile(r"^(https?://[^/]+/(.+?))-chapter-(\d+)(?:-1)?/?$")
CHAPTER_URL_ONLY_RE = re.compile(r"^https?://[^/]+/.+-chapter-(\d+)(?:-1)?/?$")
LIST_ITEM_RE = re.compile(r'<li[^>]*\bdata-num="(\d+)"[^>]*>(.*?)</li>', re.S)
ANCHOR_HREF_RE = re.compile(r'<a\b[^>]*\bhref="([^"]+)"', re.S)
PAGE_HREF_RE = re.compile(r'href="(https?://[^"]+?-chapter-(\d+)(?:-1)?/?)"')
READER_RE = re.compile(r"ts_reader\.run\((\{.*?\})\)\s*;", re.S)
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "bmp"}
RETRY_DELAYS = (1.0, 2.0)


def log(message):
    print(message, file=sys.stderr, flush=True)


def fail(message):
    log("error: " + message)
    raise SystemExit(1)


def fetch(url, referer=None, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if referer:
        request.add_header("Referer", referer)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_with_retries(url, referer=None, timeout=30, attempts=3):
    last_error = None
    for attempt in range(attempts):
        try:
            return fetch(url, referer=referer, timeout=timeout)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
    raise last_error


def series_slug_from_url(series_url):
    path = urlparse(series_url).path.strip("/")
    return path.rsplit("/", 1)[-1] if path else ""


def prefer_url(existing, new):
    if existing is None:
        return new
    if new is None:
        return existing
    if new.endswith("-1/") and not existing.endswith("-1/"):
        return new
    return existing


def parse_series_page(html, series_slug=""):
    entries = {}
    for match in LIST_ITEM_RE.finditer(html):
        number = int(match.group(1))
        anchor = ANCHOR_HREF_RE.search(match.group(2))
        url = None
        if anchor and CHAPTER_URL_ONLY_RE.match(anchor.group(1)):
            url = anchor.group(1)
        entries[number] = prefer_url(entries.get(number), url)
    for match in PAGE_HREF_RE.finditer(html):
        url = match.group(1)
        number = int(match.group(2))
        slug, _ = chapter_slug_and_number(url)
        if series_slug and slug != series_slug:
            continue
        entries[number] = prefer_url(entries.get(number), url)
    return sorted(entries.items())


def discover(series_url, count, chapter_start=None, chapter_end=None, order="newest"):
    try:
        html = fetch_with_retries(series_url, timeout=30).decode("utf-8", errors="replace")
    except Exception as error:
        fail("could not fetch series page %s: %s" % (series_url, error))
    entries = parse_series_page(html, series_slug_from_url(series_url))
    if not entries:
        fail("could not parse any chapter from series page %s" % series_url)
    by_number = dict(entries)
    locked = [number for number, url in entries if url is None]
    if chapter_start is not None:
        in_range = [(number, by_number[number]) for number in range(chapter_start, chapter_end + 1)
                    if number in by_number and by_number[number] is not None]
        missing = [number for number in range(chapter_start, chapter_end + 1)
                   if number not in by_number or by_number[number] is None]
        chosen = [{"chapter": str(number), "url": url} for number, url in in_range]
        if order == "newest":
            chosen.reverse()
        log("series page lists %d chapters, %d coin-locked, range ch%d-ch%d selected %d in %s order, skipped locked/empty %s"
            % (len(entries), len(locked), chapter_start, chapter_end, len(chosen), order, missing))
        if not chosen:
            fail("no unlocked chapter in ch%d-ch%d on %s" % (chapter_start, chapter_end, series_url))
        print(json.dumps({"chapter": chosen}, ensure_ascii=False, separators=(",", ":")))
        return 0
    ordered = sorted(entries, key=lambda item: item[0], reverse=True)
    newest = [number for number, _ in ordered[:2]]
    pool = ordered[2:]
    chosen = []
    for number, url in pool:
        if url is None:
            continue
        chosen.append({"chapter": str(number), "url": url})
        if len(chosen) >= count:
            break
    if order == "oldest":
        chosen.reverse()
    log("series page lists %d chapters, %d coin-locked, skipped newest %s, selected %d in %s order"
        % (len(entries), len(locked), newest, len(chosen), order))
    if not chosen:
        fail("no unlocked chapter left after skipping the 2 newest on %s" % series_url)
    print(json.dumps({"chapter": chosen}, ensure_ascii=False, separators=(",", ":")))
    return 0


def parse_reader_images(html):
    match = READER_RE.search(html)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    sources = data.get("sources") or []
    for source in sources:
        images = source.get("images") if isinstance(source, dict) else None
        if images:
            return [str(image) for image in images]
    return None


def image_extension(url):
    segment = urlparse(url).path.rsplit("/", 1)[-1]
    suffix = segment.rsplit(".", 1)[-1].lower() if "." in segment else ""
    if suffix in ALLOWED_EXTS:
        return suffix
    return "webp"


def chapter_slug_and_number(chapter_url):
    match = CHAPTER_URL_RE.match(chapter_url)
    if not match:
        return None, None
    return match.group(2), int(match.group(3))


def download_chapter(chapter_url, chapter, out_dir):
    slug, number = chapter_slug_and_number(chapter_url)
    if slug is None:
        fail("chapter URL does not match <series-slug>-chapter-<N>: %s" % chapter_url)
    if number != chapter:
        fail("chapter URL number %d does not match --chapter %d" % (number, chapter))
    try:
        html = fetch_with_retries(chapter_url, timeout=30).decode("utf-8", errors="replace")
    except Exception as error:
        fail("could not fetch chapter page %s: %s" % (chapter_url, error))
    images = parse_reader_images(html)
    if not images:
        fail("no reader images found on chapter page %s" % chapter_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index, image_url in enumerate(images, start=1):
        name = "%03d.%s" % (index, image_extension(image_url))
        try:
            data = fetch_with_retries(image_url, referer=chapter_url, timeout=60)
        except Exception as error:
            fail("could not download image %d of %d (%s): %s" % (index, len(images), image_url, error))
        if not data:
            fail("downloaded empty image %d of %d (%s)" % (index, len(images), image_url))
        (out_dir / name).write_bytes(data)
        files.append(name)
        log("downloaded %s (%d bytes)" % (name, len(data)))
    zip_name = "madara-ch%d.zip" % chapter
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(out_dir / name, arcname=name)
    manifest = {
        "series_slug": slug,
        "chapter": chapter,
        "chapter_url": chapter_url,
        "image_count": len(images),
        "image_urls": images,
        "files": files,
        "zip": zip_name,
    }
    (out_dir / "download.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"chapter": chapter, "images": len(images), "zip": zip_name}, ensure_ascii=False))
    return 0


def download_batch(urls_file, out_dir):
    urls = []
    seen = set()
    for line in Path(urls_file).read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    if not urls:
        fail("no chapter URLs in %s" % urls_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    failed = []
    for url in urls:
        slug, number = chapter_slug_and_number(url)
        if slug is None:
            failed.append({"chapter": None, "url": url, "error": "bad chapter URL"})
            log("skipped bad chapter URL: %s" % url)
            continue
        chapter_dir = out_dir / ("ch%d" % number)
        try:
            download_chapter(url, number, chapter_dir)
        except SystemExit:
            failed.append({"chapter": number, "url": url, "error": "download failed"})
            log("download failed for ch%d, continuing" % number)
            continue
        manifest = json.loads((chapter_dir / "download.json").read_text(encoding="utf-8"))
        for name in manifest["files"]:
            (chapter_dir / name).unlink(missing_ok=True)
        downloaded.append(number)
        log("ch%d done (%d/%d)" % (number, len(downloaded), len(urls)))
    log("batch done: %d downloaded, %d failed" % (len(downloaded), len(failed)))
    print(json.dumps({"downloaded": downloaded, "failed": failed, "out": str(out_dir)}, ensure_ascii=False))
    return 0 if downloaded else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-url")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--chapter-start", type=int, default=None)
    parser.add_argument("--chapter-end", type=int, default=None)
    parser.add_argument("--order", choices=("newest", "oldest"), default="newest")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--chapter-url")
    parser.add_argument("--chapter", type=int)
    parser.add_argument("--urls-file", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.discover:
        if not args.series_url:
            fail("--discover needs --series-url")
        if args.count < 1:
            fail("--count must be at least 1")
        if (args.chapter_start is None) != (args.chapter_end is None):
            fail("--chapter-start and --chapter-end must be given together")
        if args.chapter_start is not None:
            if args.chapter_start < 1 or args.chapter_end < args.chapter_start:
                fail("--chapter-start/--chapter-end must satisfy 1 <= start <= end")
            return discover(args.series_url, args.count, args.chapter_start, args.chapter_end, args.order)
        return discover(args.series_url, args.count, order=args.order)
    if args.urls_file is not None:
        if args.out is None:
            fail("batch mode needs --urls-file and --out")
        return download_batch(args.urls_file, args.out)
    if not args.chapter_url or args.chapter is None or args.out is None:
        fail("download mode needs --chapter-url, --chapter and --out")
    if args.chapter < 1:
        fail("--chapter must be at least 1")
    return download_chapter(args.chapter_url, args.chapter, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
