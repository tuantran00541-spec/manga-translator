from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
CHAPTER_URL_RE = re.compile(r"^https?://[^/]+/comics/(.+?)/chapter/(\d+)/?$")
ASURA_CHAPTER_ASSET_RE = re.compile(r"^https?://cdn\.asurascans\.com/asura-images/chapters/", re.IGNORECASE)
PAGE_IMG_RE = re.compile(r'<img[^>]*\balt="Page\s+(\d+)"[^>]*>', re.IGNORECASE)
SRC_ATTRS = ("data-srcset", "srcset", "data-src", "data-original", "data-lazy", "src")
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp", "bmp"}
RETRY_DELAYS = (1.0, 2.0)
MAGIC = {
    b"RIFF": "webp",
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
}


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
        except http.client.IncompleteRead as error:
            last_error = error
        except http.client.HTTPException as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
    raise last_error


def image_extension(url):
    segment = urlparse(url).path.rsplit("/", 1)[-1]
    suffix = segment.rsplit(".", 1)[-1].lower() if "." in segment else ""
    if suffix in ALLOWED_EXTS:
        return suffix
    return "webp"


def sniff_extension(data, fallback):
    for magic, ext in MAGIC.items():
        if data.startswith(magic):
            return ext
    return fallback


def chapter_slug_and_number(chapter_url):
    match = CHAPTER_URL_RE.match(chapter_url)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def parse_page_images(html):
    by_page = {}
    for tag in re.finditer(r"<img[^>]*>", html, re.IGNORECASE):
        text = tag.group(0)
        alt = re.search(r'\balt="Page\s+(\d+)"', text, re.IGNORECASE)
        if not alt:
            continue
        page = int(alt.group(1))
        if page in by_page:
            continue
        for attr in SRC_ATTRS:
            found = re.search(r'\b%s="([^"]+)"' % re.escape(attr), text, re.IGNORECASE)
            if not found:
                continue
            candidate = found.group(1).split(",")[0].strip().split(" ")[0]
            if ASURA_CHAPTER_ASSET_RE.match(candidate):
                by_page[page] = candidate
                break
    if not by_page:
        return []
    ordered = []
    page = 1
    while page in by_page:
        ordered.append(by_page[page])
        page += 1
    if page <= max(by_page):
        fail("chapter page list has a gap at page %d (pages found up to %d)" % (page, max(by_page)))
    return ordered


def load_urls_file(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        fail("could not read urls file %s: %s" % (path, error))
    if isinstance(data, dict):
        images = data.get("images")
    elif isinstance(data, list):
        images = data
    else:
        images = None
    if not images or not isinstance(images, list):
        fail("urls file %s has no image list" % path)
    return [str(url) for url in images]


def fetch_image_curl(url, referer, out_path, timeout=60):
    curl = shutil.which("curl")
    if not curl:
        return False
    command = [
        curl, "-f", "-L", "--retry", "3", "--retry-delay", "1",
        "--max-time", str(timeout), "-A", USER_AGENT, "-e", referer,
        "-C", "-", "-o", str(out_path), url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 30)
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def fetch_image(url, referer=None, timeout=60, attempts=5):
    data = bytearray()
    expected = None
    for attempt in range(attempts):
        headers = {"User-Agent": USER_AGENT}
        if referer:
            headers["Referer"] = referer
        if data:
            headers["Range"] = "bytes=%d-" % len(data)
        request = urllib.request.Request(url, headers=headers)
        clean_eof = False
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if data and response.status != 206:
                    data = bytearray()
                    expected = None
                if expected is None:
                    raw = response.headers.get("Content-Length")
                    if raw and raw.isdigit():
                        total = int(raw)
                        expected = total + len(data) if response.status == 206 else total
                while True:
                    try:
                        chunk = response.read(65536)
                    except http.client.IncompleteRead as error:
                        data += error.partial
                        break
                    if not chunk:
                        clean_eof = True
                        break
                    data += chunk
        except urllib.error.HTTPError as error:
            if error.code == 416 and data:
                return bytes(data)
            if error.code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                continue
            raise
        except (urllib.error.URLError, http.client.HTTPException) as error:
            if attempt + 1 < attempts:
                time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                continue
            raise
        if expected is not None and len(data) >= expected:
            return bytes(data)
        if expected is None and clean_eof and data:
            return bytes(data)
        if attempt + 1 < attempts:
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
    if expected is not None and len(data) < expected:
        raise http.client.IncompleteRead(bytes(data), expected - len(data))
    if not data:
        raise urllib.error.URLError("empty response")
    raise http.client.IncompleteRead(bytes(data), 0)


def download_chapter(chapter_url, chapter, out_dir, urls_file=None):
    slug, number = chapter_slug_and_number(chapter_url)
    if slug is None:
        fail("chapter URL does not match /comics/<slug>/chapter/<N>: %s" % chapter_url)
    if number != chapter:
        fail("chapter URL number %d does not match --chapter %d" % (number, chapter))
    if urls_file:
        images = load_urls_file(urls_file)
        log("using %d image URLs from %s" % (len(images), urls_file))
    else:
        try:
            html = fetch_with_retries(chapter_url, timeout=30).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            if error.code == 403:
                fail("chapter page returned 403 (Cloudflare) — harvest the image URLs with a real browser and pass --urls-file")
            fail("could not fetch chapter page %s: %s" % (chapter_url, error))
        except Exception as error:
            fail("could not fetch chapter page %s: %s" % (chapter_url, error))
        images = parse_page_images(html)
        if not images:
            fail("no reader images found on chapter page %s" % chapter_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index, image_url in enumerate(images, start=1):
        ext = image_extension(image_url)
        name = "%03d.%s" % (index, ext)
        out_path = out_dir / name
        data = None
        if fetch_image_curl(image_url, chapter_url, out_path):
            data = out_path.read_bytes()
        if data is None:
            try:
                data = fetch_image(image_url, referer=chapter_url, timeout=60)
            except Exception as error:
                fail("could not download image %d of %d (%s): %s" % (index, len(images), image_url, error))
            out_path.write_bytes(data)
        if not data:
            fail("downloaded empty image %d of %d (%s)" % (index, len(images), image_url))
        sniffed = sniff_extension(data, ext)
        if sniffed != ext:
            new_path = out_dir / ("%03d.%s" % (index, sniffed))
            out_path.rename(new_path)
            out_path = new_path
            name = new_path.name
        files.append(name)
        log("downloaded %s (%d bytes)" % (name, len(data)))
    zip_name = "asura-ch%d.zip" % chapter
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-url", required=True)
    parser.add_argument("--chapter", type=int, required=True)
    parser.add_argument("--urls-file", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return download_chapter(args.chapter_url, args.chapter, Path(args.out), urls_file=args.urls_file)


if __name__ == "__main__":
    raise SystemExit(main())
