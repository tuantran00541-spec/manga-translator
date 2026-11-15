#!/usr/bin/env python3
"""Download ONE English chapter from MangaCe as a zip the Nano
builder can consume unchanged.

Chapter URL pattern:
  https://mangace.com/chapter/<slug>-chapter-<n>/
Chapter images are plain https://cdn.black-clover.org/file/...webp URLs
embedded in the static chapter HTML (path contains /chapter-<n>/).
Usage: python3 download_en_mangace.py <slug> <chapter> <en_zips_dir>
  -> <en_zips_dir>/nano-machine-ch<NNN>.zip (images 001.ext in order)
Same output contract as download_en_mangaowl.py.
"""
import os
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
BASE = "https://mangace.com"
IMG_RE = re.compile(r'https://cdn\.black-clover\.org/file/[^\s"\'<>]+\.webp')


def fetch(url, referer=None, tries=3):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            if referer:
                req.add_header("Referer", referer)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def valid_image(data):
    """Magic-byte check: a throttled image fetch can return an HTML error
    page (HTTP 200) or truncated bytes, which only crashes later in
    build_dataset2.py (PIL UnidentifiedImageError). Catch it here.
    """
    if not data or len(data) < 12:
        return False
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def fetch_one(args):
    url, referer = args
    for attempt in range(3):
        try:
            data = fetch(url, referer)
        except Exception as e:
            print(f"  FAIL {url}: {e}", flush=True)
            data = None
        if data and valid_image(data):
            return data
        print(f"  RETRY {url}: bad image bytes (attempt {attempt + 1}/3)", flush=True)
        time.sleep(5 * (attempt + 1))
    print(f"  FAIL {url}: image never validated", flush=True)
    return None


def main():
    slug, ch, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    ch_url = f"{BASE}/chapter/{slug}-chapter-{ch}/"
    html = fetch(ch_url).decode("utf-8", "replace")
    imgs = [u for u in IMG_RE.findall(html) if f"/chapter-{ch}/" in u]
    seen, uniq = set(), []
    for u in imgs:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    if not uniq:
        print(f"ch{ch}: NO IMAGES (URL {ch_url} may be a 404/series page)", flush=True)
        sys.exit(1)
    with ThreadPoolExecutor(max_workers=6) as ex:
        datas = list(ex.map(fetch_one, [(u, ch_url) for u in uniq]))
    ok = [(u, d) for u, d in zip(uniq, datas) if d]
    print(f"ch{ch}: downloaded {len(ok)}/{len(uniq)}", flush=True)
    if len(ok) != len(uniq):
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    zpath = os.path.join(out, f"nano-machine-ch{ch:03d}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for i, (u, d) in enumerate(ok, 1):
            ext = os.path.splitext(u.split("?")[0])[1] or ".jpg"
            zf.writestr(f"{i:03d}{ext}", d)
    print(f"ch{ch}: wrote {zpath} ({len(ok)} images)", flush=True)


if __name__ == "__main__":
    main()
