#!/usr/bin/env python3
"""Download ONE English chapter from MangaOwl (Madara) as a zip the Nano
builder can consume unchanged.

Chapter URL pattern (from the series page's chapter list):
  https://mangaowl.io/read-1/<slug>/<slug>-chapter-<n>/
NOT /chapter-<n>/ — that serves the series page (soft-404 trap: HTTP 200,
0 chapter images). Chapter images are the wp-manga-chapter-img srcs in
the static HTML, hosted on mangaowl.io/wp-content/uploads/WP-manga/data/.
Usage: python3 download_en_mangaowl.py <slug> <chapter> <en_zips_dir>
  -> <en_zips_dir>/nano-machine-ch<NNN>.zip (images 001.ext in order)
The zip name matches what build_dataset2.py expects; only the contents
differ from the Nano EN zips.
"""
import os
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
BASE = "https://mangaowl.io"
IMG_RE = re.compile(r'<img[^>]*class="[^"]*wp-manga-chapter-img[^"]*"[^>]*src="([^"]+)"')
IMG_RE2 = re.compile(r'<img[^>]*src="([^"]+)"[^>]*class="[^"]*wp-manga-chapter-img[^"]*"')


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


def fetch_one(args):
    url, referer = args
    try:
        return fetch(url, referer)
    except Exception as e:
        print(f"  FAIL {url}: {e}", flush=True)
        return None


def main():
    slug, ch, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    ch_url = f"{BASE}/read-1/{slug}/{slug}-chapter-{ch}/"
    html = fetch(ch_url).decode("utf-8", "replace")
    imgs = IMG_RE.findall(html) or IMG_RE2.findall(html)
    seen, uniq = set(), []
    for u in imgs:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    if not uniq:
        print(f"ch{ch}: NO IMAGES (URL {ch_url} may be a series page, not a reader)", flush=True)
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
