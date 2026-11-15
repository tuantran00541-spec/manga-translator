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


def fetch_one(args):
    url, referer = args
    try:
        return fetch(url, referer)
    except Exception as e:
        print(f"  FAIL {url}: {e}", flush=True)
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
