#!/usr/bin/env python3
"""Download ONE English chapter of Legend of the Northern Blade from
ReadXManga as a zip the Nano builder can consume unchanged.

Chapter number -> numeric ID map lives in readxmanga-chapter-map.json
(extracted from the series page's ChapterList astro-island props;
208 entries, ch1-202.2, verified ch1=33962, ch100=33690, ch202=30307,
ch202.2=30305). Chapter page:
  https://readxmanga.com/series/legend-of-the-northern-blade-229/read/<id>
Images are <img src="https://s2.cdnrxd.xyz/229/chapters/<ch>/<uuid>/<n>.jpg">
in the static HTML, NO Referer needed.
Usage: python3 download_en_readxmanga.py <chapter> <en_zips_dir> [map_json]
  -> <en_zips_dir>/nano-machine-ch<NNN>.zip (images 001.jpg in order)
"""
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
BASE = "https://readxmanga.com/series/legend-of-the-northern-blade-229"
IMG_RE = re.compile(r'<img[^>]*src="(https://s2\.cdnrxd\.xyz/229/chapters/[^"]+\.jpg)"[^>]*>')
DEFAULT_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "readxmanga-chapter-map.json")


def fetch(url, tries=3):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last


def valid_image(data):
    if not data or len(data) < 12:
        return False
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"


def fetch_one(args):
    url = args
    for attempt in range(3):
        try:
            data = fetch(url)
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
    ch = sys.argv[1]
    out = sys.argv[2]
    map_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MAP
    with open(map_path, encoding="utf-8") as f:
        cmap = json.load(f)
    if ch not in cmap:
        print(f"ch{ch}: no ReadXManga id in map", flush=True)
        sys.exit(1)
    cid = cmap[ch]
    ch_url = f"{BASE}/read/{cid}"
    html = fetch(ch_url).decode("utf-8", "replace")
    imgs = IMG_RE.findall(html)
    seen, uniq = set(), []
    for u in imgs:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    if not uniq:
        print(f"ch{ch}: NO IMAGES (URL {ch_url})", flush=True)
        sys.exit(1)
    with ThreadPoolExecutor(max_workers=6) as ex:
        datas = list(ex.map(fetch_one, uniq))
    ok = [(u, d) for u, d in zip(uniq, datas) if d]
    print(f"ch{ch}: downloaded {len(ok)}/{len(uniq)}", flush=True)
    if len(ok) != len(uniq):
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    zpath = os.path.join(out, f"nano-machine-ch{int(float(ch)):03d}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for i, (u, d) in enumerate(ok, 1):
            ext = os.path.splitext(u.split("?")[0])[1] or ".jpg"
            zf.writestr(f"{i:03d}{ext}", d)
    print(f"ch{ch}: wrote {zpath} ({len(ok)} images)", flush=True)


if __name__ == "__main__":
    main()
