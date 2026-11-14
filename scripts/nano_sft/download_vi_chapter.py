#!/usr/bin/env python3
"""Download ONE Vietnamese Nano Machine chapter from nettruyen15s.com.

Stdlib only. Chapter images are in the static HTML as src attrs on
i*.netcdn.one hosts (plain curl works, no Cloudflare).
Usage: python3 download_vi_chapter.py <chapter> <out_dir>
  -> <out_dir>/ch<NNN>/NNN.webp (3-digit, reading order)
"""
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
BASE = "https://nettruyen15s.com"
IMG_RE = re.compile(r'src="(https://i\d+\.netcdn\.one[^"]*\.webp[^"]*)"')


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


def valid_image(path):
    try:
        with open(path, "rb") as f:
            head = f.read(16)
        return head[:4] == b"RIFF" or head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def fetch_one(args):
    url, path, referer = args
    if os.path.exists(path) and valid_image(path):
        return True
    try:
        data = fetch(url, referer)
        with open(path, "wb") as f:
            f.write(data)
        return valid_image(path)
    except Exception as e:
        print(f"  FAIL {url}: {e}", flush=True)
        return False


def main():
    ch, out = sys.argv[1], sys.argv[2]
    chdir = os.path.join(out, f"ch{ch}")
    os.makedirs(chdir, exist_ok=True)
    url = f"{BASE}/manga/nga-lao-ma-than/chapter-{ch}"
    html = fetch(url).decode("utf-8", "replace")
    imgs = sorted(set(IMG_RE.findall(html)))
    if not imgs:
        print(f"ch{ch}: NO IMAGES", flush=True)
        sys.exit(1)
    jobs = []
    for i, u in enumerate(imgs, 1):
        ext = ".webp" if ".webp" in u else ".jpg"
        jobs.append((u, os.path.join(chdir, f"{i:03d}{ext}"), url))
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(fetch_one, jobs))
    ok = sum(results)
    print(f"ch{ch}: {ok}/{len(imgs)}", flush=True)
    sys.exit(0 if ok == len(imgs) else 1)


if __name__ == "__main__":
    main()
