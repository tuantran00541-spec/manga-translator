#!/usr/bin/env python3
"""Download ONE Vietnamese chapter from nettruyen15s.com for ANY series.

Series-page lookup -> chapter id -> POST /ajax/image/list/chap/<id>
(static HTML has no images for these series; the ajax endpoint returns
them as data-original/src attrs). Falls back to the static chapter-page
regex if ajax returns nothing.
Usage: python3 download_vi_series.py <slug> <chapter> <out_dir>
  -> <out_dir>/ch<NNN>/NNN.ext (3-digit, reading order)
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
BASE = "https://nettruyen15s.com"
STATIC_IMG_RE = re.compile(r'src="(https://i\d+\.netcdn\.one[^"]*\.webp[^"]*)"')
CHAP_ID_RE = re.compile(r'chapter-(\d+)" data-id="(\d+)"')
ORIG_RE = re.compile(r'data-original="(https?://[^"]+)"')
SRC_RE = re.compile(r'src="(https?://[^"]+)"')


def fetch(url, referer=None, data=None, tries=3):
    for attempt in range(tries):
        cmd = ["curl", "-sS", "-L", "--fail", "--retry", "2", "--retry-delay", "5",
               "-m", "60", "-A", UA, url]
        if data is not None:
            cmd += ["-X", "POST", "--data-binary", "@-"]
        if referer:
            cmd += ["--referer", referer]
        r = subprocess.run(cmd, input=data, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        time.sleep(5 * (attempt + 1))
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": UA})
            if referer:
                req.add_header("Referer", referer)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
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


def valid_bytes(data):
    if not data or len(data) < 12:
        return False
    return data[:4] == b"RIFF" or data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"


def fetch_one(args):
    url, path, referer = args
    if os.path.exists(path) and valid_image(path):
        return True
    try:
        data = fetch(url, referer)
    except Exception as e:
        print(f"  FAIL {url}: {e}", flush=True)
        return False
    if not valid_bytes(data):
        if os.path.exists(path):
            os.remove(path)
        return False
    with open(path, "wb") as f:
        f.write(data)
    return valid_image(path)


def chapter_images(slug, ch):
    """Return ordered unique image URLs for one chapter (ajax first)."""
    series_url = f"{BASE}/manga/{slug}"
    html = fetch(series_url).decode("utf-8", "replace")
    ids = {int(a): b for a, b in CHAP_ID_RE.findall(html)}
    if ch not in ids:
        raise SystemExit(f"chapter {ch} not in series page (found {len(ids)} chapters)")
    chap_id = ids[ch]
    ajax = fetch(f"{BASE}/ajax/image/list/chap/{chap_id}",
                 referer=f"{BASE}/manga/{slug}/chapter-{ch}", data=b"")
    try:
        payload = json.loads(ajax.decode("utf-8", "replace"))
        page_html = payload.get("html", "")
        imgs = ORIG_RE.findall(page_html) or SRC_RE.findall(page_html)
    except Exception:
        imgs = []
    if not imgs:
        # static fallback: images embedded in the chapter page itself
        ch_html = fetch(f"{BASE}/manga/{slug}/chapter-{ch}").decode("utf-8", "replace")
        imgs = STATIC_IMG_RE.findall(ch_html)
    seen, out = set(), []
    for u in imgs:
        u = u.replace("\\/", "/")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main():
    slug, ch, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    chdir = os.path.join(out, f"ch{ch}")
    os.makedirs(chdir, exist_ok=True)
    imgs = chapter_images(slug, ch)
    if not imgs:
        print(f"ch{ch}: NO IMAGES", flush=True)
        sys.exit(1)
    ch_url = f"{BASE}/manga/{slug}/chapter-{ch}"
    jobs = []
    for i, u in enumerate(imgs, 1):
        ext = ".webp" if ".webp" in u.split("?")[0] else ".jpg"
        jobs.append((u, os.path.join(chdir, f"{i:03d}{ext}"), ch_url))
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(fetch_one, jobs))
    ok = sum(results)
    missing = [i + 1 for i, r in enumerate(results) if not r]
    if missing:
        print(f"ch{ch}: {ok}/{len(imgs)} (missing pages: {missing})", flush=True)
    else:
        print(f"ch{ch}: {ok}/{len(imgs)}", flush=True)
    sys.exit(0 if ok * 20 >= len(imgs) * 19 else 1)


if __name__ == "__main__":
    main()
