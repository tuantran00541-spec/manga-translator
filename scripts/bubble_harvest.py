"""Bubble harvest: download one Asura series' chapters with the app's own
downloader, slice with the app's slicer, detect with the app's Kiuyha
detector (run-9 ONNX) via detect_slice. Inference only: no training, no
model changes, no app changes. Outputs per-chapter box JSONs + box crops
+ a deterministic miss-hunt slice sample for later eye-checking.

Fallback (harvest-script only, app code untouched): the app's Asura
adapter only accepts chapter assets under /asura-images/chapters/.
Older "restored" chapters serve images under
/asura-images/chapters-restored/, which the app filter drops silently
(static 0 urls, JS 0 urls, empty failure list). When the app path yields
nothing for a chapter, this script re-runs the same static extraction
(same alt regex, same URL resolver, same app download_urls) with both
prefixes accepted, and records download_path="fallback" so the gap is
visible in the output instead of hidden.

Miss-hunt design (free text = text with NO box/container at all):
detector boxes alone UNDER-sample free text by construction, because
the known failure is the detector missing it entirely. No VLM API is
available on the runner, so the miss-hunt is an eye-check AFTER
collection: this script saves a deterministic sample of slices per
chapter (every 8th slice, max 6) as misshunt JPGs, and the eye-check
records detector-missed free-text instances (slice, approx box, text)
into free_text_missed.json. Detector-found boxes and eye-found missed
free text are kept as two separate counts, never merged.

Usage:
  PYTHONPATH=. python scripts/bubble_harvest.py \
    --series-url https://asurascans.com/comics/nano-machine-3ebcc5a9 \
    --slug nano-machine --chapters 1,2,3 \
    --model model/kiuyha-tuned-s.onnx --out harvest-out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

_PAGE_ALT_RE = re.compile(r"^\s*Page\s+(\d+)\b", re.IGNORECASE)
_ASSET_PREFIXES = ("/asura-images/chapters/", "/asura-images/chapters-restored/")
_MISSHUNT_EVERY = 8
_MISSHUNT_MAX = 6


def model_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_width(path: Path) -> int:
    with Image.open(path) as image:
        return image.width


def fallback_download(chapter_url: str, raw_dir: Path) -> tuple[list[Path], str]:
    """Static extraction accepting both chapter-asset prefixes.

    Mirrors AsuraScansStaticAdapter (same alt regex, same resolver, same
    app download_urls); only the asset-prefix filter is wider. Returns
    (pages, prefix_used) where prefix_used is the prefix of the first
    downloaded asset, or "" when nothing was found.
    """
    from bs4 import BeautifulSoup

    from app.downloader.asura import AsuraScansStaticAdapter
    from app.downloader.http import read_response_limited, safe_get
    from app.downloader.image_urls import best_srcset_candidate, resolve_image_candidate
    from app.downloader.registry import STATIC_ADAPTER
    from app.parameters import DOWNLOAD_JS_MIN_IMAGE_WIDTH, REMOTE_CONNECT_TIMEOUT_SECONDS
    from app.security import MAX_REMOTE_DOCUMENT_BYTES

    adapter = AsuraScansStaticAdapter()
    response = safe_get(
        chapter_url,
        headers=adapter.headers,
        timeout=REMOTE_CONNECT_TIMEOUT_SECONDS,
        stream=True,
    )
    try:
        body = read_response_limited(response, limit_bytes=MAX_REMOTE_DOCUMENT_BYTES)
    finally:
        response.close()
    soup = BeautifulSoup(body, "lxml")
    by_page: dict[int, str] = {}
    for img in soup.select("img[alt]"):
        match = _PAGE_ALT_RE.match(str(img.get("alt") or ""))
        if not match:
            continue
        url = resolve_image_candidate(
            chapter_url,
            best_srcset_candidate(img.get("data-srcset") or img.get("srcset")),
            img.get("data-src"),
            img.get("data-original"),
            img.get("data-lazy"),
            img.get("src"),
        )
        if not url:
            continue
        if not any(urlparse(url).path.startswith(p) for p in _ASSET_PREFIXES):
            continue
        by_page.setdefault(int(match.group(1)), url)
    ordered: list[str] = []
    page_number = 1
    while page_number in by_page:
        ordered.append(by_page[page_number])
        page_number += 1
    if not ordered:
        return [], ""
    prefix_used = next(
        (p for p in _ASSET_PREFIXES if urlparse(ordered[0]).path.startswith(p)), ""
    )
    paths = STATIC_ADAPTER.download_urls(ordered, raw_dir, referer=chapter_url)
    pages = [p for p in paths if _image_width(p) >= DOWNLOAD_JS_MIN_IMAGE_WIDTH]
    if not pages:
        return paths, prefix_used
    for p in set(paths) - set(pages):
        p.unlink(missing_ok=True)
    return pages, prefix_used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-url", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--chapters", required=True, help="comma-separated, e.g. 1,2,3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from app.detector.kiuyha_detector import KiuyhaTextDetector
    from app.downloader.registry import download_chapter
    from app.downloader.slicer import slice_image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    det = KiuyhaTextDetector(str(model_path))
    sha = model_sha(model_path)

    summary: dict = {
        "slug": args.slug,
        "series_url": args.series_url,
        "model": model_path.name,
        "model_sha256": sha,
        "misshunt": "every 8th slice per chapter, max 6, saved as misshunt JPGs; "
        "eye-check afterwards writes free_text_missed.json (NOT in this script)",
        "chapters": {},
    }
    any_ok = False
    for ch in [int(c) for c in args.chapters.split(",") if c.strip()]:
        ch_url = f"{args.series_url}/chapter/{ch}"
        ch_dir = out / f"chapter_{ch}"
        raw_dir = ch_dir / "raw"
        slice_dir = ch_dir / "sliced"
        crop_dir = ch_dir / "crops"
        miss_dir = ch_dir / "misshunt"
        for d in (raw_dir, slice_dir, crop_dir, miss_dir):
            d.mkdir(parents=True, exist_ok=True)
        download_path = "app"
        try:
            pages = download_chapter(ch_url, raw_dir)
            if not pages:
                raise ValueError("app downloader returned no pages")
        except Exception as exc:
            print(f"[{args.slug} ch{ch}] app path failed ({exc}); trying fallback", flush=True)
            try:
                pages, _prefix = fallback_download(ch_url, raw_dir)
                download_path = "fallback"
            except Exception as exc2:
                summary["chapters"][ch] = {"chapter_url": ch_url, "error": str(exc2)[:500]}
                print(f"[{args.slug} ch{ch}] DOWNLOAD FAILED: {exc2}", flush=True)
                continue
            if not pages:
                summary["chapters"][ch] = {
                    "chapter_url": ch_url,
                    "error": "app path and fallback both yielded no pages",
                }
                print(f"[{args.slug} ch{ch}] DOWNLOAD FAILED: no pages from either path", flush=True)
                continue
        slices: list[Path] = []
        for i, page in enumerate(pages):
            slices.extend(slice_image(page, slice_dir, f"{i:03d}"))
        ch_boxes = 0
        slice_records = []
        slice_files = sorted(slice_dir.glob("*.png"))
        misshunt_samples = [
            s.name for i, s in enumerate(slice_files) if i % _MISSHUNT_EVERY == 0
        ][:_MISSHUNT_MAX]
        for s in slice_files:
            img = np.array(Image.open(s).convert("RGB"))
            boxes = det.detect_slice(img)
            box_list = [[int(x1), int(y1), int(x2), int(y2), float(c)] for x1, y1, x2, y2, c in boxes]
            ch_boxes += len(box_list)
            pil = Image.open(s).convert("RGB")
            w, h = pil.size
            if s.name in misshunt_samples:
                sample = pil.copy()
                sample.thumbnail((900, 9000))
                sample.save(miss_dir / f"{s.stem}.jpg", quality=85)
            for i, (x1, y1, x2, y2, _c) in enumerate(box_list):
                mx1, my1 = max(0, x1 - 8), max(0, y1 - 8)
                mx2, my2 = min(w, x2 + 8), min(h, y2 + 8)
                if mx2 > mx1 and my2 > my1:
                    pil.crop((mx1, my1, mx2, my2)).save(
                        crop_dir / f"{s.stem}_box{i:03d}.jpg", quality=92
                    )
            slice_records.append({"slice": s.name, "width": w, "height": h, "boxes": box_list})
        (ch_dir / "boxes.json").write_text(json.dumps({
            "slug": args.slug,
            "chapter": ch,
            "chapter_url": ch_url,
            "download_path": download_path,
            "misshunt_samples": misshunt_samples,
            "model": model_path.name,
            "model_sha256": sha,
            "pages": len(pages),
            "slices": slice_records,
        }, indent=1))
        summary["chapters"][ch] = {
            "chapter_url": ch_url,
            "download_path": download_path,
            "pages": len(pages),
            "slices": len(slice_records),
            "total_boxes": ch_boxes,
            "misshunt_samples": misshunt_samples,
        }
        any_ok = True
        print(f"[{args.slug} ch{ch}] via={download_path} pages={len(pages)} "
              f"slices={len(slice_records)} boxes={ch_boxes}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
