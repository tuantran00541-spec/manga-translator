"""Bubble harvest: download one Asura series' chapters with the app's own
downloader, slice with the app's slicer, detect with the app's Kiuyha
detector (run-9 ONNX) via detect_slice. Inference only: no training, no
model changes, no app changes. Outputs per-chapter box JSONs + box crops.

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
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def model_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "chapters": {},
    }
    any_ok = False
    for ch in [int(c) for c in args.chapters.split(",") if c.strip()]:
        ch_url = f"{args.series_url}/chapter/{ch}"
        ch_dir = out / f"chapter_{ch}"
        raw_dir = ch_dir / "raw"
        slice_dir = ch_dir / "sliced"
        crop_dir = ch_dir / "crops"
        for d in (raw_dir, slice_dir, crop_dir):
            d.mkdir(parents=True, exist_ok=True)
        try:
            pages = download_chapter(ch_url, raw_dir)
        except Exception as exc:  # record and continue with next chapter
            summary["chapters"][ch] = {"chapter_url": ch_url, "error": str(exc)[:500]}
            print(f"[{args.slug} ch{ch}] DOWNLOAD FAILED: {exc}", flush=True)
            continue
        slices: list[Path] = []
        for i, page in enumerate(pages):
            slices.extend(slice_image(page, slice_dir, f"{i:03d}"))
        ch_boxes = 0
        slice_records = []
        for s in sorted(slice_dir.glob("*.png")):
            img = np.array(Image.open(s).convert("RGB"))
            boxes = det.detect_slice(img)
            box_list = [[int(x1), int(y1), int(x2), int(y2), float(c)] for x1, y1, x2, y2, c in boxes]
            ch_boxes += len(box_list)
            pil = Image.open(s).convert("RGB")
            w, h = pil.size
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
            "model": model_path.name,
            "model_sha256": sha,
            "pages": len(pages),
            "slices": slice_records,
        }, indent=1))
        summary["chapters"][ch] = {
            "chapter_url": ch_url,
            "pages": len(pages),
            "slices": len(slice_records),
            "total_boxes": ch_boxes,
        }
        any_ok = True
        print(f"[{args.slug} ch{ch}] pages={len(pages)} slices={len(slice_records)} boxes={ch_boxes}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
