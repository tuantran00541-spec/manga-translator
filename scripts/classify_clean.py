"""Triage a cleaned chapter: find text the clean left behind, and split it
into detector misses (no box ever covered it: fine-tune material) versus
inpaint skips (a box existed, the eraser skipped it: pipeline issue).

Reuses the leftover test from scripts/chapter_run.py (re-detect text on the
clean image, connected components on the original mask, 30% remains) and
adds one step: overlap each leftover block against the manifest boxes.
No vision model beyond the detector, no eyeballing thousands of images.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.manifest_utils import load_manifest_raw

# Same cutoffs as chapter_run.py's leftover test.
MIN_BLOCK_PX = 150
REMAIN_FRAC = 0.30
DILATE_KERNEL = np.ones((15, 15), np.uint8)
# A leftover block counts as covered by a box when the overlap is at least
# this share of the block: a real skip sits inside its box, a real miss
# sits nowhere near one.
BOX_OVERLAP_FRAC = 0.20
MAX_CROPS_PER_KIND = 8


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for b in boxes:
        if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
            mask[b.y1:b.y2, b.x1:b.x2] |= b.mask > 127
    return mask


def pair(original: np.ndarray, clean: np.ndarray, box, path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = []
    for label, image in (("original", original), ("clean", clean)):
        tile = image[y1:y2, x1:x2]
        scale = min(1.0, 360 / max(1, tile.shape[0]))
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def box_overlaps_block(box: tuple[int, int, int, int], rects) -> bool:
    """True when a leftover block overlaps any manifest box rectangle."""
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    for r in rects:
        ix1, iy1 = max(x1, r["x1"]), max(y1, r["y1"])
        ix2, iy2 = min(x2, r["x2"]), min(y2, r["y2"])
        if ix2 > ix1 and iy2 > iy1 and (ix2 - ix1) * (iy2 - iy1) >= BOX_OVERLAP_FRAC * area:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest_raw(args.chapter_id)
    pages = manifest.get("pages", [])
    args.out.mkdir(parents=True, exist_ok=True)
    detector = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    started = time.perf_counter()

    misses, skips, page_rows = [], [], []
    for number, page in enumerate(pages):
        if not page.get("clean"):
            continue
        original = read_image(Path(page["original"]))
        clean = read_image(Path(page["clean"]))
        # Full slice, not the stitch core: the core excludes the slice-joint
        # overlap, and that is exactly where half-cleaned bubbles hide
        # (ch.337 clean_003_04's upper lobe sits above core_y1). A miss at a
        # joint may be reported twice, once per adjacent slice; for triage
        # that is better than missing it.
        before = original
        after = clean
        text = text_mask(before.shape[:2], detector.text_boxes(before))
        rest = text_mask(after.shape[:2], detector.text_boxes(after))
        count, labels = cv2.connectedComponents(
            cv2.dilate(text.astype(np.uint8), DILATE_KERNEL)
        )
        rects = page.get("boxes", []) or []
        confs = [b.get("confidence") for b in rects if b.get("confidence") is not None]
        row = {
            "slice": number, "boxes": len(rects),
            "min_conf": round(min(confs), 3) if confs else None,
            "misses": 0, "skips": 0,
        }
        for i in range(1, count):
            block = (labels == i) & text
            if block.sum() < MIN_BLOCK_PX:
                continue
            if (rest & block).sum() < REMAIN_FRAC * block.sum():
                continue
            ys, xs = np.nonzero(block)
            box = (
                max(0, int(xs.min()) - 40), max(0, int(ys.min()) - 40),
                min(before.shape[1], int(xs.max()) + 40),
                min(before.shape[0], int(ys.max()) + 40),
            )
            # Box coordinates are already page-relative (full slice, no crop),
            # so they compare directly against the manifest boxes.
            page_box = box
            kind = "inpaint_skip" if box_overlaps_block(page_box, rects) else "detector_miss"
            record = {"slice": number, "box": [int(v) for v in page_box],
                      "px": int(block.sum()), "kind": kind}
            if kind == "detector_miss":
                misses.append(record)
                row["misses"] += 1
                if row["misses"] <= MAX_CROPS_PER_KIND:
                    pair(before, after, box, args.out / f"miss-slice{number:03d}-{row['misses']:02d}.jpg")
            else:
                skips.append(record)
                row["skips"] += 1
                if row["skips"] <= MAX_CROPS_PER_KIND:
                    pair(before, after, box, args.out / f"skip-slice{number:03d}-{row['skips']:02d}.jpg")
        page_rows.append(row)

    result = {
        "chapter_id": args.chapter_id,
        "slices": len(pages),
        "cleaned": sum(1 for p in pages if p.get("clean")),
        "detector_misses": len(misses),
        "inpaint_skips": len(skips),
        "no_box_slices": [r["slice"] for r in page_rows if r["boxes"] == 0],
        "seconds": round(time.perf_counter() - started, 1),
        "pages": page_rows,
        "misses": misses,
        "skips": skips,
    }
    (args.out / "classification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("pages", "misses", "skips")},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
