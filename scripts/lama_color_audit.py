"""Measure how well inpainted regions match the surrounding original colours.

Runs on a processed chapter; no model is needed. Every region that differs
between the original slice and its clean image is treated as one inpainted
region. For each region the script compares:

- seam ΔE: mean colour of the clean pixels just inside the region border
  versus the original pixels just outside it (a visible tone step if large);
- body ΔE: mean colour of the whole filled region versus the outside ring;
- texture ratio: grey-level std inside the fill divided by the std of the
  surrounding original (≈1 keeps grain/screentone, ≪1 means a smooth patch).

ΔE is CIE76 in L*a*b*: < 2 is invisible, 2-5 visible on close inspection,
> 5 an obvious colour mismatch.

Usage:
    python scripts/lama_color_audit.py <chapter_id> [--top 15] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.manifest_utils import load_manifest_raw  # noqa: E402
from app.security import validate_managed_path  # noqa: E402

MIN_REGION_PIXELS = 40
OUTER_RING = 6
INNER_RING = 3
# A flat fill that matches the background exactly changes only the glyph
# pixels; merge the strokes of one text block into a single region.
MERGE_RADIUS = 6


def _lab_mean(pixels: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float64).reshape(-1, 3)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab.mean(axis=0)


def _delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def _kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))


def audit_pair(original: np.ndarray, clean: np.ndarray) -> list[dict]:
    """Return one record per inpainted region of a page."""
    if original.shape != clean.shape:
        raise ValueError(f"original {original.shape} and clean {clean.shape} differ in size")
    changed = np.any(original != clean, axis=2).astype(np.uint8)
    merged = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, _kernel(MERGE_RADIUS))
    # The outside ring must be untouched original pixels, never a neighbour fill.
    untouched = cv2.dilate(merged, _kernel(1)) == 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    records = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < MIN_REGION_PIXELS:
            continue
        pad = OUTER_RING + 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(original.shape[1], x + w + pad), min(original.shape[0], y + h + pad)
        region = (labels[y1:y2, x1:x2] == label).astype(np.uint8)
        outside = (cv2.dilate(region, _kernel(OUTER_RING)) > 0) & untouched[y1:y2, x1:x2]
        inner = (region > 0) & (cv2.erode(region, _kernel(INNER_RING)) == 0)
        body = region > 0
        if not outside.any() or not inner.any():
            continue
        orig_crop, clean_crop = original[y1:y2, x1:x2], clean[y1:y2, x1:x2]
        ring_lab = _lab_mean(orig_crop[outside])
        seam = _delta_e(_lab_mean(clean_crop[inner]), ring_lab)
        body_de = _delta_e(_lab_mean(clean_crop[body]), ring_lab)
        ring_std = float(gray_original[y1:y2, x1:x2][outside].std())
        fill_std = float(gray_clean[y1:y2, x1:x2][body].std())
        records.append({
            "box": [x, y, x + w, y + h],
            "pixels": area,
            "seam_delta_e": round(seam, 2),
            "body_delta_e": round(body_de, 2),
            "texture_ratio": round(fill_std / ring_std, 2) if ring_std >= 2.0 else None,
            "outside_grey_std": round(ring_std, 2),
            # Lightness and chroma of the surroundings: white bubbles sit near
            # L=100 / chroma 0; coloured bubbles and art-backed text do not.
            "outside_lightness": round(float(ring_lab[0]), 1),
            "outside_chroma": round(float(np.hypot(ring_lab[1], ring_lab[2])), 1),
        })
    return records


def audit_chapter(chapter_id: str) -> list[dict]:
    manifest = load_manifest_raw(chapter_id)
    records = []
    for index, page in enumerate(manifest.get("pages") or []):
        if not page.get("clean") or not page.get("original"):
            continue
        original = read_image(validate_managed_path(page["original"], RAW_DIR / chapter_id))
        clean = read_image(validate_managed_path(page["clean"], PROCESSED_DIR / chapter_id))
        for record in audit_pair(original, clean):
            record["page"] = index
            records.append(record)
    return records


def on_plain_white(record: dict) -> bool:
    return record["outside_lightness"] >= 92 and record["outside_chroma"] <= 4


def _grade(delta_e: float) -> str:
    return "ok" if delta_e < 2 else "visible" if delta_e <= 5 else "mismatch"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("chapter_id")
    parser.add_argument("--top", type=int, default=15, help="worst regions to list")
    parser.add_argument("--out", type=Path, help="write report.json and worst-region crops here")
    args = parser.parse_args()

    manifest = load_manifest_raw(args.chapter_id)
    all_records = audit_chapter(args.chapter_id)

    if not all_records:
        print("No inpainted regions found (clean images equal the originals).")
        return 0

    grades = {"ok": 0, "visible": 0, "mismatch": 0}
    for record in all_records:
        grades[_grade(record["seam_delta_e"])] += 1
    seams = np.array([r["seam_delta_e"] for r in all_records])
    ratios = [r["texture_ratio"] for r in all_records if r["texture_ratio"] is not None]
    print(f"{len(all_records)} inpainted regions in {args.chapter_id}")
    print(f"seam ΔE  median {np.median(seams):.2f}  p90 {np.percentile(seams, 90):.2f}  max {seams.max():.2f}")
    print(f"  ok (<2): {grades['ok']}   visible (2-5): {grades['visible']}   mismatch (>5): {grades['mismatch']}")
    coloured = [r for r in all_records if not on_plain_white(r)]
    if coloured:
        coloured_seams = np.array([r["seam_delta_e"] for r in coloured])
        print(f"on coloured/art backgrounds: {len(coloured)} regions, seam ΔE median {np.median(coloured_seams):.2f}"
              f"  p90 {np.percentile(coloured_seams, 90):.2f}  >5: {int((coloured_seams > 5).sum())}")
    if ratios:
        smooth = sum(1 for r in ratios if r < 0.5)
        print(f"textured surroundings: {len(ratios)} regions, {smooth} filled as a smooth patch (texture ratio < 0.5)")
    worst = sorted(all_records, key=lambda r: r["seam_delta_e"], reverse=True)[: args.top]
    print("\nworst regions (page, box, seam ΔE, body ΔE, texture ratio):")
    for r in worst:
        print(f"  p{r['page']:03d} {r['box']}  seam {r['seam_delta_e']:5.2f}  body {r['body_delta_e']:5.2f}  texture {r['texture_ratio']}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        pages = manifest["pages"]
        for rank, r in enumerate(worst):
            page = pages[r["page"]]
            original = read_image(validate_managed_path(page["original"], RAW_DIR / args.chapter_id))
            clean = read_image(validate_managed_path(page["clean"], PROCESSED_DIR / args.chapter_id))
            x1, y1, x2, y2 = r["box"]
            m = 24
            x1, y1 = max(0, x1 - m), max(0, y1 - m)
            x2, y2 = min(original.shape[1], x2 + m), min(original.shape[0], y2 + m)
            pair = np.hstack([original[y1:y2, x1:x2], clean[y1:y2, x1:x2]])
            cv2.imwrite(str(args.out / f"{rank:02d}_p{r['page']:03d}_seam{r['seam_delta_e']:.1f}.png"), pair)
        print(f"\nreport and side-by-side crops (original | clean) written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
