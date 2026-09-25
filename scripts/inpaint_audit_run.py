"""Process a real chapter with the production pipeline and grade inpaint colour.

CI helper for GitHub Actions: downloads the chapter, cleans it with the same
pipeline as the web app (dynamic LaMa), runs scripts/lama_color_audit.py and
writes small JPEG evidence that can be committed for review:

- summary.txt / report.json: colour-fidelity numbers for every region;
- slices/NN_*.jpg: original | clean side by side for the slices whose text sat
  on coloured bubbles or artwork (the cases where colour errors show);
- regions/NN_*.jpg: the worst individual regions, cropped and enlarged.

Usage:
    python scripts/inpaint_audit_run.py <chapter_url> --out audit-results/x
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from app.image_io import read_image  # noqa: E402
from app.manifest_utils import load_manifest_raw  # noqa: E402
from app.processing_pipeline_factory import build_processing_pipeline  # noqa: E402
from app.security import validate_managed_path  # noqa: E402
import lama_color_audit  # noqa: E402

CHAPTER_ID = "a0d17c01"


def _jpeg(path: Path, image: np.ndarray, max_width: int) -> None:
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _pair(original: np.ndarray, clean: np.ndarray) -> np.ndarray:
    gap = np.full((original.shape[0], 12, 3), (0, 0, 255), np.uint8)
    return np.hstack([original, gap, clean])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-slices", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--slices", type=int, default=10, help="slice pairs to export")
    parser.add_argument("--regions", type=int, default=16, help="worst regions to export")
    args = parser.parse_args()

    shutil.rmtree(RAW_DIR / CHAPTER_ID, ignore_errors=True)
    shutil.rmtree(PROCESSED_DIR / CHAPTER_ID, ignore_errors=True)
    pipeline = build_processing_pipeline()
    started = time.perf_counter()
    manifest = pipeline.download_chapter(args.url, CHAPTER_ID, workers=args.workers)
    pages = manifest.get("pages") or []
    indices = list(range(min(len(pages), args.max_slices)))
    downloaded = time.perf_counter()
    pipeline.process_pages(CHAPTER_ID, indices, workers=args.workers)
    processed = time.perf_counter()

    args.out.mkdir(parents=True, exist_ok=True)
    report = io.StringIO()
    sys.argv = ["lama_color_audit.py", CHAPTER_ID, "--top", "20"]
    with contextlib.redirect_stdout(report):
        lama_color_audit.main()
    records = lama_color_audit.audit_chapter(CHAPTER_ID)
    header = (
        f"chapter: {args.url}\n"
        f"slices: {len(pages)} downloaded, {len(indices)} processed\n"
        f"download {downloaded - started:.0f}s, clean {processed - downloaded:.0f}s\n\n"
    )
    (args.out / "summary.txt").write_text(header + report.getvalue(), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(header + report.getvalue())

    manifest = load_manifest_raw(CHAPTER_ID)
    pages = manifest["pages"]

    def images(index: int) -> tuple[np.ndarray, np.ndarray]:
        page = pages[index]
        return (
            read_image(validate_managed_path(page["original"], RAW_DIR / CHAPTER_ID)),
            read_image(validate_managed_path(page["clean"], PROCESSED_DIR / CHAPTER_ID)),
        )

    # Slices ranked by how much inpainting happened on non-white surroundings.
    weight: dict[int, int] = {}
    for r in records:
        if not lama_color_audit.on_plain_white(r):
            weight[r["page"]] = weight.get(r["page"], 0) + r["pixels"]
    slices_dir = args.out / "slices"
    slices_dir.mkdir(exist_ok=True)
    for rank, index in enumerate(sorted(weight, key=weight.get, reverse=True)[: args.slices]):
        original, clean = images(index)
        _jpeg(slices_dir / f"{rank:02d}_slice{index:03d}.jpg", _pair(original, clean), 1600)

    regions_dir = args.out / "regions"
    regions_dir.mkdir(exist_ok=True)
    worst = sorted(records, key=lambda r: r["seam_delta_e"], reverse=True)[: args.regions]
    for rank, r in enumerate(worst):
        original, clean = images(r["page"])
        x1, y1, x2, y2 = r["box"]
        m = 40
        x1, y1 = max(0, x1 - m), max(0, y1 - m)
        x2, y2 = min(original.shape[1], x2 + m), min(original.shape[0], y2 + m)
        pair = _pair(original[y1:y2, x1:x2], clean[y1:y2, x1:x2])
        scale = min(3.0, 900 / pair.shape[1]) if pair.shape[1] < 900 else 1.0
        if scale > 1.0:
            pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tag = "white" if lama_color_audit.on_plain_white(r) else "colour"
        _jpeg(regions_dir / f"{rank:02d}_slice{r['page']:03d}_{tag}_seam{r['seam_delta_e']:.1f}.jpg", pair, 1400)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
