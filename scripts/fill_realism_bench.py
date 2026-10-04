"""One-off bench: does matching the edge colour, adding grain back or a larger LaMa side make fills look real."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

import app.inpaint.lama_inpainter as lama
from app.config import KIUYHA_TEXT_MODEL, PROCESSED_DIR
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.manifest_utils import load_manifest_raw
from app.processing_pipeline_factory import build_processing_pipeline
from app.region_policy import page_preserve_regions
from scripts.verify import leftovers, text_mask

VARIANTS = ("base", "match", "grain", "side768")
MODE = {"match": False, "grain": False}
BAND = 4  # pixels inside a hole whose edge colour is matched to the outside
MIN_AREA = 200  # smaller fills are specks
CROPS = 14
_original_fill = lama.Inpainter._lama_fill


def _nblur(image: np.ndarray, where: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    w = where.astype(np.float32)
    num = cv2.blur(image.astype(np.float32) * w[..., None], (k, k))
    den = cv2.blur(w, (k, k))
    return num / np.maximum(den, 1e-3)[..., None], den


def adjust(before: np.ndarray, painted: np.ndarray, hole: np.ndarray, seed: int) -> np.ndarray:
    """Pull a fill's edge colour to its surroundings and give it the surroundings' grain."""
    out = painted.astype(np.float32)
    if MODE["match"]:
        inner = cv2.erode(hole.astype(np.uint8), np.ones((2 * BAND + 1,) * 2, np.uint8)) > 0
        band = hole & ~inner
        outside_mean, den = _nblur(before, ~hole, 2 * BAND + 3)
        inside_mean, _ = _nblur(painted, hole, 2 * BAND + 3)
        shift = np.where((den > 0.05)[..., None], outside_mean - inside_mean, 0.0).astype(np.float32)
        target = (inner.astype(np.uint8)) * 255
        for c in range(3):
            channel = np.where(band, shift[..., c], 0).astype(np.float32)
            if inner.any():
                channel = cv2.inpaint(channel, target, 3, cv2.INPAINT_TELEA)
            shift[..., c] = cv2.GaussianBlur(channel, (0, 0), 2)
        out[hole] += shift[hole]
    if MODE["grain"]:
        def high(img):
            g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return g - cv2.GaussianBlur(g, (0, 0), 1.2)
        ring = (cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & \
               ~(cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        if ring.sum() > 50 and hole.sum() > 50:
            s_out, s_in = float(high(before)[ring].std()), float(high(out)[hole].std())
            if s_out > 2 and s_out > 1.2 * s_in:
                noise = np.random.default_rng(seed).normal(0, (s_out ** 2 - s_in ** 2) ** 0.5, hole.shape)
                out[hole] += noise[hole][:, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def patched_fill(self, image, crop, local_mask, crop_box, feather=False, hole=None):
    cx1, cy1, cx2, cy2 = crop_box
    before = image[cy1:cy2, cx1:cx2].copy()
    out = _original_fill(self, image, crop, local_mask, crop_box, feather=feather, hole=hole)
    if MODE["match"] or MODE["grain"]:
        out[cy1:cy2, cx1:cx2] = adjust(before, out[cy1:cy2, cx1:cx2], local_mask > 127, cx1 * 7919 + cy1)
    return out


def set_variant(name: str) -> None:
    MODE["match"], MODE["grain"] = name == "match", name == "grain"
    side = 768 if name == "side768" else 512
    lama.DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM, lama.DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS = side, side * side


def fills(original: np.ndarray, clean: np.ndarray) -> list[dict]:
    """Each erased area with how far its edge colour jumps and how its grain compares to the surroundings."""
    changed = np.any(cv2.absdiff(original, clean) > 6, axis=2).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    lab_o = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_c = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_c = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_o, high_c = gray_o - cv2.GaussianBlur(gray_o, (0, 0), 1.2), gray_c - cv2.GaussianBlur(gray_c, (0, 0), 1.2)
    found = []
    h, w = changed.shape
    for i in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < MIN_AREA:
            continue
        x1, y1, x2, y2 = max(0, x - 20), max(0, y - 20), min(w, x + bw + 20), min(h, y + bh + 20)
        hole = labels[y1:y2, x1:x2] == i
        outside_mean, den_o = _nblur(lab_o[y1:y2, x1:x2], ~hole, 7)
        inside_mean, den_i = _nblur(lab_c[y1:y2, x1:x2], hole, 7)
        u8 = hole.astype(np.uint8)
        edge = (cv2.dilate(u8, np.ones((3, 3), np.uint8)) > 0) & ~(cv2.erode(u8, np.ones((3, 3), np.uint8)) > 0)
        edge &= (den_o > 0.2) & (den_i > 0.2)
        if edge.sum() < 20:
            continue
        seam = float(np.linalg.norm(outside_mean[edge] - inside_mean[edge], axis=1).mean())
        ring = (cv2.dilate(u8, np.ones((31, 31), np.uint8)) > 0) & ~(cv2.dilate(u8, np.ones((7, 7), np.uint8)) > 0)
        s_out = float(high_o[y1:y2, x1:x2][ring].std()) if ring.any() else 0.0
        s_in = float(high_c[y1:y2, x1:x2][hole].std())
        found.append({"box": [x1, y1, x2, y2], "area": area, "seam": round(seam, 2),
                      "grain_out": round(s_out, 2), "grain_in": round(s_in, 2)})
    return found


def summary(items: list[dict]) -> dict:
    seams = np.array([i["seam"] for i in items]) if items else np.zeros(1)
    textured = [i["grain_in"] / i["grain_out"] for i in items if i["grain_out"] > 2]
    return {"fills": len(items), "seam_median": round(float(np.median(seams)), 2),
            "seam_p90": round(float(np.percentile(seams, 90)), 2), "seam_over_6": int((seams > 6).sum()),
            "textured_fills": len(textured),
            "grain_ratio_median": round(float(np.median(textured)), 3) if textured else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chapter-id", default="c1a90003")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lama.Inpainter._lama_fill = patched_fill

    pipeline = build_processing_pipeline()
    pages = pipeline.download_chapter(args.url, args.chapter_id, workers=2)["pages"]
    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    keep_dir = Path("/tmp/fill-variants")
    report, per_variant = {"url": args.url, "slices": len(pages)}, {}
    for name in VARIANTS:
        set_variant(name)
        started = time.perf_counter()
        pipeline.process_pages(args.chapter_id, list(range(len(pages))))
        seconds = time.perf_counter() - started
        items, text_left, ghosts = [], 0, 0
        for number, page in enumerate(load_manifest_raw(args.chapter_id)["pages"]):
            if page.get("skipped") or not page.get("clean"):
                continue
            original, clean = read_image(Path(page["original"])), read_image(Path(page["clean"]))
            target = keep_dir / name / f"{number:03d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), clean)
            for item in fills(original, clean):
                items.append({"slice": number, **item})
            keep = [(int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])) for r in page_preserve_regions(page)]
            found = leftovers(original, clean, text_mask(original.shape[:2], kiuyha.text_boxes(original)), keep)
            text_left += sum(i.kind == "text" for i in found)
            ghosts += sum(i.kind == "ghost" for i in found)
        per_variant[name] = items
        report[name] = {"clean_s": round(seconds, 1), "text_left": text_left, "ghosts": ghosts, **summary(items)}
        print(name, json.dumps(report[name]), flush=True)

    # The same fills side by side, worst edges of today's clean first.
    worst = sorted((i for i in per_variant["base"] if i["grain_out"] > 1), key=lambda i: -i["seam"])[:CROPS]
    manifest_pages = load_manifest_raw(args.chapter_id)["pages"]
    for rank, item in enumerate(worst, 1):
        x1, y1, x2, y2 = item["box"]
        tiles = [read_image(Path(manifest_pages[item["slice"]]["original"]))[y1:y2, x1:x2]]
        tiles += [cv2.imread(str(keep_dir / name / f"{item['slice']:03d}.png"))[y1:y2, x1:x2] for name in VARIANTS]
        labels = ("original",) + VARIANTS
        scale = min(1.0, 320 / max(1, tiles[0].shape[0]))
        row = []
        for label, tile in zip(labels, tiles):
            tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
            tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            row.append(tile)
        cv2.imwrite(str(args.out / f"fill-{rank:02d}-slice{item['slice']:03d}.jpg"), np.concatenate(row, axis=1),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    report["worst_base_fills"] = worst
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    shutil.rmtree(keep_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
