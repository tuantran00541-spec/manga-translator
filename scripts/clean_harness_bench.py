"""One-off bench: Kiuyha judges each erase attempt through a different odd view of the image, then a held-out set."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.inpaint.lama_inpainter import Inpainter
from app.manifest_utils import load_manifest_raw
from app.processing_pipeline_factory import build_processing_pipeline
from app.region_policy import page_preserve_regions
from scripts.verify import _blocks, text_mask

PAD = 32  # room round a block that Kiuyha sees
NEAR = 15  # repairs stay this close to the original letters
VIEWS_PER_TRY, HELD_OUT = 3, 4
MAX_PAIRS = 60


def _gamma(img, g):
    return (255 * (img / 255.0) ** g).astype(np.uint8)


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(4.0, (4, 4)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _deviation(img):
    dev = cv2.absdiff(img, cv2.medianBlur(img, 21)).max(axis=2).astype(np.float32)
    return cv2.cvtColor(255 - np.clip(dev * 4, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _saturation(img):
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    return cv2.cvtColor(255 - cv2.equalizeHist(s), cv2.COLOR_GRAY2BGR)


def _channel(img):
    c = int(np.argmax([img[..., i].std() for i in range(3)]))
    return cv2.cvtColor(cv2.equalizeHist(img[..., c]), cv2.COLOR_GRAY2BGR)


def _highpass(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.clip(128 + 6 * (g - cv2.GaussianBlur(g, (0, 0), 3)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hp, cv2.COLOR_GRAY2BGR)


def _unsharp(img):
    return cv2.addWeighted(img, 3.0, cv2.GaussianBlur(img, (0, 0), 2), -2.0, 0)


def _edges(img):
    e = cv2.dilate(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 30, 90), np.ones((2, 2), np.uint8))
    return cv2.cvtColor(255 - e, cv2.COLOR_GRAY2BGR)


# Each view returns the image Kiuyha sees and the scale back to the crop.
VIEWS = {
    "plain": lambda im: (im, 1.0),
    "clahe": lambda im: (_clahe(im), 1.0),
    "invert": lambda im: (255 - im, 1.0),
    "deviation": lambda im: (_deviation(im), 1.0),
    "dark": lambda im: (_gamma(im, 2.2), 1.0),
    "bright": lambda im: (_gamma(im, 0.45), 1.0),
    "saturation": lambda im: (_saturation(im), 1.0),
    "channel": lambda im: (_channel(im), 1.0),
    "highpass": lambda im: (_highpass(im), 1.0),
    "unsharp": lambda im: (_unsharp(im), 1.0),
    "edges": lambda im: (_edges(im), 1.0),
    "zoom_up": lambda im: (cv2.resize(im, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC), 1.6),
    "zoom_down": lambda im: (cv2.resize(im, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), 0.6),
}


class Judge:
    """Kiuyha's own boxes, without the letter model, on a view of a crop."""

    def __init__(self, kiuyha: KiuyhaTextDetector):
        self.kiuyha = kiuyha
        self.calls = 0

    def found(self, crop: np.ndarray, view: str) -> np.ndarray:
        seen, scale = VIEWS[view](np.ascontiguousarray(crop))
        self.calls += 1
        out = np.zeros(crop.shape[:2], np.float32)
        for x1, y1, x2, y2, score in self.kiuyha.detect_slice(np.ascontiguousarray(seen)):
            x1, y1, x2, y2 = (int(round(v / scale)) for v in (x1, y1, x2, y2))
            out[max(0, y1):y2, max(0, x1):x2] = np.maximum(out[max(0, y1):y2, max(0, x1):x2], float(score))
        return out

    def score(self, crop, near: np.ndarray, views) -> tuple[float, np.ndarray]:
        """Worst share of the text area some view still reads as text, and where."""
        worst, where = 0.0, np.zeros(near.shape, bool)
        for view in views:
            hit = self.found(crop, view) * near
            worst = max(worst, float(hit.sum()) / max(1, int(near.sum())))
            where |= hit > 0
        return worst, where


def candidates(original, best, near, crop_box):
    """Erases to try: what this attempt's views saw, wider, colour deviations round the letters, then all near them."""
    x1, y1, x2, y2 = crop_box
    full = np.zeros(original.shape[:2], np.uint8)

    def place(local, grow):
        mask = full.copy()
        part = local.astype(np.uint8) * 255
        if grow:
            part = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        mask[y1:y2, x1:x2] = part & (near.astype(np.uint8) * 255)
        return mask

    crop = best[y1:y2, x1:x2]
    ring = (cv2.dilate(near.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) & ~near
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = np.median(lab[ring], axis=0) if ring.any() else lab.reshape(-1, 3).mean(0)
    deviates = np.linalg.norm(lab - background, axis=2) > 18
    yield "judged", lambda where: place(where, 7), False
    yield "judged_lama", lambda where: place(where, 15), True
    yield "deviation", lambda where: place(deviates, 5), False
    yield "deviation_lama", lambda where: place(deviates, 9), True
    yield "near_lama", lambda where: place(near, 0), True


def refine(inpainter, judge, original, clean, crop_box, near, rng, pool):
    x1, y1, x2, y2 = crop_box
    best = clean
    log = {"steps": []}
    quiet = 0
    for name, build, force in candidates(original, clean, near, crop_box):
        views = rng.sample(pool, VIEWS_PER_TRY)  # a new odd view set every attempt
        before, where = judge.score(best[y1:y2, x1:x2], near, views)
        if before == 0:
            quiet += 1
            log["steps"].append({"try": name, "views": views, "skip": "clean under these views"})
            if quiet >= 2:
                break
            continue
        quiet = 0
        mask = build(where)
        if not mask.any():
            continue
        out = inpainter.inpaint_mask(best, mask, force_lama=force)
        after, _ = judge.score(out[y1:y2, x1:x2], near, views)
        log["steps"].append({"try": name, "views": views, "before": round(before, 3), "after": round(after, 3)})
        if after < 0.8 * before:
            best = out
            log.setdefault("kept", []).append(name)
    return best, log


def triple(original, base, new, box, path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = []
    for label, image in (("original", original), ("today", base), ("harness", new)):
        tile = image[y1:y2, x1:x2]
        scale = min(1.0, 360 / max(1, tile.shape[0]))
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chapter-id", default="c1a90002")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = build_processing_pipeline()
    pages = pipeline.download_chapter(args.url, args.chapter_id, workers=2)["pages"]
    started = time.perf_counter()
    pipeline.process_pages(args.chapter_id, list(range(len(pages))))
    clean_s = time.perf_counter() - started

    kiuyha = KiuyhaTextDetector(KIUYHA_TEXT_MODEL)
    judge, inpainter = Judge(kiuyha), Inpainter()
    blocks, pairs, harness_s = [], 0, 0.0
    held_before = held_after = plain_before = plain_after = 0
    for number, page in enumerate(load_manifest_raw(args.chapter_id)["pages"]):
        if page.get("skipped") or not page.get("clean"):
            continue
        core = page.get("stitch_core") or {}
        original, clean = read_image(Path(page["original"])), read_image(Path(page["clean"]))
        top = int(core.get("core_y1", 0))
        bottom = min(int(core.get("core_y2", original.shape[0])), original.shape[0])
        before = np.ascontiguousarray(original[top:bottom])
        base = np.ascontiguousarray(clean[top:bottom])
        keep = [(int(r["x1"]), int(r["y1"]) - top, int(r["x2"]), int(r["y2"]) - top) for r in page_preserve_regions(page)]
        h, w = before.shape[:2]
        new = base
        started = time.perf_counter()
        for letters, (bx1, by1, bx2, by2) in _blocks(text_mask(before.shape[:2], kiuyha.text_boxes(before))):
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if any(k[0] <= cx < k[2] and k[1] <= cy < k[3] for k in keep):
                continue
            box = (max(0, bx1 - PAD), max(0, by1 - PAD), min(w, bx2 + PAD), min(h, by2 + PAD))
            x1, y1, x2, y2 = box
            local = np.zeros((y2 - y1, x2 - x1), np.uint8)
            local[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = letters
            near = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR + 1,) * 2)) > 0
            seen = judge.score(new[y1:y2, x1:x2], near, ["plain", "clahe"])[0]
            if seen < 0.02:
                continue
            rng = random.Random(hash((number, bx1, by1)) & 0xFFFF)
            held = rng.sample(list(VIEWS), HELD_OUT)
            hb = judge.score(base[y1:y2, x1:x2], near, held)[0]
            pb = judge.score(base[y1:y2, x1:x2], near, ["plain"])[0]
            pool = [v for v in VIEWS if v not in held and v != "plain"]  # the search never sees the held-out views
            new, log = refine(inpainter, judge, before, new, box, near, rng, pool)
            ha = judge.score(new[y1:y2, x1:x2], near, held)[0]
            pa = judge.score(new[y1:y2, x1:x2], near, ["plain"])[0]
            held_before += hb > 0.02
            held_after += ha > 0.02
            plain_before += pb > 0.02
            plain_after += pa > 0.02
            blocks.append({"slice": number, "box": list(box), "held_out": held, "held_before": round(hb, 3),
                           "held_after": round(ha, 3), "plain_before": round(pb, 3), "plain_after": round(pa, 3), **log})
            if pairs < MAX_PAIRS:
                pairs += 1
                triple(before, base, new, box, args.out / f"block-{pairs:02d}-slice{number:03d}.jpg")
        harness_s += time.perf_counter() - started

    report = {"url": args.url, "slices": len(pages), "clean_s": round(clean_s, 1), "harness_s": round(harness_s, 1),
              "judge_calls": judge.calls, "blocks_tried": len(blocks),
              "blocks_kept_change": sum(1 for b in blocks if b.get("kept")),
              "held_out_reads_before": held_before, "held_out_reads_after": held_after,
              "plain_reads_before": plain_before, "plain_reads_after": plain_after, "blocks": blocks}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "blocks"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
