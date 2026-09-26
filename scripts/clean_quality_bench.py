"""Compare text detection and LaMa fill settings on a real chapter.

Downloads and slices a chapter the way the app does, then cleans the same
slices with each variant (the app's segmenter: one pass, two halves side by
side in one pass, windows at a given scale; or Kiuyha/Manga-Bubble-YOLO boxes,
native or two halves in a 1280 square, masked by Otsu inside each box) and
measures:

  detect_s / inpaint_s   time spent in each stage
  coverage              share of reference text pixels the variant's mask covers
                        (reference: tiled detection at scale 1.25 on the original)
  over_erase            share of the variant's mask farther than 10 px from text
  residual_px           text pixels the reference detector still finds after cleaning
  left_rate             share of reference text blocks still at least 30% there
                        after cleaning (the miss rate that matters)
  sharpness             gradient energy inside the filled holes / in the ring
                        around them (1.0 = as sharp as the surroundings)

and saves side-by-side crops: text only the tiled detector caught, and the
largest filled areas.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import app.detector.mask_builder as mask_builder
from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.image_io import read_image
from app.inpaint.lama_inpainter import Inpainter
from app.one_shot_cleanup import OneShotTextMaskDetector
from app.processing_pipeline_factory import build_processing_pipeline

VARIANTS = {
    "current": {"scale": 0.0, "dilate": 7},
    "collage": {"scale": 0.0, "dilate": 7, "collage": True},
    "tile075": {"scale": 0.75, "dilate": 7},
    "kiuyha_otsu": {"scale": 0.0, "dilate": 7, "kiuyha": "native"},
    "kiuyha_collage_otsu": {"scale": 0.0, "dilate": 7, "kiuyha": "collage"},
}
REFERENCE_SCALE = 1.25


def detect(detector: OneShotTextMaskDetector, image: np.ndarray, core: tuple[int, int], scale: float,
           collage: bool = False):
    detector.tile_scale = scale
    detector.collage = collage
    y1, y2 = core
    started = time.perf_counter()
    boxes, metrics = detector.detect(image[y1:y2])
    elapsed = time.perf_counter() - started
    return [replace(b, y1=b.y1 + y1, y2=b.y2 + y1) for b in boxes], elapsed, int(metrics["detector_forward_calls"])


def kiuyha_detect(model, image: np.ndarray, core: tuple[int, int], mode: str):
    """Kiuyha (YOLO26) boxes, each masked by Otsu against the box's border colour."""
    from scripts.kiuyha_mask_bench import kiuyha_boxes, kiuyha_collage_boxes, otsu_mask

    y1, y2 = core
    part = np.ascontiguousarray(image[y1:y2])
    started = time.perf_counter()
    found = kiuyha_collage_boxes(model, part) if mode == "collage" else kiuyha_boxes(model, part)
    boxes = []
    for bx1, by1, bx2, by2 in found:
        mask = otsu_mask(part, [(bx1, by1, bx2, by2)])[by1:by2, bx1:bx2]
        if mask.any():
            boxes.append(BubbleBox(bx1, by1 + y1, bx2, by2 + y1, 0.9, mask.astype(np.uint8) * 255,
                                   source_role="text_segmenter", safe_to_inpaint=True))
    return boxes, time.perf_counter() - started, 1


def text_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for b in boxes:
        if b.mask is None or b.mask.shape != (b.y2 - b.y1, b.x2 - b.x1):
            continue
        view = mask[b.y1:b.y2, b.x1:b.x2]
        np.maximum(view, b.mask[: view.shape[0], : view.shape[1]], out=view)
    return (mask > 127).astype(np.uint8) * 255


def clean(inpainter: Inpainter, image: np.ndarray, boxes, dilate: int):
    mask_builder.MASK_DILATE_KERNEL_SIZE = dilate
    mask_builder.MASK_ADAPTIVE_DILATE_KERNEL_SIZE = dilate + 2
    started = time.perf_counter()
    result = inpainter.inpaint(image, boxes)
    elapsed = time.perf_counter() - started
    hole = (np.abs(result.astype(np.int16) - image.astype(np.int16)).max(axis=2) > 0).astype(np.uint8) * 255
    return result, hole, elapsed


def sharpness(image: np.ndarray, hole: np.ndarray) -> list[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((hole > 0).astype(np.uint8))
    ratios = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 400:
            continue
        inside = labels == label
        outer = cv2.dilate(inside.astype(np.uint8), np.ones((29, 29), np.uint8)) > 0
        inner = cv2.dilate(inside.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ring = outer & ~inner & (hole == 0)
        if np.count_nonzero(ring) < 100:
            continue
        ring_energy = float(grad[ring].mean())
        if ring_energy < 4.0:  # flat paper: nothing to be sharp about
            continue
        ratios.append(float(grad[inside].mean()) / ring_energy)
    return ratios


def crop_sheet(images: list[tuple[str, np.ndarray]], box: tuple[int, int, int, int], path: Path) -> None:
    x1, y1, x2, y2 = box
    tiles = []
    for label, image in images:
        tile = image[y1:y2, x1:x2]
        scale = min(1.0, 360 / max(1, tile.shape[0]))
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), max(1, int(tile.shape[0] * scale))))
        tile = cv2.copyMakeBorder(tile, 26, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--skip", type=int, default=1, help="leading slices to skip (credits)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = build_processing_pipeline()
    manifest = pipeline.download_chapter(args.url, "c1ea0001", workers=2)
    pages = manifest["pages"][args.skip:args.skip + args.slices]
    detector = OneShotTextMaskDetector()
    from scripts.kiuyha_mask_bench import kiuyha_model
    kiuyha = kiuyha_model()
    import torch
    torch.set_num_threads(2)  # same budget as the app's OpenVINO detector threads
    inpainter = Inpainter()

    totals = {name: {"detect_s": 0.0, "inpaint_s": 0.0, "forward_calls": 0, "ref_px": 0, "covered_px": 0,
                     "mask_px": 0, "over_px": 0, "residual_px": 0, "boxes": 0, "sharpness": [],
                     "blocks": 0, "blocks_left": 0}
              for name in VARIANTS}
    caught_regions, missed_regions, fill_regions, kiuyha_missed = [], [], [], []
    for page_number, page in enumerate(pages):
        image = read_image(Path(page["original"]))
        core = page.get("stitch_core") or {}
        span = (int(core.get("core_y1", 0)), int(core.get("core_y2", image.shape[0])))
        ref_boxes, _, _ = detect(detector, image, span, REFERENCE_SCALE)
        ref = text_mask(image.shape[:2], ref_boxes)
        near_ref = cv2.dilate(ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        count, labels = cv2.connectedComponents(cv2.dilate(ref, np.ones((15, 15), np.uint8)))
        blocks = [b for b in ((labels == i) & (ref > 0) for i in range(1, count)) if np.count_nonzero(b) >= 150]
        outputs = {}
        for name, variant in VARIANTS.items():
            if variant.get("kiuyha"):
                boxes, detect_s, calls = kiuyha_detect(kiuyha, image, span, variant["kiuyha"])
            else:
                boxes, detect_s, calls = detect(detector, image, span, variant["scale"], variant.get("collage", False))
            result, hole, inpaint_s = clean(inpainter, image, boxes, variant["dilate"])
            mask = build_mask(image.shape[:2], boxes, image)
            after, _, _ = detect(detector, result, span, REFERENCE_SCALE)
            t = totals[name]
            t["detect_s"] += detect_s
            t["inpaint_s"] += inpaint_s
            t["forward_calls"] += calls
            t["boxes"] += len(boxes)
            t["ref_px"] += int(np.count_nonzero(ref))
            t["covered_px"] += int(np.count_nonzero((ref > 0) & (mask > 0)))
            t["mask_px"] += int(np.count_nonzero(mask))
            t["over_px"] += int(np.count_nonzero((mask > 0) & (near_ref == 0)))
            left_text = text_mask(image.shape[:2], after)
            t["residual_px"] += int(np.count_nonzero(left_text))
            for block in blocks:
                t["blocks"] += 1
                if np.count_nonzero(left_text[block]) >= 0.3 * np.count_nonzero(block):
                    t["blocks_left"] += 1
            t["sharpness"] += sharpness(result, hole)
            outputs[name] = (result, mask)

        # Reference text the current detector leaves unmasked but the collage masks.
        missed = ((ref > 0) & (outputs["current"][1] == 0) & (outputs["collage"][1] > 0)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(missed, np.ones((15, 15), np.uint8)))
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area >= 300:
                caught_regions.append((area, page_number, (x, y, w, h)))
        # Reference text the collage still leaves unmasked.
        left = ((ref > 0) & (outputs["collage"][1] == 0)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(left, np.ones((15, 15), np.uint8)))
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area >= 300:
                missed_regions.append((area, page_number, (x, y, w, h)))
        count, _, stats, _ = cv2.connectedComponentsWithStats((outputs["collage"][1] > 0).astype(np.uint8))
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            fill_regions.append((area, page_number, (x, y, w, h)))
        left = ((ref > 0) & (outputs["kiuyha_collage_otsu"][1] == 0)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(left, np.ones((15, 15), np.uint8)))
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area >= 300:
                kiuyha_missed.append((area, page_number, (x, y, w, h)))
        page["_outputs"] = {name: result for name, (result, _) in outputs.items()}
        page["_image"] = image
        print(f"slice {page_number + 1}/{len(pages)} done", flush=True)

    def sheet(kind: str, regions: list, limit: int) -> None:
        for rank, (_, page_number, (x, y, w, h)) in enumerate(sorted(regions, reverse=True)[:limit], start=1):
            page = pages[page_number]
            image = page["_image"]
            pad = 40
            box = (max(0, x - pad), max(0, y - pad), min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad))
            crop_sheet([("original", image)] + [(name, page["_outputs"][name]) for name in VARIANTS],
                       box, args.out / f"{kind}-{rank:02d}.jpg")

    sheet("caught", caught_regions, 8)
    sheet("missed", missed_regions, 8)
    sheet("missed-kiuyha", kiuyha_missed, 8)
    sheet("fill", fill_regions, 8)

    report = {"url": args.url, "slices": len(pages), "reference_scale": REFERENCE_SCALE, "variants": {}}
    for name, t in totals.items():
        ratios = sorted(t["sharpness"])
        report["variants"][name] = {
            **VARIANTS[name],
            "detect_s": round(t["detect_s"], 1), "inpaint_s": round(t["inpaint_s"], 1),
            "forward_calls": t["forward_calls"], "boxes": t["boxes"],
            "coverage": round(t["covered_px"] / max(1, t["ref_px"]), 4),
            "over_erase": round(t["over_px"] / max(1, t["mask_px"]), 4),
            "mask_px": t["mask_px"], "residual_px": t["residual_px"],
            "blocks": t["blocks"], "blocks_left": t["blocks_left"],
            "left_rate": round(t["blocks_left"] / max(1, t["blocks"]), 3),
            "sharpness_median": round(ratios[len(ratios) // 2], 3) if ratios else None,
            "sharpness_regions": len(ratios),
        }
    report["caught_regions"] = len(caught_regions)
    report["missed_regions"] = len(missed_regions)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
