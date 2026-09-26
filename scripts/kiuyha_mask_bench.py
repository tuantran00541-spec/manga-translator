"""How much text does each detector miss, and can Kiuyha's boxes become masks?

Runs on the same real slices as the other benches and compares text masks:

  current       the app's segmenter, whole slice in one pass
  collage       the app's segmenter, two halves side by side in one pass (default)
  kiuyha_box    Kiuyha/Manga-Bubble-YOLO (YOLO26, boxes only), the whole box erased
  kiuyha_otsu   inside each Kiuyha box: pixels far from the box's border colour
                (Otsu), minus anything touching the box edge (bubble outline, art)
  kiuyha_seg    Kiuyha boxes cut out at full size, packed into 1024x1024 sheets and
                run through the app's segmenter: pixel masks where Kiuyha found text
  collage_plus  collage, plus kiuyha_seg for the Kiuyha boxes the collage left
                less than half covered
  kiuyha_collage_box / kiuyha_collage_seg
                Kiuyha run on the two halves side by side in one 1280 square
                (one pass, text ~0.79x on a 800 px wide slice), as box / as
                segmenter masks

Per variant, after the app's 7 px mask dilation: miss rate (reference text
blocks less than half covered), stray area (mask farther than 20 px from
reference text), seconds and forward passes.
The reference is the app's segmenter in 1.25x windows, so it favours that
model; missed-*.jpg and slice-*.jpg are there to check by eye.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

KIUYHA = "Kiuyha/Manga-Bubble-YOLO"
REFERENCE_SCALE = 1.25
BOX_PAD = 16  # 8 cut off letters at the ends of wide lines
SHEET = 1024


def kiuyha_model():
    from huggingface_hub import hf_hub_download, list_repo_files
    from ultralytics import YOLO

    files = list_repo_files(KIUYHA)
    weights = sorted((f for f in files if f.endswith(".pt")), key=lambda f: ("best" not in f.lower(), len(f)))
    print(f"{KIUYHA}: using {weights[0]}", flush=True)
    return YOLO(hf_hub_download(KIUYHA, weights[0]))


def kiuyha_boxes(model, image: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    imgsz = [int(math.ceil(h / 32) * 32), int(math.ceil(w / 32) * 32)]
    result = model.predict(image, imgsz=imgsz, conf=0.25, verbose=False)[0]
    boxes = []
    for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
        boxes.append((max(0, int(x1) - BOX_PAD), max(0, int(y1) - BOX_PAD),
                      min(w, int(math.ceil(x2)) + BOX_PAD), min(h, int(math.ceil(y2)) + BOX_PAD)))
    return boxes


def kiuyha_collage_boxes(model, image: np.ndarray, size: int = 1280) -> list[tuple[int, int, int, int]]:
    """Kiuyha on the two halves side by side in one square (the model card trains
    and recommends 1280), boxes mapped back to the slice and merged in the overlap."""
    h, w = image.shape[:2]
    gap = 16
    scale = min(1.0, (size - gap) / (2.0 * w))
    overlap = int(2 * size / scale) - h
    if overlap < 256:
        overlap = min(h, 256)
        scale = min(scale, 2.0 * size / (h + overlap))
    overlap = min(overlap, h)
    if scale < 1.2 * min(size / w, size / h):
        result = model.predict(image, imgsz=size, conf=0.25, verbose=False)[0]
        raw = [tuple(v) for v in result.boxes.xyxy.tolist()]
        return [(max(0, int(x1) - BOX_PAD), max(0, int(y1) - BOX_PAD),
                 min(w, int(math.ceil(x2)) + BOX_PAD), min(h, int(math.ceil(y2)) + BOX_PAD)) for x1, y1, x2, y2 in raw]
    half = (h + overlap + 1) // 2
    halves = [(0, half), (h - half, h)]
    canvas = np.full((size, size, 3), 114, np.uint8)
    columns = []
    x = 0
    for y0, y1 in halves:
        rw, rh = min(size - x, int(w * scale)), min(size, int((y1 - y0) * scale))
        canvas[:rh, x:x + rw] = cv2.resize(image[y0:y1], (rw, rh))
        columns.append((x, rw, rh, y0, y1))
        x += rw + gap
    result = model.predict(canvas, imgsz=size, conf=0.25, verbose=False)[0]
    boxes = []
    for cx1, cy1, cx2, cy2 in result.boxes.xyxy.tolist():
        centre = (cx1 + cx2) / 2
        px, rw, rh, y0, y1 = columns[0] if centre < columns[1][0] else columns[1]
        sx, sy = rw / w, rh / (y1 - y0)
        x1 = max(0, int((max(cx1, px) - px) / sx) - BOX_PAD)
        x2 = min(w, int(math.ceil((min(cx2, px + rw) - px) / sx)) + BOX_PAD)
        top = max(0, int(min(cy1, rh) / sy) + y0 - BOX_PAD)
        bottom = min(h, int(math.ceil(min(cy2, rh) / sy)) + y0 + BOX_PAD)
        if x2 - x1 > 4 and bottom - top > 4:
            boxes.append((x1, top, x2, bottom))
    merged: list[list[int]] = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        for kept in merged:
            ix = max(0, min(kept[2], box[2]) - max(kept[0], box[0]))
            iy = max(0, min(kept[3], box[3]) - max(kept[1], box[1]))
            if ix * iy >= 0.5 * (box[2] - box[0]) * (box[3] - box[1]):
                kept[:] = [min(kept[0], box[0]), min(kept[1], box[1]), max(kept[2], box[2]), max(kept[3], box[3])]
                break
        else:
            merged.append(list(box))
    return [tuple(b) for b in merged]


def otsu_mask(image: np.ndarray, boxes) -> np.ndarray:
    mask = np.zeros(image.shape[:2], bool)
    for x1, y1, x2, y2 in boxes:
        crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
        if min(crop.shape[:2]) < 8:
            continue
        ring = np.concatenate([crop[:3].reshape(-1, 3), crop[-3:].reshape(-1, 3),
                               crop[:, :3].reshape(-1, 3), crop[:, -3:].reshape(-1, 3)])
        diff = np.linalg.norm(crop - np.median(ring, axis=0), axis=2)
        diff = np.clip(diff * (255.0 / max(1.0, float(diff.max()))), 0, 255).astype(np.uint8)
        _, fg = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        keep = np.zeros(count, bool)
        for label in range(1, count):
            x, y, bw, bh, _area = stats[label]
            keep[label] = x > 0 and y > 0 and x + bw < fg.shape[1] and y + bh < fg.shape[0]
        # Otsu takes the letters but not their outline (a white stroke around brown
        # text sits close to a light background) and LaMa then keeps the outline
        # as a ghost. Close the gaps into word blobs and grow past the outline.
        part = keep[labels].astype(np.uint8)
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        part = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        mask[y1:y2, x1:x2] |= part
    return mask


def pack(boxes) -> list[list[tuple[tuple[int, int, int, int], float, int, int]]]:
    """Shelf-pack boxes at full size (shrunk only past 1024) into square sheets."""
    items = sorted(boxes, key=lambda b: b[3] - b[1], reverse=True)
    sheets, x, y, shelf = [[]], 0, 0, 0
    for box in items:
        bw, bh = box[2] - box[0], box[3] - box[1]
        scale = min(1.0, SHEET / max(bw, bh))
        sw, sh = max(1, int(bw * scale)), max(1, int(bh * scale))
        if x + sw > SHEET:
            x, y, shelf = 0, y + shelf + 8, 0
        if y + sh > SHEET:
            sheets.append([])
            x, y, shelf = 0, 0, 0
        sheets[-1].append((box, scale, x, y))
        x, shelf = x + sw + 8, max(shelf, sh)
    return [s for s in sheets if s]


def seg_masks(detector, image: np.ndarray, boxes, letterbox: int) -> tuple[np.ndarray, int]:
    detector.tile_scale, detector.collage = 0.0, False  # each sheet is already 1024x1024
    mask = np.zeros(image.shape[:2], bool)
    sheets = pack(boxes)
    for sheet in sheets:
        canvas = np.full((SHEET, SHEET, 3), letterbox, np.uint8)
        for (x1, y1, x2, y2), scale, px, py in sheet:
            crop = image[y1:y2, x1:x2]
            sw, sh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
            canvas[py:py + sh, px:px + sw] = cv2.resize(crop, (sw, sh)) if scale != 1.0 else crop
        found, _ = detector.detect(canvas)
        for b in found:
            if b.mask is None or b.mask.shape != (b.y2 - b.y1, b.x2 - b.x1):
                continue
            full = np.zeros((SHEET, SHEET), bool)
            full[b.y1:b.y2, b.x1:b.x2] = b.mask > 127
            for (x1, y1, x2, y2), scale, px, py in sheet:
                sw, sh = max(1, int((x2 - x1) * scale)), max(1, int((y2 - y1) * scale))
                part = full[py:py + sh, px:px + sw]
                if not part.any():
                    continue
                if scale != 1.0:
                    part = cv2.resize(part.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
                mask[y1:y2, x1:x2] |= part
    return mask, len(sheets)


def uncovered(mask: np.ndarray, text: np.ndarray, box) -> bool:
    """True when ``mask`` covers less than half of the text Otsu found in ``box``
    (or, with no Otsu text, almost none of the box)."""
    x1, y1, x2, y2 = box
    inside, wanted = mask[y1:y2, x1:x2], text[y1:y2, x1:x2]
    if wanted.sum() < 30:
        return inside.mean() < 0.02
    return (inside & wanted).sum() < 0.5 * wanted.sum()


def collage_debug(seg, image: np.ndarray, block_box, tag: str, out: Path) -> dict:
    """Why did the collage miss a block the single pass caught? Raw detections at a
    low threshold, from both layouts, near the block; the collage canvas is saved."""
    h, w = image.shape[:2]
    info = {"tag": tag, "slice": [h, w], "block": list(block_box)}
    bx1, by1, bx2, by2 = block_box

    def near(boxes):
        return [{"box": [b.x1, b.y1, b.x2, b.y2], "conf": round(float(b.confidence), 3),
                 "mask_px": int(np.count_nonzero(b.mask)) if b.mask is not None else 0}
                for b in boxes if b.x2 > bx1 and b.x1 < bx2 and b.y2 > by1 and b.y1 < by2]

    outputs, transform = seg._single_forward_outputs(image)
    info["single_raw"] = near(seg._postprocess_at_threshold(outputs, transform, 0.05))
    plan = seg.collage_plan(h, w)
    info["plan"] = None if plan is None else {"scale": round(plan[0], 3), "halves": plan[1]}
    if plan is not None:
        canvas, transforms = seg._collage_canvas(image, *plan)
        outputs = seg._run_session((canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        drawn = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        info["collage_raw"] = []
        for index, t in enumerate(transforms):
            raw = seg._postprocess_at_threshold(outputs, t, 0.05)
            info["collage_raw"].append(near(raw))
            for b in raw:
                x1, y1, x2, y2 = (int(v) for v in t.canvas_box_from_page((b.x1, b.y1, b.x2, b.y2)))
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255) if b.confidence >= 0.2 else (0, 200, 255), 2)
            x1, y1, x2, y2 = (int(v) for v in t.canvas_box_from_page(block_box))
            if y2 > 0 and y1 < canvas.shape[0]:
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.imwrite(str(out / f"debug-{tag}.jpg"), drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return info


def box_mask(shape, boxes) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def overlay(image: np.ndarray, mask: np.ndarray, boxes, label: str, width: int = 300) -> np.ndarray:
    canvas = image.copy()
    canvas[mask] = (0.5 * canvas[mask] + 0.5 * np.array((0, 0, 255))).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 3)
    scale = width / canvas.shape[1]
    small = cv2.resize(canvas, (width, max(1, int(canvas.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    small = cv2.copyMakeBorder(small, 22, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(small, label[:40], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return small


def row(tiles: list[np.ndarray]) -> np.ndarray:
    height = max(t.shape[0] for t in tiles)
    return np.concatenate([cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                              value=(255, 255, 255)) for t in tiles], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--skip", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    from app.image_io import read_image
    from app.one_shot_cleanup import OneShotTextMaskDetector
    from app.parameters import DETECTOR_LETTERBOX_VALUE
    from app.processing_pipeline_factory import build_processing_pipeline

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = kiuyha_model()
    pipeline = build_processing_pipeline()
    manifest = pipeline.download_chapter(args.url, "c1ea0003", workers=2)
    pages = manifest["pages"][args.skip:args.skip + args.slices]
    seg = OneShotTextMaskDetector()

    def app_mask(image, *, scale=0.0, collage=False):
        seg.tile_scale, seg.collage = scale, collage
        boxes, metrics = seg.detect(image)
        mask = np.zeros(image.shape[:2], bool)
        for b in boxes:
            if b.mask is not None and b.mask.shape == (b.y2 - b.y1, b.x2 - b.x1):
                mask[b.y1:b.y2, b.x1:b.x2] |= b.mask > 127
        return mask, int(metrics["detector_forward_calls"])

    names = ["current", "collage", "kiuyha_box", "kiuyha_otsu", "kiuyha_seg", "collage_plus",
             "kiuyha_collage_box", "kiuyha_collage_seg"]
    totals = {n: {"seconds": 0.0, "forwards": 0, "blocks": 0, "missed": 0, "area": 0, "stray": 0} for n in names}
    missed_regions: dict[str, list] = {"collage": [], "kiuyha_seg": [], "kiuyha_collage_seg": []}
    images, debug = [], []
    for number, page in enumerate(pages, start=1):
        full = read_image(Path(page["original"]))
        core = page.get("stitch_core") or {}
        image = np.ascontiguousarray(full[int(core.get("core_y1", 0)):int(core.get("core_y2", full.shape[0]))])
        images.append(image)
        ref, _ = app_mask(image, scale=REFERENCE_SCALE)
        count, labels = cv2.connectedComponents(cv2.dilate(ref.astype(np.uint8), np.ones((15, 15), np.uint8)))
        blocks = [b for b in ((labels == i) & ref for i in range(1, count)) if b.sum() >= 150]
        near = cv2.dilate(ref.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0

        masks, seconds, forwards = {}, {}, {}
        for name, kwargs in (("current", {}), ("collage", {"collage": True})):
            started = time.perf_counter()
            masks[name], forwards[name] = app_mask(image, **kwargs)
            seconds[name] = time.perf_counter() - started

        started = time.perf_counter()
        boxes = kiuyha_boxes(model, image)
        kiuyha_s = time.perf_counter() - started
        masks["kiuyha_box"], seconds["kiuyha_box"], forwards["kiuyha_box"] = box_mask(image.shape[:2], boxes), kiuyha_s, 1
        started = time.perf_counter()
        masks["kiuyha_otsu"] = otsu_mask(image, boxes)
        seconds["kiuyha_otsu"], forwards["kiuyha_otsu"] = kiuyha_s + time.perf_counter() - started, 1
        started = time.perf_counter()
        masks["kiuyha_seg"], sheets = seg_masks(seg, image, boxes, DETECTOR_LETTERBOX_VALUE)
        seconds["kiuyha_seg"], forwards["kiuyha_seg"] = kiuyha_s + time.perf_counter() - started, 1 + sheets

        started = time.perf_counter()
        cboxes = kiuyha_collage_boxes(model, image)
        kc_s = time.perf_counter() - started
        masks["kiuyha_collage_box"], seconds["kiuyha_collage_box"], forwards["kiuyha_collage_box"] = (
            box_mask(image.shape[:2], cboxes), kc_s, 1)
        started = time.perf_counter()
        masks["kiuyha_collage_seg"], sheets = seg_masks(seg, image, cboxes, DETECTOR_LETTERBOX_VALUE)
        seconds["kiuyha_collage_seg"], forwards["kiuyha_collage_seg"] = kc_s + time.perf_counter() - started, 1 + sheets

        left = [b for b in boxes if uncovered(masks["collage"], masks["kiuyha_otsu"], b)]
        started = time.perf_counter()
        extra, sheets = seg_masks(seg, image, left, DETECTOR_LETTERBOX_VALUE) if left else (np.zeros_like(ref), 0)
        masks["collage_plus"] = masks["collage"] | extra
        seconds["collage_plus"] = seconds["collage"] + kiuyha_s + time.perf_counter() - started
        forwards["collage_plus"] = forwards["collage"] + 1 + sheets

        for index, block in enumerate(blocks):
            size = block.sum()
            if (masks["collage"] & block).sum() < 0.5 * size <= (masks["current"] & block).sum():
                ys, xs = np.nonzero(block)
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                debug.append(collage_debug(seg, image, box, f"{number:02d}-{index}", args.out))
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for name in names:
            # Judge every variant after the app's own mask dilation (ellipse 7), so
            # a stroke-tight mask is not counted as missing the blob around it.
            t, m = totals[name], cv2.dilate(masks[name].astype(np.uint8), grow) > 0
            t["seconds"] += seconds[name]
            t["forwards"] += forwards[name]
            t["area"] += int(m.sum())
            t["stray"] += int((m & ~near).sum())
            for block in blocks:
                t["blocks"] += 1
                if (m & block).sum() < 0.5 * block.sum():
                    t["missed"] += 1
                    if name in missed_regions:
                        ys, xs = np.nonzero(block)
                        missed_regions[name].append((int(block.sum()), number - 1,
                                                     (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
        tiles = [overlay(image, ref, [], "reference (1.25x windows)")]
        tiles += [overlay(image, masks[n], cboxes if n.startswith("kiuyha_collage") else
                          boxes if n.startswith("kiuyha") else [], n) for n in names]
        cv2.imwrite(str(args.out / f"slice-{number:02d}.jpg"), row(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        page["_masks"] = masks
        print(f"slice {number}/{len(pages)} done", flush=True)

    for name, regions in missed_regions.items():
        for rank, (_, index, (x1, y1, x2, y2)) in enumerate(sorted(regions, reverse=True)[:8], start=1):
            image, masks = images[index], pages[index]["_masks"]
            pad = 60
            bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
            bx2, by2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
            crop = image[by1:by2, bx1:bx2]
            tiles = [overlay(crop, np.zeros(crop.shape[:2], bool), [], "original", 360)]
            tiles += [overlay(crop, masks[n][by1:by2, bx1:bx2], [], n, 360) for n in names]
            cv2.imwrite(str(args.out / f"missed-{name}-{rank:02d}.jpg"), row(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])

    report = {"url": args.url, "slices": len(pages), "reference_scale": REFERENCE_SCALE, "variants": {},
              "collage_regressions": debug}
    for name, t in totals.items():
        report["variants"][name] = {
            "seconds": round(t["seconds"], 1), "forwards": t["forwards"],
            "blocks": t["blocks"], "missed": t["missed"],
            "miss_rate": round(t["missed"] / max(1, t["blocks"]), 3),
            "stray_area": round(t["stray"] / max(1, t["area"]), 3), "mask_px": t["area"],
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
