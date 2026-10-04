"""One-off bench: a scored second pass that erases what the detectors still read, against today's clean."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import KIUYHA_TEXT_MODEL
from app.detector import ctd_mask
from app.detector.kiuyha_detector import KiuyhaTextDetector
from app.image_io import read_image
from app.inpaint.lama_inpainter import Inpainter
from app.manifest_utils import load_manifest_raw
from app.processing_pipeline_factory import build_processing_pipeline
from app.region_policy import page_preserve_regions
from scripts.verify import GHOST, PAD_PX, leftovers, text_mask

MAX_PAIRS = 40


def residue(image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int]:
    """Pixels the letter model reads as text, and as a faint trace, inside a block."""
    x1, y1, x2, y2 = box
    p = ctd_mask.probability(np.ascontiguousarray(image[y1:y2, x1:x2]))
    return int((p > ctd_mask.THRESHOLD).sum()), int((p > GHOST).sum())


def score(image: np.ndarray, box) -> int:
    text, faint = residue(image, box)
    return 4 * text + faint


def full_mask(shape, box, local: np.ndarray, grow: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = np.zeros(shape[:2], np.uint8)
    part = local.astype(np.uint8) * 255
    if grow:
        part = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    mask[y1:y2, x1:x2] = part
    return mask


def ladder(original: np.ndarray, clean: np.ndarray, box):
    """Stronger erases to try in turn: the leftover alone, then wider, then the original letters again."""
    x1, y1, x2, y2 = box
    shape = clean.shape
    after = ctd_mask.probability(np.ascontiguousarray(clean[y1:y2, x1:x2])) > GHOST
    before = ctd_mask.probability(np.ascontiguousarray(original[y1:y2, x1:x2])) > GHOST
    yield "residue", full_mask(shape, box, after, 5), False
    yield "residue_lama", full_mask(shape, box, after, 11), True
    yield "letters_lama", full_mask(shape, box, before | after, 9), True
    grown_box, grown = ctd_mask.letter_mask(original, box)
    yield "letter_mask_wide", full_mask(shape, grown_box, grown, 15), True


def refine(inpainter: Inpainter, original: np.ndarray, clean: np.ndarray, box) -> tuple[np.ndarray, dict]:
    """Keep the attempt the letter model reads least text in; stop once it reads none."""
    best, best_score = clean, score(clean, box)
    log = {"start": best_score, "steps": []}
    for name, mask, force in ladder(original, clean, box):
        if not mask.any():
            continue
        out = inpainter.inpaint_mask(best, mask, force_lama=force)
        s = score(out, box)
        log["steps"].append({"try": name, "score": s})
        if s < best_score * 0.8:
            best, best_score = out, s
            log["kept"] = name
        if residue(best, box)[0] == 0 and best_score < 200:
            break
    log["end"] = best_score
    return best, log


def kiuyha_left(kiuyha, before: np.ndarray, after: np.ndarray) -> int:
    """Blocks of the original's text the text detector still finds on the clean slice."""
    text = text_mask(before.shape[:2], kiuyha.text_boxes(before))
    rest = text_mask(after.shape[:2], kiuyha.text_boxes(after))
    count, labels = cv2.connectedComponents(cv2.dilate(text.astype(np.uint8), np.ones((15, 15), np.uint8)))
    left = 0
    for i in range(1, count):
        block = (labels == i) & text
        if block.sum() >= 150 and (rest & block).sum() >= 0.3 * block.sum():
            left += 1
    return left


def triple(original, base, new, box, path: Path) -> None:
    x1, y1, x2, y2 = box
    h, w = original.shape[:2]
    x1, y1, x2, y2 = max(0, x1 - 40), max(0, y1 - 40), min(w, x2 + 40), min(h, y2 + 40)
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
    inpainter = Inpainter()
    totals = {"text_before": 0, "ghost_before": 0, "text_after": 0, "ghost_after": 0,
              "kiuyha_left_before": 0, "kiuyha_left_after": 0, "blocks_tried": 0, "blocks_improved": 0}
    blocks, pairs, harness_s = [], 0, 0.0
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
        letters = text_mask(before.shape[:2], kiuyha.text_boxes(before))
        found = leftovers(before, base, letters, keep)
        totals["text_before"] += sum(i.kind == "text" for i in found)
        totals["ghost_before"] += sum(i.kind == "ghost" for i in found)
        totals["kiuyha_left_before"] += kiuyha_left(kiuyha, before, base)
        new = base
        started = time.perf_counter()
        for item in found:
            x1, y1, x2, y2 = item.box
            h, w = new.shape[:2]
            box = (max(0, x1 - PAD_PX), max(0, y1 - PAD_PX), min(w, x2 + PAD_PX), min(h, y2 + PAD_PX))
            refined, log = refine(inpainter, before, new, box)
            totals["blocks_tried"] += 1
            if "kept" in log:
                totals["blocks_improved"] += 1
                new = refined
            blocks.append({"slice": number, "kind": item.kind, "box": list(box), **log})
            if pairs < MAX_PAIRS:
                pairs += 1
                triple(before, base, new, box, args.out / f"block-{pairs:02d}-slice{number:03d}.jpg")
        harness_s += time.perf_counter() - started
        after = leftovers(before, new, letters, keep)
        totals["text_after"] += sum(i.kind == "text" for i in after)
        totals["ghost_after"] += sum(i.kind == "ghost" for i in after)
        totals["kiuyha_left_after"] += kiuyha_left(kiuyha, before, new)
        changed = int(np.any(new != base, axis=2).sum())
        if changed:
            blocks.append({"slice": number, "changed_px": changed})

    report = {"url": args.url, "slices": len(pages), "clean_s": round(clean_s, 1),
              "harness_s": round(harness_s, 1), **totals, "blocks": blocks}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "blocks"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
