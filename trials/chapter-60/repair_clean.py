from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.inpaint.lama_inpainter import Inpainter

CHAPTER_ID = "f1cd0121"
CHAPTER_URL = "https://asurascans.com/comics/a-cadet-becomes-a-prophet-53fc8424/chapter/60"

# Human-review decisions. Copy operations restore non-story artwork exactly from RAW.
RAW_RESTORES = [
    {"page": 0, "bbox": None, "reason": "title/scanlation credit/promo card is non-story artwork; restore whole slice"},
    {"page": 14, "bbox": [710, 1280, 900, 1340], "reason": "ASURASCANS.COM watermark over-inpainted"},
    {"page": 22, "bbox": [658, 2813, 900, 2872], "reason": "ASURASCANS.COM watermark over-inpainted"},
    {"page": 30, "bbox": [656, 425, 900, 480], "reason": "ASURASCANS.COM watermark over-inpainted"},
    {"page": 43, "bbox": [640, 1895, 815, 2010], "reason": "decorative reaction face is artwork, not translatable text"},
    {"page": 44, "bbox": [10, 2850, 155, 2960], "reason": "bottle label is decorative artwork, not story text"},
    {"page": 44, "bbox": [40, 3075, 220, 3190], "reason": "decorative reaction face is artwork, not translatable text"},
    {"page": 50, "bbox": [658, 1960, 900, 2019], "reason": "ASURASCANS.COM watermark over-inpainted"},
    {"page": 57, "bbox": [658, 3197, 900, 3256], "reason": "ASURASCANS.COM watermark over-inpainted"},
    {"page": 61, "bbox": [658, 1488, 900, 1547], "reason": "ASURASCANS.COM watermark over-inpainted"},
    {"page": 63, "bbox": [0, 1960, 900, 2400], "reason": "StoryZAK/production credits are non-story material; preserve"},
    {"page": 64, "bbox": None, "reason": "end-card ASURASCANS.COM promo is non-story material; restore whole slice"},
]

# Confirmed story narration missed by detector. Coordinates are human-reviewed on RAW.
LAMA_REPAIR = {
    "page": 55,
    "bbox": [235, 1565, 670, 1805],
    "reason": "story free-text narration miss: I RECEIVED THE RECORDING ORB YOU SENT.",
}


def sha256_image(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def page_paths(root: Path, manifest: dict, page_index: int) -> tuple[Path, Path]:
    page = manifest["pages"][page_index]
    raw = root / "raw" / "sliced" / Path(str(page["original"])).name
    clean = root / "processed" / Path(str(page["clean"])).name
    if not raw.is_file() or not clean.is_file():
        raise FileNotFoundError(f"missing page {page_index}: raw={raw} clean={clean}")
    return raw, clean


def clip_bbox(bbox: list[int] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if bbox is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty bbox after clipping: {bbox} -> {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def preserve_pre_repair(root: Path, clean: Path) -> Path:
    dst = root / "pre-repair" / clean.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(clean, dst)
    return dst


def make_triptych(raw: np.ndarray, before: np.ndarray, after: np.ndarray, bbox, label: str) -> Image.Image:
    h, w = raw.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    pad = 55
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)

    def thumb(arr: np.ndarray) -> Image.Image:
        crop = cv2.cvtColor(arr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        im = Image.fromarray(crop)
        im.thumbnail((300, 235), Image.Resampling.LANCZOS)
        return im

    a, b, c = thumb(raw), thumb(before), thumb(after)
    card = Image.new("RGB", (940, 285), "white")
    for col, im in enumerate((a, b, c)):
        xoff = col * 313 + (300 - im.width) // 2
        card.paste(im, (xoff, 30 + (235 - im.height) // 2))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    draw.text((5, 5), label[:145], fill="black", font=font)
    draw.text((125, 268), "RAW", fill="black", font=font)
    draw.text((425, 268), "BEFORE", fill="black", font=font)
    draw.text((735, 268), "AFTER", fill="black", font=font)
    return card


def make_full_pair(before: np.ndarray, after: np.ndarray, page: int) -> Image.Image:
    def thumb(arr: np.ndarray) -> Image.Image:
        im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        im.thumbnail((310, 410), Image.Resampling.LANCZOS)
        return im
    a, b = thumb(before), thumb(after)
    card = Image.new("RGB", (650, 450), "white")
    card.paste(a, ((320 - a.width)//2, 25 + (410-a.height)//2))
    card.paste(b, (330 + (310-b.width)//2, 25 + (410-b.height)//2))
    d = ImageDraw.Draw(card); f = ImageFont.load_default()
    d.text((5, 5), f"p{page:03d} full slice", fill="black", font=f)
    d.text((135, 435), "BEFORE", fill="black", font=f)
    d.text((465, 435), "REPAIRED", fill="black", font=f)
    return card


def save_vertical(cards: list[Image.Image], path: Path) -> None:
    if not cards:
        return
    width = max(c.width for c in cards)
    height = sum(c.height for c in cards)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for card in cards:
        out.paste(card, (0, y)); y += card.height
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="downloaded 02-clean-review artifact root")
    args = ap.parse_args()
    root = args.root.resolve()
    manifest_path = root / "processed" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit(f"chapter mismatch: {manifest.get('chapter_id')}")

    operations: list[dict] = []
    proof_cards: list[Image.Image] = []
    touched_before: dict[int, np.ndarray] = {}

    # Exact RAW restoration for human-classified non-story material/artwork.
    for spec in RAW_RESTORES:
        page = int(spec["page"])
        raw_path, clean_path = page_paths(root, manifest, page)
        preserve_pre_repair(root, clean_path)
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        before = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if raw is None or before is None or raw.shape != before.shape:
            raise RuntimeError(f"invalid page images for p{page:03d}")
        touched_before.setdefault(page, before.copy())
        after = before.copy()
        x1, y1, x2, y2 = clip_bbox(spec["bbox"], raw.shape[1], raw.shape[0])
        after[y1:y2, x1:x2] = raw[y1:y2, x1:x2]
        cv2.imwrite(str(clean_path), after)
        op = {
            "page_index": page,
            "operation": "restore_raw_pixels",
            "bbox": [x1, y1, x2, y2],
            "reason": spec["reason"],
            "before_sha256_pixels": sha256_image(before),
            "after_sha256_pixels": sha256_image(after),
            "raw_sha256_pixels": sha256_image(raw),
        }
        operations.append(op)
        proof_cards.append(make_triptych(raw, before, after, [x1,y1,x2,y2], f"p{page:03d} restore RAW — {spec['reason']}"))

    # One confirmed story-text miss: targeted manual mask + production LaMa only.
    spec = LAMA_REPAIR
    page = int(spec["page"])
    raw_path, clean_path = page_paths(root, manifest, page)
    preserve_pre_repair(root, clean_path)
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    before = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
    if raw is None or before is None or raw.shape != before.shape:
        raise RuntimeError(f"invalid page images for p{page:03d}")
    touched_before.setdefault(page, before.copy())
    x1, y1, x2, y2 = clip_bbox(spec["bbox"], before.shape[1], before.shape[0])
    mask = np.zeros(before.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    inpainter = Inpainter()
    after = inpainter.inpaint_mask(before.copy(), mask, force_lama=True)
    if np.array_equal(after, before):
        raise RuntimeError("p055 LaMa repair produced no pixel change")
    cv2.imwrite(str(clean_path), after)
    op = {
        "page_index": page,
        "operation": "targeted_lama_inpaint",
        "bbox": [x1, y1, x2, y2],
        "reason": spec["reason"],
        "force_lama": True,
        "before_sha256_pixels": sha256_image(before),
        "after_sha256_pixels": sha256_image(after),
        "raw_sha256_pixels": sha256_image(raw),
        "inpaint_metrics": inpainter.last_metrics(),
    }
    operations.append(op)
    proof_cards.append(make_triptych(raw, before, after, [x1,y1,x2,y2], f"p{page:03d} targeted LaMa — {spec['reason']}"))

    # Durable proof: focused operation crops plus full touched slices.
    proof = root / "repair-proof"
    proof.mkdir(parents=True, exist_ok=True)
    save_vertical(proof_cards, proof / "focused-repairs.jpg")
    full_cards: list[Image.Image] = []
    for page in sorted(touched_before):
        _, clean_path = page_paths(root, manifest, page)
        repaired = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        full_cards.append(make_full_pair(touched_before[page], repaired, page))
    save_vertical(full_cards, proof / "full-touched-slices.jpg")

    metadata = {
        "checkpoint": "02b-clean-repaired",
        "chapter_id": CHAPTER_ID,
        "source_url": CHAPTER_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": "02-clean-review-f1cd0121",
        "human_review_before_repair": "FAIL_LOCAL_REPAIR_REQUIRED",
        "repair_scope": "LOCAL_ONLY",
        "operation_count": len(operations),
        "touched_pages": sorted(touched_before),
        "operations": operations,
        "next_action": "STOP_FOR_HUMAN_REPAIR_REVIEW",
    }
    (root / "clean-repair-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "HUMAN_CHECKPOINT_02B.txt").write_text(
        "CHECKPOINT: CLEAN LOCAL REPAIR\n"
        "STOP here. Human-review focused-repairs.jpg and full-touched-slices.jpg.\n"
        "Do not start OCR until story-text blockers are zero and restored non-story artwork/credits/promo are visually accepted.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": metadata["checkpoint"],
        "operation_count": metadata["operation_count"],
        "touched_pages": metadata["touched_pages"],
        "next_action": metadata["next_action"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
