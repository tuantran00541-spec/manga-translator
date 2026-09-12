from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.inpaint.lama_inpainter import Inpainter

CHAPTER_ID = "f1cd0121"
BASE_RUN_ID = 34672445541


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def paths(root: Path, manifest: dict, page_index: int) -> tuple[Path, Path]:
    page = manifest["pages"][page_index]
    raw = root / "raw" / "sliced" / Path(str(page["original"])).name
    clean = root / "processed" / Path(str(page["clean"])).name
    if not raw.is_file() or not clean.is_file():
        raise FileNotFoundError(f"missing p{page_index:03d}: raw={raw} clean={clean}")
    return raw, clean


def smooth_background_fill(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # p036: the remaining ghost is fully on smooth near-white background and
    # stops before the black panel border. Fit a quadratic RGB surface from the
    # surrounding white pixels instead of letting an inpainter bleed the border.
    out = image.copy().astype(np.float64)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[3252:3285, 36:98] = 255
    cx1, cy1, cx2, cy2 = 12, 3195, 99, 3320
    yy, xx = np.mgrid[cy1:cy2, cx1:cx2]
    patch = image[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    local_mask = mask[cy1:cy2, cx1:cx2] > 0
    valid = (gray > 205) & (hsv[:, :, 1] < 45) & (~local_mask)
    X = xx[valid].astype(np.float64)
    Y = yy[valid].astype(np.float64)
    A = np.column_stack([np.ones_like(X), X, Y, X * X, Y * Y, X * Y])
    ys, xs = np.where(mask > 0)
    B = np.column_stack([
        np.ones_like(xs, dtype=np.float64), xs, ys,
        xs.astype(np.float64) ** 2, ys.astype(np.float64) ** 2,
        xs.astype(np.float64) * ys.astype(np.float64),
    ])
    for c in range(3):
        coef = np.linalg.lstsq(A, patch[:, :, c][valid].astype(np.float64), rcond=None)[0]
        out[ys, xs, c] = B @ coef
    return np.clip(out, 0, 255).astype(np.uint8), mask


def p38_masks(shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = shape
    masks: list[np.ndarray] = []

    upper = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(upper, [np.array([
        [708, 2434], [766, 2434], [776, 2452], [776, 2490],
        [744, 2495], [708, 2478],
    ], dtype=np.int32)], 255)
    upper = cv2.dilate(upper, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    masks.append(upper)

    lower = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lower, [np.array([
        [704, 2482], [770, 2488], [780, 2512], [775, 2532],
        [742, 2544], [670, 2554], [670, 2520],
    ], dtype=np.int32)], 255)
    lower = cv2.dilate(lower, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    masks.append(lower)
    return masks


def focused_card(raw: np.ndarray, before: np.ndarray, after: np.ndarray, page: int, bbox: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    pad = 65
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(raw.shape[1], x2 + pad); y2 = min(raw.shape[0], y2 + pad)
    ims = []
    for arr in (raw, before, after):
        im = Image.fromarray(cv2.cvtColor(arr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        im.thumbnail((350, 320), Image.Resampling.LANCZOS)
        ims.append(im)
    card = Image.new("RGB", (1080, 370), "white")
    for i, im in enumerate(ims):
        card.paste(im, (i * 360 + (350 - im.width) // 2, 25 + (320 - im.height) // 2))
    d = ImageDraw.Draw(card); f = ImageFont.load_default()
    d.text((5, 5), f"p{page:03d} 03C ghost-residue repair", fill="black", font=f)
    for x, label in ((150, "RAW"), (505, "03B"), (865, "03C")):
        d.text((x, 350), label, fill="black", font=f)
    return card


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path); args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "processed" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("chapter_id") != CHAPTER_ID:
        raise SystemExit("chapter mismatch")

    inp = Inpainter()
    ops = []
    cards = []

    # p036 deterministic smooth-background repair; RAW remains untouched.
    raw_path, clean_path = paths(root, manifest, 36)
    raw = cv2.imread(str(raw_path)); before = cv2.imread(str(clean_path))
    if raw is None or before is None or raw.shape != before.shape:
        raise RuntimeError("bad p036")
    after, mask = smooth_background_fill(before)
    cv2.imwrite(str(clean_path), after)
    ops.append({
        "page_index": 36, "operation": "quadratic_background_reconstruction",
        "bbox": [36, 3252, 98, 3285], "mask_pixels": int(np.count_nonzero(mask)),
        "reason": "remove final blue ghost beside panel border without bleeding the black structure",
        "before_sha256_pixels": sha(before), "after_sha256_pixels": sha(after), "raw_sha256_pixels": sha(raw),
    })
    cards.append(focused_card(raw, before, after, 36, (20, 3235, 115, 3295)))

    # p038: two broader human-scoped LaMa masks over the hallucinated terminal
    # strokes. The second pass sees the first repaired result, reducing copying
    # of the original SQUEEE! geometry back into the fill.
    raw_path, clean_path = paths(root, manifest, 38)
    raw = cv2.imread(str(raw_path)); before = cv2.imread(str(clean_path))
    if raw is None or before is None or raw.shape != before.shape:
        raise RuntimeError("bad p038")
    after = before.copy()
    pass_metrics = []
    total_mask = np.zeros(before.shape[:2], dtype=np.uint8)
    for idx, mask in enumerate(p38_masks(before.shape[:2]), start=1):
        total_mask = cv2.bitwise_or(total_mask, mask)
        after = inp.inpaint_mask(after, mask, force_lama=True)
        pass_metrics.append({"pass": idx, "mask_pixels": int(np.count_nonzero(mask)), "metrics": inp.last_metrics()})
    cv2.imwrite(str(clean_path), after)
    ops.append({
        "page_index": 38, "operation": "two_pass_targeted_lama",
        "bbox": [670, 2434, 780, 2554], "mask_pixels": int(np.count_nonzero(total_mask)),
        "reason": "remove 03B LaMa ghost strokes from SQUEEE! while limiting repair to the right terminal footprint",
        "passes": pass_metrics, "before_sha256_pixels": sha(before), "after_sha256_pixels": sha(after), "raw_sha256_pixels": sha(raw),
    })
    cards.append(focused_card(raw, before, after, 38, (655, 2415, 795, 2570)))

    proof = root / "late-clean-proof-03c"; proof.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1080, 370 * len(cards)), "white")
    for i, card in enumerate(cards): sheet.paste(card, (0, i * 370))
    sheet.save(proof / "focused-ghost-residue.jpg", quality=95)

    meta = {
        "checkpoint": "03c-late-clean-residue-repair", "chapter_id": CHAPTER_ID,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_run": BASE_RUN_ID,
        "repair_scope": "MICRO_LOCAL_ONLY", "raw_changed": False, "ocr_rerun_required": False,
        "touched_pages": [36, 38], "operation_count": 2, "operations": ops,
        "next_action": "STOP_FOR_HUMAN_03C_CLEAN_REVIEW",
    }
    (root / "late-clean-repair-03c.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "HUMAN_CHECKPOINT_03C.txt").write_text(
        "CHECKPOINT: 03C CLEAN GHOST-RESIDUE MICRO-REPAIR\nReview p036/p038 RAW vs 03B vs 03C. RAW was not modified; do not rerun OCR.\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
