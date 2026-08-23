#!/usr/bin/env python3
"""CPU-only baseline runner for the production detect/box path.

This runner intentionally evaluates only independently drawn missed-GT boxes
from the reviewed GT. Reviewed detector candidates are not treated as
independent geometry ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    ba = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def match_recall(preds: list[Any], gts: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    rows = []
    matched = 0
    for gi, gt in enumerate(gts):
        best_iou = 0.0
        best_idx = None
        for pi, p in enumerate(preds):
            pb = {"x1": p.x1, "y1": p.y1, "x2": p.x2, "y2": p.y2}
            score = iou(pb, gt)
            if score > best_iou:
                best_iou = score
                best_idx = pi
        hit = best_iou >= threshold
        matched += int(hit)
        rows.append({"gt_index": gi, "best_iou": best_iou, "best_prediction_index": best_idx, "hit": hit})
    return {"gt_count": len(gts), "matched": matched, "recall": matched / len(gts) if gts else None, "rows": rows}


def resolve_image(gt_item: dict[str, Any], images_dir: Path) -> Path:
    raw = Path(gt_item["image"]["path"])
    candidates = [raw, images_dir / raw.name]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Image not found: {raw.name} under {images_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--bubble-model", type=Path, required=True)
    ap.add_argument("--text-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    os.environ["MANGA_ORT_INTRA_OP_THREADS"] = str(max(1, args.threads))
    os.environ["ENABLE_TTA"] = "1" if args.tta else "0"
    sys.path.insert(0, str(args.repo.resolve()))

    try:
        import cv2
        import numpy as np
        from app.detector.bubble_detector import YoloDetector
    except Exception as exc:
        print(f"ERROR: production detector dependencies unavailable: {exc}", file=sys.stderr)
        return 2

    with args.gt.open("r", encoding="utf-8") as f:
        gt = json.load(f)

    missing = [p for p in (args.bubble_model, args.text_model) if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: missing model: {p}", file=sys.stderr)
        return 2

    try:
        from PIL import Image
    except Exception as exc:
        print(f"ERROR: Pillow unavailable: {exc}", file=sys.stderr)
        return 2

    init_t0 = time.perf_counter()
    print(f"[INIT] Loading bubble model: {args.bubble_model}", flush=True)
    bubble_detector = YoloDetector(args.bubble_model, 0.40, use_tta=args.tta)
    bubble_init_ms = (time.perf_counter() - init_t0) * 1000.0
    print(f"[INIT] Bubble model ready: {bubble_init_ms:.1f} ms", flush=True)

    init_t1 = time.perf_counter()
    print(f"[INIT] Loading text model: {args.text_model}", flush=True)
    text_detector = YoloDetector(args.text_model, 0.20, use_tta=args.tta)
    text_init_ms = (time.perf_counter() - init_t1) * 1000.0
    print(f"[INIT] Text model ready: {text_init_ms:.1f} ms", flush=True)

    per_image = []
    bubble_lat = []
    text_lat = []

    images = gt.get("images", [])
    total_images = len(images)
    run_t0 = time.perf_counter()
    print(f"[RUN] Starting CPU baseline: {total_images} images | threads={max(1, args.threads)} | TTA={bool(args.tta)} | IoU={args.iou:.2f}", flush=True)

    for index, item in enumerate(images):
        image_path = resolve_image(item, args.images)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {image_path}")

        t0 = time.perf_counter()
        bubble_preds = bubble_detector.detect(image)
        bubble_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        text_preds = text_detector.detect(image)
        text_ms = (time.perf_counter() - t1) * 1000.0
        bubble_lat.append(bubble_ms)
        text_lat.append(text_ms)

        missed_b = item.get("missed_gt_bubbles", [])
        missed_t = item.get("missed_gt_text", [])
        bubble_match = match_recall(bubble_preds, missed_b, args.iou)
        text_match = match_recall(text_preds, missed_t, args.iou)
        elapsed_s = time.perf_counter() - run_t0
        avg_s = elapsed_s / (index + 1)
        remaining_s = avg_s * max(0, total_images - index - 1)
        bubble_hits = bubble_match["matched"]
        text_hits = text_match["matched"]
        print(
            f"[{index + 1:02d}/{total_images:02d}] {image_path.name} "
            f"({image.shape[1]}x{image.shape[0]}) | "
            f"bubble {bubble_ms:.0f} ms/{len(bubble_preds)} boxes | "
            f"text {text_ms:.0f} ms/{len(text_preds)} boxes | "
            f"missed-GT B {bubble_hits}/{bubble_match['gt_count']} "
            f"T {text_hits}/{text_match['gt_count']} | "
            f"ETA {remaining_s:.1f}s",
            flush=True,
        )
        per_image.append({
            "image": image_path.name,
            "index": index,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "bubble_prediction_count": len(bubble_preds),
            "text_prediction_count": len(text_preds),
            "missed_gt_bubbles": bubble_match,
            "missed_gt_text": text_match,
            "bubble_latency_ms": bubble_ms,
            "text_latency_ms": text_ms,
        })

    total_mb = sum(x["missed_gt_bubbles"]["gt_count"] for x in per_image)
    hit_mb = sum(x["missed_gt_bubbles"]["matched"] for x in per_image)
    total_mt = sum(x["missed_gt_text"]["gt_count"] for x in per_image)
    hit_mt = sum(x["missed_gt_text"]["matched"] for x in per_image)

    out = {
        "benchmark": "detect_box_mask_cpu_baseline",
        "corpus_images": len(per_image),
        "iou_threshold": args.iou,
        "runtime": {
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": max(1, args.threads),
            "tta": bool(args.tta),
            "bubble_model_init_ms": bubble_init_ms,
            "text_model_init_ms": text_init_ms,
            "total_run_ms": (time.perf_counter() - run_t0) * 1000.0,
        },
        "models": {
            "bubble": str(args.bubble_model),
            "text": str(args.text_model),
        },
        "missed_gt_recovery": {
            "bubble": {"gt": total_mb, "hit": hit_mb, "recall": hit_mb / total_mb if total_mb else None},
            "text": {"gt": total_mt, "hit": hit_mt, "recall": hit_mt / total_mt if total_mt else None},
        },
        "latency_ms": {
            "bubble_mean": mean(bubble_lat) if bubble_lat else None,
            "bubble_median": median(bubble_lat) if bubble_lat else None,
            "text_mean": mean(text_lat) if text_lat else None,
            "text_median": median(text_lat) if text_lat else None,
        },
        "per_image": per_image,
        "metric_scope_note": "These recall figures evaluate independently drawn missed-GT boxes only; reviewed target candidates are intentionally excluded from geometry scoring.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("[DONE] Baseline complete.", flush=True)
    print(f"[DONE] Results written to: {args.output}", flush=True)
    print(json.dumps(out["missed_gt_recovery"], indent=2), flush=True)
    print(json.dumps(out["latency_ms"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
