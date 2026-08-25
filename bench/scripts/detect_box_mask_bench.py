#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import BUBBLE_IOU_THRESHOLD, ENABLE_TTA, BUBBLE_CONF_THRESHOLD, TEXT_CONF_THRESHOLD
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.ort_utils import make_session


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_sha256(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def finite_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def box_dict(b: BubbleBox, img_shape: tuple[int, int, int] | tuple[int, int]) -> dict[str, Any]:
    h, w = img_shape[:2]
    bw = max(0, b.x2 - b.x1)
    bh = max(0, b.y2 - b.y1)
    area = bw * bh
    mask_area = int(np.count_nonzero(b.mask > 127)) if b.mask is not None else None
    return {
        "x1": int(b.x1), "y1": int(b.y1), "x2": int(b.x2), "y2": int(b.y2),
        "width": int(bw), "height": int(bh),
        "confidence": finite_float(b.confidence),
        "area": int(area),
        "page_area_ratio": finite_float(area / max(1, w * h)),
        "aspect_ratio": finite_float(bw / bh) if bh else None,
        "edge_contact": {
            "left": b.x1 <= 0,
            "top": b.y1 <= 0,
            "right": b.x2 >= w,
            "bottom": b.y2 >= h,
        },
        "mask_present": b.mask is not None,
        "mask_shape": list(b.mask.shape) if b.mask is not None else None,
        "mask_area": mask_area,
        "mask_to_box_ratio": finite_float(mask_area / max(1, area)) if mask_area is not None else None,
    }


def summarize_boxes(boxes: list[BubbleBox], img: np.ndarray) -> dict[str, Any]:
    h, w = img.shape[:2]
    areas = []
    confs = []
    mask_ratios = []
    edge_contacts = 0
    mask_missing = 0
    invalid = 0
    for b in boxes:
        bw, bh = b.x2 - b.x1, b.y2 - b.y1
        if bw <= 0 or bh <= 0 or b.x1 < 0 or b.y1 < 0 or b.x2 > w or b.y2 > h:
            invalid += 1
        areas.append(bw * bh)
        confs.append(float(b.confidence))
        if b.mask is None:
            mask_missing += 1
        else:
            mask_area = int(np.count_nonzero(b.mask > 127))
            mask_ratios.append(mask_area / max(1, bw * bh))
        if b.x1 <= 0 or b.y1 <= 0 or b.x2 >= w or b.y2 >= h:
            edge_contacts += 1

    def stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "mean": None, "median": None, "max": None}
        a = np.asarray(values, dtype=np.float64)
        return {"min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max())}

    return {
        "count": len(boxes),
        "confidence": stats(confs),
        "box_area_px": stats([float(x) for x in areas]),
        "box_area_page_ratio": stats([float(x) / max(1, w * h) for x in areas]),
        "mask_to_box_ratio": stats(mask_ratios),
        "mask_missing_count": mask_missing,
        "edge_contact_count": edge_contacts,
        "invalid_geometry_count": invalid,
    }


def preprocess_probe(detector: YoloDetector, image: np.ndarray) -> dict[str, Any]:
    h, w = image.shape[:2]
    blob, scale, pad = detector._preprocess(image)
    if blob is None:
        return {"valid": False}
    nh, nw = int(h * scale), int(w * scale)
    return {
        "valid": True,
        "source_shape": [h, w, int(image.shape[2]) if image.ndim == 3 else 1],
        "input_shape": list(blob.shape),
        "scale": float(scale),
        "resized_shape": [nh, nw],
        "pad_x": int(pad[0]),
        "pad_y": int(pad[1]),
        "effective_source_pixels_per_input_pixel": float(1.0 / scale) if scale else None,
    }


def slice_plan(image: np.ndarray) -> list[dict[str, int]]:
    h, w = image.shape[:2]
    input_size = 1024
    overlap = 200
    if h <= input_size * 1.5:
        return [{"x": 0, "y": 0, "width": w, "height": h}]
    step = input_size - overlap
    out = []
    y = 0
    while y < h:
        sh = min(input_size, h - y)
        out.append({"x": 0, "y": y, "width": w, "height": sh})
        if y + sh >= h:
            break
        y += step
    return out


def draw_boxes(image: np.ndarray, boxes: list[BubbleBox], label: str) -> np.ndarray:
    canvas = image.copy()
    for i, b in enumerate(boxes):
        cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
        text = f"{i}:{b.confidence:.2f} {b.x2-b.x1}x{b.y2-b.y1}"
        cv2.putText(canvas, text, (max(0, b.x1), max(18, b.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_mask(image: np.ndarray, boxes: list[BubbleBox], label: str) -> np.ndarray:
    canvas = image.copy()
    overlay = np.zeros_like(canvas)
    for b in boxes:
        if b.mask is None:
            continue
        x1, y1 = max(0, b.x1), max(0, b.y1)
        x2, y2 = min(canvas.shape[1], b.x2), min(canvas.shape[0], b.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        src = b.mask[: y2 - b.y1, : x2 - b.x1]
        src = src[max(0, y1 - b.y1): max(0, y1 - b.y1) + (y2-y1),
                  max(0, x1 - b.x1): max(0, x1 - b.x1) + (x2-x1)]
        if src.shape != (y2-y1, x2-x1):
            src = cv2.resize(src, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        roi = overlay[y1:y2, x1:x2]
        roi[src > 127] = (0, 255, 255) if False else (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0)
    cv2.putText(canvas, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def concat_visuals(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        raise ValueError("No visual items")
    target_h = max(x.shape[0] for x in items)
    normalized = []
    for x in items:
        if x.shape[0] != target_h:
            scale = target_h / x.shape[0]
            x = cv2.resize(x, (int(round(x.shape[1]*scale)), target_h), interpolation=cv2.INTER_AREA)
        normalized.append(x)
    return cv2.hconcat(normalized)


def session_info(model_path: Path) -> dict[str, Any]:
    sess = make_session(model_path)
    return {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "providers": list(sess.get_providers()),
        "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in sess.get_inputs()],
        "outputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in sess.get_outputs()],
    }


def build_combined(bubble_model: Path, text_model: Path, bubble_conf: float, text_conf: float, tta: bool) -> CombinedTextDetector:
    obj = CombinedTextDetector.__new__(CombinedTextDetector)
    obj.bubble_detector = YoloDetector(bubble_model, bubble_conf, use_tta=tta)
    obj.text_detector = YoloDetector(text_model, text_conf, use_tta=tta)
    return obj


def run_one(image_path: Path, bubble_model: Path, text_model: Path, bubble_conf: float, text_conf: float, tta: bool, out_dir: Path, save_masks: bool) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]
    image_id = image_path.stem

    bubble = YoloDetector(bubble_model, bubble_conf, use_tta=tta)
    text = YoloDetector(text_model, text_conf, use_tta=tta)
    combined = build_combined(bubble_model, text_model, bubble_conf, text_conf, tta)

    t0 = time.perf_counter()
    bubble_boxes = bubble.detect(image)
    bubble_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    text_boxes = text.detect(image)
    text_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    final_boxes = combined.detect(image)
    combined_ms = (time.perf_counter() - t0) * 1000

    final_mask = np.zeros((h, w), dtype=np.uint8)
    for b in final_boxes:
        if b.mask is None:
            final_mask[max(0,b.y1):min(h,b.y2), max(0,b.x1):min(w,b.x2)] = 255
            continue
        x1, y1 = max(0, b.x1), max(0, b.y1)
        x2, y2 = min(w, b.x2), min(h, b.y2)
        if x2 > x1 and y2 > y1:
            src = b.mask[:y2-b.y1, :x2-b.x1]
            sy, sx = max(0,y1-b.y1), max(0,x1-b.x1)
            src = src[sy:sy+(y2-y1), sx:sx+(x2-x1)]
            if src.shape == (y2-y1, x2-x1):
                final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], src)

    img_out = out_dir / image_id / ("tta" if tta else "plain")
    img_out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_out / "bubble_boxes.jpg"), draw_boxes(image, bubble_boxes, f"bubble {'TTA' if tta else 'plain'}"))
    cv2.imwrite(str(img_out / "text_boxes.jpg"), draw_boxes(image, text_boxes, f"text {'TTA' if tta else 'plain'}"))
    cv2.imwrite(str(img_out / "final_boxes.jpg"), draw_boxes(image, final_boxes, f"combined {'TTA' if tta else 'plain'}"))
    cv2.imwrite(str(img_out / "final_mask_overlay.jpg"), draw_mask(image, final_boxes, f"final mask {'TTA' if tta else 'plain'}"))
    if save_masks:
        cv2.imwrite(str(img_out / "final_mask.png"), final_mask)

    visual = concat_visuals([
        draw_boxes(image, bubble_boxes, f"bubble {'TTA' if tta else 'plain'}: {len(bubble_boxes)}"),
        draw_boxes(image, text_boxes, f"text {'TTA' if tta else 'plain'}: {len(text_boxes)}"),
        draw_mask(image, final_boxes, f"final mask {'TTA' if tta else 'plain'}: {len(final_boxes)}"),
    ])
    cv2.imwrite(str(img_out / "overview.jpg"), visual)

    return {
        "image": {
            "path": str(image_path),
            "sha256": image_sha256(image),
            "shape": [h, w, int(image.shape[2])],
        },
        "settings": {
            "tta": bool(tta),
            "bubble_conf": bubble_conf,
            "text_conf": text_conf,
            "bubble_iou": BUBBLE_IOU_THRESHOLD,
            "production_enable_tta": bool(ENABLE_TTA),
        },
        "preprocess": {
            "bubble": preprocess_probe(bubble, image),
            "text": preprocess_probe(text, image),
            "slice_plan": slice_plan(image),
        },
        "timing_ms": {"bubble": bubble_ms, "text": text_ms, "combined": combined_ms},
        "bubble": {
            "summary": summarize_boxes(bubble_boxes, image),
            "boxes": [box_dict(b, image.shape) for b in bubble_boxes],
        },
        "text": {
            "summary": summarize_boxes(text_boxes, image),
            "boxes": [box_dict(b, image.shape) for b in text_boxes],
        },
        "final": {
            "summary": summarize_boxes(final_boxes, image),
            "boxes": [box_dict(b, image.shape) for b in final_boxes],
            "mask_total_area": int(np.count_nonzero(final_mask > 127)),
            "mask_page_ratio": float(np.count_nonzero(final_mask > 127) / max(1, h*w)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--bubble-model", default="models/bubble_yolo.onnx")
    ap.add_argument("--text-model", default="models/text_segmenter.onnx")
    ap.add_argument("--out", default="bench/detect_box_mask")
    ap.add_argument("--conf", type=float, default=BUBBLE_CONF_THRESHOLD)
    ap.add_argument("--text-conf", type=float, default=TEXT_CONF_THRESHOLD)
    ap.add_argument("--tta", action="store_true", help="also run TTA pass")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    bubble_model = Path(args.bubble_model).resolve()
    text_model = Path(args.text_model).resolve()
    if not bubble_model.is_file():
        print(f"ERROR: missing bubble model: {bubble_model}", file=sys.stderr)
        return 2
    if not text_model.is_file():
        print(f"ERROR: missing text model: {text_model}", file=sys.stderr)
        return 2

    images = [Path(x).resolve() for x in args.images]
    if args.max_images > 0:
        images = images[:args.max_images]
    for p in images:
        if not p.is_file():
            print(f"ERROR: missing image: {p}", file=sys.stderr)
            return 2

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "benchmark": "detect_box_mask_diagnostic_v1",
        "warning": "No ground-truth annotations: precision/recall/IoU are intentionally not claimed.",
        "repo_root": str(ROOT),
        "runtime": {"python": sys.version, "pid": os.getpid()},
        "models": {"bubble": session_info(bubble_model), "text": session_info(text_model)},
        "runs": [],
    }

    for image in images:
        print(f"[RUN] {image.name}")
        for tta in ([False, True] if args.tta else [False]):
            tag = "TTA" if tta else "plain"
            print(f"  - {tag}")
            try:
                result = run_one(image, bubble_model, text_model, args.conf, args.text_conf, tta, out_dir, args.save_masks)
                report["runs"].append(result)
                print(
                    f"    bubble={result['bubble']['summary']['count']} "
                    f"text={result['text']['summary']['count']} "
                    f"final={result['final']['summary']['count']} "
                    f"mask_ratio={result['final']['mask_page_ratio']:.4f} "
                    f"time={result['timing_ms']['combined']:.1f}ms"
                )
            except Exception as exc:
                print(f"    ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                report["runs"].append({"image": str(image), "tta": tta, "error": f"{type(exc).__name__}: {exc}"})

    (out_dir / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRESULT: {out_dir / 'result.json'}")
    return 0 if not any("error" in x for x in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
