from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from app.config import BUBBLE_CONF_THRESHOLD, TEXT_CONF_THRESHOLD
from app.detector.bubble_detector import BubbleBox, YoloDetector
from app.detector.combined_detector import CombinedTextDetector
from app.detector import mask_builder
from app.inpaint import lama_inpainter
from app.inpaint.lama_inpainter import Inpainter
from app.ort_utils import make_session

INPUT_SIZE = 1024
SLICE_OVERLAP = 200


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def box_area(b: BubbleBox) -> int:
    return max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)


def mask_area(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask > 127))


def box_record(b: BubbleBox, image_shape: tuple[int, ...]) -> dict[str, Any]:
    h, w = image_shape[:2]
    area = box_area(b)
    ma = mask_area(b.mask)
    return {
        "x1": int(b.x1), "y1": int(b.y1), "x2": int(b.x2), "y2": int(b.y2),
        "width": int(max(0, b.x2 - b.x1)), "height": int(max(0, b.y2 - b.y1)),
        "confidence": float(b.confidence),
        "mask_present": b.mask is not None,
        "mask_area_px": ma,
        "mask_to_box_ratio": float(ma / area) if area else None,
        "edge": {
            "left": b.x1 <= 0, "top": b.y1 <= 0,
            "right": b.x2 >= w, "bottom": b.y2 >= h,
        },
    }


def records(boxes: list[BubbleBox], shape: tuple[int, ...]) -> list[dict[str, Any]]:
    return [box_record(b, shape) for b in boxes]


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("non-finite benchmark values")
    return {"min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max())}


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
        "pad_x": int(pad[0]), "pad_y": int(pad[1]),
    }


def slice_plan(image: np.ndarray) -> list[dict[str, int]]:
    h, w = image.shape[:2]
    if h <= INPUT_SIZE * 1.5:
        return [{"x": 0, "y": 0, "width": w, "height": h}]
    step = INPUT_SIZE - SLICE_OVERLAP
    out = []
    y = 0
    while y < h:
        sh = min(INPUT_SIZE, h - y)
        out.append({"x": 0, "y": y, "width": w, "height": sh})
        if y + sh >= h:
            break
        y += step
    return out


def iou(a: BubbleBox, b: BubbleBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    den = box_area(a) + box_area(b) - inter
    return float(inter / den) if den else 0.0


def contains_center(inner: BubbleBox, outer: BubbleBox) -> bool:
    cx = (inner.x1 + inner.x2) / 2
    cy = (inner.y1 + inner.y2) / 2
    return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2


def association_probe(bubbles: list[BubbleBox], texts: list[BubbleBox], detector: CombinedTextDetector) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    for ti, t in enumerate(texts):
        matches = [bi for bi, b in enumerate(bubbles) if detector._is_inside(t, b)]
        assignments.append({
            "text_index": ti,
            "bubble_indices": matches,
            "ambiguous": len(matches) > 1,
            "unassigned": len(matches) == 0,
        })
    return {
        "text_count": len(texts),
        "bubble_count": len(bubbles),
        "assigned_text_count": sum(bool(x["bubble_indices"]) for x in assignments),
        "unassigned_text_count": sum(x["unassigned"] for x in assignments),
        "ambiguous_text_count": sum(x["ambiguous"] for x in assignments),
        "assignments": assignments,
    }


class Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add(self, stage: str, **payload: Any) -> None:
        self.events.append({"stage": stage, **payload})


def wrap_instance_method(obj: Any, name: str, trace: Trace, stage: str):
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        payload: dict[str, Any] = {"elapsed_ms": elapsed}
        if isinstance(out, list) and all(isinstance(x, BubbleBox) for x in out):
            payload["output_count"] = len(out)
        trace.add(stage, **payload)
        return out

    setattr(obj, name, wrapped)
    return original


def run_combined_cached(combined: CombinedTextDetector, image: np.ndarray, bubble_boxes: list[BubbleBox], text_boxes: list[BubbleBox], trace: Trace) -> tuple[list[BubbleBox], float]:
    orig_b = combined.bubble_detector.detect
    orig_t = combined.text_detector.detect
    combined.bubble_detector.detect = lambda _image: bubble_boxes
    combined.text_detector.detect = lambda _image: text_boxes
    try:
        t0 = time.perf_counter()
        final = combined.detect(image)
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        combined.bubble_detector.detect = orig_b
        combined.text_detector.detect = orig_t
    trace.add("combined_postprocess_exact", elapsed_ms=elapsed, output_count=len(final))
    return final, elapsed


def instrument_mask_builder(trace: Trace) -> dict[str, Any]:
    originals = {
        "adaptive": mask_builder.adaptive_dilate_mask,
        "lama_build": lama_inpainter.build_mask,
    }

    def adaptive_wrapped(mask: np.ndarray, crop_img: np.ndarray | None = None) -> np.ndarray:
        before = mask.copy()
        t0 = time.perf_counter()
        out = originals["adaptive"](mask, crop_img)
        elapsed = (time.perf_counter() - t0) * 1000
        trace.add(
            "mask_adaptive_dilation",
            elapsed_ms=elapsed,
            input_area=mask_area(before),
            output_area=mask_area(out),
            added_area=max(0, mask_area(out) - mask_area(before)),
            kernel_effective_area_delta=max(0, mask_area(out) - mask_area(before)),
        )
        return out

    def build_wrapped(image_shape, boxes, crop_img=None):
        t0 = time.perf_counter()
        out = originals["lama_build"](image_shape, boxes, crop_img)
        elapsed = (time.perf_counter() - t0) * 1000
        fallback = sum(1 for b in boxes if b.mask is None)
        empty = sum(1 for b in boxes if b.mask is not None and mask_area(b.mask) == 0)
        trace.add(
            "mask_build_production",
            elapsed_ms=elapsed,
            box_count=len(boxes),
            mask_area=mask_area(out),
            fallback_box_count=fallback,
            empty_input_mask_count=empty,
            output_shape=list(out.shape),
        )
        return out

    mask_builder.adaptive_dilate_mask = adaptive_wrapped
    lama_inpainter.build_mask = build_wrapped
    return originals


def restore_mask_builder(originals: dict[str, Any]) -> None:
    mask_builder.adaptive_dilate_mask = originals["adaptive"]
    lama_inpainter.build_mask = originals["lama_build"]


def draw_boxes(image: np.ndarray, boxes: list[BubbleBox], label: str) -> np.ndarray:
    canvas = image.copy()
    for i, b in enumerate(boxes):
        cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{i}:{b.confidence:.2f}", (max(0, b.x1), max(18, b.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def union_box_masks(boxes: list[BubbleBox], shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = max(0, b.x1), max(0, b.y1), min(w, b.x2), min(h, b.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if b.mask is None:
            out[y1:y2, x1:x2] = 255
            continue
        sx1, sy1 = max(0, -b.x1), max(0, -b.y1)
        src = b.mask[sy1:sy1 + y2 - y1, sx1:sx1 + x2 - x1]
        if src.shape == (y2-y1, x2-x1):
            out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], src)
    return out


def instrument_stage_methods(obj: Any, trace: Trace) -> list[tuple[str, Any]]:
    names = [
        ("_cluster_free_text_boxes", "free_text_cluster"),
        ("_split_cluster_by_lines", "line_split"),
        ("_refine_and_split_tall_boxes", "tall_box_refine_split"),
        ("_apply_final_nms", "final_nms"),
        ("_merge_masks", "merge_masks"),
    ]
    originals = []
    for name, stage in names:
        if not hasattr(obj, name):
            continue
        original = getattr(obj, name)
        def make_wrapper(original_fn, stage_name):
            def wrapped(*args, **kwargs):
                t0 = time.perf_counter()
                out = original_fn(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                payload = {"elapsed_ms": elapsed}
                if isinstance(out, list):
                    payload["output_count"] = len(out)
                    if stage_name == "line_split":
                        payload["group_sizes"] = [len(x) for x in out]
                elif isinstance(out, np.ndarray):
                    payload["output_shape"] = list(out.shape)
                    payload["output_area"] = mask_area(out)
                trace.add(stage_name, **payload)
                return out
            return wrapped
        wrapper = make_wrapper(original, stage)
        setattr(obj, name, wrapper)
        originals.append((name, original))
    return originals


def restore_stage_methods(obj: Any, originals: list[tuple[str, Any]]) -> None:
    for name, original in originals:
        setattr(obj, name, original)


def instrument_inpainter_methods(obj: Inpainter, trace: Trace) -> list[tuple[str, Any]]:
    originals = []
    for name, stage in (("_cluster_boxes", "inpaint_cluster_boxes"), ("_compute_crop_region", "inpaint_crop_region")):
        original = getattr(obj, name)
        def make_wrapper(original_fn, stage_name):
            def wrapped(*args, **kwargs):
                t0 = time.perf_counter()
                out = original_fn(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                payload = {"elapsed_ms": elapsed}
                if isinstance(out, list):
                    payload["output_count"] = len(out)
                elif isinstance(out, tuple):
                    payload["output"] = list(map(int, out))
                trace.add(stage_name, **payload)
                return out
            return wrapped
        wrapper = make_wrapper(original, stage)
        setattr(obj, name, wrapper)
        originals.append((name, original))
    return originals


def run_one(image_path: Path, combined: CombinedTextDetector, inpainter: Inpainter, out_dir: Path, save_masks: bool) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]
    trace = Trace()

    t0 = time.perf_counter()
    bubble_boxes = combined.bubble_detector.detect(image)
    bubble_ms = (time.perf_counter() - t0) * 1000
    trace.add("bubble_raw", elapsed_ms=bubble_ms, output_count=len(bubble_boxes))

    t0 = time.perf_counter()
    text_boxes = combined.text_detector.detect(image)
    text_ms = (time.perf_counter() - t0) * 1000
    trace.add("text_raw", elapsed_ms=text_ms, output_count=len(text_boxes))

    assoc = association_probe(bubble_boxes, text_boxes, combined)
    combined_originals = instrument_stage_methods(combined, trace)
    try:
        final_boxes, post_ms = run_combined_cached(combined, image, bubble_boxes, text_boxes, trace)
    finally:
        restore_stage_methods(combined, combined_originals)

    originals = instrument_mask_builder(trace)
    inpainter_originals = instrument_inpainter_methods(inpainter, trace)
    try:
        original_smart = inpainter._smart_paint_region
        smart_calls = []

        def capture_smart(image_arr, local_mask, crop_box, feather=False):
            smart_calls.append({
                "crop_box": list(map(int, crop_box)),
                "mask_area": mask_area(local_mask),
                "mask_shape": list(local_mask.shape),
                "feather": bool(feather),
            })
            return image_arr

        inpainter._smart_paint_region = capture_smart
        try:
            t0 = time.perf_counter()
            _ = inpainter.inpaint(image, final_boxes)
            inpaint_mask_pipeline_ms = (time.perf_counter() - t0) * 1000
        finally:
            inpainter._smart_paint_region = original_smart
    finally:
        restore_stage_methods(inpainter, inpainter_originals)
        restore_mask_builder(originals)

    raw_final_mask = union_box_masks(final_boxes, image.shape)
    final_records = records(final_boxes, image.shape)

    invalid_mask = int(not np.isfinite(raw_final_mask).all())
    if invalid_mask:
        raise ValueError(f"non-finite final mask for {image_path}")

    img_out = out_dir / image_path.stem
    img_out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_out / "bubble_boxes.jpg"), draw_boxes(image, bubble_boxes, f"bubble: {len(bubble_boxes)}"))
    cv2.imwrite(str(img_out / "text_boxes.jpg"), draw_boxes(image, text_boxes, f"text: {len(text_boxes)}"))
    cv2.imwrite(str(img_out / "final_boxes.jpg"), draw_boxes(image, final_boxes, f"final: {len(final_boxes)}"))
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[raw_final_mask > 127] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.72, colored, 0.28, 0)
    cv2.imwrite(str(img_out / "raw_final_mask_overlay.jpg"), overlay)
    if save_masks:
        cv2.imwrite(str(img_out / "raw_final_mask.png"), raw_final_mask)

    return {
        "image": {"path": str(image_path), "sha256": sha256_file(image_path), "shape": [h, w, int(image.shape[2])]},
        "preprocess": {
            "bubble": preprocess_probe(combined.bubble_detector, image),
            "text": preprocess_probe(combined.text_detector, image),
            "slice_plan": slice_plan(image),
        },
        "timing_ms": {
            "bubble_raw": bubble_ms,
            "text_raw": text_ms,
            "combined_postprocess_exact": post_ms,
            "detector_plus_postprocess_sum": bubble_ms + text_ms + post_ms,
            "production_mask_pipeline": inpaint_mask_pipeline_ms,
        },
        "association": assoc,
        "bubble": {"count": len(bubble_boxes), "boxes": records(bubble_boxes, image.shape)},
        "text": {"count": len(text_boxes), "boxes": records(text_boxes, image.shape)},
        "final": {
            "count": len(final_boxes),
            "boxes": final_records,
            "raw_union_mask_area": mask_area(raw_final_mask),
            "raw_union_mask_page_ratio": float(mask_area(raw_final_mask) / max(1, h*w)),
        },
        "production_mask_trace": {
            "smart_paint_calls": smart_calls,
            "trace": trace.events,
        },
    }


def model_info(path: Path) -> dict[str, Any]:
    sess = make_session(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "providers": list(sess.get_providers()),
        "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in sess.get_inputs()],
        "outputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in sess.get_outputs()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--bubble-model", default="models/bubble_yolo.onnx")
    ap.add_argument("--text-model", default="models/text_segmenter.onnx")
    ap.add_argument("--out", default="bench/detect_box_mask_v2")
    ap.add_argument("--conf", type=float, default=BUBBLE_CONF_THRESHOLD)
    ap.add_argument("--text-conf", type=float, default=TEXT_CONF_THRESHOLD)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    images = sorted(Path(x).resolve() for x in args.images)
    bubble_model = Path(args.bubble_model).resolve()
    text_model = Path(args.text_model).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in images:
        if not p.is_file():
            raise SystemExit(f"ERROR: missing image: {p}")
    for p in (bubble_model, text_model):
        if not p.is_file():
            raise SystemExit(f"ERROR: missing model: {p}")

    combined = CombinedTextDetector.__new__(CombinedTextDetector)
    combined.bubble_detector = YoloDetector(bubble_model, args.conf, use_tta=args.tta)
    combined.text_detector = YoloDetector(text_model, args.text_conf, use_tta=args.tta)
    inpainter = Inpainter()

    results = []
    for p in images:
        results.append(run_one(p, combined, inpainter, out_dir, args.save_masks))
        print(f"PASS {p.name}: final={results[-1]['final']['count']} "
              f"bubble={results[-1]['bubble']['count']} text={results[-1]['text']['count']}")

    payload = {
        "schema_version": "2.0.0",
        "benchmark": "detect_box_mask_v2",
        "production_untouched": True,
        "execution_provider": list(combined.bubble_detector.session.get_providers()),
        "settings": {
            "bubble_conf": args.conf,
            "text_conf": args.text_conf,
            "tta": bool(args.tta),
            "input_size": INPUT_SIZE,
            "slice_overlap": SLICE_OVERLAP,
        },
        "models": {"bubble": model_info(bubble_model), "text": model_info(text_model)},
        "images": results,
        "summary": {
            "image_count": len(results),
            "bubble_total": sum(x["bubble"]["count"] for x in results),
            "text_total": sum(x["text"]["count"] for x in results),
            "final_total": sum(x["final"]["count"] for x in results),
            "ambiguous_text_total": sum(x["association"]["ambiguous_text_count"] for x in results),
            "unassigned_text_total": sum(x["association"]["unassigned_text_count"] for x in results),
            "final_empty_mask_count": sum(
                1 for x in results for b in x["final"]["boxes"] if b["mask_present"] and b["mask_area_px"] == 0
            ),
            "final_missing_mask_count": sum(
                1 for x in results for b in x["final"]["boxes"] if not b["mask_present"]
            ),
        },
    }
    out_json = out_dir / "result_v2.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
