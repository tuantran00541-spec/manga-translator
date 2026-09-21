from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app.detector.bubble_detector import YoloDetector
from app.image_io import read_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slice-dir", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--manga-pt", required=True)
    p.add_argument("--comic-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--sample-size", type=int, default=20)
    return p.parse_args()


def _normalize_output0(output0: np.ndarray) -> np.ndarray:
    arr = np.squeeze(output0)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr


def onnx_probe(detector: YoloDetector, image: np.ndarray, conf: float) -> dict:
    started = time.perf_counter()
    blob, transform = detector._preprocess(image, offset_x=0, offset_y=0)
    outputs = detector.session.run(None, {detector.input_name: blob})
    infer_ms = (time.perf_counter() - started) * 1000.0

    arr = _normalize_output0(outputs[0])
    proto_coeffs = int(outputs[1].shape[1]) if len(outputs) > 1 and outputs[1].ndim == 4 else 0
    num_classes = max(1, int(arr.shape[1] - 4 - proto_coeffs)) if arr.ndim == 2 else 0

    raw_text_scores = np.zeros((0,), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] and num_classes > 1:
        scores = arr[:, 4 : 4 + num_classes].astype(np.float32, copy=False)
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), class_ids]
        raw_text_scores = confs[class_ids == 1]

    raw_ge_conf = int(np.count_nonzero(raw_text_scores >= conf))
    raw_ge_025 = int(np.count_nonzero(raw_text_scores >= 0.25))
    raw_max = float(raw_text_scores.max()) if raw_text_scores.size else 0.0

    post_started = time.perf_counter()
    post = detector._postprocess(outputs, transform)
    semantic = [detector._with_semantics(box) for box in post]
    h, w = image.shape[:2]
    boxes = detector._filter_invalid(semantic, w, h)
    post_ms = (time.perf_counter() - post_started) * 1000.0

    text_boxes = [box for box in boxes if box.class_name == "text"]
    verified = [box for box in text_boxes if box.verified_mask]
    mask_pixels = int(
        sum(int(np.count_nonzero(box.mask > 0)) for box in verified if box.mask is not None)
    )

    return {
        "raw_text_candidates_ge_conf": raw_ge_conf,
        "raw_text_candidates_ge_025": raw_ge_025,
        "raw_text_max_conf": round(raw_max, 6),
        "post_nms_text_boxes": int(len(text_boxes)),
        "verified_mask_instances": int(len(verified)),
        "mask_pixels": mask_pixels,
        "infer_ms": round(infer_ms, 3),
        "postprocess_ms": round(post_ms, 3),
    }


def pick_evenly(items: list[dict], count: int) -> list[dict]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    idxs = np.linspace(0, len(items) - 1, count).round().astype(int)
    seen = set()
    out = []
    for idx in idxs:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(items[idx])
    return out


def select_sample(rows: list[dict], limit: int) -> list[dict]:
    limit = max(6, int(limit))
    zero = [r for r in rows if r["onnx"]["verified_mask_instances"] == 0]
    positive = [r for r in rows if r["onnx"]["verified_mask_instances"] > 0]

    decode_suspect = [
        r
        for r in zero
        if r["onnx"]["raw_text_candidates_ge_conf"] > 0
        or r["onnx"]["post_nms_text_boxes"] > 0
    ]
    decode_suspect.sort(
        key=lambda r: (
            r["onnx"]["post_nms_text_boxes"],
            r["onnx"]["raw_text_max_conf"],
            r["onnx"]["raw_text_candidates_ge_conf"],
        ),
        reverse=True,
    )
    raw_miss = [r for r in zero if r["onnx"]["raw_text_candidates_ge_conf"] == 0]

    n_decode = min(6, max(2, limit // 3))
    n_miss = min(6, max(2, limit // 3))
    chosen = decode_suspect[:n_decode] + pick_evenly(raw_miss, n_miss)

    used = {r["file"] for r in chosen}
    remaining = max(0, limit - len(chosen))
    for row in pick_evenly(positive, remaining):
        if row["file"] not in used:
            chosen.append(row)
            used.add(row["file"])

    if len(chosen) < limit:
        for row in rows:
            if row["file"] not in used:
                chosen.append(row)
                used.add(row["file"])
                if len(chosen) >= limit:
                    break
    return chosen[:limit]


def _text_class_id(names) -> int | None:
    if isinstance(names, dict):
        pairs = list(names.items())
    else:
        pairs = list(enumerate(names or []))
    preferred = ("text", "text region", "text_region", "text bubble", "text_bubble")
    lowered = [(int(k), str(v).strip().lower()) for k, v in pairs]
    for wanted in preferred:
        for idx, name in lowered:
            if name == wanted:
                return idx
    for idx, name in lowered:
        if "text" in name:
            return idx
    return 1 if len(lowered) > 1 else (lowered[0][0] if lowered else None)


def ultra_probe(model, image: np.ndarray, *, imgsz: int, conf: float, iou: float) -> dict:
    started = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        retina_masks=True,
        device="cpu",
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    text_id = _text_class_id(result.names)
    if result.boxes is None or text_id is None or len(result.boxes) == 0:
        return {
            "text_class_id": text_id,
            "instances": 0,
            "instances_ge_025": 0,
            "mask_instances": 0,
            "mask_pixels": 0,
            "max_conf": 0.0,
            "elapsed_ms": round(elapsed_ms, 3),
            "names": result.names,
        }

    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    selected = np.flatnonzero(classes == int(text_id))
    mask_instances = 0
    mask_pixels = 0
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.detach().cpu().numpy()
        for idx in selected:
            if idx < masks.shape[0] and bool(np.any(masks[idx] > 0.5)):
                mask_instances += 1
                mask_pixels += int(np.count_nonzero(masks[idx] > 0.5))

    selected_confs = confs[selected] if selected.size else np.zeros((0,), dtype=np.float32)
    return {
        "text_class_id": int(text_id),
        "instances": int(selected.size),
        "instances_ge_025": int(np.count_nonzero(selected_confs >= 0.25)),
        "mask_instances": int(mask_instances),
        "mask_pixels": int(mask_pixels),
        "max_conf": round(float(selected_confs.max()) if selected_confs.size else 0.0, 6),
        "elapsed_ms": round(elapsed_ms, 3),
        "names": result.names,
    }


def summarize_ms(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return {"median": 0.0, "total": 0.0}
    return {
        "median": round(float(statistics.median(vals)), 3),
        "total": round(float(sum(vals)), 3),
    }


def main():
    args = parse_args()
    paths = sorted(Path(args.slice_dir).glob("*.png"))
    if not paths:
        raise SystemExit("no slices found")

    detector = YoloDetector(
        args.onnx,
        args.conf,
        use_tta=False,
        model_role="manga109_yolo26_seg",
    )

    all_rows = []
    for idx, path in enumerate(paths):
        image = read_image(path)
        probe = onnx_probe(detector, image, args.conf)
        all_rows.append({"index": idx, "file": path.name, "onnx": probe})
        if idx % 15 == 0:
            print("ONNX", idx, path.name, json.dumps(probe, sort_keys=True))

    sample = select_sample(all_rows, args.sample_size)
    print("SAMPLE", [r["file"] for r in sample])

    from ultralytics import YOLO
    import ultralytics

    manga_model = YOLO(args.manga_pt, task="segment")
    comic_model = None
    comic_load_error = None
    try:
        comic_model = YOLO(args.comic_model, task="segment")
    except Exception as exc:
        comic_load_error = repr(exc)
        print("COMIC_MODEL_LOAD_ERROR", comic_load_error)

    by_file = {r["file"]: r for r in all_rows}
    sampled_rows = []

    for order, base in enumerate(sample):
        path = Path(args.slice_dir) / base["file"]
        image = read_image(path)
        row = {
            "sample_order": order,
            "index": base["index"],
            "file": base["file"],
            "onnx": base["onnx"],
        }
        row["manga_pt_1280_ref"] = ultra_probe(
            manga_model, image, imgsz=1280, conf=args.conf, iou=0.7
        )
        row["manga_pt_1024_parity"] = ultra_probe(
            manga_model, image, imgsz=1024, conf=args.conf, iou=0.30
        )
        if comic_model is not None:
            row["comic_pt_1280"] = ultra_probe(
                comic_model, image, imgsz=1280, conf=args.conf, iou=0.7
            )
        else:
            row["comic_pt_1280"] = {"error": comic_load_error}
        sampled_rows.append(row)
        by_file[base["file"]]["sample"] = {
            k: v for k, v in row.items() if k not in {"onnx", "index", "file", "sample_order"}
        }
        print("COMPARE", base["file"], json.dumps(row, sort_keys=True, default=str))

    zero_rows = [r for r in all_rows if r["onnx"]["verified_mask_instances"] == 0]
    zero_raw = [r for r in zero_rows if r["onnx"]["raw_text_candidates_ge_conf"] > 0]
    zero_post = [r for r in zero_rows if r["onnx"]["post_nms_text_boxes"] > 0]
    positives = [r for r in all_rows if r["onnx"]["verified_mask_instances"] > 0]

    sampled_zero = [r for r in sampled_rows if r["onnx"]["verified_mask_instances"] == 0]

    def recovered(field: str) -> int:
        total = 0
        for r in sampled_zero:
            value = r.get(field, {})
            if isinstance(value, dict) and int(value.get("mask_instances", 0) or 0) > 0:
                total += 1
        return total

    summary = {
        "slice_count": len(all_rows),
        "conf": args.conf,
        "onnx_all": {
            "positive_slices": len(positives),
            "zero_mask_slices": len(zero_rows),
            "zero_mask_with_raw_text_candidates": len(zero_raw),
            "zero_mask_with_post_nms_text_boxes": len(zero_post),
            "total_raw_text_candidates_ge_conf": int(
                sum(r["onnx"]["raw_text_candidates_ge_conf"] for r in all_rows)
            ),
            "total_post_nms_text_boxes": int(
                sum(r["onnx"]["post_nms_text_boxes"] for r in all_rows)
            ),
            "total_verified_mask_instances": int(
                sum(r["onnx"]["verified_mask_instances"] for r in all_rows)
            ),
            "infer_ms": summarize_ms([r["onnx"] for r in all_rows], "infer_ms"),
            "postprocess_ms": summarize_ms([r["onnx"] for r in all_rows], "postprocess_ms"),
        },
        "sample": {
            "count": len(sampled_rows),
            "onnx_zero_count": len(sampled_zero),
            "manga_pt_1280_recovers_onnx_zero": recovered("manga_pt_1280_ref"),
            "manga_pt_1024_recovers_onnx_zero": recovered("manga_pt_1024_parity"),
            "comic_pt_1280_recovers_onnx_zero": recovered("comic_pt_1280"),
            "files": [r["file"] for r in sampled_rows],
        },
        "comic_model_load_error": comic_load_error,
        "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
    }

    report = {
        "summary": summary,
        "sample_rows": sampled_rows,
        "all_onnx_rows": all_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("FINAL")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
