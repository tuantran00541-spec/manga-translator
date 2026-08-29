#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from japanese_ocr_benchmark import SAMPLES, _download, _levenshtein, _normalize, _payload, _percentile


def _box(poly: Any) -> dict[str, float]:
    pts = list(poly or [])
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    if not xs or not ys:
        return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0}
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": max(1.0, x2 - x1), "h": max(1.0, y2 - y1), "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
    low = int(pos)
    high = min(len(ordered) - 1, low + 1)
    f = pos - low
    return ordered[low] * (1.0 - f) + ordered[high] * f


def _horizontal_order(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not regions:
        return []
    heights = [r["box"]["h"] for r in regions]
    tolerance = max(8.0, _quantile(heights, 0.6) * 0.65)
    rows: list[dict[str, Any]] = []
    for region in sorted(regions, key=lambda r: (r["box"]["cy"], r["box"]["cx"])):
        best = None
        best_distance = None
        for row in rows:
            distance = abs(region["box"]["cy"] - row["cy"])
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best = row
                best_distance = distance
        if best is None:
            rows.append({"cy": region["box"]["cy"], "items": [region]})
        else:
            best["items"].append(region)
            best["cy"] = statistics.fmean(item["box"]["cy"] for item in best["items"])
    ordered: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["cy"]):
        ordered.extend(sorted(row["items"], key=lambda r: r["box"]["cx"]))
    return ordered


def _vertical_order(regions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    verticalish = [r for r in regions if r["box"]["h"] >= r["box"]["w"] * 1.10]
    widths = [r["box"]["w"] for r in verticalish] or [r["box"]["w"] for r in regions]
    main_width = _quantile(widths, 0.75)
    ruby_width_threshold = max(8.0, main_width * 0.58)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for region in regions:
        box = region["box"]
        score = region.get("score")
        # Ruby/furigana normally forms a much narrower vertical strip adjacent
        # to a full-size text column. Do not throw away short main columns: the
        # width test, not height, is the primary signal.
        ruby_like = box["h"] >= box["w"] * 1.05 and box["w"] < ruby_width_threshold
        tiny_low_conf = (
            score is not None
            and score < 0.35
            and len(str(region.get("text") or "")) <= 2
            and box["w"] < main_width * 0.85
        )
        if ruby_like or tiny_low_conf:
            removed.append(region)
        else:
            kept.append(region)

    # Be conservative: if filtering would remove nearly everything, fall back
    # to all regions rather than silently deleting real text.
    if len(kept) < max(1, len(regions) // 3):
        kept = list(regions)
        removed = []

    column_tolerance = max(10.0, main_width * 0.85)
    columns: list[dict[str, Any]] = []
    for region in sorted(kept, key=lambda r: (-r["box"]["cx"], r["box"]["cy"])):
        best = None
        best_distance = None
        for column in columns:
            distance = abs(region["box"]["cx"] - column["cx"])
            if distance <= column_tolerance and (best_distance is None or distance < best_distance):
                best = column
                best_distance = distance
        if best is None:
            columns.append({"cx": region["box"]["cx"], "items": [region]})
        else:
            best["items"].append(region)
            best["cx"] = statistics.fmean(item["box"]["cx"] for item in best["items"])

    ordered: list[dict[str, Any]] = []
    for column in sorted(columns, key=lambda item: item["cx"], reverse=True):
        ordered.extend(sorted(column["items"], key=lambda r: r["box"]["cy"]))
    return ordered, removed, ruby_width_threshold


def _reconstruct(texts: list[str], scores: list[Any], polygons: list[Any]) -> dict[str, Any]:
    count = min(len(texts), len(polygons))
    regions: list[dict[str, Any]] = []
    for idx in range(count):
        text = str(texts[idx] or "").strip()
        if not text:
            continue
        try:
            score = float(scores[idx]) if idx < len(scores) and scores[idx] is not None else None
        except (TypeError, ValueError):
            score = None
        regions.append({"index": idx, "text": text, "score": score, "box": _box(polygons[idx])})

    vertical_weight = sum(max(1, len(r["text"])) for r in regions if r["box"]["h"] >= r["box"]["w"] * 1.20)
    horizontal_weight = sum(max(1, len(r["text"])) for r in regions if r["box"]["w"] >= r["box"]["h"] * 1.20)
    is_vertical = vertical_weight > horizontal_weight

    if is_vertical:
        ordered, removed, threshold = _vertical_order(regions)
    else:
        ordered = _horizontal_order(regions)
        removed = []
        threshold = 0.0

    return {
        "orientation": "vertical" if is_vertical else "horizontal",
        "prediction": "".join(r["text"] for r in ordered),
        "ordered_indices": [r["index"] for r in ordered],
        "removed_indices": [r["index"] for r in removed],
        "ruby_width_threshold": threshold,
        "regions": regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="benchmark-results/japanese-ocr/images")
    parser.add_argument("--output", default="benchmark-results/japanese-ocr/ppocrv6-geometry.json")
    args = parser.parse_args()

    from paddleocr import PaddleOCR

    image_dir = Path(args.image_dir)
    paths: dict[str, Path] = {}
    for sample_id, remote_path, _gt in SAMPLES:
        paths[sample_id] = _download(sample_id, remote_path, image_dir)

    started = time.perf_counter()
    model = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        device="cpu",
    )
    init_ms = (time.perf_counter() - started) * 1000.0

    rows: list[dict[str, Any]] = []
    raw_edits = 0
    ordered_edits = 0
    char_total = 0
    raw_exact = 0
    ordered_exact = 0
    latencies: list[float] = []

    for sample_id, remote_path, gt in SAMPLES:
        path = paths[sample_id]
        started = time.perf_counter()
        outputs = model.predict(input=str(path))
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        texts: list[str] = []
        scores: list[Any] = []
        polygons: list[Any] = []
        for result in outputs:
            data = _payload(result)
            item_texts = list(data.get("rec_texts") or [])
            item_scores = list(data.get("rec_scores") or [])
            item_polygons = data.get("rec_polys")
            if item_polygons is None or len(item_polygons) == 0:
                item_polygons = data.get("dt_polys") or []
            item_polygons = list(item_polygons)
            n = min(len(item_texts), len(item_polygons))
            texts.extend(str(item_texts[i] or "").strip() for i in range(n))
            scores.extend(item_scores[i] if i < len(item_scores) else None for i in range(n))
            polygons.extend(item_polygons[:n])

        raw_prediction = "".join(text for text in texts if text)
        geometry = _reconstruct(texts, scores, polygons)
        ordered_prediction = geometry["prediction"]
        gt_norm = _normalize(gt)
        raw_norm = _normalize(raw_prediction)
        ordered_norm = _normalize(ordered_prediction)
        raw_distance = _levenshtein(gt_norm, raw_norm)
        ordered_distance = _levenshtein(gt_norm, ordered_norm)
        raw_cer = raw_distance / max(1, len(gt_norm))
        ordered_cer = ordered_distance / max(1, len(gt_norm))
        raw_edits += raw_distance
        ordered_edits += ordered_distance
        char_total += len(gt_norm)
        raw_exact += int(raw_norm == gt_norm)
        ordered_exact += int(ordered_norm == gt_norm)

        row = {
            "id": sample_id,
            "dataset_path": remote_path,
            "image": str(path),
            "ground_truth": gt,
            "ground_truth_normalized": gt_norm,
            "raw_prediction": raw_prediction,
            "raw_prediction_normalized": raw_norm,
            "raw_cer": raw_cer,
            "ordered_prediction": ordered_prediction,
            "ordered_prediction_normalized": ordered_norm,
            "ordered_cer": ordered_cer,
            "latency_ms": latency_ms,
            "orientation": geometry["orientation"],
            "ordered_indices": geometry["ordered_indices"],
            "removed_indices": geometry["removed_indices"],
            "ruby_width_threshold": geometry["ruby_width_threshold"],
            "regions": geometry["regions"],
        }
        rows.append(row)
        print("@@JP_ORDER_SAMPLE@@" + json.dumps({
            "id": sample_id,
            "orientation": row["orientation"],
            "gt": gt_norm,
            "raw": raw_norm,
            "ordered": ordered_norm,
            "raw_cer": raw_cer,
            "ordered_cer": ordered_cer,
            "removed": row["removed_indices"],
        }, ensure_ascii=False), flush=True)

    summary = {
        "engine": "ppocrv6-small-geometry-order",
        "samples": len(rows),
        "init_ms": init_ms,
        "raw_aggregate_cer": raw_edits / max(1, char_total),
        "ordered_aggregate_cer": ordered_edits / max(1, char_total),
        "raw_exact_match_rate": raw_exact / max(1, len(rows)),
        "ordered_exact_match_rate": ordered_exact / max(1, len(rows)),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "vertical_samples": sum(1 for row in rows if row["orientation"] == "vertical"),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("@@JP_ORDER_SUMMARY@@" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
