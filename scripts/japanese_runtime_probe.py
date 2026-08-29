#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from app.ocr.multi_lang_ocr import MultiLangOCR
from scripts.japanese_ocr_benchmark import (
    DATASET_LICENSE,
    DATASET_REPO,
    SAMPLES,
    _download,
    _levenshtein,
    _normalize,
    _percentile,
)


def _read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="benchmark-results/japanese-ocr/images")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    paths: dict[str, Path] = {}
    for sample_id, remote_path, _gt in SAMPLES:
        paths[sample_id] = _download(sample_id, remote_path, image_dir)

    started = time.perf_counter()
    engine = MultiLangOCR()
    facade_init_ms = (time.perf_counter() - started) * 1000.0

    rows: list[dict] = []
    latencies: list[float] = []
    edit_total = 0
    char_total = 0
    exact_count = 0
    fallback_count = 0

    for sample_id, remote_path, gt in SAMPLES:
        image = _read_rgb(paths[sample_id])
        started = time.perf_counter()
        result = engine.read_detailed(image, "ja")
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        prediction = str(result.text or "").strip()
        gt_norm = _normalize(gt)
        pred_norm = _normalize(prediction)
        edits = _levenshtein(gt_norm, pred_norm)
        cer = edits / max(1, len(gt_norm))
        exact = pred_norm == gt_norm
        used_fallback = result.model == "manga-ocr-fallback"

        edit_total += edits
        char_total += len(gt_norm)
        exact_count += int(exact)
        fallback_count += int(used_fallback)

        row = {
            "id": sample_id,
            "dataset_path": remote_path,
            "image": str(paths[sample_id]),
            "ground_truth": gt,
            "prediction": prediction,
            "ground_truth_normalized": gt_norm,
            "prediction_normalized": pred_norm,
            "edit_distance": edits,
            "cer": cer,
            "exact": exact,
            "latency_ms": latency_ms,
            "model": result.model,
            "confidence": result.confidence,
            "orientation": result.orientation,
            "region_count": result.region_count,
            "used_mangaocr_fallback": used_fallback,
        }
        rows.append(row)
        print("@@JP_RUNTIME_SAMPLE@@" + json.dumps({
            "id": sample_id,
            "gt": gt_norm,
            "pred": pred_norm,
            "cer": cer,
            "exact": exact,
            "model": result.model,
            "confidence": result.confidence,
            "orientation": result.orientation,
            "latency_ms": latency_ms,
        }, ensure_ascii=False), flush=True)

    summary = {
        "engine": "multilangocr-ppocrv6-with-ja-fallback",
        "dataset": DATASET_REPO,
        "dataset_license": DATASET_LICENSE,
        "samples": len(rows),
        "facade_init_ms": facade_init_ms,
        "aggregate_cer": edit_total / max(1, char_total),
        "exact_match_rate": exact_count / max(1, len(rows)),
        "exact_match_count": exact_count,
        "mangaocr_fallback_count": fallback_count,
        "mangaocr_fallback_rate": fallback_count / max(1, len(rows)),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "rows": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("@@JP_RUNTIME_SUMMARY@@" + json.dumps({
        key: value for key, value in summary.items() if key != "rows"
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
