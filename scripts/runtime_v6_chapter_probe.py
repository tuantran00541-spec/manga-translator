#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import unicodedata
from typing import Any

import cv2
import numpy as np

from app.ocr.multi_lang_ocr import MultiLangOCR


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buf.tofile(path)


def _crop_polygon_with_context(
    image: np.ndarray,
    polygon: list[list[int]] | list[list[float]],
    *,
    pad: int,
) -> np.ndarray:
    """Approximate OCRService's context-preserving crop."""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return image[0:0, 0:0]
    height, width = image.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + box_width + pad)
    y2 = min(height, y + box_height + pad)
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2].copy()


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").upper()
    return "".join(ch for ch in value if ch.isalnum())


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _mismatch_class(reference_norm: str, runtime_norm: str) -> str:
    if reference_norm == runtime_norm:
        return "same"
    if not runtime_norm:
        return "runtime_blank"
    if reference_norm and reference_norm in runtime_norm:
        return "runtime_contains_reference_plus_noise"
    if runtime_norm and runtime_norm in reference_norm:
        return "runtime_partial_reference"
    return "different"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppocr-summary", required=True)
    parser.add_argument(
        "--output",
        default="benchmark-results/chapter210/runtime_v6",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--target-mode", choices=("all", "centered"), default="all")
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=12,
        help="Real source-image context around the reference polygon.",
    )
    args = parser.parse_args()

    source_summary = json.loads(Path(args.ppocr_summary).read_text(encoding="utf-8"))
    rows = list(source_summary.get("rows") or [])
    if not rows:
        raise RuntimeError("PP-OCRv6 summary has no rows")

    output_root = Path(args.output)
    disagreement_dir = output_root / "disagreements"
    output_root.mkdir(parents=True, exist_ok=True)

    engine = MultiLangOCR()
    comparisons: list[dict[str, Any]] = []
    latencies: list[float] = []
    confidences: list[float] = []
    quality_counts = {"good": 0, "review": 0, "reject": 0, "unknown": 0}
    quality_reasons: dict[str, int] = {}
    mismatch_counts = {
        "same": 0,
        "runtime_contains_reference_plus_noise": 0,
        "runtime_partial_reference": 0,
        "runtime_blank": 0,
        "different": 0,
    }
    runtime_blank = 0
    reference_blank = 0
    total = 0

    crop_pad = max(0, min(int(args.crop_pad), 64))

    for page_index, row in enumerate(rows):
        source = Path(str(row["source"]))
        image = _read_image(source)
        texts = list(row.get("texts") or [])
        scores = list(row.get("scores") or [])
        polygons = list(row.get("polygons") or [])
        region_count = min(len(texts), len(polygons))

        for region_index in range(region_count):
            reference_text = str(texts[region_index] or "").strip()
            reference_score = scores[region_index] if region_index < len(scores) else None
            crop = _crop_polygon_with_context(
                image,
                polygons[region_index],
                pad=crop_pad,
            )
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            result = engine.read_detailed(
                rgb,
                args.lang,
                target_mode=args.target_mode,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            if result.confidence is not None:
                confidences.append(float(result.confidence))

            quality = str(result.quality or "unknown").lower()
            if quality not in quality_counts:
                quality = "unknown"
            quality_counts[quality] += 1
            if result.quality_reason:
                reason = str(result.quality_reason)
                quality_reasons[reason] = quality_reasons.get(reason, 0) + 1

            runtime_text = str(result.text or "").strip()
            reference_norm = _normalize(reference_text)
            runtime_norm = _normalize(runtime_text)
            mismatch_class = _mismatch_class(reference_norm, runtime_norm)
            mismatch_counts[mismatch_class] += 1
            is_same = mismatch_class == "same"
            runtime_blank += int(not runtime_norm)
            reference_blank += int(not reference_norm)
            total += 1

            comparison = {
                "source": str(source),
                "page_index": page_index,
                "region_index": region_index,
                "reference_text": reference_text,
                "reference_score": reference_score,
                "runtime_text": runtime_text,
                "runtime_confidence": result.confidence,
                "runtime_model": result.model,
                "runtime_orientation": result.orientation,
                "runtime_regions": result.region_count,
                "runtime_quality": quality,
                "runtime_quality_reason": result.quality_reason,
                "same_normalized": is_same,
                "mismatch_class": mismatch_class,
                "latency_ms": latency_ms,
                "crop_pad": crop_pad,
                "target_mode": args.target_mode,
                "polygon": polygons[region_index],
            }
            comparisons.append(comparison)

            if not is_same:
                score_label = (
                    "na"
                    if result.confidence is None
                    else f"{float(result.confidence):.2f}"
                )
                _write_image(
                    disagreement_dir
                    / f"p{page_index:03d}_r{region_index:03d}_c{score_label}.jpg",
                    crop,
                )

    same = mismatch_counts["same"]
    disagreements = [item for item in comparisons if not item["same_normalized"]]
    disagreements.sort(
        key=lambda item: (
            -(float(item["runtime_confidence"]) if item["runtime_confidence"] is not None else -1.0),
            -len(str(item["reference_text"])),
        )
    )

    summary = {
        "lang": args.lang,
        "target_mode": args.target_mode,
        "crop_pad": crop_pad,
        "regions_compared": total,
        "same_normalized_count": same,
        "same_normalized_rate": same / max(1, total),
        "disagreement_count": total - same,
        "mismatch_counts": mismatch_counts,
        "runtime_blank_count": runtime_blank,
        "reference_blank_count": reference_blank,
        "quality_counts": quality_counts,
        "quality_reasons": quality_reasons,
        "reject_rate": quality_counts["reject"] / max(1, total),
        "review_rate": quality_counts["review"] / max(1, total),
        "mean_runtime_confidence": _mean(confidences),
        "mean_region_latency_ms": _mean(latencies),
        "p95_region_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "reference_full_slice_mean_latency_ms": source_summary.get("mean_page_latency_ms"),
        "reference_full_slice_p95_latency_ms": source_summary.get("p95_page_latency_ms"),
        "top_disagreements": disagreements[:80],
        "comparisons": comparisons,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("@@RUNTIME_V6_SUMMARY@@" + json.dumps({
        key: value
        for key, value in summary.items()
        if key not in {"top_disagreements", "comparisons"}
    }, ensure_ascii=False))
    print("@@RUNTIME_V6_TOP@@")
    for item in disagreements[:30]:
        print(json.dumps({
            "source": Path(item["source"]).name,
            "region": item["region_index"],
            "reference": item["reference_text"],
            "runtime": item["runtime_text"],
            "confidence": item["runtime_confidence"],
            "quality": item["runtime_quality"],
            "reason": item["runtime_quality_reason"],
            "mismatch_class": item["mismatch_class"],
            "orientation": item["runtime_orientation"],
            "latency_ms": item["latency_ms"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
