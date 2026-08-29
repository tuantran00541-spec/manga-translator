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


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def _crop_polygon(
    image: np.ndarray,
    polygon: list[list[int]] | list[list[float]],
    pad: int = 8,
) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    height, width = image.shape[:2]
    if len(points) == 4:
        rect = _order_quad(points)
        tl, tr, br, bl = rect
        crop_width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        crop_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        if crop_width >= 4 and crop_height >= 4:
            destination = np.array(
                [
                    [0, 0],
                    [crop_width - 1, 0],
                    [crop_width - 1, crop_height - 1],
                    [0, crop_height - 1],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(rect, destination)
            crop = cv2.warpPerspective(
                image,
                matrix,
                (crop_width, crop_height),
                borderMode=cv2.BORDER_REPLICATE,
            )
            if pad > 0:
                crop = cv2.copyMakeBorder(
                    crop,
                    pad,
                    pad,
                    pad,
                    pad,
                    cv2.BORDER_REPLICATE,
                )
            return crop

    x, y, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(width, x + box_width + pad), min(height, y + box_height + pad)
    return image[y1:y2, x1:x2].copy()


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").upper()
    return "".join(ch for ch in value if ch.isalnum())


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppocr-summary", required=True)
    parser.add_argument(
        "--output",
        default="benchmark-results/chapter210/runtime_v6",
    )
    parser.add_argument("--lang", default="en")
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
    same = 0
    runtime_blank = 0
    reference_blank = 0
    total = 0

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
            crop = _crop_polygon(image, polygons[region_index])
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            result = engine.read_detailed(rgb, args.lang)
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            if result.confidence is not None:
                confidences.append(float(result.confidence))

            runtime_text = str(result.text or "").strip()
            reference_norm = _normalize(reference_text)
            runtime_norm = _normalize(runtime_text)
            is_same = reference_norm == runtime_norm
            same += int(is_same)
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
                "same_normalized": is_same,
                "latency_ms": latency_ms,
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

    disagreements = [item for item in comparisons if not item["same_normalized"]]
    disagreements.sort(
        key=lambda item: (
            -(float(item["runtime_confidence"]) if item["runtime_confidence"] is not None else -1.0),
            -len(str(item["reference_text"])),
        )
    )

    summary = {
        "lang": args.lang,
        "regions_compared": total,
        "same_normalized_count": same,
        "same_normalized_rate": same / max(1, total),
        "disagreement_count": total - same,
        "runtime_blank_count": runtime_blank,
        "reference_blank_count": reference_blank,
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
            "orientation": item["runtime_orientation"],
            "latency_ms": item["latency_ms"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
