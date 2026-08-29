#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import cv2
import numpy as np

HANGUL_RANGES = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF))


def _payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        value = dict(value)
    inner = value.get("res", value)
    return inner if isinstance(inner, dict) else value


def _contains_hangul(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in HANGUL_RANGES):
            return True
    return False


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


def _crop_polygon(image: np.ndarray, polygon: list[list[int]] | list[list[float]], pad: int = 6) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    h, w = image.shape[:2]
    if len(pts) == 4:
        rect = _order_quad(pts)
        tl, tr, br, bl = rect
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        if width >= 4 and height >= 4:
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(rect, dst)
            crop = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            if pad > 0:
                crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            return crop

    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    return image[y1:y2, x1:x2].copy()


def _recognize(model: Any, crop: np.ndarray) -> tuple[str, float | None, float]:
    started = time.perf_counter()
    output = model.predict(input=crop, batch_size=1)
    elapsed = (time.perf_counter() - started) * 1000.0
    if not output:
        return "", None, elapsed
    data = _payload(output[0])
    text = str(data.get("rec_text") or "").strip()
    score = data.get("rec_score")
    try:
        confidence = float(score) if score is not None else None
    except (TypeError, ValueError):
        confidence = None
    return text, confidence, elapsed


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppocr-summary", required=True)
    parser.add_argument("--output", default="benchmark-results/chapter210/korean_fallback")
    parser.add_argument("--v6-low-threshold", type=float, default=0.75)
    args = parser.parse_args()

    from paddleocr import TextRecognition
    import paddleocr

    source_summary = json.loads(Path(args.ppocr_summary).read_text(encoding="utf-8"))
    rows = list(source_summary.get("rows") or [])
    if not rows:
        raise RuntimeError("PP-OCRv6 summary has no rows")

    out_root = Path(args.output)
    recovered_dir = out_root / "hangul_recovered"
    disagreements_dir = out_root / "low_conf_disagreements"
    out_root.mkdir(parents=True, exist_ok=True)

    model_name = "korean_PP-OCRv5_mobile_rec"
    started = time.perf_counter()
    model = TextRecognition(model_name=model_name, device="cpu")
    init_ms = (time.perf_counter() - started) * 1000.0

    comparisons: list[dict[str, Any]] = []
    rec_latencies: list[float] = []
    ko_scores: list[float] = []
    recovered_hangul = 0
    low_conf_candidates = 0
    low_conf_improved = 0

    crop_index = 0
    for page_index, row in enumerate(rows):
        source = Path(str(row["source"]))
        image = _read_image(source)
        texts = list(row.get("texts") or [])
        scores = list(row.get("scores") or [])
        polygons = list(row.get("polygons") or [])
        region_count = min(len(texts), len(polygons))

        for region_index in range(region_count):
            v6_text = str(texts[region_index] or "").strip()
            v6_score_raw = scores[region_index] if region_index < len(scores) else None
            try:
                v6_score = float(v6_score_raw) if v6_score_raw is not None else None
            except (TypeError, ValueError):
                v6_score = None

            crop = _crop_polygon(image, polygons[region_index])
            if crop.size == 0:
                continue
            ko_text, ko_score, latency_ms = _recognize(model, crop)
            rec_latencies.append(latency_ms)
            if ko_score is not None:
                ko_scores.append(ko_score)

            has_hangul = _contains_hangul(ko_text)
            v6_low = v6_score is None or v6_score < args.v6_low_threshold
            improved = bool(v6_low and ko_score is not None and (v6_score is None or ko_score >= v6_score + 0.15))
            if v6_low:
                low_conf_candidates += 1
                if improved:
                    low_conf_improved += 1
            if has_hangul:
                recovered_hangul += 1

            item = {
                "source": str(source),
                "page_index": page_index,
                "region_index": region_index,
                "v6_text": v6_text,
                "v6_score": v6_score,
                "ko_text": ko_text,
                "ko_score": ko_score,
                "ko_contains_hangul": has_hangul,
                "v6_low_confidence": v6_low,
                "ko_confidence_improved": improved,
                "ko_latency_ms": latency_ms,
                "polygon": polygons[region_index],
            }
            comparisons.append(item)

            if has_hangul or (v6_low and ko_text != v6_text):
                label = "hangul" if has_hangul else "disagree"
                target_dir = recovered_dir if has_hangul else disagreements_dir
                score_label = "na" if ko_score is None else f"{ko_score:.2f}"
                _write_image(target_dir / f"{crop_index:04d}_{label}_ko{score_label}.jpg", crop)
            crop_index += 1

    hangul_examples = [item for item in comparisons if item["ko_contains_hangul"]]
    hangul_examples.sort(key=lambda item: item["ko_score"] if item["ko_score"] is not None else -1.0, reverse=True)
    low_disagreements = [item for item in comparisons if item["v6_low_confidence"] and item["ko_text"] != item["v6_text"]]
    low_disagreements.sort(key=lambda item: item["ko_score"] if item["ko_score"] is not None else -1.0, reverse=True)

    summary = {
        "paddleocr_version": str(getattr(paddleocr, "__version__", "unknown")),
        "model": model_name,
        "source_v6_model": f"PP-OCRv6_{source_summary.get('tier', 'unknown')}_rec",
        "init_ms": init_ms,
        "regions_rechecked": len(comparisons),
        "mean_rec_latency_ms": _mean(rec_latencies),
        "p95_rec_latency_ms": float(np.percentile(rec_latencies, 95)) if rec_latencies else None,
        "mean_ko_confidence": _mean(ko_scores),
        "v6_low_threshold": args.v6_low_threshold,
        "v6_low_conf_candidates": low_conf_candidates,
        "ko_confidence_improved_count": low_conf_improved,
        "hangul_recovered_count": recovered_hangul,
        "hangul_examples": hangul_examples[:50],
        "low_conf_disagreements": low_disagreements[:80],
        "comparisons": comparisons,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("@@KO_FALLBACK_SUMMARY@@" + json.dumps({
        key: value for key, value in summary.items() if key not in {"hangul_examples", "low_conf_disagreements", "comparisons"}
    }, ensure_ascii=False), flush=True)
    print("@@KO_HANGUL_EXAMPLES@@", flush=True)
    for item in hangul_examples[:25]:
        print(json.dumps({
            "source": Path(item["source"]).name,
            "region": item["region_index"],
            "v6_text": item["v6_text"],
            "v6_score": item["v6_score"],
            "ko_text": item["ko_text"],
            "ko_score": item["ko_score"],
            "ko_latency_ms": item["ko_latency_ms"],
        }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
