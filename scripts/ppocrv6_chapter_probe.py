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


def _as_polygons(data: dict[str, Any]) -> list[np.ndarray]:
    raw = data.get("rec_polys")
    if raw is None or len(raw) == 0:
        raw = data.get("dt_polys") or []
    polygons: list[np.ndarray] = []
    for item in raw:
        arr = np.asarray(item, dtype=np.int32).reshape(-1, 2)
        if len(arr) >= 3:
            polygons.append(arr)
    return polygons


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-file", required=True)
    parser.add_argument("--output", default="benchmark-results/chapter210/ppocrv6")
    parser.add_argument("--tier", choices=("small", "medium"), default="small")
    args = parser.parse_args()

    from paddleocr import PaddleOCR
    import paddleocr

    selected = [Path(line.strip()) for line in Path(args.selected_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not selected:
        raise RuntimeError("No selected pages")

    out_root = Path(args.output)
    previews = out_root / "previews"
    crops = out_root / "low_conf_crops"
    out_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{args.tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{args.tier}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        device="cpu",
    )
    init_ms = (time.perf_counter() - started) * 1000.0

    rows: list[dict[str, Any]] = []
    all_scores: list[float] = []
    latencies: list[float] = []
    total_regions = 0

    for page_no, path in enumerate(selected):
        image = _read_image(path)
        started = time.perf_counter()
        outputs = model.predict(input=str(path))
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        texts: list[str] = []
        scores: list[float] = []
        polygons: list[np.ndarray] = []
        for result in outputs:
            data = _payload(result)
            item_texts = list(data.get("rec_texts") or [])
            item_scores = list(data.get("rec_scores") or [])
            item_polys = _as_polygons(data)
            for idx, text in enumerate(item_texts):
                text = str(text or "").strip()
                texts.append(text)
                if idx < len(item_scores):
                    try:
                        scores.append(float(item_scores[idx]))
                    except (TypeError, ValueError):
                        scores.append(float("nan"))
                else:
                    scores.append(float("nan"))
            polygons.extend(item_polys)

        finite_scores = [v for v in scores if np.isfinite(v)]
        all_scores.extend(finite_scores)
        total_regions += len(texts)

        preview = image.copy()
        low_count = 0
        for idx, poly in enumerate(polygons):
            cv2.polylines(preview, [poly], True, (0, 255, 0), 2)
            score = scores[idx] if idx < len(scores) else float("nan")
            if np.isfinite(score) and score < 0.65:
                low_count += 1
                x, y, w, h = cv2.boundingRect(poly)
                pad = 6
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
                if x2 > x1 and y2 > y1:
                    _write_image(crops / f"p{page_no:02d}_r{idx:03d}_{score:.2f}.jpg", image[y1:y2, x1:x2])

        _write_image(previews / f"page_{page_no:02d}.jpg", preview)
        row = {
            "source": str(path),
            "regions": len(texts),
            "mean_confidence": _mean(finite_scores),
            "low_confidence_lt_0_65": low_count,
            "latency_ms": latency_ms,
            "texts": texts,
            "scores": [None if not np.isfinite(v) else v for v in scores],
            "polygons": [poly.tolist() for poly in polygons],
        }
        rows.append(row)
        print("@@PPOCR_PAGE@@" + json.dumps({
            "source": path.name,
            "regions": row["regions"],
            "mean_confidence": row["mean_confidence"],
            "low_confidence_lt_0_65": low_count,
            "latency_ms": latency_ms,
            "texts": texts[:30],
        }, ensure_ascii=False), flush=True)

    summary = {
        "paddleocr_version": str(getattr(paddleocr, "__version__", "unknown")),
        "tier": args.tier,
        "init_ms": init_ms,
        "pages_tested": len(rows),
        "total_regions": total_regions,
        "mean_confidence": _mean(all_scores),
        "mean_page_latency_ms": _mean(latencies),
        "p95_page_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "rows": rows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("@@PPOCR_SUMMARY@@" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
