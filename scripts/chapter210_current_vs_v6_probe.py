#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import time
import unicodedata

import cv2
import numpy as np

from app.ocr.multi_lang_ocr import MultiLangOCR


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(path)


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def crop_polygon(image: np.ndarray, polygon, pad: int = 8) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    h, w = image.shape[:2]
    if len(pts) == 4:
        rect = order_quad(pts)
        tl, tr, br, bl = rect
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        if width >= 4 and height >= 4:
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(rect, dst)
            crop = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            return cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    return image[y1:y2, x1:x2].copy()


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    return "".join(ch for ch in value if ch.isalnum())


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    distance = previous[-1]
    return max(0.0, 1.0 - distance / max(len(a), len(b)))


def ascii_word_chars(text: str) -> int:
    return sum(ch.isascii() and ch.isalnum() for ch in str(text or ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppocr-summary", required=True)
    parser.add_argument("--output", default="benchmark-results/chapter210/current_vs_v6")
    args = parser.parse_args()

    source = json.loads(Path(args.ppocr_summary).read_text(encoding="utf-8"))
    rows = list(source.get("rows") or [])
    if not rows:
        raise RuntimeError("PP-OCRv6 summary has no rows")

    out = Path(args.output)
    disagree_dir = out / "disagreements"
    v6_wins_dir = out / "v6_likely_wins"
    current_wins_dir = out / "current_likely_wins"
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    engine = MultiLangOCR()
    # Force model init outside region timing, using a tiny benign crop.
    warm = np.full((96, 320, 3), 255, dtype=np.uint8)
    cv2.putText(warm, "warmup", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    engine.read(warm, "en")
    init_ms = (time.perf_counter() - started) * 1000.0

    comparisons = []
    latencies = []
    exact = 0
    current_blank = 0
    v6_blank = 0
    disagreements = 0
    v6_likely_wins = 0
    current_likely_wins = 0
    current_chars = 0
    v6_chars = 0
    crop_index = 0

    for page_index, row in enumerate(rows):
        page = read_image(Path(row["source"]))
        texts = list(row.get("texts") or [])
        scores = list(row.get("scores") or [])
        polygons = list(row.get("polygons") or [])
        for region_index in range(min(len(texts), len(polygons))):
            crop = crop_polygon(page, polygons[region_index])
            if crop.size == 0:
                continue
            v6_text = str(texts[region_index] or "").strip()
            score_raw = scores[region_index] if region_index < len(scores) else None
            try:
                v6_score = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                v6_score = None

            t0 = time.perf_counter()
            current_text = engine.read(crop, "en").strip()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            ncur, nv6 = norm(current_text), norm(v6_text)
            sim = similarity(current_text, v6_text)
            same = ncur == nv6
            exact += int(same)
            current_blank += int(not ncur)
            v6_blank += int(not nv6)
            current_chars += len(ncur)
            v6_chars += len(nv6)
            if not same:
                disagreements += 1

            # Without ground truth these are conservative review buckets, not correctness labels.
            likely_v6 = bool(
                nv6 and (
                    not ncur or
                    (v6_score is not None and v6_score >= 0.88 and len(nv6) >= len(ncur) + 3 and sim < 0.72)
                )
            )
            likely_current = bool(
                ncur and (
                    not nv6 or
                    ((v6_score is None or v6_score < 0.60) and len(ncur) >= len(nv6) + 3 and ascii_word_chars(current_text) >= 3)
                )
            )
            v6_likely_wins += int(likely_v6)
            current_likely_wins += int(likely_current)

            item = {
                "page_index": page_index,
                "source": row["source"],
                "region_index": region_index,
                "v6_text": v6_text,
                "v6_score": v6_score,
                "current_text": current_text,
                "normalized_similarity": sim,
                "same_normalized": same,
                "current_latency_ms": latency_ms,
                "likely_v6": likely_v6,
                "likely_current": likely_current,
                "polygon": polygons[region_index],
            }
            comparisons.append(item)

            if not same:
                if likely_v6:
                    target = v6_wins_dir
                elif likely_current:
                    target = current_wins_dir
                else:
                    target = disagree_dir
                conf = "na" if v6_score is None else f"{v6_score:.2f}"
                write_image(target / f"{crop_index:04d}_p{page_index:02d}_r{region_index:03d}_v6{conf}.jpg", crop)
            crop_index += 1

    ranked = sorted(
        (x for x in comparisons if not x["same_normalized"]),
        key=lambda x: (x["likely_v6"], x["likely_current"], 1.0 - x["normalized_similarity"], x["v6_score"] or 0.0),
        reverse=True,
    )
    summary = {
        "baseline": "production MultiLangOCR / paddleocr 2.8.1 / lang=en",
        "candidate": f"paddleocr {source.get('paddleocr_version')} PP-OCRv6_{source.get('tier')}",
        "regions_compared": len(comparisons),
        "current_init_ms": init_ms,
        "current_mean_region_latency_ms": statistics.fmean(latencies) if latencies else None,
        "current_p95_region_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "same_normalized_count": exact,
        "same_normalized_rate": exact / max(1, len(comparisons)),
        "disagreement_count": disagreements,
        "current_blank_count": current_blank,
        "v6_blank_count": v6_blank,
        "v6_likely_win_review_count": v6_likely_wins,
        "current_likely_win_review_count": current_likely_wins,
        "current_normalized_chars": current_chars,
        "v6_normalized_chars": v6_chars,
        "candidate_full_slice_mean_latency_ms": source.get("mean_page_latency_ms"),
        "candidate_full_slice_p95_latency_ms": source.get("p95_page_latency_ms"),
        "top_disagreements": ranked[:100],
        "comparisons": comparisons,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("@@CURRENT_VS_V6_SUMMARY@@" + json.dumps({k:v for k,v in summary.items() if k not in {"top_disagreements", "comparisons"}}, ensure_ascii=False), flush=True)
    print("@@CURRENT_VS_V6_TOP@@", flush=True)
    for item in ranked[:40]:
        print(json.dumps({k:item[k] for k in ["page_index","region_index","v6_text","v6_score","current_text","normalized_similarity","likely_v6","likely_current","current_latency_ms"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
