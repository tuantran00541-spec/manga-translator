from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.pipeline import ChapterPipeline


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/detector-scheduler/report.json")
SAMPLE_COUNT = 12
PROFILES = (
    ("control_1w_2t_1s", 1, 2, 1),
    ("candidate_2w_1t_1s", 2, 1, 1),
    ("candidate_2w_1t_2s", 2, 1, 2),
)


def _mask_hash(mask) -> str | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _box_signature(boxes) -> list[tuple]:
    values = []
    for box in boxes:
        values.append(
            (
                int(box.x1), int(box.y1), int(box.x2), int(box.y2),
                round(float(box.confidence), 8),
                str(box.source_model), str(box.source_role), int(box.class_id),
                str(box.class_name), str(box.semantic_type), str(box.mask_source),
                bool(box.safe_to_inpaint), bool(box.ocr_eligible),
                bool(box.needs_review), box.deferred_reason,
                _mask_hash(box.mask),
            )
        )
    return sorted(values, key=repr)


def _pct(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return round((1.0 - after / before) * 100.0, 2)


def _set_profile_env(label: str, threads: int, streams: int) -> None:
    os.environ["MANGA_ORT_PROVIDER"] = "openvino"
    os.environ["MANGA_ORT_REQUIRE_PROVIDER"] = "1"
    os.environ["MANGA_ORT_OPENVINO_SCOPE"] = "detectors"
    os.environ["MANGA_ORT_OPENVINO_THREADS"] = str(int(threads))
    os.environ["MANGA_ORT_OPENVINO_STREAMS"] = str(int(streams))
    os.environ["MANGA_ORT_OPENVINO_CACHE_DIR"] = f"/tmp/manga-openvino-cache-{label}"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _process_one(detector, index: int, image: np.ndarray) -> tuple[int, float, list[tuple], dict]:
    started = time.perf_counter()
    boxes = detector.detect(image, parallel=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics = detector.last_metrics()
    signature = _box_signature(boxes)
    return index, elapsed_ms, signature, metrics


def _run_profile(
    label: str,
    workers: int,
    threads: int,
    streams: int,
    images: dict[int, np.ndarray],
    warmup: np.ndarray,
) -> dict:
    _set_profile_env(label, threads, streams)
    init_started = time.perf_counter()
    detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    init_ms = (time.perf_counter() - init_started) * 1000.0

    detector.detect(warmup, parallel=False)

    signatures: dict[str, list[tuple]] = {}
    page_elapsed: list[float] = []
    rows: list[dict] = []
    batch_started = time.perf_counter()

    if workers == 1:
        for index, image in images.items():
            idx, elapsed_ms, signature, metrics = _process_one(detector, index, image)
            signatures[str(idx)] = signature
            page_elapsed.append(elapsed_ms)
            rows.append(metrics)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="detector-page") as pool:
            futures = {
                pool.submit(_process_one, detector, index, image): index
                for index, image in images.items()
            }
            for future in as_completed(futures):
                idx, elapsed_ms, signature, metrics = future.result()
                signatures[str(idx)] = signature
                page_elapsed.append(elapsed_ms)
                rows.append(metrics)

    wall_ms = (time.perf_counter() - batch_started) * 1000.0

    def metric_mean(name: str) -> float:
        values = [float(row.get(name) or 0.0) for row in rows]
        return round(statistics.mean(values), 3) if values else 0.0

    result = {
        "label": label,
        "workers": int(workers),
        "threads": int(threads),
        "streams": int(streams),
        "session_init_ms": round(init_ms, 3),
        "batch_wall_ms": round(wall_ms, 3),
        "throughput_slices_per_min": round(len(images) * 60000.0 / max(1.0, wall_ms), 3),
        "page_latency_mean_ms": round(statistics.mean(page_elapsed), 3),
        "page_latency_median_ms": round(statistics.median(page_elapsed), 3),
        "metrics_mean_ms": {
            key: metric_mean(key)
            for key in (
                "bubble_model_ms",
                "text_model_ms",
                "focus_prefetch_mser_ms",
                "mser_ms",
                "text_grayscale_fallback_ms",
                "total_ms",
            )
        },
        "focus_chip_calls": int(sum(int(row.get("focus_chip_calls") or 0) for row in rows)),
        "result_boxes": int(sum(int(row.get("result_boxes") or 0) for row in rows)),
        "review_boxes": int(sum(int(row.get("review_boxes") or 0) for row in rows)),
        "box_signatures": signatures,
    }
    del detector
    gc.collect()
    return result


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"detector-scheduler-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    if total < SAMPLE_COUNT + 1:
        raise RuntimeError(f"expected >= {SAMPLE_COUNT + 1} slices, got {total}")

    indices = sorted({
        round(i * (total - 1) / (SAMPLE_COUNT - 1))
        for i in range(SAMPLE_COUNT)
    })
    warmup_index = next(i for i in range(total) if i not in set(indices))
    images = {index: read_image(Path(pages[index]["original"])) for index in indices}
    warmup = read_image(Path(pages[warmup_index]["original"]))

    profiles: dict[str, dict] = {}
    for label, workers, threads, streams in PROFILES:
        print(f"DETECTOR_SCHEDULER_START={label}", flush=True)
        profiles[label] = _run_profile(
            label,
            workers,
            threads,
            streams,
            images,
            warmup,
        )
        print("DETECTOR_SCHEDULER_PROFILE=" + json.dumps(profiles[label], ensure_ascii=False), flush=True)

    control = profiles["control_1w_2t_1s"]
    quality = {}
    speedup = {}
    for label, result in profiles.items():
        if label == "control_1w_2t_1s":
            continue
        mismatch = [
            key for key, value in control["box_signatures"].items()
            if result["box_signatures"].get(key) != value
        ]
        quality[label] = {"mismatch_pages": mismatch}
        speedup[label] = {
            "batch_wall_reduction_pct": _pct(control["batch_wall_ms"], result["batch_wall_ms"]),
            "throughput_gain_pct": round(
                (result["throughput_slices_per_min"] / max(1e-9, control["throughput_slices_per_min"]) - 1.0) * 100.0,
                2,
            ),
            "page_latency_change_pct": round(
                (result["page_latency_mean_ms"] / max(1e-9, control["page_latency_mean_ms"]) - 1.0) * 100.0,
                2,
            ),
        }

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "warmup_index": warmup_index,
        "profiles": profiles,
        "quality": quality,
        "speedup": speedup,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DETECTOR_SCHEDULER_AB=" + json.dumps(report, ensure_ascii=False), flush=True)

    mismatches = {
        label: item["mismatch_pages"]
        for label, item in quality.items()
        if item["mismatch_pages"]
    }
    if mismatches:
        raise RuntimeError(
            "detector scheduling changed geometry/masks: " + json.dumps(mismatches, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
