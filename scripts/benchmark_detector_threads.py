from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

import app.ort_utils as ort_utils
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.pipeline import ChapterPipeline


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/detector-thread-sweep/report.json")
SAMPLE_COUNT = 10
THREAD_PROFILES = (
    ("threads_2_current", 2, False),
    ("threads_4", 4, False),
    ("latency_auto", None, True),
    ("threads_1", 1, False),
    ("threads_2_repeat", 2, False),
)


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return round(statistics.mean(values), 3) if values else 0.0


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


def _auto_openvino_provider_options() -> dict[str, str]:
    config: dict[str, dict[str, str]] = {
        "CPU": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32",
        }
    }
    cache_dir = os.environ.get("MANGA_ORT_OPENVINO_CACHE_DIR", "").strip()
    if cache_dir:
        config["CPU"]["CACHE_DIR"] = cache_dir
        config["CPU"]["CACHE_MODE"] = "OPTIMIZE_SPEED"
    return {
        "device_type": "CPU",
        "load_config": json.dumps(config, separators=(",", ":")),
    }


def _run_profile(
    label: str,
    threads: int | None,
    auto: bool,
    paths: dict[int, Path],
    indices: list[int],
    warmup_path: Path,
    base_cache_dir: str,
) -> dict:
    original_provider_options = ort_utils._openvino_provider_options
    original_cache = os.environ.get("MANGA_ORT_OPENVINO_CACHE_DIR")
    profile_cache = Path(base_cache_dir or "/tmp/manga-openvino-cache") / label
    profile_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MANGA_ORT_OPENVINO_CACHE_DIR"] = str(profile_cache)

    if auto:
        ort_utils._openvino_provider_options = _auto_openvino_provider_options
        os.environ.pop("MANGA_ORT_OPENVINO_THREADS", None)
        os.environ.pop("MANGA_ORT_OPENVINO_STREAMS", None)
    else:
        os.environ["MANGA_ORT_OPENVINO_THREADS"] = str(int(threads))
        os.environ["MANGA_ORT_OPENVINO_STREAMS"] = "1"

    detector = None
    try:
        init_started = time.perf_counter()
        detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
        init_ms = (time.perf_counter() - init_started) * 1000.0

        warm = read_image(warmup_path)
        detector.detect(warm, parallel=False)
        del warm

        rows: list[dict] = []
        signatures: dict[str, list[tuple]] = {}
        elapsed: list[float] = []
        for index in indices:
            image = read_image(paths[index])
            started = time.perf_counter()
            boxes = detector.detect(image, parallel=False)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            elapsed.append(elapsed_ms)
            rows.append(detector.last_metrics())
            signatures[str(index)] = _box_signature(boxes)
            del image, boxes

        wall_ms = sum(elapsed)
        return {
            "label": label,
            "threads": threads,
            "openvino_auto_latency": bool(auto),
            "session_init_ms": round(init_ms, 3),
            "detector_wall_ms": round(wall_ms, 3),
            "slices_per_min": round(len(indices) * 60000.0 / max(1.0, wall_ms), 3),
            "elapsed_mean_ms": round(statistics.mean(elapsed), 3),
            "elapsed_median_ms": round(statistics.median(elapsed), 3),
            "metrics_mean_ms": {
                key: _mean(rows, key)
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
    finally:
        ort_utils._openvino_provider_options = original_provider_options
        if original_cache is None:
            os.environ.pop("MANGA_ORT_OPENVINO_CACHE_DIR", None)
        else:
            os.environ["MANGA_ORT_OPENVINO_CACHE_DIR"] = original_cache
        if detector is not None:
            del detector
        gc.collect()


def _diff(reference: dict, candidate: dict) -> list[str]:
    return [
        key
        for key, value in reference["box_signatures"].items()
        if value != candidate["box_signatures"].get(key)
    ]


def _speedup_pct(control_ms: float, candidate_ms: float) -> float:
    if control_ms <= 0:
        return 0.0
    return round((1.0 - candidate_ms / control_ms) * 100.0, 2)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"detector-thread-sweep-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    if total < SAMPLE_COUNT + 1:
        raise RuntimeError(f"expected >= {SAMPLE_COUNT + 1} slices, got {total}")

    indices = sorted({round(i * (total - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    paths = {index: Path(pages[index]["original"]) for index in indices}
    warmup_index = next(index for index in range(total) if index not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])
    base_cache_dir = os.environ.get("MANGA_ORT_OPENVINO_CACHE_DIR", "/tmp/manga-openvino-cache")

    profiles: dict[str, dict] = {}
    for label, threads, auto in THREAD_PROFILES:
        print(f"THREAD_SWEEP_START={label}", flush=True)
        profiles[label] = _run_profile(
            label, threads, auto, paths, indices, warmup_path, base_cache_dir
        )
        print("THREAD_SWEEP_RESULT=" + json.dumps(profiles[label], ensure_ascii=False), flush=True)

    control = profiles["threads_2_current"]
    quality = {
        label: {"mismatch_pages": _diff(control, result)}
        for label, result in profiles.items()
        if label != "threads_2_current"
    }
    speedup = {
        label: {
            "detector_wall_reduction_pct": _speedup_pct(control["detector_wall_ms"], result["detector_wall_ms"]),
            "bubble_model_reduction_pct": _speedup_pct(
                control["metrics_mean_ms"]["bubble_model_ms"], result["metrics_mean_ms"]["bubble_model_ms"]
            ),
            "text_model_reduction_pct": _speedup_pct(
                control["metrics_mean_ms"]["text_model_ms"], result["metrics_mean_ms"]["text_model_ms"]
            ),
        }
        for label, result in profiles.items()
        if label not in {"threads_2_current", "threads_2_repeat"}
    }
    repeat = profiles["threads_2_repeat"]
    control_drift_pct = round(
        (repeat["detector_wall_ms"] / max(1.0, control["detector_wall_ms"]) - 1.0) * 100.0,
        2,
    )

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "warmup_index": warmup_index,
        "profiles": profiles,
        "quality": quality,
        "speedup": speedup,
        "control_repeat_drift_pct": control_drift_pct,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DETECTOR_THREAD_SWEEP=" + json.dumps(report, ensure_ascii=False), flush=True)

    mismatches = {
        label: diff["mismatch_pages"]
        for label, diff in quality.items()
        if diff["mismatch_pages"]
    }
    if mismatches:
        raise RuntimeError(
            "OpenVINO thread profile changed detector geometry/masks: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
