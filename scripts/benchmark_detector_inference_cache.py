#!/usr/bin/env python3
"""Measure exact duplicate detector inference opportunities.

This is intentionally a probe, not a production switch.  It runs the same
source slices with cache ``off``, ``shadow`` and ``reuse``.  ``shadow`` still
executes the model after a hit, which lets the report detect nondeterministic or
stale cached results before ``reuse`` is considered for a real pipeline.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import psutil

from app.benchmarking.manifest import build_manifest
from app.config import BUBBLE_DETECTOR_MODEL, TEXT_SEGMENTER_MODEL
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.downloader.registry import download_chapter
from app.downloader.slicer import slice_image
from app.image_io import read_image


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/detector-inference-cache/report.json")
SAMPLE_COUNT = 8


def _box_signature(boxes) -> list[tuple]:
    rows = []
    for box in boxes:
        mask = box.mask
        mask_hash = (
            hashlib.sha256(np.asarray(mask).tobytes()).hexdigest()
            if mask is not None
            else None
        )
        rows.append(
            (
                int(box.x1),
                int(box.y1),
                int(box.x2),
                int(box.y2),
                round(float(box.confidence), 8),
                int(box.class_id),
                str(box.source_model),
                str(box.source_role),
                str(box.semantic_type),
                str(box.mask_source),
                bool(box.safe_to_inpaint),
                bool(box.ocr_eligible),
                bool(box.needs_review),
                box.deferred_reason,
                mask_hash,
            )
        )
    return sorted(rows, key=repr)


def _models(detector):
    return (
        getattr(detector, "_bubble_model", detector.bubble_detector),
        getattr(detector, "_text_model", detector.text_detector),
    )


def _set_cache_mode(detector, mode: str) -> None:
    for model in _models(detector):
        cache = getattr(model, "_inference_cache", None)
        if cache is None:
            continue
        cache.mode = str(mode)
        cache.clear()


def _cache_snapshot(detector) -> dict[str, dict]:
    out = {}
    for label, model in zip(("bubble", "text"), _models(detector)):
        snapshot = getattr(model, "inference_cache_metrics", lambda: {})()
        out[label] = dict(snapshot)
    return out


def _rss_bytes() -> int:
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def _run_profile(
    mode: str,
    paths: dict[int, Path],
    indices: list[int],
    warmup_path: Path,
    repeats: int = 2,
) -> dict:
    detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    # Keep model/session warm-up out of the measured cache counters.
    _set_cache_mode(detector, "off")

    warmup = read_image(warmup_path)
    detector.detect(warmup, parallel=False)
    _set_cache_mode(detector, mode)
    del warmup

    rows = []
    signatures: dict[str, list] = {}
    repeat_wall_ms = []
    peak_rss = _rss_bytes()
    for repeat in range(max(1, int(repeats))):
        started = time.perf_counter()
        for index in indices:
            image = read_image(paths[index])
            call_started = time.perf_counter()
            boxes = detector.detect(image, parallel=False)
            elapsed_ms = (time.perf_counter() - call_started) * 1000.0
            signatures[f"{repeat}:{index}"] = _box_signature(boxes)
            rows.append(
                {
                    "repeat": repeat,
                    "index": index,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "detector_metrics": dict(detector.last_metrics()),
                    "rss_bytes": _rss_bytes(),
                }
            )
            peak_rss = max(peak_rss, rows[-1]["rss_bytes"])
            del image, boxes
        repeat_wall_ms.append(round((time.perf_counter() - started) * 1000.0, 3))

    cache = _cache_snapshot(detector)
    all_elapsed = [float(row["elapsed_ms"]) for row in rows]
    return {
        "mode": mode,
        "repeats": int(repeats),
        "wall_ms": round(sum(repeat_wall_ms), 3),
        "repeat_wall_ms": repeat_wall_ms,
        "mean_ms": round(statistics.mean(all_elapsed), 3) if all_elapsed else 0.0,
        "median_ms": round(statistics.median(all_elapsed), 3) if all_elapsed else 0.0,
        "peak_rss_bytes": peak_rss,
        "cache": cache,
        "rows": rows,
        "signatures": signatures,
    }


def _mismatches(control: dict, candidate: dict, indices: list[int]) -> list[str]:
    mismatches = []
    for index in indices:
        control_signature = control["signatures"].get(f"0:{index}")
        candidate_signature = candidate["signatures"].get(f"0:{index}")
        if control_signature != candidate_signature:
            mismatches.append(str(index))
    return mismatches


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chapter_id = hashlib.sha256(f"detector-cache-{time.time_ns()}".encode()).hexdigest()[:8]
    raw_dir = Path("data/raw") / chapter_id / "source"
    sliced_dir = Path("data/raw") / chapter_id / "sliced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sliced_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = download_chapter(CHAPTER_URL, raw_dir)
    if not raw_paths:
        raise RuntimeError("chapter download returned no source images")

    sliced: dict[int, list[Path]] = {}
    for index, source in enumerate(raw_paths):
        sliced[index] = [Path(path) for path in slice_image(source, sliced_dir, f"{index:03d}")]
    pages = [path for index in range(len(raw_paths)) for path in sliced[index]]
    if len(pages) < SAMPLE_COUNT + 1:
        raise RuntimeError(f"expected at least {SAMPLE_COUNT + 1} slices, got {len(pages)}")

    indices = sorted({round(i * (len(pages) - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    paths = {index: pages[index] for index in indices}
    warmup_index = next(index for index in range(len(pages)) if index not in set(indices))
    warmup_path = pages[warmup_index]

    profiles = {}
    for mode in ("off", "shadow", "reuse"):
        print(f"DETECTOR_CACHE_START={mode}", flush=True)
        profiles[mode] = _run_profile(mode, paths, indices, warmup_path)
        print("DETECTOR_CACHE_PROFILE=" + json.dumps(profiles[mode], ensure_ascii=False), flush=True)
        gc.collect()

    control = profiles["off"]
    quality = {
        "shadow_first_repeat_mismatch": _mismatches(control, profiles["shadow"], indices),
        "reuse_first_repeat_mismatch": _mismatches(control, profiles["reuse"], indices),
        "reuse_repeat_mismatch": [
            str(index)
            for index in indices
            if profiles["reuse"]["signatures"].get(f"0:{index}")
            != profiles["reuse"]["signatures"].get(f"1:{index}")
        ],
        "shadow_cache_mismatch_count": sum(
            int(snapshot.get("shadow_mismatches") or 0)
            for snapshot in profiles["shadow"]["cache"].values()
        ),
    }
    report = {
        "benchmark": "detector-inference-cache",
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "source_images": len(raw_paths),
        "slices": len(pages),
        "sample_indices": indices,
        "warmup_index": warmup_index,
        "profiles": profiles,
        "quality": quality,
        "manifest": build_manifest(
            benchmark="detector-inference-cache",
            dataset=CHAPTER_URL,
            repo_sha=os.getenv("GITHUB_SHA", "unknown"),
            model_paths={
                "bubble_detector": BUBBLE_DETECTOR_MODEL,
                "text_segmenter": TEXT_SEGMENTER_MODEL,
            },
            parameters={
                "cache_probe_modes": ["off", "shadow", "reuse"],
                "sample_count": SAMPLE_COUNT,
            },
            cases=[],
        ),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DETECTOR_CACHE_AB=" + json.dumps(report, ensure_ascii=False), flush=True)

    if any(quality.values()):
        raise RuntimeError("content-addressed reuse was not exact: " + json.dumps(quality, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
