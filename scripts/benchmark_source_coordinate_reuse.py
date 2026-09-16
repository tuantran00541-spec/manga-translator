#!/usr/bin/env python3
"""Shadow benchmark for source-page detector reuse.

The production path detects each slice core and uses a shared seam pass.  This
probe asks a deliberately stronger question: can one detector pass over a raw
source page be projected into every slice without changing box/mask evidence?
It never writes projected detections to production artifacts and never fails a
run merely because the hypothesis is wrong; mismatches become registry cases.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil

from app.benchmarking.failure_registry import (
    FailureRegistry,
    build_failure_case,
)
from app.benchmarking.manifest import build_manifest, sha256_file
from app.benchmarking.source_coordinate_ledger import (
    SourceCoordinateLedger,
    SourceTile,
    clip_box_to_slice,
    source_overlap_pixels,
)
from app.config import BUBBLE_DETECTOR_MODEL, TEXT_SEGMENTER_MODEL
from app.detector.bubble_detector import BubbleBox
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.downloader.registry import download_chapter
from app.downloader.slicer import slice_image
from app.image_io import read_image


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/source-coordinate-reuse/report.json")
SAMPLE_SOURCES = 4


def _rss_bytes() -> int:
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def _absolute(box: BubbleBox, y_offset: int) -> BubbleBox:
    return BubbleBox(
        int(box.x1),
        int(box.y1) + int(y_offset),
        int(box.x2),
        int(box.y2) + int(y_offset),
        float(box.confidence),
        None if box.mask is None else np.asarray(box.mask).copy(),
        source_model=box.source_model,
        class_id=box.class_id,
        class_name=box.class_name,
        semantic_type=box.semantic_type,
        mask_source=box.mask_source,
        safe_to_inpaint=box.safe_to_inpaint,
        ocr_eligible=box.ocr_eligible,
        needs_review=box.needs_review,
        source_role=box.source_role,
        deferred_reason=box.deferred_reason,
    )


def _core_clip(box: BubbleBox, core_y1: int, core_y2: int, width: int) -> BubbleBox | None:
    """Clip to an owned source core for a fair per-slice comparison."""
    return clip_box_to_slice(
        box,
        slice_box=(0, int(core_y1), int(width), int(core_y2)),
    )


def _signature(boxes: list[BubbleBox], *, y_offset: int = 0) -> list[tuple]:
    rows = []
    for box in boxes:
        mask_hash = (
            hashlib.sha256(np.asarray(box.mask).tobytes()).hexdigest()
            if box.mask is not None
            else None
        )
        rows.append(
            (
                int(box.x1),
                int(box.y1) + int(y_offset),
                int(box.x2),
                int(box.y2) + int(y_offset),
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


def _detector_counts(detector) -> dict[str, int]:
    counts = {}
    for label, model in (
        ("bubble", getattr(detector, "_bubble_model", detector.bubble_detector)),
        ("text", getattr(detector, "_text_model", detector.text_detector)),
    ):
        counts[f"{label}_inference_calls"] = int(
            getattr(model, "inference_call_count", 0)
        )
    return counts


def _slice_meta(raw_paths: list[Path], sliced_dir: Path) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for index, raw_path in enumerate(raw_paths):
        result[index] = [
            item
            if isinstance(item, dict)
            else {"path": item}
            for item in slice_image(raw_path, sliced_dir, f"{index:03d}", return_metadata=True)
        ]
    return result


def _run_source(
    detector: SequentialFastResidueAdaptiveFocusCombinedTextDetector,
    source_path: Path,
    metas: list[dict],
    source_page: int,
    ledger: SourceCoordinateLedger,
) -> dict:
    image = read_image(source_path)
    height, width = image.shape[:2]
    source_sha = sha256_file(source_path)
    signature = "adaptive-combined-v1"
    tile = SourceTile(
        source_sha,
        source_page,
        (0, 0, int(width), int(height)),
        signature,
    )

    core_tiles = []
    for meta in metas:
        source_y1 = int(meta.get("source_y1", 0))
        source_y2 = int(meta.get("source_y2", height))
        core_y1 = int(meta.get("core_source_y1", source_y1))
        core_y2 = int(meta.get("core_source_y2", source_y2))
        core_tiles.append((0, core_y1, width, core_y2))
    requested, unique = source_overlap_pixels(core_tiles)
    ledger.record_source_pixels(requested=requested, unique=unique)

    before = _detector_counts(detector)
    baseline_started = time.perf_counter()
    baseline_by_slice: dict[int, list[BubbleBox]] = {}
    baseline_rows = []
    for slice_index, meta in enumerate(metas):
        core_y1 = int(meta.get("core_source_y1", meta.get("source_y1", 0)))
        core_y2 = int(meta.get("core_source_y2", meta.get("source_y2", height)))
        if core_y2 <= core_y1:
            baseline_by_slice[slice_index] = []
            continue
        call_started = time.perf_counter()
        boxes = detector.detect(image[core_y1:core_y2, :], parallel=False)
        baseline_by_slice[slice_index] = [_absolute(box, core_y1) for box in boxes]
        baseline_rows.append(
            {
                "slice_index": slice_index,
                "elapsed_ms": round((time.perf_counter() - call_started) * 1000.0, 3),
                "boxes": len(boxes),
            }
        )
    baseline_wall_ms = (time.perf_counter() - baseline_started) * 1000.0
    baseline_counts = _detector_counts(detector)

    source_started = time.perf_counter()
    source_boxes = detector.detect(image, parallel=False)
    source_wall_ms = (time.perf_counter() - source_started) * 1000.0
    source_counts = _detector_counts(detector)
    ledger.store(tile, source_boxes)

    mismatches = []
    comparison_rows = []
    for slice_index, meta in enumerate(metas):
        source_y1 = int(meta.get("source_y1", 0))
        source_y2 = int(meta.get("source_y2", height))
        core_y1 = int(meta.get("core_source_y1", source_y1))
        core_y2 = int(meta.get("core_source_y2", source_y2))
        baseline = [
            box
            for box in baseline_by_slice.get(slice_index, [])
            if _core_clip(box, core_y1, core_y2, width) is not None
        ]
        candidate = []
        for box in source_boxes:
            clipped = _core_clip(box, core_y1, core_y2, width)
            if clipped is None:
                continue
            candidate.append(_absolute(clipped, core_y1))
        baseline_signature = _signature(baseline)
        candidate_signature = _signature(candidate)
        mismatch = baseline_signature != candidate_signature
        if mismatch:
            mismatches.append(slice_index)
        comparison_rows.append(
            {
                "slice_index": slice_index,
                "source_y1": source_y1,
                "source_y2": source_y2,
                "core_source_y1": core_y1,
                "core_source_y2": core_y2,
                "baseline_boxes": len(baseline),
                "source_projected_boxes": len(candidate),
                "exact": not mismatch,
            }
        )

    del image, source_boxes, baseline_by_slice
    gc.collect()
    return {
        "source_page": int(source_page),
        "source_path": source_path.as_posix(),
        "source_sha256": source_sha,
        "shape": [int(height), int(width)],
        "slice_count": len(metas),
        "source_tile": {
            "key": tile.key,
            "bbox": [0, 0, int(width), int(height)],
        },
        "source_pixels": {
            "per_slice_core_requested": requested,
            "per_slice_core_unique": unique,
            "source_pass_requested": int(width * height),
        },
        "baseline": {
            "wall_ms": round(baseline_wall_ms, 3),
            "inference_calls": {
                label: baseline_counts[f"{label}_inference_calls"]
                - before[f"{label}_inference_calls"]
                for label in ("bubble", "text")
            },
            "rows": baseline_rows,
        },
        "source_page_pass": {
            "wall_ms": round(source_wall_ms, 3),
            "inference_calls": {
                label: source_counts[f"{label}_inference_calls"]
                - baseline_counts[f"{label}_inference_calls"]
                for label in ("bubble", "text")
            },
        },
        "mismatched_slices": mismatches,
        "comparisons": comparison_rows,
        "peak_rss_bytes": _rss_bytes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sources", type=int, default=SAMPLE_SOURCES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.sample_sources < 1:
        raise ValueError("--sample-sources must be >= 1")

    chapter_id = hashlib.sha256(f"source-coordinate-{time.time_ns()}".encode()).hexdigest()[:8]
    raw_dir = Path("data/raw") / chapter_id / "source"
    sliced_dir = Path("data/raw") / chapter_id / "sliced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sliced_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = download_chapter(CHAPTER_URL, raw_dir)
    if not raw_paths:
        raise RuntimeError("chapter download returned no source images")
    metas = _slice_meta(raw_paths, sliced_dir)
    candidates = [
        (index, path, metas[index])
        for index, path in enumerate(raw_paths)
        if len(metas[index]) >= 2
    ]
    candidates = candidates[: int(args.sample_sources)]
    if not candidates:
        raise RuntimeError("chapter has no multi-slice source page for probe")

    detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    warmup_path = Path(metas[candidates[0][0]][0]["path"])
    detector.detect(read_image(warmup_path), parallel=False)
    ledger = SourceCoordinateLedger(mode="shadow", max_entries=max(2, len(candidates)))

    rows = []
    for source_page, source_path, page_metas in candidates:
        print(f"SOURCE_COORDINATE_START={source_page}", flush=True)
        row = _run_source(detector, Path(source_path), page_metas, source_page, ledger)
        rows.append(row)
        print("SOURCE_COORDINATE_ROW=" + json.dumps(row, ensure_ascii=False), flush=True)

    registry = FailureRegistry(args.out.parent / "failure-registry.jsonl")
    mismatch_cases = 0
    for row in rows:
        for slice_index in row["mismatched_slices"]:
            mismatch_cases += 1
            registry.append(
                build_failure_case(
                    source_sha256=row["source_sha256"],
                    source_page=row["source_page"],
                    slice_index=int(slice_index),
                    stage="source_coordinate_reuse",
                    taxonomy="detector_fn",
                    partition="hard",
                    candidate="source-page-adaptive-v1",
                    baseline={"comparison": row["comparisons"][slice_index]},
                    observed={"comparison": row["comparisons"][slice_index]},
                    metrics={
                        "baseline_wall_ms": row["baseline"]["wall_ms"],
                        "source_page_wall_ms": row["source_page_pass"]["wall_ms"],
                    },
                    notes="Source-page projection differs from owned per-slice core; keep per-slice fallback.",
                )
            )

    report = {
        "benchmark": "source-coordinate-reuse-shadow",
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "sample_sources": len(candidates),
        "rows": rows,
        "ledger": ledger.snapshot(),
        "failure_registry": registry.summary(),
        "manifest": build_manifest(
            benchmark="source-coordinate-reuse-shadow",
            dataset=CHAPTER_URL,
            repo_sha=os.getenv("GITHUB_SHA", "unknown"),
            model_paths={
                "bubble_detector": BUBBLE_DETECTOR_MODEL,
                "text_segmenter": TEXT_SEGMENTER_MODEL,
            },
            parameters={
                "sample_sources": int(args.sample_sources),
                "mode": "shadow",
                "projection": "owned-source-core",
            },
            cases=[],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_COORDINATE_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    # The probe is evidence gathering.  Any mismatch is intentionally recorded
    # and keeps the workflow green; production reuse remains opt-in until a
    # later gate proves exactness on smoke, hard and holdout partitions.
    return 0 if mismatch_cases >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

