"""Real-image benchmark for the deliberately minimal cleanup experiment.

Two modes run in separate processes on identical production slices:

current
    Current main detector core + current adaptive inpainter.

simple
    One text-segmenter forward over the whole slice. If and only if that pass
    produces no verified mask, retry the same full slice once at lower
    confidence. Then use the existing AdaptiveFastInpainter unchanged.

The simple mode still avoids bubble detector, recovery, residue verification
and TTA. The bounded retry exists only to recover zero-box tail misses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detector.bubble_detector import YoloDetector
from app.downloader.slicer import slice_image
from app.image_io import read_image


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _prepare_slices(raw_dir: Path, slice_dir: Path, limit: int) -> list[Path]:
    raw_paths = sorted(
        path
        for path in raw_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not raw_paths:
        raise RuntimeError(f"No benchmark inputs in {raw_dir}")

    slice_dir.mkdir(parents=True, exist_ok=True)
    all_slices: list[Path] = []
    for index, path in enumerate(raw_paths):
        records = slice_image(
            path,
            slice_dir,
            f"{index:03d}",
            return_metadata=True,
        )
        for record in records:
            all_slices.append(Path(record["path"]))

    if not all_slices:
        raise RuntimeError("Production slicer produced no benchmark slices")

    limit = int(limit)
    if limit <= 0 or limit >= len(all_slices):
        return all_slices
    limit = max(1, limit)

    # Evenly spread the sample instead of accidentally measuring only one page.
    positions = np.linspace(0, len(all_slices) - 1, num=limit)
    indices = sorted({int(round(value)) for value in positions})
    while len(indices) < limit:
        for candidate in range(len(all_slices)):
            if candidate not in indices:
                indices.append(candidate)
                if len(indices) == limit:
                    break
    return [all_slices[index] for index in sorted(indices[:limit])]


class ForwardCounter:
    def __init__(self):
        self.counts = Counter()
        self.lock = threading.Lock()
        self.original = YoloDetector._detect_single_plain

    def install(self) -> None:
        counter = self
        original = self.original

        def counted(detector, image, offset_x, offset_y):
            with counter.lock:
                counter.counts[str(getattr(detector, "source_model", "unknown"))] += 1
            return original(detector, image, offset_x, offset_y)

        YoloDetector._detect_single_plain = counted

    def snapshot(self) -> Counter:
        with self.lock:
            return Counter(self.counts)


def _delta_counts(before: Counter, after: Counter) -> dict[str, int]:
    names = sorted(set(before) | set(after))
    return {
        name: int(after.get(name, 0) - before.get(name, 0))
        for name in names
        if after.get(name, 0) != before.get(name, 0)
    }


def _save_evidence(
    evidence_dir: Path,
    index: int,
    original: np.ndarray,
    clean: np.ndarray,
    mask: np.ndarray,
    *,
    stride: int,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    # Masks are always retained because the chapter comparison needs every slice.
    cv2.imwrite(str(evidence_dir / f"{index:02d}-mask.png"), mask)
    # Full-resolution original/clean pairs are only visual evidence. Keep them
    # sparse on chapter-scale runs so the artifact stays reasonably bounded.
    if stride <= 1 or index % stride == 0:
        cv2.imwrite(str(evidence_dir / f"{index:02d}-original.png"), original)
        cv2.imwrite(str(evidence_dir / f"{index:02d}-clean.png"), clean)


def _page_safety(original: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    changed = np.any(original != clean, axis=2)
    authority = mask > 127
    return {
        "mask_pixels": int(np.count_nonzero(authority)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "outside_mask_changed": int(np.count_nonzero(changed & (~authority))),
    }


def _run_simple(
    paths: list[Path],
    evidence_dir: Path,
    padding: int,
    counter: ForwardCounter,
    evidence_stride: int,
) -> dict:
    from app.one_shot_cleanup import OneShotCleanupPipeline

    pipeline = OneShotCleanupPipeline(padding=padding)
    prepare = getattr(pipeline.inpainter, "prepare_for_page_workers", None)
    if callable(prepare):
        prepare(1)
    pipeline.inpainter.preload()

    rows = []
    total_started = time.perf_counter()
    for index, path in enumerate(paths):
        image = read_image(path)
        before = counter.snapshot()
        started = time.perf_counter()
        result = pipeline.clean(image)
        wall_ms = (time.perf_counter() - started) * 1000.0
        after = counter.snapshot()
        forwards = _delta_counts(before, after)
        forward_total = int(sum(forwards.values()))

        safety = _page_safety(image, result.image, result.mask)
        if safety["outside_mask_changed"]:
            raise RuntimeError(
                f"simple mode changed {safety['outside_mask_changed']} pixels outside authority"
            )
        expected_forwards = int(result.metrics.get("detector_forward_calls", 0))
        if forward_total != expected_forwards:
            raise RuntimeError(
                f"detector forward accounting mismatch: observed {forward_total}, "
                f"metrics {expected_forwards}: {forwards}"
            )
        row = {
            "index": index,
            "file": path.name,
            "shape": [int(image.shape[0]), int(image.shape[1])],
            "wall_ms": round(wall_ms, 3),
            "detector_ms": float(result.metrics.get("detector_ms", 0.0)),
            "inpaint_ms": float(result.metrics.get("inpaint_ms", 0.0)),
            "detector_forward_calls": forward_total,
            "detector_forwards_by_model": forwards,
            "lama_model_runs": int(result.metrics.get("lama_model_runs", 0)),
            "detector_boxes": int(result.metrics.get("detector_boxes", 0)),
            "accepted_mask_boxes": int(result.metrics.get("accepted_mask_boxes", 0)),
            "fallback_triggered": int(result.metrics.get("fallback_triggered", 0)),
            "fallback_forward_calls": int(
                result.metrics.get("fallback_forward_calls", 0)
            ),
            "fallback_boxes": int(result.metrics.get("fallback_boxes", 0)),
            "smart_fill_regions": int(result.metrics.get("smart_fill_regions", 0)),
            "bubble_fast_fill_regions": int(
                result.metrics.get("bubble_fast_fill_regions", 0)
            ),
            "clusters": int(result.metrics.get("clusters", 0)),
            **safety,
        }
        rows.append(row)
        _save_evidence(
            evidence_dir,
            index,
            image,
            result.image,
            result.mask,
            stride=evidence_stride,
        )
        print(json.dumps(row), flush=True)

    return {
        "mode": "simple",
        "pipeline": "OnePassLowConfRetryAdaptiveFastInpainter",
        "pages": rows,
        "total_wall_ms": round((time.perf_counter() - total_started) * 1000.0, 3),
    }


def _run_current(
    paths: list[Path],
    evidence_dir: Path,
    counter: ForwardCounter,
    evidence_stride: int,
) -> dict:
    from app.detector.mask_builder import build_mask
    from app.detector.sequential_fast_residue_detector import (
        SequentialFastResidueAdaptiveFocusCombinedTextDetector,
    )
    from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter

    detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    inpainter = AdaptiveFastInpainter()
    prepare = getattr(inpainter, "prepare_for_page_workers", None)
    if callable(prepare):
        prepare(1)
    inpainter.preload()

    rows = []
    total_started = time.perf_counter()
    for index, path in enumerate(paths):
        image = read_image(path)

        before = counter.snapshot()
        detect_started = time.perf_counter()
        boxes = detector.detect(image)
        detector_ms = (time.perf_counter() - detect_started) * 1000.0
        after = counter.snapshot()
        forwards = _delta_counts(before, after)
        forward_total = int(sum(forwards.values()))

        safe_boxes = [
            box
            for box in boxes
            if bool(getattr(box, "safe_to_inpaint", False))
            and bool(getattr(box, "verified_mask", False))
            and not getattr(box, "deferred_reason", None)
        ]
        mask = build_mask(image.shape[:2], safe_boxes, image) if safe_boxes else np.zeros(
            image.shape[:2], dtype=np.uint8
        )

        inpaint_started = time.perf_counter()
        clean = inpainter.inpaint(image, safe_boxes)
        inpaint_ms = (time.perf_counter() - inpaint_started) * 1000.0
        inpaint_metrics = inpainter.last_metrics()

        safety = _page_safety(image, clean, mask)
        row = {
            "index": index,
            "file": path.name,
            "shape": [int(image.shape[0]), int(image.shape[1])],
            "wall_ms": round(detector_ms + inpaint_ms, 3),
            "detector_ms": round(detector_ms, 3),
            "inpaint_ms": round(inpaint_ms, 3),
            "detector_forward_calls": forward_total,
            "detector_forwards_by_model": forwards,
            "lama_model_runs": int(inpaint_metrics.get("lama_model_runs", 0)),
            "detector_boxes": int(len(boxes)),
            "safe_boxes": int(len(safe_boxes)),
            **safety,
        }
        rows.append(row)
        _save_evidence(
            evidence_dir,
            index,
            image,
            clean,
            mask,
            stride=evidence_stride,
        )
        print(json.dumps(row), flush=True)

    return {
        "mode": "current",
        "pipeline": "CurrentMainCleanupCore",
        "pages": rows,
        "total_wall_ms": round((time.perf_counter() - total_started) * 1000.0, 3),
    }


def _finalize(report: dict, paths: list[Path], padding: int) -> dict:
    pages = report["pages"]
    report.update(
        {
            "sample_count": len(paths),
            "padding": int(padding),
            "files": [path.name for path in paths],
            "sum_page_wall_ms": round(sum(float(row["wall_ms"]) for row in pages), 3),
            "sum_detector_ms": round(sum(float(row["detector_ms"]) for row in pages), 3),
            "sum_inpaint_ms": round(sum(float(row["inpaint_ms"]) for row in pages), 3),
            "detector_forward_calls": int(
                sum(int(row["detector_forward_calls"]) for row in pages)
            ),
            "lama_model_runs": int(sum(int(row["lama_model_runs"]) for row in pages)),
            "fallback_triggered_slices": int(
                sum(int(row.get("fallback_triggered", 0)) for row in pages)
            ),
            "fallback_forward_calls": int(
                sum(int(row.get("fallback_forward_calls", 0)) for row in pages)
            ),
            "mask_pixels": int(sum(int(row["mask_pixels"]) for row in pages)),
            "changed_pixels": int(sum(int(row["changed_pixels"]) for row in pages)),
            "outside_mask_changed": int(
                sum(int(row["outside_mask_changed"]) for row in pages)
            ),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "simple"), required=True)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "benchmark-results" / "profile-input",
    )
    parser.add_argument(
        "--slice-dir",
        type=Path,
        default=ROOT / "benchmark-results" / "one-shot" / "slices",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum slices to benchmark; 0 means the full chapter.",
    )
    parser.add_argument("--padding", type=int, default=64)
    parser.add_argument(
        "--evidence-stride",
        type=int,
        default=1,
        help="Save original/clean PNGs every N slices; masks are always saved.",
    )
    args = parser.parse_args()

    paths = _prepare_slices(args.raw_dir, args.slice_dir, args.limit)
    evidence_dir = args.evidence_dir or (
        ROOT / "benchmark-results" / "one-shot" / args.mode
    )

    counter = ForwardCounter()
    counter.install()

    if args.mode == "simple":
        report = _run_simple(
            paths,
            evidence_dir,
            args.padding,
            counter,
            max(1, int(args.evidence_stride)),
        )
    else:
        report = _run_current(
            paths,
            evidence_dir,
            counter,
            max(1, int(args.evidence_stride)),
        )

    report = _finalize(report, paths, args.padding)
    _write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "pages"}), flush=True)


if __name__ == "__main__":
    main()
