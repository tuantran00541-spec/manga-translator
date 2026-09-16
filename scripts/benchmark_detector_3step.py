from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path

from app.benchmarking.failure_registry import FailureRegistry, build_failure_case
from app.benchmarking.manifest import build_manifest, sha256_file
from app.config import BUBBLE_DETECTOR_MODEL, PROCESSED_DIR, TEXT_SEGMENTER_MODEL
from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.detector.fast_residue_detector import (
    FastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.manifest_utils import load_manifest_raw, save_manifest_raw
from app.optimized_pipeline import OptimizedChapterPipeline


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/detector-3step/report.json")
SAMPLE_COUNT = 16


class SequentialNoGateDetector(FastResidueAdaptiveFocusCombinedTextDetector):
    """Current branch detector before the three-step experiment."""

    def __init__(self):
        super().__init__()
        self._residue_flat_gate_enabled = False

    def detect(self, image, *, parallel: bool = False):
        # Explicitly bypass ParallelAdaptiveFocusCombinedTextDetector while
        # keeping the same adaptive focus, MSER cache and residue coalescing.
        return AdaptiveFocusCombinedTextDetector.detect(
            self,
            image,
            parallel=False,
        )


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return round(statistics.mean(values), 3) if values else 0.0


def _box_signature(page: dict) -> list[tuple]:
    fields = (
        "x1",
        "y1",
        "x2",
        "y2",
        "source_model",
        "source_role",
        "class_name",
        "semantic_type",
        "mask_source",
        "safe_to_inpaint",
        "ocr_eligible",
        "needs_review",
        "deferred_reason",
    )
    values = []
    for box in page.get("boxes", []):
        values.append(tuple(box.get(name) for name in fields))
    return sorted(values, key=repr)


def _residue_signature(page: dict) -> list[tuple]:
    fields = (
        "x1",
        "y1",
        "x2",
        "y2",
        "source_model",
        "source_role",
        "class_name",
        "semantic_type",
        "deferred_reason",
    )
    values = []
    for box in page.get("residue_regions", []):
        values.append(tuple(box.get(name) for name in fields))
    return sorted(values, key=repr)


def _reset_run_state(chapter_id: str, base_manifest: dict) -> None:
    processed_dir = PROCESSED_DIR / chapter_id
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_manifest_raw(chapter_id, copy.deepcopy(base_manifest))


def _snapshot(
    pipeline: OptimizedChapterPipeline,
    chapter_id: str,
    base_manifest: dict,
    indices: list[int],
    label: str,
    detector: FastResidueAdaptiveFocusCombinedTextDetector,
) -> dict:
    _reset_run_state(chapter_id, base_manifest)
    detector.residue_metrics_snapshot(reset=True)
    pipeline._detector = detector

    started = time.perf_counter()
    pipeline.process_pages(chapter_id, indices, workers=1)
    wall_ms = (time.perf_counter() - started) * 1000.0

    manifest = load_manifest_raw(chapter_id)
    timing_rows: list[dict] = []
    detector_rows: list[dict] = []
    box_signatures: dict[str, list[tuple]] = {}
    residue_signatures: dict[str, list[tuple]] = {}
    clean_hashes: dict[str, str] = {}

    for index in indices:
        page = manifest["pages"][index]
        metrics = page.get("processing_metrics") or {}
        timing_rows.append(dict(metrics.get("timing_ms") or {}))
        detector_rows.append(dict(metrics.get("detector") or {}))
        box_signatures[str(index)] = _box_signature(page)
        residue_signatures[str(index)] = _residue_signature(page)
        clean = read_image(Path(page["clean"]))
        clean_hashes[str(index)] = hashlib.sha256(clean.tobytes()).hexdigest()

    def detector_mean(name: str) -> float:
        return _mean(detector_rows, name)

    second_mser = []
    for row in detector_rows:
        second_mser.append(
            max(
                0.0,
                float(row.get("mser_ms") or 0.0)
                - float(row.get("focus_prefetch_mser_ms") or 0.0),
            )
        )

    result = {
        "label": label,
        "wall_ms": round(wall_ms, 3),
        "slices_per_min": round(len(indices) * 60000.0 / max(1.0, wall_ms), 3),
        "timing_mean_ms": {
            key: _mean(timing_rows, key)
            for key in ("detect", "auto_inpaint", "residue_verify", "total")
        },
        "detector_mean_ms": {
            key: detector_mean(key)
            for key in (
                "bubble_model_ms",
                "text_model_ms",
                "mser_ms",
                "focus_prefetch_mser_ms",
                "parallel_prefetch_wall_ms",
                "parallel_prefetch_overlap_ms",
                "total_ms",
            )
        },
        "second_mser_filter_mean_ms": round(
            statistics.mean(second_mser) if second_mser else 0.0,
            3,
        ),
        "focus_chip_calls": int(
            sum(int(row.get("focus_chip_calls") or 0) for row in detector_rows)
        ),
        "parallel_prefetch_pages": int(
            sum(bool(row.get("parallel_prefetch_enabled")) for row in detector_rows)
        ),
        "residue_metrics": detector.residue_metrics_snapshot(reset=True),
        "box_signatures": box_signatures,
        "residue_signatures": residue_signatures,
        "clean_hashes": clean_hashes,
        "shared_seam": copy.deepcopy(
            (manifest.get("last_processing_run") or {}).get("shared_seam") or {}
        ),
    }
    return result


def _reduction_pct(before: float, after: float) -> float:
    before = float(before)
    after = float(after)
    if before <= 0.0:
        return 0.0
    return round((1.0 - after / before) * 100.0, 2)


def _quality_diff(reference: dict, candidate: dict) -> dict:
    keys = sorted(reference["box_signatures"])
    box_mismatch = [
        key
        for key in keys
        if reference["box_signatures"][key] != candidate["box_signatures"].get(key)
    ]
    residue_mismatch = [
        key
        for key in keys
        if reference["residue_signatures"][key]
        != candidate["residue_signatures"].get(key)
    ]
    clean_mismatch = [
        key
        for key in keys
        if reference["clean_hashes"][key] != candidate["clean_hashes"].get(key)
    ]
    return {
        "box_mismatch_pages": box_mismatch,
        "residue_mismatch_pages": residue_mismatch,
        "clean_hash_mismatch_pages": clean_mismatch,
    }


def _failure_source_sha(base_manifest: dict, chapter_id: str, page_index: int) -> str:
    """Use immutable slice bytes when available, with a deterministic fallback."""
    pages = base_manifest.get("pages") or []
    page = pages[page_index] if 0 <= page_index < len(pages) else {}
    source = page.get("original") if isinstance(page, dict) else None
    digest = sha256_file(source) if source else None
    if digest:
        return digest
    return hashlib.sha256(f"{chapter_id}:page:{page_index}".encode("utf-8")).hexdigest()


def _record_quality_failures(
    registry: FailureRegistry,
    *,
    base_manifest: dict,
    chapter_id: str,
    reference: dict,
    candidate: dict,
    quality: dict,
    candidate_name: str,
) -> int:
    """Distil a quality diff into stable, reviewable registry cases."""
    added = 0
    categories = (
        ("box_mismatch_pages", "detector", "detector_fp"),
        ("residue_mismatch_pages", "residue_verify", "residue"),
        ("clean_hash_mismatch_pages", "inpaint", "inpaint_texture_damage"),
    )
    pages = base_manifest.get("pages") or []
    for field, stage, default_taxonomy in categories:
        for raw_index in quality.get(field, []):
            page_index = int(raw_index)
            key = str(page_index)
            baseline_count = len(reference.get("box_signatures", {}).get(key, []))
            observed_count = len(candidate.get("box_signatures", {}).get(key, []))
            taxonomy = default_taxonomy
            if field == "box_mismatch_pages" and observed_count < baseline_count:
                taxonomy = "detector_fn"
            page = pages[page_index] if 0 <= page_index < len(pages) else {}
            case = build_failure_case(
                source_sha256=_failure_source_sha(base_manifest, chapter_id, page_index),
                source_page=page.get("source_page") if isinstance(page, dict) else None,
                slice_index=page_index,
                stage=stage,
                taxonomy=taxonomy,
                partition="hard",
                candidate=candidate_name,
                baseline={
                    "box_count": baseline_count,
                    "residue_count": len(reference.get("residue_signatures", {}).get(key, [])),
                    "clean_hash": reference.get("clean_hashes", {}).get(key),
                },
                observed={
                    "box_count": observed_count,
                    "residue_count": len(candidate.get("residue_signatures", {}).get(key, [])),
                    "clean_hash": candidate.get("clean_hashes", {}).get(key),
                },
                evidence={"quality_field": field, "page_index": page_index},
                notes="Optimization changed the frozen 3-step signature; keep as hard evidence and do not promote.",
            )
            added += int(registry.append(case))
    return added


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = OptimizedChapterPipeline()
    chapter_id = hashlib.sha256(
        f"detector-3step-{time.time_ns()}".encode()
    ).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    total = len(manifest.get("pages", []))
    if total < SAMPLE_COUNT:
        raise RuntimeError(f"expected >= {SAMPLE_COUNT} slices, got {total}")
    indices = sorted(
        {
            round(i * (total - 1) / (SAMPLE_COUNT - 1))
            for i in range(SAMPLE_COUNT)
        }
    )
    base_manifest = copy.deepcopy(load_manifest_raw(chapter_id))

    # Keep inpainting identical and warm across detector variants. The benchmark
    # reports detector/residue timings separately; clean hashes remain an
    # end-to-end guard that parallel scheduling does not alter destructive output.
    shared_inpainter = FastInpainter()
    shared_inpainter.preload()
    pipeline._inpainter = shared_inpainter

    control_detector = SequentialNoGateDetector()
    control = _snapshot(
        pipeline,
        chapter_id,
        base_manifest,
        indices,
        "control_sequential_cache_coalesce",
        control_detector,
    )
    pipeline._detector = None
    del control_detector
    gc.collect()

    parallel_detector = FastResidueAdaptiveFocusCombinedTextDetector()
    parallel_detector._residue_flat_gate_enabled = False
    parallel = _snapshot(
        pipeline,
        chapter_id,
        base_manifest,
        indices,
        "parallel_prefetch_no_flat_gate",
        parallel_detector,
    )
    pipeline._detector = None
    del parallel_detector
    gc.collect()

    candidate_detector = FastResidueAdaptiveFocusCombinedTextDetector()
    candidate = _snapshot(
        pipeline,
        chapter_id,
        base_manifest,
        indices,
        "parallel_prefetch_plus_flat_gate",
        candidate_detector,
    )

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "control": control,
        "parallel": parallel,
        "candidate": candidate,
    }
    report["quality"] = {
        "parallel_vs_control": _quality_diff(control, parallel),
        "candidate_vs_control": _quality_diff(control, candidate),
    }
    report["speedup"] = {
        "parallel_detect_reduction_pct": _reduction_pct(
            control["timing_mean_ms"]["detect"],
            parallel["timing_mean_ms"]["detect"],
        ),
        "parallel_detector_internal_reduction_pct": _reduction_pct(
            control["detector_mean_ms"]["total_ms"],
            parallel["detector_mean_ms"]["total_ms"],
        ),
        "flat_gate_residue_reduction_pct": _reduction_pct(
            parallel["timing_mean_ms"]["residue_verify"],
            candidate["timing_mean_ms"]["residue_verify"],
        ),
        "combined_detect_plus_residue_reduction_pct": _reduction_pct(
            control["timing_mean_ms"]["detect"]
            + control["timing_mean_ms"]["residue_verify"],
            candidate["timing_mean_ms"]["detect"]
            + candidate["timing_mean_ms"]["residue_verify"],
        ),
        "candidate_total_reduction_pct": _reduction_pct(
            control["timing_mean_ms"]["total"],
            candidate["timing_mean_ms"]["total"],
        ),
    }
    registry = FailureRegistry(OUT.parent / "failure-registry.jsonl")
    failure_cases_added = 0
    for label, result in (
        ("parallel-prefetch-v1", report["quality"]["parallel_vs_control"]),
        ("flat-gate-v1", report["quality"]["candidate_vs_control"]),
    ):
        failure_cases_added += _record_quality_failures(
            registry,
            base_manifest=base_manifest,
            chapter_id=chapter_id,
            reference=control,
            candidate=parallel if label.startswith("parallel") else candidate,
            quality=result,
            candidate_name=label,
        )
    report["failure_cases_added"] = failure_cases_added
    report["failure_registry"] = registry.summary()
    report["manifest"] = build_manifest(
        benchmark="detector-3step-ab",
        dataset=CHAPTER_URL,
        repo_sha=os.getenv("GITHUB_SHA", "unknown"),
        model_paths={
            "bubble_detector": BUBBLE_DETECTOR_MODEL,
            "text_segmenter": TEXT_SEGMENTER_MODEL,
        },
        parameters={
            "sample_count": SAMPLE_COUNT,
            "candidate": "parallel_prefetch_plus_flat_gate",
        },
        cases=registry.read(),
    )

    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("DETECTOR_3STEP_AB=" + json.dumps(report, ensure_ascii=False), flush=True)

    for comparison in report["quality"].values():
        if any(comparison.values()):
            raise RuntimeError(
                "detector optimization changed boxes, residue evidence, or clean pixels: "
                + json.dumps(report["quality"], ensure_ascii=False)
            )


if __name__ == "__main__":
    main()
