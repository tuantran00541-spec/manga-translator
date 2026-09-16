#!/usr/bin/env python3
"""Cross-chapter validation for the conservative detector router.

Rules are learned on one chapter and applied unchanged to a different chapter.
This prevents the small calibration sample used by the historical router probe
from being mistaken for production evidence.  The candidate never changes
inpaint or export ownership; authority/review/count diffs are reported and
indexed as hard cases.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

from app.benchmarking.failure_registry import FailureRegistry, build_failure_case
from app.benchmarking.manifest import build_manifest, sha256_file
from app.config import BUBBLE_DETECTOR_MODEL, TEXT_SEGMENTER_MODEL
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.pipeline import ChapterPipeline

import scripts.benchmark_combined_safe_stack as stack


TRAIN_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
HOLDOUT_URL = "https://asurascans.com/comics/logging-10000-years-into-the-future-53fc8424/chapter/351"
OUT = Path("benchmark-results/router-holdout/report.json")
SAMPLE_COUNT = 8


def _sample_manifest(manifest: dict, sample_count: int) -> tuple[dict[int, Path], Path, list[int]]:
    pages = list(manifest.get("pages") or [])
    if len(pages) < 2:
        raise RuntimeError("chapter manifest does not contain enough slices")
    count = max(2, min(int(sample_count), len(pages) - 1))
    indices = sorted(
        {
            round(i * (len(pages) - 1) / (count - 1))
            for i in range(count)
        }
    )
    used = set(indices)
    warmup_index = next(index for index in range(len(pages)) if index not in used)
    return (
        {index: Path(pages[index]["original"]) for index in indices},
        Path(pages[warmup_index]["original"]),
        indices,
    )


def _train_router(
    manifest: dict,
    *,
    sample_count: int,
) -> tuple[dict, dict, dict]:
    paths, warmup_path, indices = _sample_manifest(manifest, sample_count)
    control = stack._run(
        SequentialFastResidueAdaptiveFocusCombinedTextDetector(),
        paths,
        indices,
        warmup_path,
        "train_control",
    )
    gc.collect()

    bubble_probe = stack._run(
        stack.CombinedSafeDetector(force_bubble_fast=True),
        paths,
        indices,
        warmup_path,
        "train_bubble_probe",
    )
    bubble_rule, bubble_probe_quality, _bubble_features = stack._learn_rule(
        bubble_probe, control, "bubble_feature_"
    )
    gc.collect()

    text_probe = stack._run(
        stack.CombinedSafeDetector(
            bubble_rule=bubble_rule,
            force_skip_fullpass=True,
        ),
        paths,
        indices,
        warmup_path,
        "train_text_probe",
    )
    router_rule, text_probe_quality, _router_features = stack._learn_rule(
        text_probe, control, "router_feature_"
    )
    gc.collect()

    return {
        "sample_indices": indices,
        "control": control,
        "bubble_probe": bubble_probe,
        "bubble_probe_quality": bubble_probe_quality,
        "text_probe": text_probe,
        "text_probe_quality": text_probe_quality,
    }, bubble_rule, router_rule


def _evaluate_router(
    manifest: dict,
    *,
    bubble_rule: dict,
    router_rule: dict,
    sample_count: int,
) -> dict:
    paths, warmup_path, indices = _sample_manifest(manifest, sample_count)
    control = stack._run(
        SequentialFastResidueAdaptiveFocusCombinedTextDetector(),
        paths,
        indices,
        warmup_path,
        "holdout_control",
    )
    gc.collect()
    candidate = stack._run(
        stack.CombinedSafeDetector(
            bubble_rule=bubble_rule,
            router_rule=router_rule,
        ),
        paths,
        indices,
        warmup_path,
        "holdout_router_candidate",
    )
    quality = stack._mismatch(control, candidate)
    return {
        "sample_indices": indices,
        "control": control,
        "candidate": candidate,
        "quality": quality,
        "exact": stack._exact(quality),
        "speedup": {
            "wall_reduction_pct": stack.bubble_base._pct(
                control["wall_ms"], candidate["wall_ms"]
            ),
            "bubble_model_reduction_pct": stack.bubble_base._pct(
                control["bubble_model_mean_ms"], candidate["bubble_model_mean_ms"]
            ),
            "text_model_reduction_pct": stack.bubble_base._pct(
                control["text_model_mean_ms"], candidate["text_model_mean_ms"]
            ),
            "skip_fullpass_pages": candidate["skip_fullpass_pages"],
            "keep_fullpass_pages": candidate["keep_fullpass_pages"],
            "bubble_fastpath_pages": candidate["bubble_fastpath_pages"],
            "bubble_fallback_pages": candidate["bubble_fallback_pages"],
        },
    }


def _record_quality_failures(
    registry: FailureRegistry,
    manifest: dict,
    quality: dict,
    *,
    metrics: dict,
    candidate: str,
    partition: str,
    stage: str,
    note: str,
) -> int:
    pages = list(manifest.get("pages") or [])
    cases = 0
    taxonomy_by_field = {
        "authority": "mask_overreach",
        "review": "detector_fp",
        "counts": "detector_fp",
    }
    for field, indices in quality.items():
        taxonomy = taxonomy_by_field.get(field, "other")
        for index_text in indices:
            index = int(index_text)
            if not (0 <= index < len(pages)):
                continue
            source_path = Path(pages[index]["original"])
            added = registry.append(
                build_failure_case(
                    source_sha256=sha256_file(source_path),
                    source_page=pages[index].get("source_page"),
                    slice_index=pages[index].get("slice_index", index),
                    stage=stage,
                    taxonomy=taxonomy,
                    partition=partition,
                    candidate=candidate,
                    baseline={"field": field, "slice_index": index},
                    observed={"field": field, "slice_index": index},
                    metrics=metrics,
                    notes=note,
                )
            )
            cases += int(added)
    return cases


def main() -> int:
    started = time.perf_counter()
    pipeline = ChapterPipeline()
    train_id = hashlib.sha256(f"router-train-{time.time_ns()}".encode()).hexdigest()[:8]
    train_manifest = pipeline.download_chapter(TRAIN_URL, train_id, workers=2)
    print(f"ROUTER_TRAIN_CHAPTER={train_id}", flush=True)
    train, bubble_rule, router_rule = _train_router(
        train_manifest,
        sample_count=SAMPLE_COUNT,
    )
    gc.collect()

    holdout_id = hashlib.sha256(f"router-holdout-{time.time_ns()}".encode()).hexdigest()[:8]
    holdout_manifest = pipeline.download_chapter(HOLDOUT_URL, holdout_id, workers=2)
    print(f"ROUTER_HOLDOUT_CHAPTER={holdout_id}", flush=True)
    evaluation = _evaluate_router(
        holdout_manifest,
        bubble_rule=bubble_rule,
        router_rule=router_rule,
        sample_count=SAMPLE_COUNT,
    )

    registry = FailureRegistry(OUT.parent / "failure-registry.jsonl")
    train_failure_cases = (
        _record_quality_failures(
            registry,
            train_manifest,
            train["bubble_probe_quality"],
            metrics={"wall_ms": train["bubble_probe"]["wall_ms"]},
            candidate="bubble-fastpath-probe-v1",
            partition="calibration",
            stage="router_train_bubble_probe",
            note="Bubble fastpath changed the calibration signature; keep this as a hard example for router fitting.",
        )
        + _record_quality_failures(
            registry,
            train_manifest,
            train["text_probe_quality"],
            metrics={"wall_ms": train["text_probe"]["wall_ms"]},
            candidate="text-fullpass-probe-v1",
            partition="calibration",
            stage="router_train_text_probe",
            note="Text full-pass elision changed the calibration signature; keep this as a hard example for router fitting.",
        )
    )
    holdout_failure_cases = _record_quality_failures(
        registry,
        holdout_manifest,
        evaluation["quality"],
        metrics=evaluation["speedup"],
        candidate="combined-safe-stack-v1",
        partition="holdout",
        stage="router_holdout",
        note="Cross-chapter router rule changed the frozen holdout signature; do not promote.",
    )
    failure_cases = train_failure_cases + holdout_failure_cases
    report = {
        "benchmark": "router-cross-chapter-holdout",
        "train_url": TRAIN_URL,
        "holdout_url": HOLDOUT_URL,
        "train": {"chapter_id": train_id, **train},
        "rules": {"bubble": bubble_rule, "text": router_rule},
        "holdout": {"chapter_id": holdout_id, **evaluation},
        "failure_cases_added": failure_cases,
        "failure_registry": registry.summary(),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "manifest": build_manifest(
            benchmark="router-cross-chapter-holdout",
            dataset={"train": TRAIN_URL, "holdout": HOLDOUT_URL},
            repo_sha=os.getenv("GITHUB_SHA", "unknown"),
            model_paths={
                "bubble_detector": BUBBLE_DETECTOR_MODEL,
                "text_segmenter": TEXT_SEGMENTER_MODEL,
            },
            parameters={
                "sample_count": SAMPLE_COUNT,
                "split": "by-source-chapter",
                "candidate": "combined-safe-stack-v1",
            },
            cases=[],
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ROUTER_HOLDOUT_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)
    # A failed holdout is useful evidence, not a CI infrastructure failure.
    # Production promotion remains blocked by the explicit `exact` field.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
