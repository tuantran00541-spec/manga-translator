"""Full-chapter research benchmark with MSER removed from focus prefetch only.

The production adaptive detector currently runs SecondaryTextRecovery once to
seed focus bands, then CombinedTextDetector runs recovery again on the final
merged results. This experiment removes only the first call. The final recovery,
text full pass, focus refinement, grayscale fallback, mask authority, and
inpainting paths remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort
import psutil

from app.detector.adaptive_focus_detector import (
    AdaptiveFocusCombinedTextDetector,
    _adaptive_detect,
    _focus_text_detect,
)
from app.detector.combined_detector import CombinedTextDetector
from app.manifest_utils import load_manifest_raw
from app.pipeline import ChapterPipeline
from app.parameters import parameter_snapshot


BASELINE_WALL_S = 647.327
BASELINE_SLICES = 101
BASELINE_BOXES = 423
BASELINE_SAFE = 260
DEFAULT_CHAPTER = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"


class NoPrefetchMserAdaptiveDetector(AdaptiveFocusCombinedTextDetector):
    """Adaptive focus detector using bubble proposals only for focus planning."""

    def detect(self, image, *, parallel: bool = False):
        started_at = time.perf_counter()

        bubble_started = time.perf_counter()
        bubble_boxes = _adaptive_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - bubble_started) * 1000.0

        text_started = time.perf_counter()
        text_boxes, focus_metrics = _focus_text_detect(
            self._text_model,
            image,
            list(bubble_boxes),
        )
        text_ms = (time.perf_counter() - text_started) * 1000.0

        # Reuse neural outputs inside the normal CombinedTextDetector path.
        # Crucially, CombinedTextDetector still executes its final MSER recovery
        # against the fully merged result boxes, preserving cleanup authority.
        with self.bubble_detector.prefetched(image, bubble_boxes):
            with self.text_detector.prefetched(image, text_boxes):
                result = CombinedTextDetector.detect(self, image, parallel=False)

        metrics = dict(getattr(self._metrics_local, "value", {}) or {})
        metrics["bubble_model_ms"] = round(bubble_ms, 3)
        metrics["text_model_ms"] = round(text_ms, 3)
        metrics["focus_prefetch_mser_ms"] = 0.0
        metrics["focus_prefetch_proposals"] = 0
        metrics.update(focus_metrics)
        metrics["total_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0,
            3,
        )
        self._metrics_local.value = metrics
        return result


class NoPrefetchMserPipeline(ChapterPipeline):
    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = NoPrefetchMserAdaptiveDetector()
        return self._detector


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _box_breakdown(manifest: dict) -> dict:
    semantic = Counter()
    source = Counter()
    mask_source = Counter()
    safe = 0
    review = 0
    total = 0
    for page in manifest.get("pages", []):
        for box in page.get("boxes", []):
            total += 1
            semantic[str(box.get("semantic_type", "unknown"))] += 1
            source[str(box.get("source_model", "unknown"))] += 1
            mask_source[str(box.get("mask_source", "unknown"))] += 1
            safe += int(bool(box.get("safe_to_inpaint")))
            review += int(bool(box.get("needs_review")))
    return {
        "total": total,
        "safe": safe,
        "review": review,
        "semantic": dict(sorted(semantic.items())),
        "source_model": dict(sorted(source.items())),
        "mask_source": dict(sorted(mask_source.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter-url", default=DEFAULT_CHAPTER)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results/no-prefetch-mser/report.json",
    )
    args = parser.parse_args()

    pipeline = NoPrefetchMserPipeline()
    process = psutil.Process()
    chapter_id = hashlib.sha256(
        f"no-prefetch-mser-{time.time_ns()}".encode()
    ).hexdigest()[:8]

    report = {
        "source_sha": os.getenv("GITHUB_SHA"),
        "variant": "no_focus_prefetch_mser",
        "workers": args.workers,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "parameters": parameter_snapshot(),
        "baseline": {
            "wall_s": BASELINE_WALL_S,
            "slices": BASELINE_SLICES,
            "boxes": BASELINE_BOXES,
            "safe": BASELINE_SAFE,
        },
    }

    ingest_started = time.perf_counter()
    manifest = pipeline.download_chapter(
        args.chapter_url,
        chapter_id,
        workers=args.workers,
    )
    ingest_ms = (time.perf_counter() - ingest_started) * 1000.0
    indices = list(range(len(manifest.get("pages", []))))
    if not indices:
        raise RuntimeError("Benchmark input produced zero slices")

    rss_before = process.memory_info().rss / 2**20
    process_started = time.perf_counter()
    pipeline.process_pages(chapter_id, indices, workers=args.workers)
    wall_ms = (time.perf_counter() - process_started) * 1000.0
    rss_after = process.memory_info().rss / 2**20

    manifest = load_manifest_raw(chapter_id)
    slices = len(manifest.get("pages", []))
    source_pages = len({page.get("source_page") for page in manifest.get("pages", [])})
    clean_pages = sum(bool(page.get("clean")) for page in manifest.get("pages", []))
    boxes = _box_breakdown(manifest)
    wall_s = wall_ms / 1000.0

    comparable = slices == BASELINE_SLICES
    report.update(
        {
            "source": f"chapter:{args.chapter_url}",
            "chapter_id": chapter_id,
            "source_pages": source_pages,
            "slices": slices,
            "clean_pages": clean_pages,
            "ingest_ms": round(ingest_ms, 3),
            "process_wall_ms": round(wall_ms, 3),
            "process_wall_s": round(wall_s, 3),
            "throughput_ms_per_slice": round(wall_ms / slices, 3),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
            "boxes": boxes,
            "comparable_to_baseline": comparable,
            "delta_vs_baseline": {
                "wall_s": round(wall_s - BASELINE_WALL_S, 3) if comparable else None,
                "wall_pct": round((wall_s / BASELINE_WALL_S - 1.0) * 100.0, 2) if comparable else None,
                "boxes": boxes["total"] - BASELINE_BOXES if comparable else None,
                "safe": boxes["safe"] - BASELINE_SAFE if comparable else None,
            },
            "last_processing_run": manifest.get("last_processing_run"),
        }
    )
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
