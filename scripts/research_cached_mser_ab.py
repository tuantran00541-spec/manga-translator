"""Same-run A/B benchmark for reusing raw MSER regions across adaptive focus.

The production adaptive detector calls SecondaryTextRecovery twice per page:
first for focus proposals and later against the final merged boxes. Both calls
need different filtering semantics, so caching the final BubbleBox list would be
incorrect. This experiment caches only cv2.MSER.detectRegions() output inside a
single AdaptiveFocusCombinedTextDetector.detect() call. All downstream recovery
filtering, mask construction, final NMS, and inpaint authority execute normally.

A cached full-chapter run is executed first, followed by an uncached baseline on
the same runner, same model sessions, and same downloaded source images. Running
the baseline second makes the speed comparison conservative for the cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime as ort
import psutil

from app.config import RAW_DIR
from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.downloader.registry import download_chapter as fetch_chapter_images
from app.manifest_utils import load_manifest_raw
from app.pipeline import ChapterPipeline
from app.parameters import parameter_snapshot


DEFAULT_CHAPTER = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"


class PairedMserRegionsProxy:
    """Transparent MSER proxy with optional per-page one-shot region reuse."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self.enabled = False
        self.reset_stats()

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._stats = {
                "detect_regions_calls": 0,
                "underlying_calls": 0,
                "cache_hits": 0,
                "underlying_ms": 0.0,
                "estimated_saved_ms": 0.0,
            }

    def stats(self) -> dict:
        with self._stats_lock:
            return {
                "detect_regions_calls": int(self._stats["detect_regions_calls"]),
                "underlying_calls": int(self._stats["underlying_calls"]),
                "cache_hits": int(self._stats["cache_hits"]),
                "underlying_ms": round(float(self._stats["underlying_ms"]), 3),
                "estimated_saved_ms": round(float(self._stats["estimated_saved_ms"]), 3),
            }

    @contextmanager
    def paired(self):
        previous = getattr(self._local, "state", None)
        self._local.state = {
            "enabled": bool(self.enabled),
            "result": None,
            "first_ms": 0.0,
        }
        try:
            yield
        finally:
            self._local.state = previous

    def detectRegions(self, gray):
        state = getattr(self._local, "state", None)
        with self._stats_lock:
            self._stats["detect_regions_calls"] += 1

        if state and state["enabled"] and state["result"] is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
                self._stats["estimated_saved_ms"] += float(state["first_ms"])
            return state["result"]

        started = time.perf_counter()
        result = self._inner.detectRegions(gray)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._stats_lock:
            self._stats["underlying_calls"] += 1
            self._stats["underlying_ms"] += elapsed_ms

        if state and state["enabled"] and state["result"] is None:
            state["result"] = result
            state["first_ms"] = elapsed_ms
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CacheToggleAdaptiveDetector(AdaptiveFocusCombinedTextDetector):
    def __init__(self) -> None:
        super().__init__()
        self.mser_regions_proxy = PairedMserRegionsProxy(self.recovery._mser)
        self.recovery._mser = self.mser_regions_proxy

    def detect(self, image, *, parallel: bool = False):
        with self.mser_regions_proxy.paired():
            return super().detect(image, parallel=parallel)


class CacheTogglePipeline(ChapterPipeline):
    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = CacheToggleAdaptiveDetector()
        return self._detector


def _digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _box_signature(page: dict) -> str:
    records = []
    for box in page.get("boxes", []):
        mask = str(box.get("mask") or "")
        records.append(
            {
                "xyxy": [int(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2")],
                "confidence": round(float(box.get("confidence", 0.0)), 6),
                "semantic_type": str(box.get("semantic_type", "")),
                "source_model": str(box.get("source_model", "")),
                "mask_source": str(box.get("mask_source", "")),
                "safe_to_inpaint": bool(box.get("safe_to_inpaint")),
                "ocr_eligible": bool(box.get("ocr_eligible")),
                "needs_review": bool(box.get("needs_review")),
                "mask_sha": hashlib.sha256(mask.encode("utf-8")).hexdigest(),
            }
        )
    records.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _breakdown(manifest: dict) -> dict:
    semantic = Counter()
    source_model = Counter()
    mask_source = Counter()
    total = safe = review = 0
    for page in manifest.get("pages", []):
        for box in page.get("boxes", []):
            total += 1
            safe += int(bool(box.get("safe_to_inpaint")))
            review += int(bool(box.get("needs_review")))
            semantic[str(box.get("semantic_type", "unknown"))] += 1
            source_model[str(box.get("source_model", "unknown"))] += 1
            mask_source[str(box.get("mask_source", "unknown"))] += 1
    return {
        "total": total,
        "safe": safe,
        "review": review,
        "semantic": dict(sorted(semantic.items())),
        "source_model": dict(sorted(source_model.items())),
        "mask_source": dict(sorted(mask_source.items())),
    }


def _run_variant(
    pipeline: CacheTogglePipeline,
    raw_paths: list[Path],
    *,
    label: str,
    cache_enabled: bool,
    workers: int,
) -> dict:
    proxy = pipeline.detector.mser_regions_proxy
    proxy.enabled = bool(cache_enabled)
    proxy.reset_stats()

    chapter_id = hashlib.sha256(f"cached-mser-{label}-{time.time_ns()}".encode()).hexdigest()[:8]
    build_started = time.perf_counter()
    manifest = pipeline._build_chapter_from_raw_paths(
        chapter_id,
        raw_paths,
        source_url=None,
        workers=workers,
    )
    build_ms = (time.perf_counter() - build_started) * 1000.0
    indices = list(range(len(manifest.get("pages", []))))
    if not indices:
        raise RuntimeError(f"{label}: zero slices")

    original_sha = [_digest_file(Path(page["original"])) for page in manifest["pages"]]
    started = time.perf_counter()
    pipeline.process_pages(chapter_id, indices, workers=workers)
    wall_ms = (time.perf_counter() - started) * 1000.0
    manifest = load_manifest_raw(chapter_id)

    return {
        "label": label,
        "cache_enabled": bool(cache_enabled),
        "chapter_id": chapter_id,
        "source_pages": len({page.get("source_page") for page in manifest.get("pages", [])}),
        "slices": len(manifest.get("pages", [])),
        "clean_pages": sum(bool(page.get("clean")) for page in manifest.get("pages", [])),
        "build_ms": round(build_ms, 3),
        "wall_ms": round(wall_ms, 3),
        "wall_s": round(wall_ms / 1000.0, 3),
        "throughput_ms_per_slice": round(wall_ms / max(1, len(indices)), 3),
        "boxes": _breakdown(manifest),
        "mser_cache": proxy.stats(),
        "original_sha": original_sha,
        "box_signature": [_box_signature(page) for page in manifest.get("pages", [])],
        "clean_sha": [
            _digest_file(Path(page["clean"])) if page.get("clean") else None
            for page in manifest.get("pages", [])
        ],
        "last_processing_run": manifest.get("last_processing_run"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter-url", default=DEFAULT_CHAPTER)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results/cached-mser-ab/report.json",
    )
    args = parser.parse_args()

    input_dir = ROOT / "benchmark-results/cached-mser-input"
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = fetch_chapter_images(args.chapter_url, input_dir)
    if not raw_paths:
        raise RuntimeError("No chapter images downloaded")

    pipeline = CacheTogglePipeline()
    process = psutil.Process()

    # One real page warms detector/OpenVINO/LaMa once before either timed full run.
    warm_manifest = pipeline._build_chapter_from_raw_paths(
        "mserwarm",
        raw_paths[:1],
        source_url=None,
        workers=args.workers,
    )
    pipeline.detector.mser_regions_proxy.enabled = False
    pipeline.process_pages("mserwarm", [0], workers=1)
    gc.collect()

    rss_before = process.memory_info().rss / 2**20

    # Cache runs first. Baseline runs second and therefore receives any residual
    # OS/model-cache advantage; this makes a measured cache win conservative.
    cached = _run_variant(
        pipeline,
        raw_paths,
        label="cached",
        cache_enabled=True,
        workers=args.workers,
    )
    gc.collect()
    baseline = _run_variant(
        pipeline,
        raw_paths,
        label="baseline",
        cache_enabled=False,
        workers=args.workers,
    )
    rss_after = process.memory_info().rss / 2**20

    same_slices = cached["slices"] == baseline["slices"]
    original_mismatch_pages = []
    box_mismatch_pages = []
    clean_mismatch_pages = []
    if same_slices:
        for index in range(cached["slices"]):
            if cached["original_sha"][index] != baseline["original_sha"][index]:
                original_mismatch_pages.append(index)
            if cached["box_signature"][index] != baseline["box_signature"][index]:
                box_mismatch_pages.append(index)
            if cached["clean_sha"][index] != baseline["clean_sha"][index]:
                clean_mismatch_pages.append(index)

    delta_ms = cached["wall_ms"] - baseline["wall_ms"]
    report = {
        "source_sha": os.getenv("GITHUB_SHA"),
        "source": f"chapter:{args.chapter_url}",
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "workers": args.workers,
        "parameters": parameter_snapshot(),
        "raw_pages": len(raw_paths),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "cached": cached,
        "baseline": baseline,
        "comparison": {
            "same_slices": same_slices,
            "wall_delta_s_cached_minus_baseline": round(delta_ms / 1000.0, 3),
            "wall_delta_pct": round(delta_ms / baseline["wall_ms"] * 100.0, 2),
            "box_total_delta": cached["boxes"]["total"] - baseline["boxes"]["total"],
            "safe_delta": cached["boxes"]["safe"] - baseline["boxes"]["safe"],
            "original_mismatch_pages": original_mismatch_pages,
            "box_mismatch_pages": box_mismatch_pages,
            "clean_mismatch_pages": clean_mismatch_pages,
            "exact_detector_match": same_slices and not box_mismatch_pages,
            "exact_clean_match": same_slices and not clean_mismatch_pages,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
