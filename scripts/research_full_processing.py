"""Full detector -> inpaint benchmark for the production pipeline.

This is research-only instrumentation. It deliberately stops before OCR,
translation, rendering, and export so wall time represents the processing lane
we are optimizing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import onnxruntime as ort
import psutil
import requests

from app.manifest_utils import load_manifest_raw
from app.optimized_pipeline import OptimizedChapterPipeline
from app.parameters import parameter_snapshot

ROOT = Path(__file__).resolve().parent.parent


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _aggregate_page_metrics(manifest: dict) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for page in manifest.get("pages", []):
        metrics = page.get("processing_metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                groups[key].append(float(value))
    return {
        key: {
            "count": len(values),
            "sum": round(sum(values), 3),
            "mean": round(statistics.mean(values), 3),
            "p95": round(sorted(values)[min(len(values) - 1, int(0.95 * (len(values) - 1)))], 3),
            "max": round(max(values), 3),
        }
        for key, values in sorted(groups.items())
        if values
    }


def _load_input(pipeline: OptimizedChapterPipeline, args) -> tuple[dict, str]:
    chapter_id = hashlib.sha256(f"research-full-{time.time_ns()}".encode()).hexdigest()[:8]
    if args.zip_url:
        response = requests.get(args.zip_url, timeout=120)
        response.raise_for_status()
        manifest = pipeline.create_chapter_from_uploads(
            chapter_id,
            [(Path(args.zip_url.split("?", 1)[0]).name or "benchmark.zip", response.content)],
            workers=args.workers,
        )
        return manifest, f"zip:{args.zip_url}"
    manifest = pipeline.download_chapter(args.chapter_url, chapter_id, workers=args.workers)
    return manifest, f"chapter:{args.chapter_url}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chapter-url",
        default="https://asurascans.com/comics/killer-pietro-08677664/chapter/120",
    )
    parser.add_argument("--zip-url", default=os.environ.get("MANGA_BENCHMARK_ZIP_URL", ""))
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark-results/research-full/report.json",
    )
    args = parser.parse_args()

    pipeline = OptimizedChapterPipeline()
    process = psutil.Process()
    report: dict = {
        "source_sha": os.getenv("GITHUB_SHA"),
        "workers": args.workers,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "parameters": parameter_snapshot(),
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("MANGA_") or key in {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
        },
    }

    started = time.perf_counter()
    manifest, source = _load_input(pipeline, args)
    ingest_ms = (time.perf_counter() - started) * 1000.0
    page_indices = list(range(len(manifest.get("pages", []))))
    if not page_indices:
        raise RuntimeError("Benchmark input produced zero slices")

    rss_before = process.memory_info().rss / 2**20
    started = time.perf_counter()
    pipeline.process_pages(manifest["chapter_id"], page_indices, workers=args.workers)
    process_wall_ms = (time.perf_counter() - started) * 1000.0
    rss_after = process.memory_info().rss / 2**20

    manifest = load_manifest_raw(manifest["chapter_id"])
    source_pages = len({page.get("source_page") for page in manifest.get("pages", [])})
    slice_count = len(manifest.get("pages", []))
    clean_count = sum(bool(page.get("clean")) for page in manifest.get("pages", []))
    box_count = sum(len(page.get("boxes", [])) for page in manifest.get("pages", []))
    safe_count = sum(
        1
        for page in manifest.get("pages", [])
        for box in page.get("boxes", [])
        if box.get("safe_to_inpaint")
    )

    report.update(
        {
            "source": source,
            "chapter_id": manifest["chapter_id"],
            "source_pages": source_pages,
            "slices": slice_count,
            "clean_pages": clean_count,
            "boxes": box_count,
            "safe_to_inpaint": safe_count,
            "ingest_ms": round(ingest_ms, 3),
            "process_wall_ms": round(process_wall_ms, 3),
            "process_wall_s": round(process_wall_ms / 1000.0, 3),
            "throughput_slices_per_s": round(slice_count / (process_wall_ms / 1000.0), 4),
            "throughput_ms_per_slice": round(process_wall_ms / slice_count, 3),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
            "page_metrics": _aggregate_page_metrics(manifest),
            "last_processing_run": manifest.get("last_processing_run"),
        }
    )
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
