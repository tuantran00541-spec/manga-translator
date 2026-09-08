"""Bounded real-image CPU profile of the production detector/inpaint path.

Timer rows are inclusive and may overlap across threads: never add them to
infer elapsed time. Configuration comparisons run in separate processes.

The profiler deliberately does not wrap the LaMa ONNX session. The fixed-LaMa
runtime uses session capabilities to decide serialization/recycling behavior;
wrapping that session used to change the control flow being measured (F15).
LaMa execution is instead timed at the Inpainter method boundary and via the
production inpaint metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Timers:
    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()

    @contextmanager
    def span(self, name, **details):
        start = time.perf_counter()
        try:
            yield
        finally:
            row = {"name": name, "ms": (time.perf_counter() - start) * 1000,
                   "thread": threading.current_thread().name, **details}
            with self.lock:
                self.rows.append(row)

    def wrap(self, owner, method, name):
        descriptor = vars(owner)[method]
        original = getattr(owner, method)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.span(name):
                return original(*args, **kwargs)

        setattr(owner, method, staticmethod(measured) if isinstance(descriptor, staticmethod)
                else measured)

    def summary(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["name"]].append(row["ms"])
        return {name: {"calls": len(values), "sum_ms": round(sum(values), 3),
                       "median_ms": round(statistics.median(values), 3),
                       "max_ms": round(max(values), 3)}
                for name, values in sorted(groups.items())}


def instrument():
    import app.ort_utils as ort_utils
    import app.detector.bubble_detector as yolo
    import app.inpaint.lama_inpainter as lama
    from app.detector.combined_detector import CombinedTextDetector
    from app.detector.recovery import SecondaryTextRecovery

    timers = Timers()
    make_session = ort_utils.make_session

    class Session:
        """Detector-only timing proxy; attribute access remains transparent."""

        def __init__(self, session, name):
            self.session, self.name = session, name

        def run(self, outputs, feed, *args, **kwargs):
            shapes = {key: list(value.shape) for key, value in feed.items()}
            with timers.span("onnx." + self.name, inputs=shapes):
                return self.session.run(outputs, feed, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.session, name)

    def measured_session(path, **kwargs):
        with timers.span("load." + Path(path).name):
            session = make_session(path, **kwargs)
        return Session(session, Path(path).name)

    # Only detector sessions are wrapped. Inpainter keeps the exact production
    # session object so fixed-session serialization/recycling decisions match an
    # unprofiled run.
    yolo.make_session = measured_session
    for method in ("detect", "_preprocess", "_postprocess", "_decode_mask", "_nms", "_nms_boxes"):
        timers.wrap(yolo.YoloDetector, method, "yolo." + method)
    for method in ("detect", "_flat_bubble_text_fallback", "_merge_masks",
                   "_refine_and_split_tall_boxes", "_apply_final_nms"):
        timers.wrap(CombinedTextDetector, method, "combined." + method)
    timers.wrap(SecondaryTextRecovery, "detect", "recovery.detect")
    for method in ("_cluster_boxes", "_smart_fill_color", "_smart_paint_region",
                   "_lama_fill_single_dynamic", "_lama_fill_single", "_lama_fill_tiled",
                   "_run_lama"):
        if method in vars(lama.Inpainter):
            timers.wrap(lama.Inpainter, method, "inpaint." + method)
    original_mask = lama.build_mask

    def measured_mask(*args, **kwargs):
        with timers.span("mask.build"):
            return original_mask(*args, **kwargs)

    lama.build_mask = measured_mask
    return timers


def prepare(args):
    from app.downloader.registry import ASURA_STATIC_ADAPTER
    from PIL import Image

    urls = ASURA_STATIC_ADAPTER.extract_image_urls(args.chapter_url)
    if len(urls) < 4:
        raise RuntimeError("Expected several original chapter images; refusing a blank benchmark")
    # Distributed originals, excluding the first/last cover or credit image.
    chosen = sorted(set([1, len(urls) // 2, len(urls) - 2]))
    selected = [urls[index] for index in chosen]
    paths = ASURA_STATIC_ADAPTER.download_urls(selected, args.raw_dir, referer=args.chapter_url)
    if len(paths) != len(selected):
        raise RuntimeError("Incomplete source-image download")
    records = []
    for path in paths:
        with Image.open(path) as image:
            records.append({"file": path.name, "size": list(image.size), "sha256": digest(path)})
    write_json(args.raw_dir / "sources.json", {"chapter_url": args.chapter_url,
               "source_indices": chosen, "urls": selected, "files": records})
    print(json.dumps(records), flush=True)


def _provider_snapshot(pipeline):
    detector = pipeline.detector
    result = {
        "bubble_yolo.onnx": list(detector.bubble_detector.session.get_providers()),
        "text_segmenter.onnx": list(detector.text_detector.session.get_providers()),
    }
    inpainter = pipeline.inpainter
    if bool(getattr(inpainter, "session_loaded", False)) and inpainter.session is not None:
        result[Path(inpainter.lama_model_path).name] = list(inpainter.session.get_providers())
    return result


def run(args):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import psutil
    from threadpoolctl import threadpool_info
    from app.manifest_utils import load_manifest_raw
    from app.ort_utils import _configured_intra_op_threads, _cpu_count
    from app.parameters import parameter_snapshot

    timers = instrument()
    from app.optimized_pipeline import OptimizedChapterPipeline
    from app.pipeline import read_image
    from scripts.model_e2e_gate import _authority_mask

    raw_paths = sorted(path for path in args.raw_dir.iterdir()
                       if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not raw_paths:
        raise RuntimeError("No original images")
    pipeline = OptimizedChapterPipeline()
    process = psutil.Process()
    memory = {"peak_rss_mb": 0.0}
    stop = threading.Event()

    def sample():
        while not stop.wait(0.1):
            memory["peak_rss_mb"] = max(memory["peak_rss_mb"], process.memory_info().rss / 2**20)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    report = {"source_sha": os.getenv("GITHUB_SHA"), "profile": args.profile,
              "pipeline": "OptimizedChapterPipeline", "workers": args.workers,
              "python": sys.version, "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "effective_cpu_count": _cpu_count(),
              "ort_threads": _configured_intra_op_threads(), "opencv_threads": cv2.getNumThreads(),
              "onnxruntime": ort.__version__, "available_providers": ort.get_available_providers(),
              "numpy": np.__version__, "threadpools": threadpool_info(),
              "parameters": parameter_snapshot(),
              "profiler_semantics": {"detector_session_wrapped": True,
                                     "lama_session_wrapped": False},
              "env": {key: value for key, value in os.environ.items()
                      if key.startswith("MANGA_") or key in {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"}},
              "models": {path.name: digest(path) for path in (ROOT / "models").glob("*.onnx")},
              "sources": json.loads((args.raw_dir / "sources.json").read_text()), "runs": []}
    try:
        for repeat in range(args.repeats):
            chapter_id = hashlib.sha256(f"{args.profile}-{repeat}-{time.time_ns()}".encode()).hexdigest()[:8]
            started = time.perf_counter()
            manifest = pipeline._build_chapter_from_raw_paths(chapter_id, raw_paths, source_url=None, workers=args.workers)
            ingest_ms = (time.perf_counter() - started) * 1000
            count = len(manifest["pages"])
            # Include adjacent slices, with coverage across the three originals.
            indices = sorted(set([min(1, count - 1), min(2, count - 1), count // 2, max(0, count - 2)]))
            timers.rows.clear()
            started = time.perf_counter()
            pipeline.process_pages(chapter_id, indices, workers=args.workers)
            wall_ms = (time.perf_counter() - started) * 1000
            manifest = load_manifest_raw(chapter_id)
            inpainter = pipeline.inpainter
            row = {"repeat": repeat, "cold": repeat == 0, "ingest_ms": ingest_ms,
                   "wall_ms": wall_ms, "indices": indices, "slice_count": count,
                   "provider_placement": _provider_snapshot(pipeline),
                   "inpaint_runtime": {
                       "model": Path(getattr(inpainter, "lama_model_path", "")).name or None,
                       "dynamic": bool(getattr(inpainter, "dynamic_lama", False)),
                       "serialized_inference": bool(getattr(inpainter, "serialized_inference", False)),
                       "session_type": type(getattr(inpainter, "session", None)).__name__,
                   },
                   "timers": timers.summary(), "events": list(timers.rows),
                   "last_processing_run": manifest.get("last_processing_run"), "pages": []}
            for index in indices:
                page = manifest["pages"][index]
                original = read_image(Path(page["original"]))
                clean = read_image(Path(page["clean"]))
                mask = _authority_mask(original, page.get("boxes", []), pipeline.inpainter)
                changed = np.any(original != clean, axis=2)
                outside = int(np.count_nonzero(changed & (mask <= 127)))
                item = {"index": index, "size": [page["width"], page["height"]],
                        "metrics": page.get("processing_metrics"), "boxes": len(page.get("boxes", [])),
                        "mask_pixels": int(np.count_nonzero(mask)), "changed_pixels": int(changed.sum()),
                        "outside_mask_changed": outside, "original_sha": digest(page["original"]),
                        "clean_sha": hashlib.sha256(clean.tobytes()).hexdigest(),
                        "mask_sha": hashlib.sha256(mask.tobytes()).hexdigest()}
                row["pages"].append(item)
                if repeat == args.repeats - 1:
                    evidence = args.output.parent / args.profile
                    evidence.mkdir(parents=True, exist_ok=True)
                    for label, image in [("original", original), ("clean", clean), ("mask", mask)]:
                        cv2.imwrite(str(evidence / f"{index}-{label}.png"), image)
                if outside:
                    raise RuntimeError(f"Changed {outside} pixels outside authorized mask on page {index}")
            report["runs"].append(row)
            report.update(memory)
            write_json(args.output, report)
            print(json.dumps({"profile": args.profile, "repeat": repeat, "wall_ms": wall_ms,
                              "provider_placement": row["provider_placement"],
                              "inpaint_runtime": row["inpaint_runtime"],
                              "timers": row["timers"], **memory}), flush=True)
        if not any(page["changed_pixels"] for row in report["runs"] for page in row["pages"]):
            raise RuntimeError("Sample did not exercise cleanup")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        sampler.join(timeout=1)
        report.update(memory)
        write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--chapter-url", default="https://asurascans.com/comics/killer-pietro-08677664/chapter/120")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "benchmark-results/profile-input")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--repeats", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results/profile/baseline.json")
    args = parser.parse_args()
    prepare(args) if args.prepare else run(args)


if __name__ == "__main__":
    main()
