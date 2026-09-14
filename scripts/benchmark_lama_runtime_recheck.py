from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from app.image_io import read_image
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.inpaint.runtime_tuned_inpainter import RuntimeTunedFastInpainter
from app.manifest_utils import load_manifest_raw
from app.optimized_pipeline import OptimizedChapterPipeline
from app.pipeline import ChapterPipeline
from app.runtime_responsiveness import responsive_process_workers, visible_cpu_count
from scripts.model_e2e_gate import _authority_mask

URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/lama-runtime-recheck/report.json")
RUNTIME_KEYS = (
    "MANGA_LAMA_RUNTIME_ARENA",
    "MANGA_LAMA_RUNTIME_MEM_PATTERN",
    "MANGA_LAMA_RUNTIME_THREADS",
    "MANGA_LAMA_RUNTIME_DYNAMIC_BLOCK_BASE",
    "MANGA_LAMA_RUNTIME_TENSOR_POOL",
    "MANGA_LAMA_RUNTIME_TENSOR_POOL_CAPACITY",
)
MIB = 1024 * 1024


class RSSSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = max(0.01, float(interval))
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.samples: list[int] = []
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = [int(self.process.memory_info().rss)]
        self.stop_event.clear()

        def _run() -> None:
            while not self.stop_event.wait(self.interval):
                self.samples.append(int(self.process.memory_info().rss))

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, float]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.samples.append(int(self.process.memory_info().rss))
        return {
            "start_mib": round(self.samples[0] / MIB, 3),
            "peak_mib": round(max(self.samples) / MIB, 3),
            "end_mib": round(self.samples[-1] / MIB, 3),
            "growth_mib": round((max(self.samples) - self.samples[0]) / MIB, 3),
            "retained_mib": round((self.samples[-1] - self.samples[0]) / MIB, 3),
            "sample_count": len(self.samples),
        }


def rss_mib() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / MIB, 3)


def clear_runtime_env() -> None:
    for key in RUNTIME_KEYS:
        os.environ.pop(key, None)


def make_runtime(*, threads: int, block: int, arena: bool = True, mem_pattern: bool = True):
    clear_runtime_env()
    os.environ["MANGA_LAMA_RUNTIME_ARENA"] = "1" if arena else "0"
    os.environ["MANGA_LAMA_RUNTIME_MEM_PATTERN"] = "1" if mem_pattern else "0"
    os.environ["MANGA_LAMA_RUNTIME_THREADS"] = str(int(threads))
    os.environ["MANGA_LAMA_RUNTIME_DYNAMIC_BLOCK_BASE"] = str(int(block))
    # Literature/runtime recheck deliberately leaves exact-shape I/O pool off:
    # previous benchmark showed negative value on dynamic CPU ROI inference.
    os.environ["MANGA_LAMA_RUNTIME_TENSOR_POOL"] = "0"
    return RuntimeTunedFastInpainter()


def release_inpainter(pipeline: OptimizedChapterPipeline, inpainter) -> float:
    pipeline._inpainter = None
    try:
        inpainter.session = None
    except Exception:
        pass
    del inpainter
    gc.collect()
    return rss_mib()


def mean_timing(rows: list[dict], name: str) -> float:
    values = [float(row.get(name) or 0.0) for row in rows]
    return round(statistics.mean(values), 3) if values else 0.0


def snapshot(
    pipeline: OptimizedChapterPipeline,
    chapter_id: str,
    indices: list[int],
    *,
    label: str,
    inpainter,
    requested_workers: int,
    forced_workers: bool,
    baseline_images: dict[int, np.ndarray] | None,
) -> tuple[dict, dict[int, np.ndarray]]:
    pipeline._inpainter = inpainter
    inpainter.preload()
    config = (
        inpainter.runtime_config()
        if hasattr(inpainter, "runtime_config")
        else {"runtime_session_active": False, "tensor_pool": False}
    )
    if hasattr(inpainter, "runtime_config") and not config.get("runtime_session_active"):
        raise RuntimeError(f"{label}: tuned dynamic session did not activate")
    if bool(config.get("tensor_pool")):
        raise RuntimeError(f"{label}: literature recheck must keep tensor pool disabled")

    preload_rss = rss_mib()
    sampler = RSSSampler()
    sampler.start()
    started = time.perf_counter()
    if forced_workers:
        # Bypass OptimizedChapterPipeline's responsiveness cap on purpose so the
        # benchmark can expose oversubscription risk. The production wrapper is
        # still measured separately through responsive mode.
        ChapterPipeline.process_pages(
            pipeline,
            chapter_id,
            indices,
            workers=requested_workers,
        )
        effective_workers = int(requested_workers)
    else:
        effective_workers = int(responsive_process_workers(requested_workers))
        pipeline.process_pages(
            chapter_id,
            indices,
            workers=requested_workers,
        )
    wall_ms = (time.perf_counter() - started) * 1000.0
    memory = sampler.stop()
    gc.collect()
    memory["preload_mib"] = preload_rss
    memory["post_gc_same_session_mib"] = rss_mib()

    current = load_manifest_raw(chapter_id)
    pages = current.get("pages", [])
    timing: list[dict] = []
    inpaint_metrics: dict[str, float] = {}
    authority_outside = 0
    residue = 0
    boxes = safe = review = deferred = 0
    clean_images: dict[int, np.ndarray] = {}
    max_abs_diff = 0
    diff_sum = 0
    diff_values = 0
    changed_values = 0

    for idx in indices:
        page = pages[idx]
        metrics = page.get("processing_metrics") or {}
        timing.append(copy.deepcopy(metrics.get("timing_ms") or {}))
        for key, value in (metrics.get("auto_inpaint") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                inpaint_metrics[key] = inpaint_metrics.get(key, 0.0) + float(value)
        residue += int((metrics.get("detector") or {}).get("post_inpaint_residue") or 0)
        page_boxes = page.get("boxes", [])
        boxes += len(page_boxes)
        safe += sum(bool(box.get("safe_to_inpaint")) for box in page_boxes)
        review += sum(bool(box.get("needs_review")) for box in page_boxes)
        deferred += sum(bool(box.get("deferred_reason")) for box in page_boxes)
        original = read_image(Path(page["original"]))
        clean = read_image(Path(page["clean"]))
        clean_images[idx] = clean.copy()
        authority = _authority_mask(original, page_boxes, inpainter)
        changed = np.any(original != clean, axis=2)
        authority_outside += int(np.count_nonzero(changed & (authority <= 127)))
        if baseline_images is not None:
            diff = np.abs(clean.astype(np.int16) - baseline_images[idx].astype(np.int16))
            max_abs_diff = max(max_abs_diff, int(diff.max(initial=0)))
            diff_sum += int(diff.sum())
            diff_values += int(diff.size)
            changed_values += int(np.count_nonzero(diff))

    row = {
        "label": label,
        "config": config,
        "requested_workers": int(requested_workers),
        "effective_workers": int(effective_workers),
        "forced_workers": bool(forced_workers),
        "wall_ms": round(wall_ms, 3),
        "slices_per_min": round(len(indices) * 60000.0 / max(1.0, wall_ms), 3),
        "timing_mean_ms": {
            key: mean_timing(timing, key)
            for key in ("detect", "auto_inpaint", "residue_verify", "total")
        },
        "inpaint_metrics": {key: round(value, 3) for key, value in sorted(inpaint_metrics.items())},
        "memory": memory,
        "post_inpaint_residue": residue,
        "authority_outside_changed": authority_outside,
        "boxes": boxes,
        "safe": safe,
        "review": review,
        "deferred": deferred,
        "quality_vs_current_w1": (
            {
                "max_abs_channel_diff": max_abs_diff,
                "mean_abs_channel_diff": round(diff_sum / max(1, diff_values), 8),
                "changed_channel_values": changed_values,
            }
            if baseline_images is not None
            else None
        ),
    }
    return row, clean_images


def synthetic_case(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((h, w), dtype=np.uint16)
    b = ((x * 3 + y * 5 + 17) % 256).astype(np.uint8)
    g = ((x * 7 + y * 2 + 53) % 256).astype(np.uint8)
    r = ((x * 5 + y * 11 + 101) % 256).astype(np.uint8)
    canvas = np.dstack((b, g, r))
    mask = np.zeros((h, w), dtype=np.uint8)
    cy1, cy2 = max(8, h // 3), min(h - 8, h * 2 // 3)
    cx1, cx2 = max(8, w // 3), min(w - 8, w * 2 // 3)
    mask[cy1:cy2, cx1:cx2] = 255
    # Thin disconnected support catches dynamic-shape/memory-reuse mistakes that
    # a single solid rectangle can miss.
    mask[max(1, h // 5):min(h - 1, h // 5 + 3), max(1, w // 6):min(w - 1, w * 5 // 6)] = 255
    return canvas, mask


def stress_alternating_shapes(label: str, inpainter) -> tuple[dict, dict[tuple[int, int], np.ndarray]]:
    inpainter.preload()
    shapes = [(128, 128), (512, 512), (160, 224), (512, 128), (128, 128)]
    cases = {shape: synthetic_case(*shape) for shape in sorted(set(shapes))}
    process = psutil.Process(os.getpid())
    gc.collect()
    start_rss = int(process.memory_info().rss)
    peak_rss = start_rss
    first_hash: dict[tuple[int, int], str] = {}
    last_hash: dict[tuple[int, int], str] = {}
    first_outputs: dict[tuple[int, int], np.ndarray] = {}
    call_ms: list[float] = []

    inpainter._begin_metrics()
    for _ in range(2):
        for shape in shapes:
            canvas, mask = cases[shape]
            started = time.perf_counter()
            output = inpainter._run_lama(canvas.copy(), mask)
            call_ms.append((time.perf_counter() - started) * 1000.0)
            digest = hashlib.sha256(output.tobytes()).hexdigest()
            first_hash.setdefault(shape, digest)
            first_outputs.setdefault(shape, output.copy())
            last_hash[shape] = digest
            peak_rss = max(peak_rss, int(process.memory_info().rss))

    gc.collect()
    end_rss = int(process.memory_info().rss)
    stable = all(first_hash[shape] == last_hash[shape] for shape in first_hash)
    report = {
        "label": label,
        "shape_sequence": [list(shape) for shape in shapes],
        "repeats": 2,
        "calls": len(call_ms),
        "mean_call_ms": round(statistics.mean(call_ms), 3),
        "median_call_ms": round(statistics.median(call_ms), 3),
        "max_call_ms": round(max(call_ms), 3),
        "rss_start_mib": round(start_rss / MIB, 3),
        "rss_peak_mib": round(peak_rss / MIB, 3),
        "rss_end_mib": round(end_rss / MIB, 3),
        "rss_peak_growth_mib": round((peak_rss - start_rss) / MIB, 3),
        "rss_retained_mib": round((end_rss - start_rss) / MIB, 3),
        "repeat_output_stable": stable,
        "first_hashes": {f"{h}x{w}": first_hash[(h, w)] for h, w in first_hash},
        "last_hashes": {f"{h}x{w}": last_hash[(h, w)] for h, w in last_hash},
    }
    return report, first_outputs


def quality_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "max_abs_channel_diff": int(diff.max(initial=0)),
        "mean_abs_channel_diff": round(float(diff.mean()), 8),
        "changed_channel_values": int(np.count_nonzero(diff)),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = OptimizedChapterPipeline()
    chapter_id = hashlib.sha256(f"lama-runtime-recheck-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(URL, chapter_id, workers=2)
    total = len(manifest.get("pages", []))
    if total < 6:
        raise RuntimeError(f"expected >=6 slices, got {total}")
    indices = sorted({round(i * (total - 1) / 5) for i in range(6)})

    cpu_visible = int(visible_cpu_count())
    variants: list[dict] = []

    clear_runtime_env()
    current_before = FastInpainter()
    current_w1, baseline_images = snapshot(
        pipeline,
        chapter_id,
        indices,
        label="current_w1_before",
        inpainter=current_before,
        requested_workers=1,
        forced_workers=False,
        baseline_images=None,
    )
    current_w1["rss_after_session_release_mib"] = release_inpainter(pipeline, current_before)
    variants.append(current_w1)

    specs = [
        ("arena_mem_t2_w1", 2, 0, 1, False),
        ("arena_mem_block4_t2_w1", 2, 4, 1, False),
        ("arena_mem_block4_t4_w1", 4, 4, 1, False),
    ]
    for label, threads, block, workers, forced in specs:
        candidate = make_runtime(threads=threads, block=block)
        row, _ = snapshot(
            pipeline,
            chapter_id,
            indices,
            label=label,
            inpainter=candidate,
            requested_workers=workers,
            forced_workers=forced,
            baseline_images=baseline_images,
        )
        row["rss_after_session_release_mib"] = release_inpainter(pipeline, candidate)
        variants.append(row)

    clear_runtime_env()
    current_w2 = FastInpainter()
    row, _ = snapshot(
        pipeline,
        chapter_id,
        indices,
        label="current_forced_w2",
        inpainter=current_w2,
        requested_workers=2,
        forced_workers=True,
        baseline_images=baseline_images,
    )
    row["rss_after_session_release_mib"] = release_inpainter(pipeline, current_w2)
    variants.append(row)

    for label, threads in (
        ("arena_mem_block4_t2_forced_w2", 2),
        ("arena_mem_block4_t4_forced_w2", 4),
    ):
        candidate = make_runtime(threads=threads, block=4)
        row, _ = snapshot(
            pipeline,
            chapter_id,
            indices,
            label=label,
            inpainter=candidate,
            requested_workers=2,
            forced_workers=True,
            baseline_images=baseline_images,
        )
        row["rss_after_session_release_mib"] = release_inpainter(pipeline, candidate)
        variants.append(row)

    clear_runtime_env()
    current_after = FastInpainter()
    after, _ = snapshot(
        pipeline,
        chapter_id,
        indices,
        label="current_w1_after",
        inpainter=current_after,
        requested_workers=1,
        forced_workers=False,
        baseline_images=baseline_images,
    )
    after["rss_after_session_release_mib"] = release_inpainter(pipeline, current_after)
    variants.append(after)

    # Alternating-shape stress is isolated from chapter timing. It specifically
    # probes dynamic-shape correctness and arena high-water behavior.
    clear_runtime_env()
    stress_current = FastInpainter()
    current_stress, current_outputs = stress_alternating_shapes("current", stress_current)
    del stress_current
    gc.collect()

    stress_candidate = make_runtime(threads=4, block=4)
    tuned_stress, tuned_outputs = stress_alternating_shapes("arena_mem_block4_t4", stress_candidate)
    del stress_candidate
    gc.collect()

    stress_quality: dict[str, dict] = {}
    for shape in sorted(current_outputs):
        stress_quality[f"{shape[0]}x{shape[1]}"] = quality_diff(
            current_outputs[shape], tuned_outputs[shape]
        )

    control_auto = statistics.median(
        [
            current_w1["timing_mean_ms"]["auto_inpaint"],
            after["timing_mean_ms"]["auto_inpaint"],
        ]
    )
    control_wall = statistics.median([current_w1["wall_ms"], after["wall_ms"]])
    current_forced_w2 = next(row for row in variants if row["label"] == "current_forced_w2")
    for row in variants:
        row["auto_inpaint_speedup_vs_w1_control_pct"] = round(
            (control_auto / max(1.0, row["timing_mean_ms"]["auto_inpaint"]) - 1.0) * 100.0,
            2,
        )
        row["wall_speedup_vs_w1_control_pct"] = round(
            (control_wall / max(1.0, row["wall_ms"]) - 1.0) * 100.0,
            2,
        )
        if row["forced_workers"] and row["effective_workers"] == 2:
            row["wall_speedup_vs_forced_w2_control_pct"] = round(
                (current_forced_w2["wall_ms"] / max(1.0, row["wall_ms"]) - 1.0) * 100.0,
                2,
            )
        else:
            row["wall_speedup_vs_forced_w2_control_pct"] = None

    candidates_w1 = [
        row
        for row in variants
        if row["label"].startswith("arena_") and not row["forced_workers"]
    ]
    candidates_w2 = [
        row
        for row in variants
        if row["label"].startswith("arena_") and row["forced_workers"]
    ]
    winner_w1 = min(candidates_w1, key=lambda row: row["wall_ms"])
    winner_w2 = min(candidates_w2, key=lambda row: row["wall_ms"])

    report = {
        "chapter_url": URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "visible_cpu_count": cpu_visible,
        "responsive_workers_if_request_2": int(responsive_process_workers(2)),
        "control_auto_inpaint_median_ms": round(control_auto, 3),
        "control_wall_median_ms": round(control_wall, 3),
        "variants": variants,
        "winner_w1": winner_w1["label"],
        "winner_forced_w2": winner_w2["label"],
        "alternating_shape_stress": {
            "current": current_stress,
            "candidate": tuned_stress,
            "quality_candidate_vs_current": stress_quality,
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("LAMA_RUNTIME_RECHECK=" + json.dumps(report, ensure_ascii=False), flush=True)

    # Quality and memory-safety gates. Arena retention itself is measured, not
    # treated as failure, unless it becomes catastrophically large for this test.
    for row in variants:
        if row["label"].startswith("arena_"):
            if row["authority_outside_changed"]:
                raise RuntimeError(f"{row['label']}: changed pixels escaped authority mask")
            if row["boxes"] != current_w1["boxes"] or row["safe"] != current_w1["safe"]:
                raise RuntimeError(f"{row['label']}: detector decisions changed")
            if row["post_inpaint_residue"] > current_w1["post_inpaint_residue"]:
                raise RuntimeError(f"{row['label']}: residue regressed")
            quality = row["quality_vs_current_w1"] or {}
            if quality.get("max_abs_channel_diff", 0) > 2:
                raise RuntimeError(f"{row['label']}: pixel delta exceeds tolerance: {quality}")
            if quality.get("mean_abs_channel_diff", 0.0) > 0.02:
                raise RuntimeError(f"{row['label']}: mean pixel delta exceeds tolerance: {quality}")
            if float(row["memory"]["growth_mib"]) > 1024.0:
                raise RuntimeError(f"{row['label']}: RSS growth exceeded 1 GiB safety ceiling")

    if not current_stress["repeat_output_stable"]:
        raise RuntimeError("current runtime is not stable across alternating shapes")
    if not tuned_stress["repeat_output_stable"]:
        raise RuntimeError("tuned runtime is not stable across alternating shapes")
    if tuned_stress["rss_peak_growth_mib"] > 1024.0:
        raise RuntimeError("tuned alternating-shape stress exceeded 1 GiB RSS growth")
    for shape, quality in stress_quality.items():
        if quality["max_abs_channel_diff"] > 2 or quality["mean_abs_channel_diff"] > 0.02:
            raise RuntimeError(f"alternating-shape quality regressed at {shape}: {quality}")


if __name__ == "__main__":
    main()
