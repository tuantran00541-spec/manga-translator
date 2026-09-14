from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.pipeline import ChapterPipeline
import app.ort_utils as ort_utils
import scripts.benchmark_bubble_uncertainty as audit_base


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/openvino-precision/report.json")
SAMPLE_COUNT = 16
PROFILES = (
    ("f32", "f32"),
    ("auto", None),
    ("bf16", "bf16"),
)


def _cpu_flags() -> dict:
    text = ""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        pass
    keys = ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "avx_vnni", "avx512_vnni")
    return {key: (key in text) for key in keys}


def _provider_factory(tag: str, precision: str | None):
    def _options() -> dict[str, str]:
        cpu = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": "2",
            "CACHE_DIR": f"/tmp/manga-openvino-cache-{tag}",
            "CACHE_MODE": "OPTIMIZE_SPEED",
        }
        if precision:
            cpu["INFERENCE_PRECISION_HINT"] = precision
        return {
            "device_type": "CPU",
            "load_config": json.dumps({"CPU": cpu}, separators=(",", ":")),
        }
    return _options


def _run_profile(paths: dict[int, Path], indices: list[int], warmup_path: Path, tag: str, precision: str | None) -> dict:
    original = ort_utils._openvino_provider_options
    ort_utils._openvino_provider_options = _provider_factory(tag, precision)
    try:
        detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
        warm = read_image(warmup_path)
        detector.detect(warm, parallel=False)
        del warm

        elapsed = []
        rows = []
        authority = {}
        review = {}
        counts = {}
        for index in indices:
            image = read_image(paths[index]); h, w = image.shape[:2]
            t = time.perf_counter(); boxes = detector.detect(image, parallel=False); elapsed.append((time.perf_counter()-t)*1000.0)
            rows.append(dict(getattr(detector._metrics_local, "value", {}) or {}))
            authority[str(index)] = hashlib.sha256(audit_base._authority_mask(boxes, h, w).tobytes()).hexdigest()
            review[str(index)] = audit_base._review_signature(boxes)
            counts[str(index)] = {
                "boxes": len(boxes),
                "safe": sum(bool(b.safe_to_inpaint) for b in boxes),
                "review": sum(bool(b.needs_review) for b in boxes),
            }
            del image, boxes

        def mean_metric(name: str) -> float:
            values = [float(row.get(name) or 0.0) for row in rows]
            return round(statistics.mean(values), 3) if values else 0.0

        result = {
            "tag": tag,
            "precision_hint": precision,
            "supported": True,
            "wall_ms": round(sum(elapsed), 3),
            "mean_ms": round(statistics.mean(elapsed), 3),
            "median_ms": round(statistics.median(elapsed), 3),
            "bubble_model_mean_ms": mean_metric("bubble_model_ms"),
            "text_model_mean_ms": mean_metric("text_model_ms"),
            "authority_hashes": authority,
            "review_signatures": review,
            "counts": counts,
        }
        del detector
        gc.collect()
        return result
    except Exception as exc:
        return {
            "tag": tag,
            "precision_hint": precision,
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        ort_utils._openvino_provider_options = original


def _quality(control: dict, candidate: dict) -> dict:
    if not candidate.get("supported"):
        return {"exact": False, "unsupported": True, "authority": [], "review": [], "counts": []}
    authority = [k for k,v in control["authority_hashes"].items() if candidate["authority_hashes"].get(k) != v]
    review = [k for k,v in control["review_signatures"].items() if candidate["review_signatures"].get(k) != v]
    counts = [k for k,v in control["counts"].items() if candidate["counts"].get(k) != v]
    return {
        "exact": not authority and not review and not counts,
        "unsupported": False,
        "authority": authority,
        "review": review,
        "counts": counts,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"ov-precision-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({round(i*(total-1)/(SAMPLE_COUNT-1)) for i in range(SAMPLE_COUNT)})
    paths = {i: Path(pages[i]["original"]) for i in indices}
    warmup_index = next(i for i in range(total) if i not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    profiles = []
    for tag, precision in PROFILES:
        profiles.append(_run_profile(paths, indices, warmup_path, tag, precision))

    control = next(profile for profile in profiles if profile["tag"] == "f32")
    if not control.get("supported"):
        raise RuntimeError("F32 control failed: " + str(control.get("error")))

    comparisons = {}
    for candidate in profiles:
        if candidate["tag"] == "f32":
            continue
        quality = _quality(control, candidate)
        comparisons[candidate["tag"]] = {
            "quality": quality,
            "wall_reduction_pct": (
                audit_base._pct(control["wall_ms"], candidate["wall_ms"])
                if candidate.get("supported") else None
            ),
            "bubble_model_reduction_pct": (
                audit_base._pct(control["bubble_model_mean_ms"], candidate["bubble_model_mean_ms"])
                if candidate.get("supported") else None
            ),
            "text_model_reduction_pct": (
                audit_base._pct(control["text_model_mean_ms"], candidate["text_model_mean_ms"])
                if candidate.get("supported") else None
            ),
            "promotion_eligible": bool(
                candidate.get("supported")
                and quality["exact"]
                and candidate["wall_ms"] < control["wall_ms"]
            ),
        }

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "sample_indices": indices,
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_flags": _cpu_flags(),
        },
        "profiles": profiles,
        "comparisons": comparisons,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("OPENVINO_PRECISION_AB=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
