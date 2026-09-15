from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

from app.config import BUBBLE_DETECTOR_MODEL
from app.detector.sequential_fast_residue_detector import SequentialFastResidueAdaptiveFocusCombinedTextDetector
from app.pipeline import ChapterPipeline
import app.detector.bubble_detector as bubble_mod
import app.ort_utils as ort_utils
import scripts.benchmark_bubble_uncertainty as audit_base
import scripts.benchmark_combined_safe_stack as combined

CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-bf16-hybrid/report.json")
SAMPLE_COUNT = 16

# Rules frozen from the exact combined-safe-stack calibration run. This benchmark
# does not relearn them; it measures whether bubble-only BF16 composes cleanly.
BUBBLE_RULE = {
    "clauses": [
        [
            {"feature": "max_gap", "op": ">=", "threshold": 0.52864875791357},
            {"feature": "max_gap", "op": "<=", "threshold": 0.536875},
        ],
        [
            {"feature": "max_height", "op": ">=", "threshold": 0.40568588521812793},
            {"feature": "gabor_peak_ratio", "op": "<=", "threshold": 1.1685056754942347},
        ],
    ]
}
ROUTER_RULE = {
    "clauses": [[{"feature": "bubble_area_ratio", "op": "<=", "threshold": 0.00478125}]]
}


def _provider_factory(tag: str, precision: str):
    def _options() -> dict[str, str]:
        return {
            "device_type": "CPU",
            "load_config": json.dumps(
                {
                    "CPU": {
                        "PERFORMANCE_HINT": "LATENCY",
                        "NUM_STREAMS": "1",
                        "INFERENCE_NUM_THREADS": "2",
                        "INFERENCE_PRECISION_HINT": precision,
                        "CACHE_DIR": f"/tmp/manga-openvino-cache-{tag}",
                        "CACHE_MODE": "OPTIMIZE_SPEED",
                    }
                },
                separators=(",", ":"),
            ),
        }
    return _options


def _build(detector_factory, *, bubble_precision: str):
    """Construct bubble with requested precision and text segmenter strictly F32."""
    original_make = bubble_mod.make_session
    original_options = ort_utils._openvino_provider_options
    bubble_name = Path(BUBBLE_DETECTOR_MODEL).name

    def selective_make(model_path, *args, **kwargs):
        name = Path(model_path).name
        precision = bubble_precision if name == bubble_name else "f32"
        tag = f"bubble-{precision}" if name == bubble_name else "text-f32"
        previous = ort_utils._openvino_provider_options
        ort_utils._openvino_provider_options = _provider_factory(tag, precision)
        try:
            return original_make(model_path, *args, **kwargs)
        finally:
            ort_utils._openvino_provider_options = previous

    bubble_mod.make_session = selective_make
    try:
        return detector_factory()
    finally:
        bubble_mod.make_session = original_make
        ort_utils._openvino_provider_options = original_options


def _quality(control: dict, candidate: dict) -> dict:
    q = combined._mismatch(control, candidate)
    q["exact"] = combined._exact(q)
    return q


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"bubble-bf16-hybrid-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({round(i * (total - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    paths = {i: Path(pages[i]["original"]) for i in indices}
    warmup_index = next(i for i in range(total) if i not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control = combined._run(
        _build(SequentialFastResidueAdaptiveFocusCombinedTextDetector, bubble_precision="f32"),
        paths, indices, warmup_path, "control_f32",
    )
    gc.collect()

    bf16 = combined._run(
        _build(SequentialFastResidueAdaptiveFocusCombinedTextDetector, bubble_precision="bf16"),
        paths, indices, warmup_path, "bubble_bf16_text_f32",
    )
    bf16_quality = _quality(control, bf16)
    gc.collect()

    safe_f32 = combined._run(
        _build(lambda: combined.CombinedSafeDetector(bubble_rule=BUBBLE_RULE, router_rule=ROUTER_RULE), bubble_precision="f32"),
        paths, indices, warmup_path, "combined_safe_f32",
    )
    safe_f32_quality = _quality(control, safe_f32)
    gc.collect()

    safe_bf16 = combined._run(
        _build(lambda: combined.CombinedSafeDetector(bubble_rule=BUBBLE_RULE, router_rule=ROUTER_RULE), bubble_precision="bf16"),
        paths, indices, warmup_path, "combined_safe_bubble_bf16",
    )
    safe_bf16_quality = _quality(control, safe_bf16)

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "sample_indices": indices,
        "control": control,
        "bubble_bf16": bf16,
        "bubble_bf16_quality": bf16_quality,
        "combined_safe_f32": safe_f32,
        "combined_safe_f32_quality": safe_f32_quality,
        "combined_safe_bubble_bf16": safe_bf16,
        "combined_safe_bubble_bf16_quality": safe_bf16_quality,
        "speedup": {
            "bubble_bf16_wall_reduction_pct": audit_base._pct(control["wall_ms"], bf16["wall_ms"]),
            "bubble_bf16_model_reduction_pct": audit_base._pct(control["bubble_model_mean_ms"], bf16["bubble_model_mean_ms"]),
            "combined_safe_f32_wall_reduction_pct": audit_base._pct(control["wall_ms"], safe_f32["wall_ms"]),
            "combined_safe_bf16_wall_reduction_pct": audit_base._pct(control["wall_ms"], safe_bf16["wall_ms"]),
            "combined_safe_bf16_bubble_reduction_pct": audit_base._pct(control["bubble_model_mean_ms"], safe_bf16["bubble_model_mean_ms"]),
            "combined_safe_bf16_text_reduction_pct": audit_base._pct(control["text_model_mean_ms"], safe_bf16["text_model_mean_ms"]),
        },
        "promotion_eligible_on_sample": bool(safe_bf16_quality["exact"] and safe_bf16["wall_ms"] < control["wall_ms"]),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BUBBLE_BF16_HYBRID=" + json.dumps(report, ensure_ascii=False), flush=True)

    # Do not fail the workflow for quality mismatch: the artifact is the result.
    # Only infrastructure/runtime errors should fail before this point.


if __name__ == "__main__":
    main()
