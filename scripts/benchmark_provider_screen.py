from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

from app.detector.sequential_fast_residue_detector import SequentialFastResidueAdaptiveFocusCombinedTextDetector
from app.pipeline import ChapterPipeline
import scripts.benchmark_bubble_uncertainty as audit_base
import scripts.benchmark_combined_safe_stack as combined

CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/provider-screen/report.json")
SAMPLE_COUNT = 8


def _build(provider: str):
    old_provider = os.environ.get("MANGA_ORT_PROVIDER")
    old_threads = os.environ.get("MANGA_ORT_INTRA_OP_THREADS")
    old_require = os.environ.get("MANGA_ORT_REQUIRE_PROVIDER")
    os.environ["MANGA_ORT_PROVIDER"] = provider
    os.environ["MANGA_ORT_INTRA_OP_THREADS"] = "2"
    os.environ["MANGA_ORT_REQUIRE_PROVIDER"] = "1" if provider == "openvino" else "0"
    try:
        return SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    finally:
        if old_provider is None: os.environ.pop("MANGA_ORT_PROVIDER", None)
        else: os.environ["MANGA_ORT_PROVIDER"] = old_provider
        if old_threads is None: os.environ.pop("MANGA_ORT_INTRA_OP_THREADS", None)
        else: os.environ["MANGA_ORT_INTRA_OP_THREADS"] = old_threads
        if old_require is None: os.environ.pop("MANGA_ORT_REQUIRE_PROVIDER", None)
        else: os.environ["MANGA_ORT_REQUIRE_PROVIDER"] = old_require


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"provider-screen-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted({round(i * (total - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)})
    paths = {i: Path(pages[i]["original"]) for i in indices}
    warmup_index = next(i for i in range(total) if i not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    openvino = combined._run(_build("openvino"), paths, indices, warmup_path, "openvino_f32")
    gc.collect()
    cpu = combined._run(_build("cpu"), paths, indices, warmup_path, "ort_cpu")
    quality = combined._mismatch(openvino, cpu)
    exact = combined._exact(quality)
    reduction = audit_base._pct(openvino["wall_ms"], cpu["wall_ms"])

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "sample_indices": indices,
        "openvino": openvino,
        "cpu": cpu,
        "quality": {**quality, "exact": exact},
        "speedup": {
            "cpu_wall_reduction_pct": reduction,
            "cpu_bubble_reduction_pct": audit_base._pct(openvino["bubble_model_mean_ms"], cpu["bubble_model_mean_ms"]),
            "cpu_text_reduction_pct": audit_base._pct(openvino["text_model_mean_ms"], cpu["text_model_mean_ms"]),
        },
        "full_validation_worthwhile": bool(exact and reduction >= 2.0),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PROVIDER_SCREEN=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
