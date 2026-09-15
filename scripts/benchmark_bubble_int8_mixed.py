from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import nncf
import onnx

from app.config import BUBBLE_DETECTOR_MODEL
from app.pipeline import ChapterPipeline
import scripts.benchmark_bubble_int8 as base
import scripts.benchmark_bubble_uncertainty as audit_base

CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-int8-mixed/report.json")
INT8_MODEL = Path("benchmark-results/bubble-int8-mixed/bubble_yolo_int8_mixed.onnx")
SAMPLE_COUNT = 16
CALIBRATION_COUNT = 32


def _quantize(calibration_paths: list[Path]) -> dict:
    source = Path(BUBBLE_DETECTOR_MODEL)
    model = onnx.load_model(str(source))
    input_name = model.graph.input[0].name
    dataset = nncf.Dataset(
        calibration_paths,
        lambda path: base._preprocess_path(Path(path), input_name),
    )
    started = time.perf_counter()
    quantized = nncf.quantize(
        model,
        dataset,
        subset_size=len(calibration_paths),
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        fast_bias_correction=False,
    )
    INT8_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(quantized, str(INT8_MODEL))
    return {
        "quantization_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "calibration_count": len(calibration_paths),
        "source_bytes": source.stat().st_size,
        "int8_bytes": INT8_MODEL.stat().st_size,
        "size_reduction_pct": round((1.0 - INT8_MODEL.stat().st_size / float(max(1, source.stat().st_size))) * 100.0, 2),
        "preset": "mixed",
        "target_device": "cpu",
        "fast_bias_correction": False,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"bubble-int8-mixed-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    sample_indices = sorted({round(i*(total-1)/(SAMPLE_COUNT-1)) for i in range(SAMPLE_COUNT)})
    calibration_indices = sorted({round(i*(total-1)/(CALIBRATION_COUNT-1)) for i in range(CALIBRATION_COUNT)})
    sample_paths = {i: Path(pages[i]["original"]) for i in sample_indices}
    calibration_paths = [Path(pages[i]["original"]) for i in calibration_indices]
    warmup_index = next(i for i in range(total) if i not in set(sample_indices))
    warmup_path = Path(pages[warmup_index]["original"])

    try:
        quantization = _quantize(calibration_paths)
    except Exception as exc:
        report = {
            "chapter_url": CHAPTER_URL,
            "chapter_id": chapter_id,
            "sample_indices": sample_indices,
            "calibration_indices": calibration_indices,
            "quantization_error": f"{type(exc).__name__}: {exc}",
        }
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("BUBBLE_INT8_MIXED=" + json.dumps(report, ensure_ascii=False), flush=True)
        return

    control = base._run(sample_paths, sample_indices, warmup_path, Path(BUBBLE_DETECTOR_MODEL), "control_fp32_bubble")
    gc.collect()
    candidate = base._run(sample_paths, sample_indices, warmup_path, INT8_MODEL, "candidate_mixed_int8_bubble")
    quality = base._quality(control, candidate)
    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "sample_indices": sample_indices,
        "calibration_indices": calibration_indices,
        "quantization": quantization,
        "control": control,
        "candidate": candidate,
        "quality": quality,
        "speedup": {
            "wall_reduction_pct": audit_base._pct(control["wall_ms"], candidate["wall_ms"]),
            "bubble_model_reduction_pct": audit_base._pct(control["bubble_model_mean_ms"], candidate["bubble_model_mean_ms"]),
            "text_model_reduction_pct": audit_base._pct(control["text_model_mean_ms"], candidate["text_model_mean_ms"]),
        },
        "promotion_eligible": bool(quality["exact"] and candidate["wall_ms"] < control["wall_ms"]),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BUBBLE_INT8_MIXED=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
