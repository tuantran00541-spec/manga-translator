from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import nncf
import numpy as np
import onnx

from app.config import BUBBLE_DETECTOR_MODEL
from app.detector.bubble_detector import LetterboxTransform
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
import app.detector.combined_detector as combined_detector_module
from app.image_io import read_image
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_LETTERBOX_VALUE
from app.pipeline import ChapterPipeline
import scripts.benchmark_bubble_uncertainty as audit_base


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-int8/report.json")
INT8_MODEL = Path("benchmark-results/bubble-int8/bubble_yolo_int8.onnx")
SAMPLE_COUNT = 16
CALIBRATION_COUNT = 32


def _preprocess_path(path: Path, input_name: str) -> dict[str, np.ndarray]:
    image = read_image(path)
    h, w = image.shape[:2]
    transform = LetterboxTransform.create(
        w,
        h,
        DETECTOR_INPUT_SIZE,
        DETECTOR_INPUT_SIZE,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (transform.resized_w, transform.resized_h))
    canvas = np.full(
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3),
        DETECTOR_LETTERBOX_VALUE,
        dtype=np.uint8,
    )
    y1, y2 = transform.pad_y, transform.pad_y + transform.resized_h
    x1, x2 = transform.pad_x, transform.pad_x + transform.resized_w
    canvas[y1:y2, x1:x2] = resized
    blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return {input_name: blob}


def _quantize(calibration_paths: list[Path]) -> dict:
    source = Path(BUBBLE_DETECTOR_MODEL)
    model = onnx.load_model(str(source))
    input_name = model.graph.input[0].name
    dataset = nncf.Dataset(
        calibration_paths,
        lambda path: _preprocess_path(Path(path), input_name),
    )
    started = time.perf_counter()
    quantized = nncf.quantize(
        model,
        dataset,
        subset_size=len(calibration_paths),
    )
    INT8_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(quantized, str(INT8_MODEL))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "quantization_ms": round(elapsed_ms, 3),
        "calibration_count": len(calibration_paths),
        "source_bytes": source.stat().st_size,
        "int8_bytes": INT8_MODEL.stat().st_size,
        "size_reduction_pct": round(
            (1.0 - INT8_MODEL.stat().st_size / float(max(1, source.stat().st_size))) * 100.0,
            2,
        ),
    }


def _run(paths: dict[int, Path], indices: list[int], warmup_path: Path, model_path: Path, label: str) -> dict:
    original = combined_detector_module.BUBBLE_DETECTOR_MODEL
    combined_detector_module.BUBBLE_DETECTOR_MODEL = model_path
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
            vals = [float(row.get(name) or 0.0) for row in rows]
            return round(statistics.mean(vals), 3) if vals else 0.0

        result = {
            "label": label,
            "model_path": str(model_path),
            "wall_ms": round(sum(elapsed), 3),
            "mean_ms": round(statistics.mean(elapsed), 3),
            "median_ms": round(statistics.median(elapsed), 3),
            "bubble_model_mean_ms": mean_metric("bubble_model_ms"),
            "text_model_mean_ms": mean_metric("text_model_ms"),
            "focus_chip_calls": int(sum(int(row.get("focus_chip_calls") or 0) for row in rows)),
            "authority_hashes": authority,
            "review_signatures": review,
            "counts": counts,
        }
        del detector
        gc.collect()
        return result
    finally:
        combined_detector_module.BUBBLE_DETECTOR_MODEL = original


def _quality(control: dict, candidate: dict) -> dict:
    authority = [k for k,v in control["authority_hashes"].items() if candidate["authority_hashes"].get(k) != v]
    review = [k for k,v in control["review_signatures"].items() if candidate["review_signatures"].get(k) != v]
    counts = [k for k,v in control["counts"].items() if candidate["counts"].get(k) != v]
    return {
        "exact": not authority and not review and not counts,
        "authority": authority,
        "review": review,
        "counts": counts,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"bubble-int8-{time.time_ns()}".encode()).hexdigest()[:8]
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
        print("BUBBLE_INT8_AB=" + json.dumps(report, ensure_ascii=False), flush=True)
        return

    control = _run(sample_paths, sample_indices, warmup_path, Path(BUBBLE_DETECTOR_MODEL), "control_fp32_bubble")
    candidate = _run(sample_paths, sample_indices, warmup_path, INT8_MODEL, "candidate_int8_bubble")
    quality = _quality(control, candidate)
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
        "promotion_eligible": bool(
            quality["exact"] and candidate["wall_ms"] < control["wall_ms"]
        ),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BUBBLE_INT8_AB=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
