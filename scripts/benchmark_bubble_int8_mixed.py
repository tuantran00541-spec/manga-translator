from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import nncf
import onnx
from onnx import version_converter

from app.config import BUBBLE_DETECTOR_MODEL
from app.detector.bubble_detector import YoloDetector
from app.image_io import read_image
from app.parameters import BUBBLE_PROPOSAL_CONF_THRESHOLD
from app.pipeline import ChapterPipeline
import scripts.benchmark_bubble_int8 as base
import scripts.benchmark_bubble_uncertainty as audit_base

CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/bubble-int8-mixed/report.json")
OPSET13_MODEL = Path("benchmark-results/bubble-int8-mixed/bubble_yolo_opset13.onnx")
INT8_MODEL = Path("benchmark-results/bubble-int8-mixed/bubble_yolo_int8_mixed.onnx")
SAMPLE_COUNT = 16
CALIBRATION_COUNT = 32
TARGET_OPSET = 13


def _default_opset(model: onnx.ModelProto) -> int:
    for item in model.opset_import:
        if item.domain in ("", "ai.onnx"):
            return int(item.version)
    return 0


def _upgrade_opset() -> dict:
    source = Path(BUBBLE_DETECTOR_MODEL)
    model = onnx.load_model(str(source))
    source_opset = _default_opset(model)
    started = time.perf_counter()
    if source_opset == TARGET_OPSET:
        upgraded = model
    else:
        upgraded = version_converter.convert_version(model, TARGET_OPSET)
    onnx.checker.check_model(upgraded)
    OPSET13_MODEL.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(upgraded, str(OPSET13_MODEL))
    return {
        "source_opset": source_opset,
        "target_opset": _default_opset(upgraded),
        "conversion_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "source_bytes": source.stat().st_size,
        "converted_bytes": OPSET13_MODEL.stat().st_size,
    }


def _quantize(calibration_paths: list[Path]) -> dict:
    source = OPSET13_MODEL
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
        "source_opset": _default_opset(model),
        "quantized_opset": _default_opset(quantized),
        "source_bytes": source.stat().st_size,
        "int8_bytes": INT8_MODEL.stat().st_size,
        "size_reduction_pct": round((1.0 - INT8_MODEL.stat().st_size / float(max(1, source.stat().st_size))) * 100.0, 2),
        "preset": "mixed",
        "target_device": "cpu",
        "fast_bias_correction": False,
        "expected_weight_granularity": "per_channel_for_opset_ge_13",
    }


def _iou(a, b) -> float:
    ix1 = max(int(a.x1), int(b.x1))
    iy1 = max(int(a.y1), int(b.y1))
    ix2 = min(int(a.x2), int(b.x2))
    iy2 = min(int(a.y2), int(b.y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (int(a.x2) - int(a.x1)) * (int(a.y2) - int(a.y1)))
    area_b = max(1, (int(b.x2) - int(b.x1)) * (int(b.y2) - int(b.y1)))
    return inter / float(max(1, area_a + area_b - inter))


def _box_row(box) -> dict:
    return {
        "xyxy": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
        "confidence": round(float(box.confidence), 6),
        "class_id": int(box.class_id),
        "class_name": str(box.class_name),
    }


def _proposal_diagnostics(sample_paths: dict[int, Path]) -> dict:
    control_detector = YoloDetector(
        Path(BUBBLE_DETECTOR_MODEL),
        BUBBLE_PROPOSAL_CONF_THRESHOLD,
        use_tta=False,
        model_role="bubble_detector",
    )
    candidate_detector = YoloDetector(
        INT8_MODEL,
        BUBBLE_PROPOSAL_CONF_THRESHOLD,
        use_tta=False,
        model_role="bubble_detector",
    )
    threshold = float(BUBBLE_PROPOSAL_CONF_THRESHOLD)
    rows = {}
    all_best_ious: list[float] = []
    all_conf_deltas: list[float] = []
    threshold_crossings = 0
    unmatched_control_total = 0
    unmatched_candidate_total = 0

    for index, path in sample_paths.items():
        image = read_image(path)
        control_boxes = control_detector.detect(image)
        candidate_boxes = candidate_detector.detect(image)
        del image

        matched_candidate: set[int] = set()
        matches = []
        unmatched_control = []
        for control_box in control_boxes:
            candidates = [
                (j, candidate_box, _iou(control_box, candidate_box))
                for j, candidate_box in enumerate(candidate_boxes)
                if j not in matched_candidate and int(candidate_box.class_id) == int(control_box.class_id)
            ]
            if not candidates:
                unmatched_control.append(_box_row(control_box))
                continue
            j, candidate_box, best_iou = max(candidates, key=lambda item: item[2])
            if best_iou < 0.10:
                unmatched_control.append(_box_row(control_box))
                continue
            matched_candidate.add(j)
            conf_delta = float(candidate_box.confidence) - float(control_box.confidence)
            edge_shift = max(
                abs(int(candidate_box.x1) - int(control_box.x1)),
                abs(int(candidate_box.y1) - int(control_box.y1)),
                abs(int(candidate_box.x2) - int(control_box.x2)),
                abs(int(candidate_box.y2) - int(control_box.y2)),
            )
            near_threshold = min(
                abs(float(control_box.confidence) - threshold),
                abs(float(candidate_box.confidence) - threshold),
            ) <= 0.03
            crossed = (
                (float(control_box.confidence) >= threshold) !=
                (float(candidate_box.confidence) >= threshold)
            )
            threshold_crossings += int(crossed)
            all_best_ious.append(float(best_iou))
            all_conf_deltas.append(abs(conf_delta))
            matches.append({
                "iou": round(float(best_iou), 6),
                "confidence_delta": round(conf_delta, 6),
                "edge_shift_px": int(edge_shift),
                "near_proposal_threshold": bool(near_threshold),
                "threshold_crossed": bool(crossed),
                "control": _box_row(control_box),
                "candidate": _box_row(candidate_box),
            })

        unmatched_candidate = [
            _box_row(box)
            for j, box in enumerate(candidate_boxes)
            if j not in matched_candidate
        ]
        unmatched_control_total += len(unmatched_control)
        unmatched_candidate_total += len(unmatched_candidate)
        rows[str(index)] = {
            "control_count": len(control_boxes),
            "candidate_count": len(candidate_boxes),
            "matched": len(matches),
            "mean_match_iou": round(statistics.mean([m["iou"] for m in matches]), 6) if matches else 0.0,
            "min_match_iou": round(min([m["iou"] for m in matches]), 6) if matches else 0.0,
            "max_abs_confidence_delta": round(max([abs(m["confidence_delta"]) for m in matches]), 6) if matches else 0.0,
            "max_edge_shift_px": max([m["edge_shift_px"] for m in matches], default=0),
            "unmatched_control": unmatched_control,
            "unmatched_candidate": unmatched_candidate,
            "matches": matches,
        }

    return {
        "proposal_threshold": threshold,
        "summary": {
            "mean_match_iou": round(statistics.mean(all_best_ious), 6) if all_best_ious else 0.0,
            "mean_abs_confidence_delta": round(statistics.mean(all_conf_deltas), 6) if all_conf_deltas else 0.0,
            "threshold_crossings": int(threshold_crossings),
            "unmatched_control": int(unmatched_control_total),
            "unmatched_candidate": int(unmatched_candidate_total),
        },
        "samples": rows,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"bubble-int8-opset13-{time.time_ns()}".encode()).hexdigest()[:8]
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
        opset_conversion = _upgrade_opset()
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
    converted = base._run(sample_paths, sample_indices, warmup_path, OPSET13_MODEL, "control_fp32_opset13")
    converted_quality = base._quality(control, converted)
    gc.collect()
    proposal_diagnostics = _proposal_diagnostics(sample_paths)
    candidate = base._run(sample_paths, sample_indices, warmup_path, INT8_MODEL, "candidate_opset13_perchannel_int8_bubble")
    quality = base._quality(control, candidate)
    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "sample_indices": sample_indices,
        "calibration_indices": calibration_indices,
        "opset_conversion": opset_conversion,
        "quantization": quantization,
        "converted_fp32": converted,
        "converted_fp32_quality": converted_quality,
        "proposal_diagnostics": proposal_diagnostics,
        "control": control,
        "candidate": candidate,
        "quality": quality,
        "speedup": {
            "converted_wall_reduction_pct": audit_base._pct(control["wall_ms"], converted["wall_ms"]),
            "wall_reduction_pct": audit_base._pct(control["wall_ms"], candidate["wall_ms"]),
            "bubble_model_reduction_pct": audit_base._pct(control["bubble_model_mean_ms"], candidate["bubble_model_mean_ms"]),
            "text_model_reduction_pct": audit_base._pct(control["text_model_mean_ms"], candidate["text_model_mean_ms"]),
        },
        "promotion_eligible": bool(
            converted_quality["exact"] and quality["exact"] and candidate["wall_ms"] < control["wall_ms"]
        ),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BUBBLE_INT8_MIXED=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
