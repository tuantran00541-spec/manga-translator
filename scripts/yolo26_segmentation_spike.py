#!/usr/bin/env python3
"""Shadow-benchmark a manga-specific YOLO26s segmentation model.

The candidate is never installed as a production model and never receives
destructive authority. It reuses the production YoloDetector mask decoder,
geometry, NMS, and hysteresis while bypassing the production one-class contract
only inside this benchmark because the candidate has frame/text/balloon classes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CANDIDATE_CLASS_NAMES = ("frame", "text", "balloon")
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)


def mask_metrics(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> dict:
    reference = np.asarray(reference_mask) > 0
    candidate = np.asarray(candidate_mask) > 0
    if reference.shape != candidate.shape:
        raise ValueError(
            f"mask shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )

    reference_pixels = int(np.count_nonzero(reference))
    candidate_pixels = int(np.count_nonzero(candidate))
    intersection_pixels = int(np.count_nonzero(reference & candidate))
    union_pixels = int(np.count_nonzero(reference | candidate))
    false_positive_pixels = int(np.count_nonzero(candidate & ~reference))

    pixel_recall = (
        intersection_pixels / float(reference_pixels)
        if reference_pixels
        else 1.0
    )
    pixel_precision = (
        intersection_pixels / float(candidate_pixels)
        if candidate_pixels
        else 1.0
    )
    pixel_iou = (
        intersection_pixels / float(union_pixels)
        if union_pixels
        else 1.0
    )
    return {
        "reference_pixels": reference_pixels,
        "candidate_pixels": candidate_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "false_positive_pixels": false_positive_pixels,
        "pixel_recall": pixel_recall,
        "pixel_precision": pixel_precision,
        "pixel_iou": pixel_iou,
    }


def candidate_mask_union(shape: tuple[int, ...], boxes) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        mask = getattr(box, "mask", None)
        if mask is None or getattr(mask, "ndim", 0) != 2:
            continue
        x1, y1, x2, y2 = (
            int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        )
        dst_x1 = max(0, min(width, x1))
        dst_y1 = max(0, min(height, y1))
        dst_x2 = max(dst_x1, min(width, x2))
        dst_y2 = max(dst_y1, min(height, y2))
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            continue

        src_x1 = dst_x1 - x1
        src_y1 = dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        crop = mask[src_y1:src_y2, src_x1:src_x2]
        if crop.shape != (dst_y2 - dst_y1, dst_x2 - dst_x1):
            continue
        target = result[dst_y1:dst_y2, dst_x1:dst_x2]
        result[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(
            target,
            crop.astype(np.uint8, copy=False),
        )
    return result


def _read_images(raw_dir: Path, max_pages: int) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[: max(1, int(max_pages))]
    if not paths:
        raise RuntimeError(f"No benchmark images in {raw_dir}")

    images: list[tuple[Path, np.ndarray]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read benchmark image: {path}")
        images.append((path, image))
    return images


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in raw.split(",") if item.strip()}))
    if not values:
        raise ValueError("Expected at least one float")
    return values


def _center_hits(reference_boxes, candidate_boxes) -> int:
    hits = 0
    for reference in reference_boxes:
        cx = (float(reference.x1) + float(reference.x2)) * 0.5
        cy = (float(reference.y1) + float(reference.y2)) * 0.5
        if any(
            float(candidate.x1) <= cx <= float(candidate.x2)
            and float(candidate.y1) <= cy <= float(candidate.y2)
            for candidate in candidate_boxes
        ):
            hits += 1
    return hits


class BenchmarkSegmentationDetector:
    """Use production decode semantics without relaxing production contracts."""

    def __init__(
        self,
        model_path: Path,
        *,
        conf_threshold: float,
        input_size: int,
        provider_override: str,
    ) -> None:
        from app.detector.bubble_detector import YoloDetector
        from app.ort_utils import make_session

        self._decoder = object.__new__(YoloDetector)
        decoder = self._decoder
        decoder.model_path = str(model_path)
        decoder.source_model = model_path.name
        decoder.model_role = "text_segmenter"
        decoder.input_size = int(input_size)
        decoder.session = make_session(
            model_path,
            provider_override=provider_override,
        )
        inputs = decoder.session.get_inputs()
        outputs = decoder.session.get_outputs()
        if len(inputs) != 1:
            raise RuntimeError(f"{model_path}: expected one input, got {len(inputs)}")
        input_shape = list(inputs[0].shape)
        if input_shape != [1, 3, int(input_size), int(input_size)]:
            raise RuntimeError(
                f"{model_path}: expected static [1,3,{input_size},{input_size}], "
                f"got {input_shape}"
            )
        if len(outputs) < 2 or len(outputs[1].shape) != 4:
            raise RuntimeError(
                f"{model_path}: expected YOLO segmentation output + prototypes, "
                f"got {[list(item.shape) for item in outputs]}"
            )

        decoder.contract = SimpleNamespace(
            input_name=inputs[0].name,
            class_names=CANDIDATE_CLASS_NAMES,
        )
        decoder.input_name = inputs[0].name
        decoder.conf_threshold = float(conf_threshold)
        decoder.use_tta = False

    @property
    def session(self):
        return self._decoder.session

    def detect(self, image: np.ndarray):
        return self._decoder.detect(image)


def run(args) -> dict:
    from app.config import TEXT_SEGMENTER_MODEL
    from app.detector.bubble_detector import YoloDetector
    from app.parameters import TEXT_CONF_THRESHOLD

    images = _read_images(args.raw_dir, args.max_pages)
    sizes = _parse_ints(args.sizes)
    thresholds = _parse_floats(args.thresholds)
    min_threshold = min(thresholds)

    baseline = YoloDetector(
        TEXT_SEGMENTER_MODEL,
        TEXT_CONF_THRESHOLD,
        model_role="text_segmenter",
        provider_override=args.provider,
    )
    candidates = {
        size: BenchmarkSegmentationDetector(
            args.candidate_dir / f"yolo26s_manga_seg_{size}.onnx",
            conf_threshold=min_threshold,
            input_size=size,
            provider_override=args.provider,
        )
        for size in sizes
    }

    warmup = images[0][1]
    baseline.detect(warmup)
    for candidate in candidates.values():
        candidate.detect(warmup)

    baseline_ms = 0.0
    candidate_ms = {size: 0.0 for size in sizes}
    total_reference_pixels = 0
    total_reference_boxes = 0
    total_page_pixels = 0
    aggregates = {
        (size, threshold): {
            "candidate_pixels": 0,
            "intersection_pixels": 0,
            "union_pixels": 0,
            "false_positive_pixels": 0,
            "center_hits": 0,
            "candidate_boxes": 0,
        }
        for size in sizes
        for threshold in thresholds
    }
    per_page = []

    for index, (path, image) in enumerate(images):
        started = time.perf_counter()
        reference_boxes = [
            box
            for box in baseline.detect(image)
            if box.verified_mask and box.safe_to_inpaint
        ]
        baseline_ms += (time.perf_counter() - started) * 1000.0
        reference_mask = candidate_mask_union(image.shape, reference_boxes)
        reference_pixels = int(np.count_nonzero(reference_mask > 0))
        total_reference_pixels += reference_pixels
        total_reference_boxes += len(reference_boxes)
        total_page_pixels += int(image.shape[0] * image.shape[1])

        page_row = {
            "index": index,
            "file": path.name,
            "reference_boxes": len(reference_boxes),
            "reference_pixels": reference_pixels,
            "candidates": {},
        }

        for size, candidate in candidates.items():
            started = time.perf_counter()
            detected = candidate.detect(image)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate_ms[size] += elapsed_ms
            text_boxes = [
                box
                for box in detected
                if int(box.class_id) == int(args.text_class_id)
                and box.verified_mask
            ]
            page_row["candidates"][str(size)] = {
                "detect_ms": elapsed_ms,
                "text_boxes_at_min_conf": len(text_boxes),
            }

            for threshold in thresholds:
                selected = [
                    box
                    for box in text_boxes
                    if float(box.confidence) >= float(threshold)
                ]
                candidate_mask = candidate_mask_union(image.shape, selected)
                metrics = mask_metrics(reference_mask, candidate_mask)
                aggregate = aggregates[(size, threshold)]
                aggregate["candidate_pixels"] += metrics["candidate_pixels"]
                aggregate["intersection_pixels"] += metrics["intersection_pixels"]
                aggregate["union_pixels"] += metrics["union_pixels"]
                aggregate["false_positive_pixels"] += metrics["false_positive_pixels"]
                aggregate["center_hits"] += _center_hits(reference_boxes, selected)
                aggregate["candidate_boxes"] += len(selected)

        per_page.append(page_row)

    page_count = max(1, len(images))
    baseline_mean_ms = baseline_ms / page_count
    rows = []
    for (size, threshold), aggregate in sorted(aggregates.items()):
        candidate_pixels = int(aggregate["candidate_pixels"])
        intersection_pixels = int(aggregate["intersection_pixels"])
        union_pixels = int(aggregate["union_pixels"])
        false_positive_pixels = int(aggregate["false_positive_pixels"])
        pixel_recall = (
            intersection_pixels / float(total_reference_pixels)
            if total_reference_pixels
            else 1.0
        )
        pixel_precision = (
            intersection_pixels / float(candidate_pixels)
            if candidate_pixels
            else 1.0
        )
        pixel_iou = (
            intersection_pixels / float(union_pixels)
            if union_pixels
            else 1.0
        )
        center_recall = (
            int(aggregate["center_hits"]) / float(total_reference_boxes)
            if total_reference_boxes
            else 1.0
        )
        mean_ms = candidate_ms[size] / page_count
        rows.append({
            "size": int(size),
            "threshold": float(threshold),
            "pixel_recall": pixel_recall,
            "pixel_precision": pixel_precision,
            "pixel_iou": pixel_iou,
            "center_recall": center_recall,
            "candidate_area_ratio": (
                candidate_pixels / float(total_page_pixels)
                if total_page_pixels
                else 0.0
            ),
            "false_positive_pixels_vs_reference": false_positive_pixels,
            "mean_text_boxes_per_page": aggregate["candidate_boxes"] / page_count,
            "candidate_mean_ms": mean_ms,
            "candidate_to_baseline_latency": mean_ms / max(1e-9, baseline_mean_ms),
            "destructive_authority_enabled": False,
        })

    return {
        "pages": len(images),
        "text_class_id": int(args.text_class_id),
        "class_names": list(CANDIDATE_CLASS_NAMES),
        "sizes": list(sizes),
        "thresholds": list(thresholds),
        "reference": {
            "model": Path(TEXT_SEGMENTER_MODEL).name,
            "providers": list(baseline.session.get_providers()),
            "total_ms": baseline_ms,
            "mean_ms": baseline_mean_ms,
            "reference_boxes": total_reference_boxes,
            "authority_pixels": total_reference_pixels,
        },
        "candidates": {
            str(size): {
                "model": f"yolo26s_manga_seg_{size}.onnx",
                "providers": list(candidates[size].session.get_providers()),
                "total_ms": candidate_ms[size],
                "mean_ms": candidate_ms[size] / page_count,
            }
            for size in sizes
        },
        "rows": rows,
        "per_page": per_page,
        "safety": {
            "benchmark_only": True,
            "production_model_replaced": False,
            "destructive_authority_granted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--text-class-id", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=32)
    parser.add_argument("--sizes", default="640,768,1024")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--provider", default="openvino")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = sorted(
        report["rows"],
        key=lambda row: (
            -row["pixel_recall"],
            -row["pixel_precision"],
            row["candidate_to_baseline_latency"],
        ),
    )
    print(json.dumps({
        "pages": report["pages"],
        "reference": report["reference"],
        "candidates": report["candidates"],
        "best_rows": rows[:12],
        "safety": report["safety"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
