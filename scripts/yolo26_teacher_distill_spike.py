#!/usr/bin/env python3
"""Build a benchmark-only YOLO segmentation dataset from production authority masks.

The production YOLOv8m text segmenter remains the teacher and destructive-authority
oracle. This spike only materializes verified, safe teacher masks as one-class
YOLO polygons for a YOLO26s shadow fine-tune; it never replaces production models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
    return {
        "reference_pixels": reference_pixels,
        "candidate_pixels": candidate_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "false_positive_pixels": false_positive_pixels,
        "pixel_recall": (
            intersection_pixels / float(reference_pixels)
            if reference_pixels
            else 1.0
        ),
        "pixel_precision": (
            intersection_pixels / float(candidate_pixels)
            if candidate_pixels
            else 1.0
        ),
        "pixel_iou": (
            intersection_pixels / float(union_pixels)
            if union_pixels
            else 1.0
        ),
    }


def _union_box_mask(
    result: np.ndarray,
    box,
) -> None:
    height, width = result.shape
    mask = getattr(box, "mask", None)
    if mask is None or getattr(mask, "ndim", 0) != 2:
        return

    x1, y1, x2, y2 = (
        int(box.x1),
        int(box.y1),
        int(box.x2),
        int(box.y2),
    )
    dst_x1 = max(0, min(width, x1))
    dst_y1 = max(0, min(height, y1))
    dst_x2 = max(dst_x1, min(width, x2))
    dst_y2 = max(dst_y1, min(height, y2))
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    src_x1 = dst_x1 - x1
    src_y1 = dst_y1 - y1
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    crop = np.asarray(mask)[src_y1:src_y2, src_x1:src_x2]
    if crop.shape != (dst_y2 - dst_y1, dst_x2 - dst_x1):
        return
    result[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(
        result[dst_y1:dst_y2, dst_x1:dst_x2],
        crop.astype(np.uint8, copy=False),
    )


def teacher_authority_mask(shape: tuple[int, ...], boxes) -> np.ndarray:
    """Union only production-verified masks that already have erase authority."""
    height, width = int(shape[0]), int(shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        if not bool(getattr(box, "verified_mask", False)):
            continue
        if not bool(getattr(box, "safe_to_inpaint", False)):
            continue
        _union_box_mask(result, box)
    return result


def _rectangle_polygon_for_component(points: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2).astype(np.int32))
    x2 = max(x, x + w - 1)
    y2 = max(y, y + h - 1)
    return np.asarray(
        [[x, y], [x2, y], [x2, y2], [x, y2]],
        dtype=np.float32,
    )


def mask_to_yolo_polygons(mask: np.ndarray) -> list[list[float]]:
    """Convert a binary authority union into one normalized polygon per component."""
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.ndim != 2:
        raise ValueError(f"mask must be rank-2; got {binary.shape}")
    height, width = binary.shape
    if height <= 0 or width <= 0:
        return []

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(
        contours,
        key=lambda contour: cv2.boundingRect(contour)[:2][::-1],
    )

    polygons: list[list[float]] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.size == 0:
            continue
        if len(points) < 3:
            points = _rectangle_polygon_for_component(points)
        if len(points) < 3:
            continue

        normalized: list[float] = []
        for x, y in points:
            normalized.extend(
                [
                    min(1.0, max(0.0, float(x) / float(width))),
                    min(1.0, max(0.0, float(y) / float(height))),
                ]
            )
        if len(normalized) >= 6:
            polygons.append(normalized)
    return polygons


def rasterize_yolo_polygons(
    polygons: list[list[float]],
    shape: tuple[int, ...],
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return result

    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        coords = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.empty_like(coords, dtype=np.int32)
        points[:, 0] = np.clip(
            np.rint(coords[:, 0] * width).astype(np.int32),
            0,
            width - 1,
        )
        points[:, 1] = np.clip(
            np.rint(coords[:, 1] * height).astype(np.int32),
            0,
            height - 1,
        )
        cv2.fillPoly(result, [points], 255)
    return result


def dataset_split_for_index(index: int, *, validation_stride: int = 4) -> str:
    stride = int(validation_stride)
    if stride < 2:
        raise ValueError("validation_stride must be >= 2")
    return "val" if int(index) % stride == 0 else "train"


def _read_images(raw_dir: Path, max_pages: int) -> list[Path]:
    paths = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[: max(1, int(max_pages))]
    if not paths:
        raise RuntimeError(f"No benchmark images in {raw_dir}")
    return paths


def _format_label(polygons: list[list[float]]) -> str:
    rows = []
    for polygon in polygons:
        coords = " ".join(f"{value:.8f}" for value in polygon)
        rows.append(f"0 {coords}")
    return ("\n".join(rows) + "\n") if rows else ""


def materialize_teacher_dataset(args) -> dict:
    from app.config import TEXT_SEGMENTER_MODEL
    from app.detector.bubble_detector import YoloDetector
    from app.parameters import TEXT_CONF_THRESHOLD

    image_paths = _read_images(args.raw_dir, args.max_pages)
    detector = YoloDetector(
        TEXT_SEGMENTER_MODEL,
        TEXT_CONF_THRESHOLD,
        model_role="text_segmenter",
        provider_override=args.provider,
    )

    output_dir = args.output_dir.resolve()
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Exclude one-time graph compilation from the per-page teacher timing.
    warmup = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if warmup is None:
        raise RuntimeError(f"Failed to read benchmark image: {image_paths[0]}")
    detector.detect(warmup)

    total_reference_pixels = 0
    total_reconstructed_pixels = 0
    total_intersection_pixels = 0
    total_union_pixels = 0
    total_false_positive_pixels = 0
    total_polygons = 0
    total_authority_boxes = 0
    split_counts = {"train": 0, "val": 0}
    pages = []

    for index, source in enumerate(image_paths):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read benchmark image: {source}")

        detected = detector.detect(image)
        authority_boxes = [
            box
            for box in detected
            if box.verified_mask and box.safe_to_inpaint
        ]
        authority = teacher_authority_mask(image.shape, authority_boxes)
        polygons = mask_to_yolo_polygons(authority)
        reconstructed = rasterize_yolo_polygons(polygons, authority.shape)
        metrics = mask_metrics(authority, reconstructed)
        split = dataset_split_for_index(
            index,
            validation_stride=args.validation_stride,
        )

        stem = f"{index:04d}"
        image_target = output_dir / "images" / split / f"{stem}{source.suffix.lower()}"
        label_target = output_dir / "labels" / split / f"{stem}.txt"
        shutil.copy2(source, image_target)
        label_target.write_text(_format_label(polygons), encoding="utf-8")

        split_counts[split] += 1
        total_polygons += len(polygons)
        total_authority_boxes += len(authority_boxes)
        total_reference_pixels += metrics["reference_pixels"]
        total_reconstructed_pixels += metrics["candidate_pixels"]
        total_intersection_pixels += metrics["intersection_pixels"]
        total_union_pixels += metrics["union_pixels"]
        total_false_positive_pixels += metrics["false_positive_pixels"]
        pages.append(
            {
                "index": index,
                "source": source.name,
                "split": split,
                "authority_boxes": len(authority_boxes),
                "polygons": len(polygons),
                **metrics,
            }
        )

    if split_counts["train"] == 0 or split_counts["val"] == 0:
        raise RuntimeError(f"dataset split is empty: {split_counts}")

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: text_comic",
                "",
            ]
        ),
        encoding="utf-8",
    )

    aggregate = {
        "reference_pixels": total_reference_pixels,
        "candidate_pixels": total_reconstructed_pixels,
        "intersection_pixels": total_intersection_pixels,
        "union_pixels": total_union_pixels,
        "false_positive_pixels": total_false_positive_pixels,
        "pixel_recall": (
            total_intersection_pixels / float(total_reference_pixels)
            if total_reference_pixels
            else 1.0
        ),
        "pixel_precision": (
            total_intersection_pixels / float(total_reconstructed_pixels)
            if total_reconstructed_pixels
            else 1.0
        ),
        "pixel_iou": (
            total_intersection_pixels / float(total_union_pixels)
            if total_union_pixels
            else 1.0
        ),
    }
    report = {
        "pages": len(image_paths),
        "split_counts": split_counts,
        "authority_boxes": total_authority_boxes,
        "polygons": total_polygons,
        "polygon_reconstruction": aggregate,
        "teacher": {
            "model": Path(TEXT_SEGMENTER_MODEL).name,
            "providers": list(detector.session.get_providers()),
            "destructive_authority_source": True,
        },
        "student_target": {
            "class_names": ["text_comic"],
            "destructive_authority_enabled": False,
            "benchmark_only": True,
        },
        "dataset_yaml": str(dataset_yaml),
        "per_page": pages,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=32)
    parser.add_argument("--validation-stride", type=int, default=4)
    parser.add_argument("--provider", default="openvino")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = materialize_teacher_dataset(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pages": report["pages"],
        "split_counts": report["split_counts"],
        "authority_boxes": report["authority_boxes"],
        "polygons": report["polygons"],
        "polygon_reconstruction": report["polygon_reconstruction"],
        "teacher": report["teacher"],
        "student_target": report["student_target"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
