#!/usr/bin/env python3
"""Measure whether a lightweight YOLO26 manga text detector can safely seed ROIs.

This is a benchmark-only spike. YOLO26 proposals never become destructive
authority. The validated production text segmenter remains the reference for
stroke masks; this script asks whether padded YOLO26 text boxes cover those
reference pixels well enough to justify a later ROI/cascade experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20)
DEFAULT_PADS = (0, 16, 32, 64, 96)


def build_proposal_union(
    shape: tuple[int, ...],
    proposals: list[tuple[int, int, int, int]],
    *,
    pad: int,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    pad = max(0, int(pad))
    for x1, y1, x2, y2 in proposals:
        x1 = max(0, min(width, int(x1) - pad))
        y1 = max(0, min(height, int(y1) - pad))
        x2 = max(x1, min(width, int(x2) + pad))
        y2 = max(y1, min(height, int(y2) + pad))
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 255
    return result


def mask_coverage(reference_mask: np.ndarray, proposal_union: np.ndarray) -> float:
    reference = np.asarray(reference_mask) > 0
    total = int(np.count_nonzero(reference))
    if total == 0:
        return 1.0
    covered = int(np.count_nonzero(reference & (np.asarray(proposal_union) > 0)))
    return covered / float(total)


def _center_hits(
    references: list[tuple[int, int, int, int]],
    proposals: list[tuple[int, int, int, int]],
    *,
    pad: int,
) -> int:
    pad = max(0, int(pad))
    hits = 0
    for x1, y1, x2, y2 in references:
        cx = (float(x1) + float(x2)) * 0.5
        cy = (float(y1) + float(y2)) * 0.5
        if any(
            (px1 - pad) <= cx <= (px2 + pad)
            and (py1 - pad) <= cy <= (py2 + pad)
            for px1, py1, px2, py2 in proposals
        ):
            hits += 1
    return hits


def center_recall(
    references: list[tuple[int, int, int, int]],
    proposals: list[tuple[int, int, int, int]],
    *,
    pad: int,
) -> float:
    if not references:
        return 1.0
    return _center_hits(references, proposals, pad=pad) / float(len(references))


def merge_proposal_sets(
    *proposal_sets: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge proposal geometry while preserving first-seen order."""
    merged: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for proposal_set in proposal_sets:
        for proposal in proposal_set:
            geometry = tuple(int(value) for value in proposal)
            if len(geometry) != 4 or geometry in seen:
                continue
            seen.add(geometry)
            merged.append(geometry)
    return merged


def _authority_mask(shape: tuple[int, ...], boxes) -> np.ndarray:
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
        result[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(
            result[dst_y1:dst_y2, dst_x1:dst_x2],
            crop.astype(np.uint8, copy=False),
        )
    return result


def _boxes(boxes) -> list[tuple[int, int, int, int]]:
    return [
        (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
        for box in boxes
    ]


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


def run(args) -> dict:
    from app.config import BUBBLE_DETECTOR_MODEL, TEXT_SEGMENTER_MODEL
    from app.detector.bubble_detector import YoloDetector
    from app.detector.recovery import SecondaryTextRecovery
    from app.parameters import BUBBLE_PROPOSAL_CONF_THRESHOLD, TEXT_CONF_THRESHOLD

    images = _read_images(args.raw_dir, args.max_pages)
    thresholds = _parse_floats(args.thresholds)
    pads = _parse_ints(args.pads)
    min_threshold = min(thresholds)

    baseline = YoloDetector(
        TEXT_SEGMENTER_MODEL,
        TEXT_CONF_THRESHOLD,
        model_role="text_segmenter",
        provider_override=args.provider,
    )
    bubble_detector = YoloDetector(
        BUBBLE_DETECTOR_MODEL,
        BUBBLE_PROPOSAL_CONF_THRESHOLD,
        model_role="bubble_detector",
        provider_override=args.provider,
    )
    recovery = SecondaryTextRecovery()
    candidate_paths = {
        640: args.candidate_640,
        1024: args.candidate_1024,
    }
    candidates = {
        size: YoloDetector(
            path,
            min_threshold,
            use_tta=False,
            model_role="bubble_detector",
            input_size=size,
            provider_override=args.provider,
        )
        for size, path in candidate_paths.items()
    }

    # Exclude one-time OpenVINO graph compilation and allocator warmup from the
    # steady-state detector comparison.
    warmup = images[0][1]
    baseline.detect(warmup)
    bubble_detector.detect(warmup)
    recovery.detect(warmup, existing=[])
    for detector in candidates.values():
        detector.detect(warmup)

    baseline_ms = 0.0
    bubble_ms = 0.0
    mser_ms = 0.0
    candidate_ms = {size: 0.0 for size in candidates}
    total_authority_pixels = 0
    total_reference_boxes = 0
    total_page_pixels = 0
    source_stacks = ("yolo26", "bubble+yolo26", "bubble+yolo26+mser")
    aggregates = {
        (source_stack, size, threshold, pad): {
            "covered_authority_pixels": 0,
            "covered_centers": 0,
            "proposal_union_pixels": 0,
            "proposal_count": 0,
        }
        for source_stack in source_stacks
        for size in candidates
        for threshold in thresholds
        for pad in pads
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

        reference_mask = _authority_mask(image.shape, reference_boxes)
        reference_geometry = _boxes(reference_boxes)
        authority_pixels = int(np.count_nonzero(reference_mask > 0))
        page_pixels = int(image.shape[0] * image.shape[1])
        total_authority_pixels += authority_pixels
        total_reference_boxes += len(reference_geometry)
        total_page_pixels += page_pixels

        bubble_started = time.perf_counter()
        bubble_boxes = bubble_detector.detect(image)
        bubble_ms += (time.perf_counter() - bubble_started) * 1000.0
        bubble_geometry = _boxes(bubble_boxes)

        mser_started = time.perf_counter()
        mser_boxes = recovery.detect(image, existing=[])
        mser_ms += (time.perf_counter() - mser_started) * 1000.0
        mser_geometry = _boxes(mser_boxes)

        page_row = {
            "index": index,
            "file": path.name,
            "reference_boxes": len(reference_geometry),
            "authority_pixels": authority_pixels,
            "bubble_proposals": len(bubble_geometry),
            "mser_proposals": len(mser_geometry),
            "candidates": {},
        }

        for size, detector in candidates.items():
            started = time.perf_counter()
            detected = detector.detect(image)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate_ms[size] += elapsed_ms
            text_boxes = [
                box for box in detected if int(box.class_id) == int(args.text_class_id)
            ]
            size_row = {
                "detect_ms": elapsed_ms,
                "text_proposals_at_min_conf": len(text_boxes),
            }
            page_row["candidates"][str(size)] = size_row

            for threshold in thresholds:
                selected = [
                    box for box in text_boxes
                    if float(box.confidence) >= float(threshold)
                ]
                yolo26_geometry = _boxes(selected)
                stack_geometry = {
                    "yolo26": yolo26_geometry,
                    "bubble+yolo26": merge_proposal_sets(
                        bubble_geometry,
                        yolo26_geometry,
                    ),
                    "bubble+yolo26+mser": merge_proposal_sets(
                        bubble_geometry,
                        yolo26_geometry,
                        mser_geometry,
                    ),
                }
                for source_stack, proposal_geometry in stack_geometry.items():
                    for pad in pads:
                        union = build_proposal_union(
                            image.shape,
                            proposal_geometry,
                            pad=pad,
                        )
                        key = (source_stack, size, threshold, pad)
                        aggregate = aggregates[key]
                        aggregate["covered_authority_pixels"] += int(
                            np.count_nonzero(
                                (reference_mask > 0) & (union > 0)
                            )
                        )
                        aggregate["covered_centers"] += _center_hits(
                            reference_geometry,
                            proposal_geometry,
                            pad=pad,
                        )
                        aggregate["proposal_union_pixels"] += int(
                            np.count_nonzero(union > 0)
                        )
                        aggregate["proposal_count"] += len(proposal_geometry)

        per_page.append(page_row)

    rows = []
    page_count = max(1, len(images))
    baseline_mean_ms = baseline_ms / page_count
    bubble_mean_ms = bubble_ms / page_count
    mser_mean_ms = mser_ms / page_count
    for (source_stack, size, threshold, pad), aggregate in sorted(aggregates.items()):
        authority_pixel_coverage = (
            aggregate["covered_authority_pixels"] / float(total_authority_pixels)
            if total_authority_pixels
            else 1.0
        )
        recall = (
            aggregate["covered_centers"] / float(total_reference_boxes)
            if total_reference_boxes
            else 1.0
        )
        area_ratio = (
            aggregate["proposal_union_pixels"] / float(total_page_pixels)
            if total_page_pixels
            else 0.0
        )
        candidate_mean_ms = candidate_ms[size] / page_count
        stack_mean_ms = candidate_mean_ms
        if "bubble" in source_stack:
            stack_mean_ms += bubble_mean_ms
        if "mser" in source_stack:
            stack_mean_ms += mser_mean_ms
        rows.append({
            "source_stack": source_stack,
            "size": size,
            "threshold": threshold,
            "pad": pad,
            "authority_pixel_coverage": authority_pixel_coverage,
            "center_recall": recall,
            "proposal_area_ratio": area_ratio,
            "mean_proposals_per_page": aggregate["proposal_count"] / page_count,
            "candidate_mean_ms": candidate_mean_ms,
            "candidate_to_baseline_latency": candidate_mean_ms / max(1e-9, baseline_mean_ms),
            "stack_mean_ms": stack_mean_ms,
            "stack_to_baseline_latency": stack_mean_ms / max(1e-9, baseline_mean_ms),
            "strict_promising": bool(
                authority_pixel_coverage >= 1.0
                and recall >= 1.0
                and stack_mean_ms <= baseline_mean_ms * 0.30
            ),
        })

    return {
        "pages": len(images),
        "text_class_id": int(args.text_class_id),
        "thresholds": list(thresholds),
        "pads": list(pads),
        "reference": {
            "model": Path(TEXT_SEGMENTER_MODEL).name,
            "providers": list(baseline.session.get_providers()),
            "total_ms": baseline_ms,
            "mean_ms": baseline_mean_ms,
            "reference_boxes": total_reference_boxes,
            "authority_pixels": total_authority_pixels,
        },
        "proposal_sources": {
            "bubble": {
                "model": Path(BUBBLE_DETECTOR_MODEL).name,
                "providers": list(bubble_detector.session.get_providers()),
                "total_ms": bubble_ms,
                "mean_ms": bubble_mean_ms,
            },
            "mser": {
                "model": "opencv_mser",
                "total_ms": mser_ms,
                "mean_ms": mser_mean_ms,
            },
        },
        "candidates": {
            str(size): {
                "model": Path(candidate_paths[size]).name,
                "providers": list(candidates[size].session.get_providers()),
                "total_ms": candidate_ms[size],
                "mean_ms": candidate_ms[size] / page_count,
            }
            for size in candidates
        },
        "rows": rows,
        "strict_promising_rows": [row for row in rows if row["strict_promising"]],
        "per_page": per_page,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--candidate-640", type=Path, required=True)
    parser.add_argument("--candidate-1024", type=Path, required=True)
    parser.add_argument("--text-class-id", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=32)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--pads",
        default=",".join(str(value) for value in DEFAULT_PADS),
    )
    parser.add_argument("--provider", default="openvino")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pages": report["pages"],
        "reference": report["reference"],
        "proposal_sources": report["proposal_sources"],
        "candidates": report["candidates"],
        "strict_promising_rows": report["strict_promising_rows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
