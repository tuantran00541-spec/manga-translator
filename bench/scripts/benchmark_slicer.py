from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.downloader.slicer import (
    SAFE_CUT_BAND,
    _find_cut_rows,
    _get_content_row_mask,
)
from app.detector.bubble_detector import YoloDetector

try:
    import psutil
except ImportError:
    psutil = None


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": _percentile(list(values), 50),
        "p95": _percentile(list(values), 95),
        "max": max(values),
    }


def _image_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "sliced" not in path.parts
    )


def _load_gray(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _load_color(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _measure_page(path: Path, strategy: str) -> tuple[dict, float, float]:
    started = time.perf_counter()
    gray = _load_gray(path)
    height, width = gray.shape[:2]
    if strategy == "current":
        unsafe_rows = _get_content_row_mask(gray, height, width)
    else:
        unsafe_rows = np.ones(height, dtype=bool)
    cuts = _find_cut_rows(gray, height, width, unsafe_rows)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    boundaries = [0, *cuts, height]
    slices = [
        {"y_start": start, "y_end": end, "height": end - start}
        for start, end in zip(boundaries, boundaries[1:])
    ]
    continuity = all(
        left["y_end"] == right["y_start"] for left, right in zip(slices, slices[1:])
    )
    coverage = sum(item["height"] for item in slices)

    unsafe_cuts = None
    if strategy == "current":
        unsafe_cuts = [
            cut
            for cut in cuts
            if unsafe_rows[max(0, cut - SAFE_CUT_BAND) : cut + SAFE_CUT_BAND + 1].any()
        ]
    rss_mb = 0.0
    if psutil is not None:
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    return (
        {
            "path": str(path),
            "width": width,
            "height": height,
            "slice_count": len(slices),
            "slices": slices,
            "cut_positions": cuts,
            "unsafe_cuts_current_content_mask": unsafe_cuts,
            "reconstruction": {
                "continuous": continuity,
                "covered_rows": coverage,
                "duplicate_rows": max(0, coverage - height),
                "missing_rows": max(0, height - coverage),
            },
        },
        elapsed_ms,
        rss_mb,
    )


def _detector_cut_metrics(
    paths: list[Path],
    pages: list[dict],
    model_path: Path,
    comparison_report: Path | None,
) -> dict:
    detector = YoloDetector(model_path, conf_threshold=0.4)
    comparison_cuts: dict[str, list[int]] = {}
    if comparison_report:
        comparison = json.loads(comparison_report.read_text(encoding="utf-8"))
        comparison_cuts = {
            str(page["path"]): page["cut_positions"]
            for page in comparison["page_details"]
        }

    total_boxes = 0
    current_crossed_boxes = 0
    compared_crossed_boxes = 0
    started = time.perf_counter()
    for path, page in zip(paths, pages):
        boxes = detector.detect(_load_color(path))
        total_boxes += len(boxes)
        current_cuts = page["cut_positions"]
        current_crossed_boxes += sum(
            any(box.y1 < cut < box.y2 for cut in current_cuts) for box in boxes
        )
        if comparison_cuts:
            other_cuts = comparison_cuts.get(page["path"], [])
            compared_crossed_boxes += sum(
                any(box.y1 < cut < box.y2 for cut in other_cuts) for box in boxes
            )

    result = {
        "model": str(model_path),
        "source_page_boxes": total_boxes,
        "crossed_boxes": current_crossed_boxes,
        "runtime_ms": round((time.perf_counter() - started) * 1_000, 3),
    }
    if comparison_cuts:
        result["comparison_report"] = str(comparison_report)
        result["comparison_crossed_boxes"] = compared_crossed_boxes
    return result


def benchmark(
    root: Path,
    repeat: int,
    strategy: str,
    detector_model: Path | None = None,
    comparison_report: Path | None = None,
) -> dict:
    paths = _image_paths(root)
    if not paths:
        raise ValueError(f"No supported images found under {root}")

    runs: list[list[dict]] = []
    all_timings: list[float] = []
    peak_rss_mb = 0.0
    for _ in range(repeat):
        run: list[dict] = []
        for path in paths:
            result, elapsed_ms, rss_mb = _measure_page(path, strategy)
            run.append(result)
            all_timings.append(elapsed_ms)
            peak_rss_mb = max(peak_rss_mb, rss_mb)
        runs.append(run)

    pages = runs[0]
    slice_heights = [item["height"] for page in pages for item in page["slices"]]
    slice_counts = [page["slice_count"] for page in pages]
    unsafe_cuts = [
        cut
        for page in pages
        for cut in (page["unsafe_cuts_current_content_mask"] or [])
    ]
    report = {
        "input": str(root),
        "strategy": strategy,
        "repeat": repeat,
        "pages": len(pages),
        "total_slices": sum(slice_counts),
        "slice_count_per_page": _summary(slice_counts),
        "slice_height": _summary(slice_heights),
        "runtime_ms_per_page": _summary(all_timings),
        "total_runtime_ms": round(sum(all_timings), 3),
        "peak_rss_mb": round(peak_rss_mb, 3) if psutil is not None else None,
        "unsafe_cuts_current_content_mask": (
            len(unsafe_cuts) if strategy == "current" else None
        ),
        "all_reconstructions_continuous": all(
            page["reconstruction"]["continuous"]
            and page["reconstruction"]["duplicate_rows"] == 0
            and page["reconstruction"]["missing_rows"] == 0
            for page in pages
        ),
        "page_details": pages,
    }
    if detector_model:
        report["detector_cut_metrics"] = _detector_cut_metrics(
            paths, pages, detector_model, comparison_report
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure deterministic long-page slicing without writing derived images.")
    parser.add_argument("input", type=Path, help="Image file or directory of source pages")
    parser.add_argument("--repeat", type=int, default=3, help="Planner repetitions (default: 3)")
    parser.add_argument(
        "--strategy",
        choices=("current", "score-only"),
        default="current",
        help="Planner path to benchmark (default: current)",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        help="Optional bubble YOLO model for measured cut-crossing metrics",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        help="Optional slicer JSON report whose cuts are evaluated with the same detector boxes",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    report = benchmark(
        args.input,
        args.repeat,
        args.strategy,
        args.detector_model,
        args.comparison_report,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
