from __future__ import annotations

import argparse
import json
import re
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


DATA_ROOT = (PROJECT_ROOT / "data").resolve()
BENCH_ROOT = (PROJECT_ROOT / "bench").resolve()
MODELS_ROOT = (PROJECT_ROOT / "models").resolve()
RESULTS_ROOT = (BENCH_ROOT / "results").resolve()
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_RESULT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_bounded_path(
    raw: str | Path,
    *,
    allowed_roots: tuple[Path, ...],
    must_exist: bool = True,
    must_be_file: bool = False,
) -> Path:
    text = str(raw).strip()
    if not text or "\x00" in text:
        raise argparse.ArgumentTypeError("path is empty or invalid")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not any(_is_within(resolved, root) for root in allowed_roots):
        roots = ", ".join(str(root.relative_to(PROJECT_ROOT)) for root in allowed_roots)
        raise argparse.ArgumentTypeError(f"path must stay inside: {roots}")
    if must_exist and not resolved.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {resolved}")
    if must_be_file and not resolved.is_file():
        raise argparse.ArgumentTypeError(f"expected a file: {resolved}")
    return resolved


def _input_path(raw: str) -> Path:
    return _resolve_bounded_path(raw, allowed_roots=(DATA_ROOT, BENCH_ROOT))


def _model_path(raw: str) -> Path:
    return _resolve_bounded_path(
        raw,
        allowed_roots=(MODELS_ROOT,),
        must_exist=True,
        must_be_file=True,
    )


def _comparison_path(raw: str) -> Path:
    return _resolve_bounded_path(
        raw,
        allowed_roots=(RESULTS_ROOT,),
        must_exist=True,
        must_be_file=True,
    )


def _result_path(raw: str) -> Path:
    name = raw.strip()
    if not _RESULT_NAME_RE.fullmatch(name) or Path(name).name != name:
        raise argparse.ArgumentTypeError(
            "--output must be a JSON filename such as slicer-current.json"
        )
    return RESULTS_ROOT / name


def _safe_discovered_image(path: Path) -> Path | None:
    resolved = path.resolve()
    if not any(_is_within(resolved, root) for root in (DATA_ROOT, BENCH_ROOT)):
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return resolved


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
    root = _resolve_bounded_path(root, allowed_roots=(DATA_ROOT, BENCH_ROOT))
    if root.is_file():
        safe = _safe_discovered_image(root)
        return [safe] if safe is not None else []

    paths: list[Path] = []
    for path in root.rglob("*"):
        if "sliced" in path.parts:
            continue
        safe = _safe_discovered_image(path)
        if safe is not None:
            paths.append(safe)
    return sorted(set(paths))


def _load_gray(path: Path) -> np.ndarray:
    safe = _resolve_bounded_path(
        path,
        allowed_roots=(DATA_ROOT, BENCH_ROOT),
        must_exist=True,
        must_be_file=True,
    )
    encoded = np.fromfile(str(safe), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {safe}")
    return image


def _load_color(path: Path) -> np.ndarray:
    safe = _resolve_bounded_path(
        path,
        allowed_roots=(DATA_ROOT, BENCH_ROOT),
        must_exist=True,
        must_be_file=True,
    )
    encoded = np.fromfile(str(safe), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {safe}")
    return image


def _measure_page(path: Path, strategy: str) -> tuple[dict, float, float]:
    started = time.perf_counter()
    gray = _load_gray(path)
    height, width = gray.shape[:2]
    unsafe_rows = (
        _get_content_row_mask(gray, height, width)
        if strategy == "current"
        else np.ones(height, dtype=bool)
    )
    cuts = _find_cut_rows(gray, height, width, unsafe_rows)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    boundaries = [0, *cuts, height]
    slices = [
        {"y_start": start, "y_end": end, "height": end - start}
        for start, end in zip(boundaries, boundaries[1:])
    ]
    coverage = sum(item["height"] for item in slices)
    continuity = all(
        left["y_end"] == right["y_start"]
        for left, right in zip(slices, slices[1:])
    )
    unsafe_cuts = None
    if strategy == "current":
        unsafe_cuts = [
            cut
            for cut in cuts
            if unsafe_rows[
                max(0, cut - SAFE_CUT_BAND) : cut + SAFE_CUT_BAND + 1
            ].any()
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
    safe_model = _resolve_bounded_path(
        model_path,
        allowed_roots=(MODELS_ROOT,),
        must_exist=True,
        must_be_file=True,
    )
    detector = YoloDetector(safe_model, conf_threshold=0.4)
    comparison_cuts: dict[str, list[int]] = {}
    if comparison_report is not None:
        safe_report = _resolve_bounded_path(
            comparison_report,
            allowed_roots=(RESULTS_ROOT,),
            must_exist=True,
            must_be_file=True,
        )
        comparison = json.loads(safe_report.read_text(encoding="utf-8"))
        comparison_cuts = {
            str(page["path"]): page["cut_positions"]
            for page in comparison.get("page_details", [])
            if isinstance(page, dict)
            and isinstance(page.get("path"), str)
            and isinstance(page.get("cut_positions"), list)
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
        "model": str(safe_model),
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
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if strategy not in {"current", "score-only"}:
        raise ValueError("unsupported slicer strategy")

    safe_root = _resolve_bounded_path(root, allowed_roots=(DATA_ROOT, BENCH_ROOT))
    paths = _image_paths(safe_root)
    if not paths:
        raise ValueError(f"No supported images found under {safe_root}")

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
        "input": str(safe_root),
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
    if detector_model is not None:
        report["detector_cut_metrics"] = _detector_cut_metrics(
            paths, pages, detector_model, comparison_report
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure deterministic long-page slicing. Input is restricted to "
            "data/ or bench/; reports are written only under bench/results/."
        )
    )
    parser.add_argument(
        "input",
        type=_input_path,
        help="Repository-local image file or directory under data/ or bench/",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=("current", "score-only"),
        default="current",
    )
    parser.add_argument(
        "--detector-model",
        type=_model_path,
        help="Optional model under models/ for measured cut-crossing metrics",
    )
    parser.add_argument(
        "--comparison-report",
        type=_comparison_path,
        help="Optional prior JSON report under bench/results/",
    )
    parser.add_argument(
        "--output",
        type=_result_path,
        help="JSON filename written under bench/results/",
    )
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
    if args.output is not None:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
