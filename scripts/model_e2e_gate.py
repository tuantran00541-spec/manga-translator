from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Iterable

import cv2
import numpy as np
import psutil

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_REPORT_ROOT = _REPO_ROOT / "benchmark-results"


def _source_revision() -> str | None:
    path = _REPO_ROOT / "SOURCE_SHA.txt"
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    value = os.getenv("GITHUB_SHA", "").strip()
    return value or None


class GateFailure(RuntimeError):
    pass


def _write_dynamic_report(text: str) -> None:
    _REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (_REPORT_ROOT / "model-e2e-dynamic.json").write_text(text + "\n", encoding="utf-8")


def _write_fixed_report(text: str) -> None:
    _REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (_REPORT_ROOT / "model-e2e-fixed.json").write_text(text + "\n", encoding="utf-8")


def _rss_mb() -> float:
    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (IndexError, OSError, ValueError):
        pass
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@contextmanager
def _peak_rss_sampler(interval: float = 0.05):
    stop = threading.Event()
    result = {"peak_mb": _rss_mb()}

    def sample() -> None:
        while not stop.wait(interval):
            result["peak_mb"] = max(result["peak_mb"], _rss_mb())

    thread = threading.Thread(target=sample, name="model-e2e-rss", daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join(timeout=1.0)
        result["peak_mb"] = max(result["peak_mb"], _rss_mb())


def _raw_images(raw_dir: Path, limit: int) -> list[Path]:
    paths = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise GateFailure(f"No source images found in {raw_dir}")
    return paths


def _chunked(values: list[int], size: int) -> Iterable[list[int]]:
    size = max(1, int(size))
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _box_from_record(record: dict):
    from app.detector.bubble_detector import BubbleBox
    from app.mask_store import decode_mask_value

    box = BubbleBox(
        int(record["x1"]), int(record["y1"]),
        int(record["x2"]), int(record["y2"]),
        float(record.get("confidence", 0.0) or 0.0),
        decode_mask_value(record.get("mask")),
        source_model=str(record.get("source_model") or "unknown"),
        class_id=int(record.get("class_id") or 0),
        class_name=str(record.get("class_name") or "unknown"),
        semantic_type=str(record.get("semantic_type") or "unknown"),
        mask_source=str(record.get("mask_source") or "none"),
        safe_to_inpaint=bool(record.get("safe_to_inpaint")),
        ocr_eligible=bool(record.get("ocr_eligible")),
        needs_review=bool(record.get("needs_review")),
    )
    if record.get("geometry_overridden"):
        box.allow_rectangle_fallback = True
    return box


def _authority_mask(image: np.ndarray, records: list[dict], inpainter) -> np.ndarray:
    """Reconstruct the exact automatic mask path used by Inpainter.inpaint."""
    from app.detector.bubble_detector import BubbleBox
    from app.detector.mask_builder import build_mask

    h, w = image.shape[:2]
    effective = [
        _box_from_record(record)
        for record in records
        if isinstance(record, dict)
        and not record.get("removed")
        and not (record.get("overlap_context_only") and not record.get("geometry_overridden"))
        and (record.get("safe_to_inpaint") or record.get("geometry_overridden"))
    ]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for cluster in inpainter._cluster_boxes(effective):
        x1 = min(box.x1 for box in cluster)
        y1 = min(box.y1 for box in cluster)
        x2 = max(box.x2 for box in cluster)
        y2 = max(box.y2 for box in cluster)
        cx1, cy1, cx2, cy2 = inpainter._compute_crop_region(x1, y1, x2, y2, w, h)
        local_boxes = []
        for box in cluster:
            local = BubbleBox(
                box.x1 - cx1, box.y1 - cy1,
                box.x2 - cx1, box.y2 - cy1,
                box.confidence, box.mask,
                source_model=box.source_model,
                class_id=box.class_id,
                class_name=box.class_name,
                semantic_type=box.semantic_type,
                mask_source=box.mask_source,
                safe_to_inpaint=bool(box.safe_to_inpaint),
                ocr_eligible=bool(box.ocr_eligible),
                needs_review=bool(box.needs_review),
            )
            if bool(getattr(box, "allow_rectangle_fallback", False)):
                local.allow_rectangle_fallback = True
            local_boxes.append(local)
        crop = image[cy1:cy2, cx1:cx2]
        local_mask = build_mask(crop.shape[:2], local_boxes, crop)
        full_mask[cy1:cy2, cx1:cx2] = np.maximum(
            full_mask[cy1:cy2, cx1:cx2], local_mask
        )
    return full_mask


def _check_detector_provenance(page_index: int, records: list[dict], failures: list[str]) -> None:
    from app.detector.mask_builder import AUTO_DESTRUCTIVE_MASK_SOURCES

    for box_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("removed"):
            continue
        safe = bool(record.get("safe_to_inpaint"))
        mask_source = str(record.get("mask_source") or "none")
        manual = bool(record.get("manual") or record.get("geometry_overridden"))
        if safe and not manual and mask_source not in AUTO_DESTRUCTIVE_MASK_SOURCES:
            failures.append(
                f"page {page_index} box {box_index}: unsafe automatic mask provenance {mask_source!r}"
            )
        if safe and not manual and not record.get("mask"):
            failures.append(
                f"page {page_index} box {box_index}: safe_to_inpaint without persisted mask"
            )
        if str(record.get("source_model") or "").lower().startswith("paddle"):
            failures.append(
                f"page {page_index} box {box_index}: Paddle output entered detector mask records"
            )


def _check_pixel_safety(
    page_index: int,
    page: dict,
    inpainter,
    tolerance: int,
) -> dict:
    from app.pipeline import read_image

    original_path = Path(str(page.get("original") or ""))
    clean_path = Path(str(page.get("clean") or ""))
    if not original_path.is_file():
        raise GateFailure(f"page {page_index}: original image missing: {original_path}")
    if not clean_path.is_file():
        return {
            "page_index": page_index,
            "mask_pixels": 0,
            "changed_pixels": 0,
            "outside_changed_pixels": 0,
            "outside_max_delta": 0,
            "clean_missing": True,
        }

    original = read_image(original_path)
    clean = read_image(clean_path)
    if original.shape != clean.shape:
        raise GateFailure(
            f"page {page_index}: clean shape {clean.shape} != original {original.shape}"
        )
    authority = _authority_mask(original, page.get("boxes", []) or [], inpainter)
    delta = np.max(
        np.abs(clean.astype(np.int16) - original.astype(np.int16)), axis=2
    )
    outside = authority <= 127
    outside_bad = outside & (delta > int(tolerance))
    return {
        "page_index": page_index,
        "mask_pixels": int(np.count_nonzero(authority > 127)),
        "changed_pixels": int(np.count_nonzero(delta > 0)),
        "outside_changed_pixels": int(np.count_nonzero(outside_bad)),
        "outside_max_delta": int(delta[outside].max()) if np.any(outside) else 0,
        "clean_missing": False,
    }


def _check_cache_identity(lang: str) -> dict:
    from app.ocr.identity import engine_identity

    normalized = (lang or "").strip().lower()
    if normalized in {"ja", "japan"}:
        return {"applicable": False, "changed": None}
    previous = os.environ.get("MANGA_OCR_TARGET_SELECTION")
    try:
        engine_identity.cache_clear()
        os.environ["MANGA_OCR_TARGET_SELECTION"] = "centered"
        centered = engine_identity(lang)
        engine_identity.cache_clear()
        os.environ["MANGA_OCR_TARGET_SELECTION"] = "all"
        all_mode = engine_identity(lang)
    finally:
        if previous is None:
            os.environ.pop("MANGA_OCR_TARGET_SELECTION", None)
        else:
            os.environ["MANGA_OCR_TARGET_SELECTION"] = previous
        engine_identity.cache_clear()
    return {
        "applicable": True,
        "changed": centered != all_mode,
        "centered": centered,
        "all": all_mode,
    }


def _check_ocr_route(lang: str, result: dict, failures: list[str]) -> None:
    model = str(result.get("model") or "")
    normalized = (lang or "").strip().lower()
    if normalized in {"ja", "japan"} and model != "manga-ocr":
        failures.append(f"Japanese OCR routed to {model!r}, expected 'manga-ocr'")
    elif normalized in {"ko", "korean"} and model != "korean_PP-OCRv5_mobile_rec":
        failures.append(
            f"Korean OCR routed to {model!r}, expected 'korean_PP-OCRv5_mobile_rec'"
        )
    elif normalized not in {"ja", "japan", "ko", "korean"} and not model.startswith("PP-OCRv6_"):
        failures.append(f"{lang} OCR routed to {model!r}, expected PP-OCRv6 recognizer")


def _run_ocr(chapter_id: str, pipeline, lang: str, max_boxes: int, require: bool) -> dict:
    failures: list[str] = []
    cache_identity = _check_cache_identity(lang)
    if cache_identity.get("applicable") and not cache_identity.get("changed"):
        failures.append("OCR cache identity did not change between centered and all target modes")

    try:
        from app.ocr.multi_lang_ocr import MultiLangOCR
        from app.ocr.service import OCRService

        engine = MultiLangOCR()
        expected_mode = os.getenv("MANGA_OCR_TARGET_SELECTION", "centered").strip().lower() or "centered"
        if expected_mode not in {"centered", "all"}:
            expected_mode = "centered"
        if getattr(engine, "_paddle_target_mode", expected_mode) != expected_mode:
            failures.append("MultiLangOCR target mode does not match configured OCR target selection")

        service = OCRService(engine, pipeline)
        plan = service.plan_chapter(chapter_id)
        if max_boxes > 0:
            plan = plan[:max_boxes]
        if require and not plan:
            failures.append("OCR gate has no eligible boxes; no real OCR evidence was produced")
        results: list[dict] = []
        latencies: list[float] = []
        for page_index, box_id in plan:
            started = time.perf_counter()
            result = service.inspect_box_id(chapter_id, page_index, box_id, lang)
            latencies.append((time.perf_counter() - started) * 1000.0)
            results.append(result)
            _check_ocr_route(lang, result, failures)

        quality = {"good": 0, "review": 0, "reject": 0, "unknown": 0}
        for result in results:
            key = str(result.get("quality") or "unknown")
            quality[key if key in quality else "unknown"] += 1
        return {
            "status": "pass" if not failures else "fail",
            "planned_boxes": len(plan),
            "completed_boxes": len(results),
            "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "quality": quality,
            "cache_identity": cache_identity,
            "failures": failures,
        }
    except Exception as exc:
        if require:
            return {
                "status": "fail",
                "planned_boxes": 0,
                "completed_boxes": 0,
                "cache_identity": cache_identity,
                "failures": [f"OCR runtime failed: {type(exc).__name__}: {exc}"],
            }
        return {
            "status": "skipped",
            "planned_boxes": 0,
            "completed_boxes": 0,
            "cache_identity": cache_identity,
            "reason": f"{type(exc).__name__}: {exc}",
            "failures": failures,
        }


def run(args: argparse.Namespace) -> dict:
    if args.lama_mode == "fixed":
        os.environ["MANGA_USE_DYNAMIC_LAMA"] = "0"
    else:
        os.environ["MANGA_USE_DYNAMIC_LAMA"] = "1"

    from app.config import (
        BUBBLE_DETECTOR_MODEL, TEXT_SEGMENTER_MODEL,
        LAMA_DYNAMIC_MODEL, LAMA_MODEL,
    )
    from app.manifest_utils import load_manifest_raw
    from app.pipeline import ChapterPipeline
    from app.security import validate_chapter_id

    validate_chapter_id(args.chapter_id)

    required_models = [
        BUBBLE_DETECTOR_MODEL,
        TEXT_SEGMENTER_MODEL,
        LAMA_DYNAMIC_MODEL if args.lama_mode == "dynamic" else LAMA_MODEL,
    ]
    missing = [str(path) for path in required_models if not path.is_file()]
    if missing:
        raise GateFailure("Missing model files: " + ", ".join(missing))

    pipeline = ChapterPipeline()
    started = time.perf_counter()
    rss_start = _rss_mb()
    with _peak_rss_sampler() as memory:
        if args.chapter_url:
            manifest = pipeline.download_chapter(
                args.chapter_url, args.chapter_id, workers=args.workers
            )
        else:
            raw_paths = _raw_images(args.raw_dir, args.max_source_images)
            manifest = pipeline._build_chapter_from_raw_paths(
                args.chapter_id, raw_paths, source_url=None, workers=args.workers
            )

        all_page_indices = list(range(len(manifest.get("pages", []))))
        page_indices = all_page_indices[max(0, args.start_page):]
        if args.max_pages > 0:
            page_indices = page_indices[:args.max_pages]
        if not page_indices:
            raise GateFailure("No generated slices selected for the gate")
        process_chunk_ms: list[float] = []
        for chunk in _chunked(page_indices, args.workers):
            t0 = time.perf_counter()
            pipeline.process_pages(args.chapter_id, chunk, workers=args.workers)
            process_chunk_ms.append((time.perf_counter() - t0) * 1000.0)

        manifest = load_manifest_raw(args.chapter_id)
        provenance_failures: list[str] = []
        pixel_rows: list[dict] = []
        for page_index in page_indices:
            page = manifest["pages"][page_index]
            records = page.get("boxes", []) or []
            _check_detector_provenance(page_index, records, provenance_failures)
            pixel_rows.append(
                _check_pixel_safety(
                    page_index, page, pipeline.inpainter, args.outside_pixel_tolerance
                )
            )

        model_mode_failures: list[str] = []
        if args.lama_mode == "dynamic" and not pipeline.inpainter.dynamic_lama:
            model_mode_failures.append("dynamic LaMa gate fell back to fixed LaMa")
        if args.lama_mode == "fixed" and pipeline.inpainter.dynamic_lama:
            model_mode_failures.append("fixed LaMa gate unexpectedly loaded a dynamic model")
        if args.lama_mode == "fixed" and type(pipeline.inpainter.session).__name__ != "_SerializedSession":
            model_mode_failures.append("fixed LaMa session is not on the serialized compatibility path")

        ocr = _run_ocr(
            args.chapter_id,
            pipeline,
            args.source_lang,
            args.max_ocr_boxes,
            args.require_ocr,
        )

    pixel_failures = [
        f"page {row['page_index']}: {row['outside_changed_pixels']} pixels changed outside effective mask "
        f"(max delta {row['outside_max_delta']})"
        for row in pixel_rows
        if row["outside_changed_pixels"] > 0
    ]
    cleanup_evidence = {
        "mask_pixels": sum(row["mask_pixels"] for row in pixel_rows),
        "changed_pixels": sum(row["changed_pixels"] for row in pixel_rows),
        "pages_with_mask": sum(int(row["mask_pixels"] > 0) for row in pixel_rows),
        "pages_with_changes": sum(int(row["changed_pixels"] > 0) for row in pixel_rows),
    }
    cleanup_failures: list[str] = []
    if not args.allow_empty_cleanup:
        if cleanup_evidence["mask_pixels"] <= 0:
            cleanup_failures.append(
                "model E2E produced no authorized cleanup mask pixels; inpaint path was not exercised"
            )
        elif cleanup_evidence["changed_pixels"] <= 0:
            cleanup_failures.append(
                "model E2E produced an authorized mask but changed no pixels; cleanup effectiveness was not exercised"
            )

    rss_peak = float(memory["peak_mb"])
    memory_failures = []
    if args.max_rss_mb > 0 and rss_peak > args.max_rss_mb:
        memory_failures.append(
            f"peak RSS {rss_peak:.1f} MiB exceeded limit {args.max_rss_mb:.1f} MiB"
        )

    all_failures = (
        provenance_failures
        + pixel_failures
        + cleanup_failures
        + memory_failures
        + model_mode_failures
        + list(ocr.get("failures") or [])
    )
    if args.require_ocr and ocr.get("status") != "pass" and not ocr.get("failures"):
        all_failures.append(f"OCR status is {ocr.get('status')!r} while --require-ocr is set")

    box_counts = {"total": 0, "safe": 0, "review": 0, "ocr_eligible": 0}
    for page_index in page_indices:
        for record in manifest["pages"][page_index].get("boxes", []) or []:
            if not isinstance(record, dict) or record.get("removed"):
                continue
            box_counts["total"] += 1
            box_counts["safe"] += int(bool(record.get("safe_to_inpaint")))
            box_counts["review"] += int(bool(record.get("needs_review")))
            box_counts["ocr_eligible"] += int(bool(record.get("ocr_eligible")))

    return {
        "status": "pass" if not all_failures else "fail",
        "source_revision": _source_revision(),
        "chapter_id": args.chapter_id,
        "chapter_url": args.chapter_url,
        "source_lang": args.source_lang,
        "lama_mode": args.lama_mode,
        "workers": args.workers,
        "start_page": args.start_page,
        "source_pages": (
            max((int(page.get("source_page", -1)) for page in manifest.get("pages", [])), default=-1) + 1
        ),
        "slices_total": len(manifest.get("pages", [])),
        "slices_tested": len(page_indices),
        "box_counts": box_counts,
        "processing": {
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "mean_chunk_ms": statistics.fmean(process_chunk_ms) if process_chunk_ms else None,
            "p95_chunk_ms": float(np.percentile(process_chunk_ms, 95)) if process_chunk_ms else None,
            "rss_start_mb": rss_start,
            "rss_peak_mb": rss_peak,
        },
        "pixel_safety": {
            "tolerance": args.outside_pixel_tolerance,
            "pages": pixel_rows,
            "outside_changed_pixels": sum(row["outside_changed_pixels"] for row in pixel_rows),
        },
        "cleanup_evidence": cleanup_evidence,
        "ocr": ocr,
        "failures": all_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU real-model E2E gate for detector -> mask -> inpaint -> OCR integration."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-dir", type=Path, help="Directory of ordered source images.")
    source.add_argument("--chapter-url", help="Download and process a chapter URL through production importer.")
    parser.add_argument("--chapter-id", default="e2e00001", help="Production chapter id: exactly 8 lowercase hex characters.")
    parser.add_argument("--source-lang", default="en", choices=["en", "ch", "ja", "korean"])
    parser.add_argument("--lama-mode", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-source-images", type=int, default=0, help="0 means all source images.")
    parser.add_argument("--start-page", type=int, default=0, help="First generated slice index to process.")
    parser.add_argument("--max-pages", type=int, default=0, help="0 means all selected generated slices.")
    parser.add_argument("--max-ocr-boxes", type=int, default=0, help="0 means all OCR-eligible boxes.")
    parser.add_argument("--require-ocr", action="store_true")
    parser.add_argument("--outside-pixel-tolerance", type=int, default=0)
    parser.add_argument(
        "--allow-empty-cleanup",
        action="store_true",
        help="Allow a smoke dataset that exercises no authorized cleanup pixels.",
    )
    parser.add_argument("--max-rss-mb", type=float, default=4096.0)
    parser.add_argument(
        "--report-json",
        action="store_true",
        help="Write the established report name under benchmark-results/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.workers = max(1, min(int(args.workers), 2))
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "status": "fail",
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        if args.lama_mode == "fixed":
            _write_fixed_report(text)
        else:
            _write_dynamic_report(text)
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
