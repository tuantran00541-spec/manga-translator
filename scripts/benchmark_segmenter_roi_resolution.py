"""A/B static text-segmenter resolutions on post-inpaint residue ROIs.

The benchmark intentionally leaves the production 1024px primary detector
unchanged. It first creates cleaned pages with the current production detector
and inpaint path, then replays the *same* residue-verification inputs through
1024/768/640/512 text-segmenter models.

Only review evidence is compared here. A lower-resolution candidate is never
promoted to destructive mask authority by this benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time


def _center(region: dict) -> tuple[float, float]:
    return (
        (float(region["x1"]) + float(region["x2"])) * 0.5,
        (float(region["y1"]) + float(region["y2"])) * 0.5,
    )


def _residue_hits(regions: list[dict]) -> list[dict]:
    return [
        region
        for region in regions
        if region.get("deferred_reason") == "post_inpaint_text_residue"
    ]


def match_residue_hits(
    baseline: list[dict],
    candidate: list[dict],
) -> list[dict]:
    """Return baseline residue hits without a nearby candidate counterpart."""
    wanted = _residue_hits(baseline)
    available = _residue_hits(candidate)
    used: set[int] = set()
    unmatched: list[dict] = []

    for source in wanted:
        sx, sy = _center(source)
        best_index = None
        best_distance = None
        for index, target in enumerate(available):
            if index in used:
                continue
            tx, ty = _center(target)
            distance = math.hypot(tx - sx, ty - sy)
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance

        tolerance = max(
            24.0,
            0.25
            * max(
                float(source["x2"]) - float(source["x1"]),
                float(source["y2"]) - float(source["y1"]),
            ),
        )
        if (
            best_index is None
            or best_distance is None
            or best_distance > tolerance
        ):
            unmatched.append(
                {
                    "baseline": source,
                    "nearest_distance": best_distance,
                    "tolerance": tolerance,
                }
            )
        else:
            used.add(best_index)

    return unmatched


def pixel_amplification(
    *,
    model_size: int,
    calls: int,
    source_pixels: int,
) -> float:
    """Tensor pixels consumed per scheduled source pixel."""
    return (
        int(calls) * int(model_size) * int(model_size)
        / float(max(1, int(source_pixels)))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_record(box) -> dict:
    fields = (
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "source_model",
        "source_role",
        "class_name",
        "semantic_type",
        "deferred_reason",
    )
    return {field: getattr(box, field) for field in fields}


def _reason_counts(regions_by_page: dict[str, list[dict]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for regions in regions_by_page.values():
        for region in regions:
            reason = str(region.get("deferred_reason") or "none")
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _compare_regions(
    baseline: dict[str, list[dict]],
    candidate: dict[str, list[dict]],
) -> dict:
    unmatched_baseline: list[dict] = []
    extra_candidate: list[dict] = []

    for page, baseline_regions in baseline.items():
        misses = match_residue_hits(
            baseline_regions,
            candidate.get(page, []),
        )
        for row in misses:
            unmatched_baseline.append({"page": int(page), **row})

    for page, candidate_regions in candidate.items():
        extras = match_residue_hits(
            candidate_regions,
            baseline.get(page, []),
        )
        for row in extras:
            extra_candidate.append(
                {
                    "page": int(page),
                    "candidate": row["baseline"],
                    "nearest_distance": row["nearest_distance"],
                    "tolerance": row["tolerance"],
                }
            )

    return {
        "unmatched_baseline_hits": unmatched_baseline,
        "extra_candidate_hits": extra_candidate,
    }


def _parse_variant(value: str) -> tuple[int, Path]:
    raw_size, raw_path = value.split("=", 1)
    size = int(raw_size)
    if size <= 0:
        raise argparse.ArgumentTypeError("variant size must be positive")
    path = Path(raw_path)
    return size, path


def run(args: argparse.Namespace) -> dict:
    # Heavy imports are deliberately local: the pure benchmark contract above
    # runs in CI before OpenCV/ORT/Ultralytics are installed.
    import cv2
    import numpy as np

    from app.detector.bubble_detector import (
        BubbleBox,
        LetterboxTransform,
        YoloDetector,
    )
    from app.detector.fast_residue_detector import (
        FastResidueAdaptiveFocusCombinedTextDetector,
    )
    from app.detector.sequential_fast_residue_detector import (
        SequentialFastResidueAdaptiveFocusCombinedTextDetector,
    )
    from app.image_io import read_image
    from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
    from app.manifest_utils import load_manifest_raw
    from app.mask_store import decode_mask_value
    from app.model_contracts import validate_detector_session
    from app.ort_utils import make_session
    from app.parameters import (
        DETECTOR_LETTERBOX_VALUE,
        TEXT_CONF_THRESHOLD,
    )
    from app.pipeline import ChapterPipeline

    class SizedTextSegmenter(YoloDetector):
        """Benchmark-only static-size text segmenter using production decoding."""

        def __init__(self, model_path: Path, input_size: int):
            self.model_path = str(model_path)
            self.source_model = Path(model_path).name
            self.model_role = "text_segmenter"
            self.session = make_session(model_path)
            self.contract = validate_detector_session(
                self.session,
                role=self.model_role,
                configured_input_size=int(input_size),
            )
            self.input_name = self.contract.input_name
            self.conf_threshold = TEXT_CONF_THRESHOLD
            self.use_tta = False

        def _preprocess(
            self,
            image: np.ndarray,
            *,
            offset_x: int = 0,
            offset_y: int = 0,
        ):
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                return None, None

            input_w = int(self.contract.input_width)
            input_h = int(self.contract.input_height)
            transform = LetterboxTransform.create(
                w,
                h,
                input_w,
                input_h,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            img_rgb = (
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.ndim == 3 and image.shape[2] == 3
                else image
            )
            resized = cv2.resize(
                img_rgb,
                (transform.resized_w, transform.resized_h),
            )
            canvas = np.full(
                (input_h, input_w, 3),
                DETECTOR_LETTERBOX_VALUE,
                dtype=np.uint8,
            )
            y1 = transform.pad_y
            y2 = y1 + transform.resized_h
            x1 = transform.pad_x
            x2 = x1 + transform.resized_w
            canvas[y1:y2, x1:x2] = resized
            blob = (
                (canvas.astype(np.float32) / 255.0)
                .transpose(2, 0, 1)[None]
            )
            return blob, transform

    def make_verifier(text_detector):
        verifier = FastResidueAdaptiveFocusCombinedTextDetector.__new__(
            FastResidueAdaptiveFocusCombinedTextDetector
        )
        verifier.text_detector = text_detector
        verifier._residue_metrics_local = threading.local()
        verifier._residue_metrics_lock = threading.Lock()
        verifier._residue_totals = {}
        verifier._residue_flat_gate_enabled = True
        return verifier

    def authorized_boxes(page: dict) -> list[BubbleBox]:
        boxes: list[BubbleBox] = []
        for record in page.get("boxes", []):
            if (
                not isinstance(record, dict)
                or record.get("removed")
                or record.get("overlap_context_only")
                or record.get("deferred_reason")
            ):
                continue
            geometry_overridden = bool(record.get("geometry_overridden"))
            if not (
                bool(record.get("safe_to_inpaint"))
                or geometry_overridden
            ):
                continue

            try:
                x1 = int(record["x1"])
                y1 = int(record["y1"])
                x2 = int(record["x2"])
                y2 = int(record["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            mask = decode_mask_value(record.get("mask"))
            box = BubbleBox(
                x1,
                y1,
                x2,
                y2,
                float(record.get("confidence", 1.0)),
                mask,
                source_model=str(record.get("source_model") or "unknown"),
                class_id=int(record.get("class_id") or 0),
                class_name=str(record.get("class_name") or "unknown"),
                semantic_type=str(record.get("semantic_type") or "unknown"),
                mask_source=str(record.get("mask_source") or "none"),
                safe_to_inpaint=True,
                ocr_eligible=bool(record.get("ocr_eligible")),
                needs_review=bool(record.get("needs_review")),
                source_role=str(record.get("source_role") or "unknown"),
                deferred_reason=None,
            )
            if geometry_overridden:
                box.allow_rectangle_fallback = True
            boxes.append(box)
        return boxes

    # Seed real cleaned inputs with the current production detector and current
    # inpaint implementation, but use the base chapter pipeline so residue
    # evidence is observed rather than automatically repaired in a later pass.
    seed = ChapterPipeline()
    seed_detector = SequentialFastResidueAdaptiveFocusCombinedTextDetector()
    seed._detector = seed_detector
    seed._inpainter = AdaptiveFastInpainter()
    preload = getattr(seed.inpainter, "preload", None)
    if callable(preload):
        preload()

    chapter_id = hashlib.sha256(
        f"roi-resolution-{time.time_ns()}".encode()
    ).hexdigest()[:8]
    manifest = seed.download_chapter(
        args.chapter_url,
        chapter_id,
        workers=2,
    )
    pages = manifest.get("pages", [])
    if len(pages) < args.sample_count:
        raise RuntimeError(
            f"expected at least {args.sample_count} slices, got {len(pages)}"
        )
    indices = sorted(
        {
            round(i * (len(pages) - 1) / (args.sample_count - 1))
            for i in range(args.sample_count)
        }
    )

    seed_started = time.perf_counter()
    seed.process_pages(chapter_id, indices, workers=1)
    seed_wall_ms = (time.perf_counter() - seed_started) * 1000.0
    current = load_manifest_raw(chapter_id)

    inputs: dict[int, tuple[Path, list[BubbleBox]]] = {}
    persisted_regions: dict[str, list[dict]] = {}
    for index in indices:
        page = current["pages"][index]
        clean_value = page.get("clean")
        if not clean_value:
            raise RuntimeError(f"page {index} has no clean artifact")
        inputs[index] = (
            Path(clean_value),
            authorized_boxes(page),
        )
        persisted_regions[str(index)] = list(page.get("residue_regions") or [])

    def warm_detector(text_detector) -> None:
        for clean_path, boxes in inputs.values():
            if not boxes:
                continue
            image = read_image(clean_path)
            h, w = image.shape[:2]
            source = next((box for box in boxes if box.verified_mask), None)
            if source is None:
                continue
            x1 = max(0, min(w, int(source.x1)))
            y1 = max(0, min(h, int(source.y1)))
            x2 = max(x1 + 1, min(w, int(source.x2)))
            y2 = max(y1 + 1, min(h, int(source.y2)))
            crop = image[y1:y2, x1:x2]
            if crop.size:
                text_detector._detect_single_plain(crop, x1, y1)
                return
        raise RuntimeError("no verified-mask ROI available for model warmup")

    def replay(
        text_detector,
        *,
        model_size: int,
        load_ms: float,
        model_path: Path,
    ) -> dict:
        warm_detector(text_detector)
        verifier = make_verifier(text_detector)
        verifier.residue_metrics_snapshot(reset=True)

        by_page: dict[str, list[dict]] = {}
        elapsed: list[float] = []
        for index in indices:
            clean_path, boxes = inputs[index]
            image = read_image(clean_path)
            started = time.perf_counter()
            result = verifier.verify_post_inpaint_residue(image, boxes)
            elapsed.append((time.perf_counter() - started) * 1000.0)
            by_page[str(index)] = [_box_record(box) for box in result]

        metrics = verifier.residue_metrics_snapshot()
        source_pixels = int(metrics.get("source_pixels", 0))
        calls = int(metrics.get("model_calls", 0))
        return {
            "model_size": int(model_size),
            "model": model_path.as_posix(),
            "model_sha256": _sha256(model_path),
            "session_load_ms": round(float(load_ms), 3),
            "total_ms": round(sum(elapsed), 3),
            "mean_ms": round(statistics.mean(elapsed), 3),
            "median_ms": round(statistics.median(elapsed), 3),
            "metrics": metrics,
            "reasons": _reason_counts(by_page),
            "regions": by_page,
            "model_to_source_pixel_ratio": round(
                pixel_amplification(
                    model_size=model_size,
                    calls=calls,
                    source_pixels=source_pixels,
                ),
                3,
            ),
        }

    production_model = Path(seed_detector.text_detector.model_path)
    control = replay(
        seed_detector.text_detector,
        model_size=1024,
        load_ms=0.0,
        model_path=production_model,
    )

    # The replay must reproduce the residue evidence persisted by the production
    # page pass. If it cannot, the experiment is not a valid isolated A/B.
    control_vs_seed = _compare_regions(
        persisted_regions,
        control["regions"],
    )
    seed_reasons = _reason_counts(persisted_regions)
    control_reasons = control["reasons"]
    control_valid = bool(
        not control_vs_seed["unmatched_baseline_hits"]
        and not control_vs_seed["extra_candidate_hits"]
        and seed_reasons.get("post_inpaint_verification_budget", 0)
        == control_reasons.get("post_inpaint_verification_budget", 0)
        and seed_reasons.get("post_inpaint_verification_size", 0)
        == control_reasons.get("post_inpaint_verification_size", 0)
    )
    if not control_valid:
        raise RuntimeError(
            "1024 residue replay did not reproduce production evidence: "
            + json.dumps(control_vs_seed, ensure_ascii=False)
        )

    baseline_hit_count = len(
        [
            region
            for regions in control["regions"].values()
            for region in _residue_hits(regions)
        ]
    )

    variants: dict[str, dict] = {}
    control_quality = {
        "status": "pass" if baseline_hit_count else "blocked",
        "baseline_hit_count": baseline_hit_count,
        "candidate_hit_count": baseline_hit_count,
        "unmatched_baseline_hits": [],
        "extra_candidate_hits": [],
        "promotion_eligible": False,
        "promotion_blockers": [
            "single-chapter detector evidence is not residue ground truth",
            "lower-resolution model is benchmark-only and review-only",
        ],
    }
    variants["1024"] = {
        **control,
        "speedup_pct": 0.0,
        "quality": control_quality,
    }

    for size, model_path in args.variant:
        if not model_path.is_file():
            raise RuntimeError(f"missing {size}px model: {model_path}")
        load_started = time.perf_counter()
        detector = SizedTextSegmenter(model_path, size)
        load_ms = (time.perf_counter() - load_started) * 1000.0
        candidate = replay(
            detector,
            model_size=size,
            load_ms=load_ms,
            model_path=model_path,
        )
        comparison = _compare_regions(
            control["regions"],
            candidate["regions"],
        )
        candidate_hit_count = len(
            [
                region
                for regions in candidate["regions"].values()
                for region in _residue_hits(regions)
            ]
        )

        deferred_match = all(
            candidate["reasons"].get(reason, 0)
            == control["reasons"].get(reason, 0)
            for reason in (
                "post_inpaint_verification_budget",
                "post_inpaint_verification_size",
            )
        )
        if baseline_hit_count == 0:
            status = "blocked"
        elif (
            comparison["unmatched_baseline_hits"]
            or comparison["extra_candidate_hits"]
            or not deferred_match
        ):
            status = "fail"
        else:
            status = "pass"

        quality = {
            "status": status,
            "baseline_hit_count": baseline_hit_count,
            "candidate_hit_count": candidate_hit_count,
            **comparison,
            "deferred_semantics_match": deferred_match,
            "promotion_eligible": False,
            "promotion_blockers": [
                "single-chapter detector evidence is not residue ground truth",
                "hard/holdout residue labels are required before production",
                "candidate is benchmark-only and has no destructive authority",
            ],
        }
        variants[str(size)] = {
            **candidate,
            "speedup_pct": round(
                (control["total_ms"] / max(1e-9, candidate["total_ms"]) - 1.0)
                * 100.0,
                3,
            ),
            "quality": quality,
        }

    report = {
        "benchmark": "text-segmenter-roi-resolution-v1",
        "chapter_url": args.chapter_url,
        "chapter_id": chapter_id,
        "total_slices": len(pages),
        "sample_indices": indices,
        "seed_pipeline_wall_ms": round(seed_wall_ms, 3),
        "seed_reasons": seed_reasons,
        "control_replay_valid": control_valid,
        "variants": variants,
        "decision": {
            "root_cause": (
                "small residue ROIs are currently inflated into static "
                "1024x1024 text-segmenter tensors"
            ),
            "scope": "post-inpaint residue verification only",
            "production_changed": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chapter-url",
        default="https://asurascans.com/comics/killer-pietro-08677664/chapter/120",
    )
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument(
        "--variant",
        type=_parse_variant,
        action="append",
        default=[],
        metavar="SIZE=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count < 2:
        parser.error("--sample-count must be >= 2")
    if not args.variant:
        parser.error("at least one --variant SIZE=PATH is required")

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "SEGMENTER_ROI_RESOLUTION="
        + json.dumps(report, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
