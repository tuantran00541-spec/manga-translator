from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from app.detector.adaptive_focus_detector import _adaptive_detect, _focus_text_detect
from app.detector.combined_detector import CombinedTextDetector
from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image
from app.parameters import DETECTOR_INPUT_SIZE, DETECTOR_TALL_IMAGE_FACTOR
from app.pipeline import ChapterPipeline
import scripts.benchmark_bubble_uncertainty as bubble_base
import scripts.benchmark_text_fullpass_router as text_base


CHAPTER_URL = "https://asurascans.com/comics/killer-pietro-08677664/chapter/120"
OUT = Path("benchmark-results/combined-safe-stack/report.json")
SAMPLE_COUNT = 16


def _metrics_features(row: dict, prefix: str) -> dict[str, float]:
    return {
        name[len(prefix):]: float(value)
        for name, value in row.items()
        if name.startswith(prefix)
    }


def _run(detector, paths: dict[int, Path], indices: list[int], warmup_path: Path, label: str) -> dict:
    warm = read_image(warmup_path)
    detector.detect(warm, parallel=False)
    del warm

    elapsed: list[float] = []
    rows: dict[str, dict] = {}
    authority: dict[str, str] = {}
    review: dict[str, list] = {}
    counts: dict[str, dict] = {}

    for index in indices:
        image = read_image(paths[index])
        h, w = image.shape[:2]
        started = time.perf_counter()
        boxes = detector.detect(image, parallel=False)
        elapsed.append((time.perf_counter() - started) * 1000.0)

        # Preserve raw floats. Production telemetry intentionally integer-casts
        # many counters, which is unsuitable for fitting the uncertainty rules.
        rows[str(index)] = dict(getattr(detector._metrics_local, "value", {}) or {})
        authority[str(index)] = hashlib.sha256(
            bubble_base._authority_mask(boxes, h, w).tobytes()
        ).hexdigest()
        review[str(index)] = bubble_base._review_signature(boxes)
        counts[str(index)] = {
            "boxes": len(boxes),
            "safe": sum(bool(box.safe_to_inpaint) for box in boxes),
            "review": sum(bool(box.needs_review) for box in boxes),
        }
        del image, boxes

    def mean_metric(name: str) -> float:
        values = [float(row.get(name) or 0.0) for row in rows.values()]
        return round(statistics.mean(values), 3) if values else 0.0

    return {
        "label": label,
        "wall_ms": round(sum(elapsed), 3),
        "mean_ms": round(statistics.mean(elapsed), 3),
        "median_ms": round(statistics.median(elapsed), 3),
        "bubble_model_mean_ms": mean_metric("bubble_model_ms"),
        "text_model_mean_ms": mean_metric("text_model_ms"),
        "focus_chip_calls": int(sum(int(row.get("focus_chip_calls") or 0) for row in rows.values())),
        "text_fullpass_calls": int(sum(int(row.get("text_fullpass_calls") or 0) for row in rows.values())),
        "bubble_fastpath_pages": int(sum(int(row.get("bubble_gate_fastpath") or 0) for row in rows.values())),
        "bubble_fallback_pages": int(sum(int(row.get("bubble_gate_fallback") or 0) for row in rows.values())),
        "skip_fullpass_pages": int(sum(int(row.get("text_router_skip_fullpass") or 0) for row in rows.values())),
        "keep_fullpass_pages": int(sum(int(row.get("text_router_keep_fullpass") or 0) for row in rows.values())),
        "rows": rows,
        "authority_hashes": authority,
        "review_signatures": review,
        "counts": counts,
    }


def _mismatch(control: dict, candidate: dict) -> dict:
    return {
        "authority": [
            key
            for key, value in control["authority_hashes"].items()
            if candidate["authority_hashes"].get(key) != value
        ],
        "review": [
            key
            for key, value in control["review_signatures"].items()
            if candidate["review_signatures"].get(key) != value
        ],
        "counts": [
            key
            for key, value in control["counts"].items()
            if candidate["counts"].get(key) != value
        ],
    }


def _exact(quality: dict) -> bool:
    return not quality["authority"] and not quality["review"] and not quality["counts"]


def _finish(
    detector,
    image: np.ndarray,
    bubbles,
    recovery,
    text_boxes,
    focus_metrics: dict,
    deferred,
    *,
    started: float,
    bubble_ms: float,
    pregate_mser_ms: float,
    bubble_audit_ms: float,
    router_audit_ms: float,
    bubble_fast: bool,
    skip_fullpass: bool,
    bubble_features: dict[str, float],
    router_features: dict[str, float],
):
    with detector.bubble_detector.prefetched(image, bubbles):
        with detector.text_detector.prefetched(image, text_boxes):
            result = CombinedTextDetector.detect(detector, image, parallel=False)
    if deferred:
        result.extend(deferred)

    metrics = dict(getattr(detector._metrics_local, "value", {}) or {})
    metrics.update(focus_metrics)
    metrics.update(
        {
            "bubble_model_ms": round(bubble_ms, 3),
            "pregate_mser_ms": round(pregate_mser_ms, 3),
            "bubble_audit_ms": round(bubble_audit_ms, 3),
            "text_router_audit_ms": round(router_audit_ms, 3),
            "bubble_gate_fastpath": int(bubble_fast),
            "bubble_gate_fallback": int(not bubble_fast),
            "text_router_skip_fullpass": int(skip_fullpass),
            "text_router_keep_fullpass": int(not skip_fullpass),
            "text_fullpass_calls": int(not skip_fullpass),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    )
    metrics.update({f"bubble_feature_{key}": float(value) for key, value in bubble_features.items()})
    metrics.update({f"router_feature_{key}": float(value) for key, value in router_features.items()})
    detector._metrics_local.value = metrics
    return result


class CombinedSafeDetector(SequentialFastResidueAdaptiveFocusCombinedTextDetector):
    """Compose the exact-gated bubble fastpath and exact-gated text router.

    The same class is also used for calibration probes. Each probe changes only
    one routing decision and leaves the other stage on its conservative path,
    so interaction errors are measured rather than assumed away.
    """

    def __init__(
        self,
        *,
        bubble_rule: dict | None = None,
        router_rule: dict | None = None,
        force_bubble_fast: bool = False,
        force_skip_fullpass: bool = False,
    ):
        super().__init__()
        self.bubble_rule = bubble_rule
        self.router_rule = router_rule
        self.force_bubble_fast = bool(force_bubble_fast)
        self.force_skip_fullpass = bool(force_skip_fullpass)

    def detect(self, image: np.ndarray, *, parallel: bool = False):
        if image.shape[0] <= DETECTOR_INPUT_SIZE * DETECTOR_TALL_IMAGE_FACTOR:
            return super().detect(image, parallel=False)

        started = time.perf_counter()

        # Recovery's primitive extraction is cached by image identity, so this
        # conservative pre-gate is reused by the later recovery pass.
        t = time.perf_counter()
        pre_recovery = self.recovery.detect(image, existing=[])
        pregate_mser_ms = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        bubble_features = bubble_base._pre_features(image, pre_recovery)
        bubble_audit_ms = (time.perf_counter() - t) * 1000.0

        if self.force_bubble_fast:
            bubble_fast = True
        elif self.bubble_rule is None:
            bubble_fast = False
        else:
            bubble_fast = bubble_base._rule_fast(bubble_features, self.bubble_rule)

        t = time.perf_counter()
        if bubble_fast:
            bubbles = bubble_base._single_pass_bubble_detect(self._bubble_model, image)
        else:
            bubbles = _adaptive_detect(self._bubble_model, image)
        bubble_ms = (time.perf_counter() - t) * 1000.0

        recovery = self.recovery.detect(image, existing=bubbles)

        t = time.perf_counter()
        router_features = text_base._pre_features(image, bubbles, recovery)
        router_audit_ms = (time.perf_counter() - t) * 1000.0

        if self.force_skip_fullpass:
            skip_fullpass = True
        elif self.router_rule is None:
            skip_fullpass = False
        else:
            skip_fullpass = bubble_base._rule_fast(router_features, self.router_rule)

        t = time.perf_counter()
        proposals = list(bubbles) + list(recovery)
        if skip_fullpass:
            text_boxes, focus_metrics, deferred = text_base._focus_without_fullpass(
                self._text_model, image, proposals
            )
        else:
            text_boxes, focus_metrics, deferred = _focus_text_detect(
                self._text_model, image, proposals
            )
            focus_metrics = dict(focus_metrics)
            focus_metrics.update(
                {
                    "text_router_skip_fullpass": 0,
                    "text_router_keep_fullpass": 1,
                    "text_fullpass_calls": 1,
                }
            )
        text_ms = (time.perf_counter() - t) * 1000.0
        focus_metrics = dict(focus_metrics)
        focus_metrics["text_model_ms"] = round(text_ms, 3)

        return _finish(
            self,
            image,
            bubbles,
            recovery,
            text_boxes,
            focus_metrics,
            deferred,
            started=started,
            bubble_ms=bubble_ms,
            pregate_mser_ms=pregate_mser_ms,
            bubble_audit_ms=bubble_audit_ms,
            router_audit_ms=router_audit_ms,
            bubble_fast=bubble_fast,
            skip_fullpass=skip_fullpass,
            bubble_features=bubble_features,
            router_features=router_features,
        )


def _learn_rule(probe: dict, control: dict, feature_prefix: str) -> tuple[dict, dict, dict]:
    quality = _mismatch(control, probe)
    unsafe = set().union(*(set(values) for values in quality.values()))
    features = {
        page: _metrics_features(row, feature_prefix)
        for page, row in probe["rows"].items()
        if any(name.startswith(feature_prefix) for name in row)
    }
    return bubble_base._select_safe_rule(features, unsafe), quality, features


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ChapterPipeline()
    chapter_id = hashlib.sha256(f"combined-safe-stack-{time.time_ns()}".encode()).hexdigest()[:8]
    manifest = pipeline.download_chapter(CHAPTER_URL, chapter_id, workers=2)
    pages = manifest.get("pages", [])
    total = len(pages)
    indices = sorted(
        {
            round(i * (total - 1) / (SAMPLE_COUNT - 1))
            for i in range(SAMPLE_COUNT)
        }
    )
    paths = {index: Path(pages[index]["original"]) for index in indices}
    warmup_index = next(index for index in range(total) if index not in set(indices))
    warmup_path = Path(pages[warmup_index]["original"])

    control = _run(
        SequentialFastResidueAdaptiveFocusCombinedTextDetector(),
        paths,
        indices,
        warmup_path,
        "control_current_detector",
    )
    gc.collect()

    # Stage 1: learn the already-validated uncertainty-gated bubble fastpath.
    bubble_probe = _run(
        CombinedSafeDetector(force_bubble_fast=True),
        paths,
        indices,
        warmup_path,
        "probe_all_tall_bubble_fast",
    )
    bubble_rule, bubble_probe_quality, bubble_features = _learn_rule(
        bubble_probe, control, "bubble_feature_"
    )
    gc.collect()

    bubble_candidate = _run(
        CombinedSafeDetector(bubble_rule=bubble_rule),
        paths,
        indices,
        warmup_path,
        "candidate_bubble_gate_only",
    )
    bubble_quality = _mismatch(control, bubble_candidate)
    gc.collect()

    # Stage 2: probe text full-pass elision on top of the learned bubble gate.
    # This is deliberate: the router is calibrated against the proposal
    # distribution it will actually see in the combined stack.
    text_probe = _run(
        CombinedSafeDetector(
            bubble_rule=bubble_rule,
            force_skip_fullpass=True,
        ),
        paths,
        indices,
        warmup_path,
        "probe_skip_all_fullpasses_on_bubble_gate",
    )
    router_rule, text_probe_quality, router_features = _learn_rule(
        text_probe, control, "router_feature_"
    )
    gc.collect()

    candidate = _run(
        CombinedSafeDetector(
            bubble_rule=bubble_rule,
            router_rule=router_rule,
        ),
        paths,
        indices,
        warmup_path,
        "candidate_combined_safe_stack",
    )
    quality = _mismatch(control, candidate)

    report = {
        "chapter_url": CHAPTER_URL,
        "chapter_id": chapter_id,
        "total_slices": total,
        "sample_indices": indices,
        "control": control,
        "bubble_probe": bubble_probe,
        "bubble_probe_quality": bubble_probe_quality,
        "bubble_rule": bubble_rule,
        "bubble_features": bubble_features,
        "bubble_candidate": bubble_candidate,
        "bubble_candidate_quality": bubble_quality,
        "text_probe": text_probe,
        "text_probe_quality": text_probe_quality,
        "router_rule": router_rule,
        "router_features": router_features,
        "candidate": candidate,
        "candidate_quality": quality,
        "speedup": {
            "bubble_only_wall_reduction_pct": bubble_base._pct(
                control["wall_ms"], bubble_candidate["wall_ms"]
            ),
            "combined_wall_reduction_pct": bubble_base._pct(
                control["wall_ms"], candidate["wall_ms"]
            ),
            "combined_bubble_model_reduction_pct": bubble_base._pct(
                control["bubble_model_mean_ms"], candidate["bubble_model_mean_ms"]
            ),
            "combined_text_model_reduction_pct": bubble_base._pct(
                control["text_model_mean_ms"], candidate["text_model_mean_ms"]
            ),
            "bubble_fastpath_pages": candidate["bubble_fastpath_pages"],
            "bubble_fallback_pages": candidate["bubble_fallback_pages"],
            "skip_fullpass_pages": candidate["skip_fullpass_pages"],
            "keep_fullpass_pages": candidate["keep_fullpass_pages"],
            "focus_chip_calls": candidate["focus_chip_calls"],
        },
        "promotion_eligible_on_calibration_sample": bool(
            _exact(bubble_quality)
            and _exact(quality)
            and candidate["wall_ms"] < control["wall_ms"]
        ),
        "note": (
            "Rules are calibrated on this sample. A cross-chapter validation gate "
            "is still required before production promotion."
        ),
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("COMBINED_SAFE_STACK=" + json.dumps(report, ensure_ascii=False), flush=True)

    if bubble_quality["authority"]:
        raise RuntimeError(
            "bubble gate changed destructive authority masks: "
            + json.dumps(bubble_quality["authority"])
        )
    if quality["authority"]:
        raise RuntimeError(
            "combined safe stack changed destructive authority masks: "
            + json.dumps(quality["authority"])
        )


if __name__ == "__main__":
    main()
