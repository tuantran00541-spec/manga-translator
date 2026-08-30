#!/usr/bin/env python3
"""Research-only A/B gate for text-mask refinement.

A candidate can never be eligible for promotion unless enough mask samples have
human-reviewed ``truth_mask`` evidence. The default is intentionally strict:
100% truth coverage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from app.detector.mask_builder import adaptive_dilate_mask
from stroke_mask_refinement import refine_stroke_mask


@dataclass(frozen=True)
class Sample:
    raw: dict[str, Any]
    base_dir: Path

    @property
    def id(self) -> str:
        return str(self.raw.get("id") or "")

    def path(self, key: str, required: bool = True) -> Path | None:
        value = self.raw.get(key)
        if value in (None, ""):
            if required:
                raise ValueError(f"Sample {self.id!r} is missing {key!r}")
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (self.base_dir / path).resolve()


def load_manifest(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"Manifest row {line_no} must be an object")
            if str(raw.get("task") or "mask").lower() != "mask":
                continue
            sample = Sample(raw=raw, base_dir=path.parent)
            if not sample.id or sample.id in seen:
                raise ValueError(f"Missing/duplicate mask sample id at row {line_no}: {sample.id!r}")
            seen.add(sample.id)
            rows.append(sample)
    if not rows:
        raise ValueError("Manifest contains no mask samples")
    return rows


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buf.tofile(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(clean) if clean else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if predicted.shape != truth.shape:
        raise ValueError("predicted mask and truth mask dimensions differ")
    pred, gt = predicted > 127, truth > 127
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "iou": tp / max(1, tp + fp + fn),
        "false_positive_share": fp / max(1, int(np.count_nonzero(pred))),
    }


def outside_share(mask: np.ndarray, envelope: np.ndarray | None) -> float | None:
    if envelope is None:
        return None
    if mask.shape != envelope.shape:
        raise ValueError("mask and safe envelope dimensions differ")
    pred = mask > 127
    return int(np.count_nonzero(pred & ~(envelope > 127))) / max(
        1, int(np.count_nonzero(pred))
    )


def evaluate(samples: list[Sample], args: argparse.Namespace, output: Path) -> dict[str, Any]:
    total_samples = len(samples)
    truth_samples = sum(1 for sample in samples if sample.path("truth_mask", False) is not None)
    truth_coverage = truth_samples / max(1, total_samples)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        image_path = sample.path("image", False)
        image = read_image(image_path) if image_path else None
        seed = read_image(sample.path("seed_mask"), cv2.IMREAD_GRAYSCALE)
        truth_path = sample.path("truth_mask", False)
        envelope_path = sample.path("safe_envelope", False)
        truth = read_image(truth_path, cv2.IMREAD_GRAYSCALE) if truth_path else None
        envelope = read_image(envelope_path, cv2.IMREAD_GRAYSCALE) if envelope_path else None
        if image is not None and image.shape[:2] != seed.shape:
            raise ValueError(f"Sample {sample.id}: image and seed_mask dimensions differ")
        if truth is not None and truth.shape != seed.shape:
            raise ValueError(f"Sample {sample.id}: truth_mask and seed_mask dimensions differ")
        if envelope is not None and envelope.shape != seed.shape:
            raise ValueError(f"Sample {sample.id}: safe_envelope and seed_mask dimensions differ")

        started = time.perf_counter()
        baseline = adaptive_dilate_mask(seed.copy(), image)
        base_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        candidate, stats = refine_stroke_mask(
            seed,
            image,
            safe_envelope=envelope,
            min_radius=args.stroke_min_radius,
            max_radius=args.stroke_max_radius,
        )
        cand_ms = (time.perf_counter() - started) * 1000.0

        for name, mask, latency in (
            ("current-adaptive-dilate", baseline, base_ms),
            ("stroke-width-refinement", candidate, cand_ms),
        ):
            row: dict[str, Any] = {
                "id": sample.id,
                "variant": name,
                "tags": list(sample.raw.get("tags") or []),
                "has_truth_mask": truth is not None,
                "source_pixels": int(np.count_nonzero(seed > 127)),
                "mask_pixels": int(np.count_nonzero(mask > 127)),
                "growth_ratio": int(np.count_nonzero(mask > 127))
                / max(1, int(np.count_nonzero(seed > 127))),
                "outside_safe_share": outside_share(mask, envelope),
                "latency_ms": latency,
            }
            if truth is not None:
                row.update(binary_metrics(mask, truth))
            if name == "stroke-width-refinement":
                row.update(
                    {
                        "components": stats.components,
                        "max_radius_used": stats.max_radius_used,
                        "mean_radius_used": stats.mean_radius_used,
                    }
                )
            rows.append(row)
            if args.save_artifacts:
                write_image(output / "mask_artifacts" / f"{sample.id}__{name}.png", mask)

    summary: dict[str, Any] = {}
    for variant in ("current-adaptive-dilate", "stroke-width-refinement"):
        subset = [row for row in rows if row["variant"] == variant]
        truth_subset = [row for row in subset if row["has_truth_mask"]]
        summary[variant] = {
            "samples": len(subset),
            "truth_samples": len(truth_subset),
            "truth_coverage": len(truth_subset) / max(1, len(subset)),
            "mean_f1": mean(row.get("f1") for row in subset),
            "mean_iou": mean(row.get("iou") for row in subset),
            "mean_recall": mean(row.get("recall") for row in subset),
            "mean_false_positive_share": mean(
                row.get("false_positive_share") for row in subset
            ),
            "mean_outside_safe_share": mean(
                row.get("outside_safe_share") for row in subset
            ),
            "mean_growth_ratio": mean(row.get("growth_ratio") for row in subset),
            "latency_p95_ms": percentile(
                [float(row["latency_ms"]) for row in subset], 95
            ),
        }

    base = summary["current-adaptive-dilate"]
    candidate = summary["stroke-width-refinement"]
    reasons: list[str] = []
    missing_truth = total_samples - truth_samples
    if truth_samples == 0:
        reasons.append("ground-truth mask evidence missing for all samples")
    if truth_coverage + 1e-12 < args.min_truth_coverage:
        reasons.append(
            "ground-truth mask missing for "
            f"{missing_truth}/{total_samples} samples "
            f"(coverage={truth_coverage:.3f}, required={args.min_truth_coverage:.3f})"
        )
    if base["mean_f1"] is not None and candidate["mean_f1"] is not None:
        if candidate["mean_f1"] + args.mask_f1_tolerance < base["mean_f1"]:
            reasons.append("mean F1 regressed")
    if base["mean_recall"] is not None and candidate["mean_recall"] is not None:
        if candidate["mean_recall"] + args.mask_recall_tolerance < base["mean_recall"]:
            reasons.append("recall regressed")
    if (
        base["mean_false_positive_share"] is not None
        and candidate["mean_false_positive_share"] is not None
        and candidate["mean_false_positive_share"]
        > base["mean_false_positive_share"] + args.mask_fp_tolerance
    ):
        reasons.append("artwork-overreach increased")
    if (
        candidate["mean_outside_safe_share"] is not None
        and candidate["mean_outside_safe_share"] > 0
    ):
        reasons.append("safe envelope crossed")

    coverage = {
        "truth_samples": truth_samples,
        "total_samples": total_samples,
        "truth_coverage": truth_coverage,
        "required_truth_coverage": args.min_truth_coverage,
    }
    gate = {
        "eligible_for_next_stage": not reasons,
        "reasons": reasons,
        **coverage,
    }
    write_jsonl(output / "mask_rows.jsonl", rows)
    return {"coverage": coverage, "summary": summary, "gate": gate}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--stroke-min-radius", type=int, default=1)
    parser.add_argument("--stroke-max-radius", type=int, default=6)
    parser.add_argument("--mask-f1-tolerance", type=float, default=0.005)
    parser.add_argument("--mask-fp-tolerance", type=float, default=0.005)
    parser.add_argument("--mask-recall-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--min-truth-coverage",
        type=float,
        default=1.0,
        help="Required human-reviewed truth_mask coverage; default 1.0 (100%%).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_truth_coverage <= 1.0:
        raise SystemExit("--min-truth-coverage must be between 0 and 1")
    if args.stroke_min_radius < 0 or args.stroke_max_radius < args.stroke_min_radius:
        raise SystemExit("invalid stroke radius bounds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(args.manifest.resolve())
    report = {
        "manifest": str(args.manifest.resolve()),
        "generated_at_unix": time.time(),
        **evaluate(samples, args, output),
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["gate"]["eligible_for_next_stage"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
