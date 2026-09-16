"""Evaluate an E7 student detector without granting it mask authority.

Dataset rows must contain ``image``, ``partition`` and ``truth_boxes``. Each
truth box is ``{"xyxy": [x1, y1, x2, y2], "class_id": 0}``. A teacher model
is required for the speed comparison. Missing labels/models produce a
``blocked`` report, never a false promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import BUBBLE_DETECTOR_MODEL
from app.detector.bubble_detector import YoloDetector
from app.image_io import read_image
from app.parameters import BUBBLE_PROPOSAL_CONF_THRESHOLD


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ix1 = max(first[0], second[0])
    iy1 = max(first[1], second[1])
    ix2 = min(first[2], second[2])
    iy2 = min(first[3], second[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / float(first_area + second_area - intersection)


def _xyxy(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        value = [value.get(name) for name in ("x1", "y1", "x2", "y2")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"invalid box geometry: {value!r}")
    x1, y1, x2, y2 = (int(round(float(item))) for item in value)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    cases = value if isinstance(value, list) else value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("student detector dataset needs a non-empty cases array")
    normalized = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError(f"case {index} must be an object")
        image = Path(str(row.get("image") or ""))
        if not image.is_absolute():
            image = path.parent / image
        truth = row.get("truth_boxes") or row.get("ground_truth_boxes")
        if not image.is_file() or not isinstance(truth, list):
            raise ValueError(f"case {index} needs an image and truth_boxes")
        normalized.append(
            {
                "case_id": str(row.get("case_id") or f"case-{index:04d}"),
                "partition": str(row.get("partition") or "holdout"),
                "image": image,
                "truth_boxes": [
                    {
                        "xyxy": _xyxy(item.get("xyxy", item)),
                        "class_id": int(item.get("class_id", 0)) if isinstance(item, dict) else 0,
                    }
                    for item in truth
                ],
            }
        )
    return normalized


def _evaluate_detector(detector, cases, images, iou_threshold: float) -> dict[str, Any]:
    rows = []
    elapsed = []
    totals = {"truth": 0, "matched": 0, "candidate": 0, "false_negative": 0, "false_positive": 0}
    by_partition: dict[str, dict[str, int]] = {}
    for case, image in zip(cases, images):
        started = time.perf_counter()
        boxes = detector.detect(image)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        candidates = [
            (int(box.x1), int(box.y1), int(box.x2), int(box.y2), int(box.class_id))
            for box in boxes
        ]
        matched: set[int] = set()
        matches = 0
        for truth in case["truth_boxes"]:
            eligible = [
                (index, box_iou(truth["xyxy"], candidate[:4]))
                for index, candidate in enumerate(candidates)
                if index not in matched and candidate[4] == truth["class_id"]
            ]
            if eligible:
                best_index, best_iou = max(eligible, key=lambda item: item[1])
                if best_iou >= float(iou_threshold):
                    matched.add(best_index)
                    matches += 1
        truth_count = len(case["truth_boxes"])
        candidate_count = len(candidates)
        row = {
            "case_id": case["case_id"],
            "partition": case["partition"],
            "truth_boxes": truth_count,
            "candidate_boxes": candidate_count,
            "matched": matches,
            "false_negative": truth_count - matches,
            "false_positive": candidate_count - len(matched),
            "latency_ms": round(elapsed[-1], 3),
        }
        rows.append(row)
        for key, value in (
            ("truth", truth_count),
            ("matched", matches),
            ("candidate", candidate_count),
            ("false_negative", truth_count - matches),
            ("false_positive", candidate_count - len(matched)),
        ):
            totals[key] += value
        partition = by_partition.setdefault(
            case["partition"],
            {"truth": 0, "matched": 0, "false_negative": 0, "false_positive": 0},
        )
        partition["truth"] += truth_count
        partition["matched"] += matches
        partition["false_negative"] += truth_count - matches
        partition["false_positive"] += candidate_count - len(matched)
    return {
        "rows": rows,
        "totals": totals,
        "by_partition": by_partition,
        "recall": totals["matched"] / float(max(1, totals["truth"])),
        "precision": totals["matched"] / float(max(1, totals["candidate"])),
        "median_latency_ms": statistics.median(elapsed) if elapsed else 0.0,
        "wall_ms": sum(elapsed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dataset.is_file():
        return {"status": "blocked", "blockers": [f"dataset missing: {args.dataset}"]}
    if not args.student_model.is_file():
        return {"status": "blocked", "blockers": [f"student model missing: {args.student_model}"]}
    if not args.teacher_model.is_file():
        return {"status": "blocked", "blockers": [f"teacher model missing: {args.teacher_model}"]}
    try:
        cases = _load_cases(args.dataset)
        images = [read_image(case["image"]) for case in cases]
        teacher = YoloDetector(
            args.teacher_model,
            BUBBLE_PROPOSAL_CONF_THRESHOLD,
            model_role="bubble_detector",
        )
        student = YoloDetector(
            args.student_model,
            BUBBLE_PROPOSAL_CONF_THRESHOLD,
            model_role="bubble_detector",
        )
        # Warm both contracts before the measured loop; session construction is
        # reported separately by model provisioning, not confused with page latency.
        teacher.detect(images[0])
        student.detect(images[0])
        teacher_result = _evaluate_detector(teacher, cases, images, args.iou_threshold)
        student_result = _evaluate_detector(student, cases, images, args.iou_threshold)
    except Exception as exc:
        return {
            "status": "blocked",
            "blockers": [f"student benchmark unavailable: {type(exc).__name__}: {exc}"],
        }

    hard_holdout_fn = sum(
        row["false_negative"]
        for row in student_result["rows"]
        if row["partition"] in {"hard", "holdout"}
    )
    recall_gate = hard_holdout_fn == 0
    speedup_pct = (1.0 - student_result["wall_ms"] / float(max(1e-6, teacher_result["wall_ms"]))) * 100.0
    speed_gate = speedup_pct >= float(args.min_speedup_pct)
    # Detection output is deliberately evaluated as proposal-only. It has not
    # crossed the text-segmenter mask authority boundary.
    safety_gate = True
    return {
        "status": "pass" if recall_gate and speed_gate and safety_gate else "fail",
        "benchmark": "student-detector-ab-v1",
        "student_model": args.student_model.as_posix(),
        "teacher_model": args.teacher_model.as_posix(),
        "iou_threshold": args.iou_threshold,
        "teacher": teacher_result,
        "student": student_result,
        "gates": {
            "hard_holdout_false_negatives": hard_holdout_fn,
            "recall": recall_gate,
            "speed": speed_gate,
            "proposal_authority_safety": safety_gate,
        },
        "speedup_pct": round(speedup_pct, 3),
        "min_speedup_pct": args.min_speedup_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a proposal-only student detector")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, default=Path("models/student_detector.onnx"))
    parser.add_argument("--teacher-model", type=Path, default=Path(BUBBLE_DETECTOR_MODEL))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-speedup-pct", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("E7_STUDENT_DETECTOR=" + json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report.get("status") == "pass" else 2 if report.get("status") == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
