#!/usr/bin/env python3
"""Audit speech GT without modifying it.

This tool deliberately treats reviewed detector candidates as REVIEW LABELS,
not independent geometry ground truth. IoU evaluation is only possible for
independent GT boxes (missed_gt_bubbles / missed_gt_text or a future explicit
GT geometry field).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LABELS = {"target_speech_bubble", "non_target", "uncertain"}


def box_from(obj):
    if not isinstance(obj, dict):
        return None
    for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
        if all(k in obj for k in keys):
            try:
                return tuple(float(obj[k]) for k in keys)
            except (TypeError, ValueError):
                return None
    return None


def validate_box(box):
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    data = json.loads(args.gt.read_text(encoding="utf-8"))
    images = data.get("images", [])
    report = {
        "schema_version": data.get("schema_version"),
        "image_count": len(images),
        "bubble_candidates": Counter(),
        "text_candidates": Counter(),
        "missed_gt_bubbles": 0,
        "missed_gt_text": 0,
        "independent_bubble_boxes": 0,
        "independent_text_boxes": 0,
        "candidate_geometry_available": 0,
        "warnings": [],
    }

    for idx, image in enumerate(images):
        for kind, key, counter_name in (
            ("bubble", "raw_bubble_candidates", "bubble_candidates"),
            ("text", "raw_text_candidates", "text_candidates"),
        ):
            for cand in image.get(key, []):
                label = cand.get("label")
                if label not in LABELS:
                    report["warnings"].append(
                        f"image[{idx}] {kind}: invalid/missing label={label!r}"
                    )
                else:
                    report[counter_name][label] += 1
                if validate_box(box_from(cand)):
                    report["candidate_geometry_available"] += 1

        for key, count_key, box_key in (
            ("missed_gt_bubbles", "missed_gt_bubbles", "independent_bubble_boxes"),
            ("missed_gt_text", "missed_gt_text", "independent_text_boxes"),
        ):
            for obj in image.get(key, []):
                report[count_key] += 1
                if validate_box(box_from(obj)):
                    report[box_key] += 1

    report["bubble_candidates"] = dict(report["bubble_candidates"])
    report["text_candidates"] = dict(report["text_candidates"])
    report["iou_ready"] = (
        report["independent_bubble_boxes"] > 0
        and report["independent_bubble_boxes"] == report["missed_gt_bubbles"]
    )
    report["note"] = (
        "Reviewed target candidates are not independent box GT. "
        "Use independent missed_gt boxes for IoU; do not score a detector "
        "candidate against itself."
    )

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"schema_version: {report['schema_version']}")
    print(f"images: {report['image_count']}")
    print(f"bubble candidates: {report['bubble_candidates']}")
    print(f"text candidates: {report['text_candidates']}")
    print(f"missed_gt_bubbles: {report['missed_gt_bubbles']} "
          f"(valid boxes: {report['independent_bubble_boxes']})")
    print(f"missed_gt_text: {report['missed_gt_text']} "
          f"(valid boxes: {report['independent_text_boxes']})")
    print(f"candidate geometry entries: {report['candidate_geometry_available']}")
    print(f"IoU-ready independent bubble GT: {report['iou_ready']}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    print(f"NOTE: {report['note']}")


if __name__ == "__main__":
    main()
