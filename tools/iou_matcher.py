#!/usr/bin/env python3
"""Greedy IoU matcher for independent GT boxes.

Intentionally generic: detector output is supplied as JSON so the benchmark
can separate detection from production pipeline changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match(predictions, gt, threshold):
    pairs = []
    for pi, p in enumerate(predictions):
        for gi, g in enumerate(gt):
            pairs.append((iou(p["box"], g["box"]), pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for score, pi, gi in pairs:
        if score < threshold or pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append({"pred_index": pi, "gt_index": gi, "iou": score})
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "matches": matches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("ground_truth", type=Path)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    pred = json.loads(args.predictions.read_text(encoding="utf-8"))
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    result = match(pred, gt, args.iou)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
