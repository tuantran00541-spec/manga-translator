from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from app.detector.bubble_detector import YoloDetector


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def make_detector():
    detector = YoloDetector.__new__(YoloDetector)
    detector.conf_threshold = 0.10
    detector.source_model = "text_segmenter.onnx"
    detector.model_role = "text_segmenter"
    detector.contract = SimpleNamespace(class_names=("text_comic",))
    return detector


def synthetic_case(count=48):
    prototypes = np.full((8, 256, 256), -4.0, np.float32)
    prototypes[0, :, :128] = 4.0
    prototypes[1, :128, :] = 4.0
    prototypes[2, 64:192, 64:192] = 4.0
    candidates = []
    for index in range(count):
        group = index % 3
        base_x = 120 + group * 260
        base_y = 160 + group * 220
        jitter = index % 5
        x1, y1 = base_x + jitter, base_y + jitter
        x2, y2 = x1 + 180, y1 + 120
        score = 0.95 - index * 0.004
        coeff = np.zeros(8, np.float32)
        coeff[index % 3] = 1.0
        # Model-space geometry; source and model coordinates intentionally match
        # for this postprocess-only fixture.
        geometry = (x1, y1, x2, y2)
        candidates.append((x1, y1, x2, y2, score, 0, 1, geometry, coeff))
    return candidates, prototypes


def reference_old(detector, candidates, prototypes):
    decoded = []
    for candidate in candidates:
        x1, y1, x2, y2 = map(int, candidate[:4])
        score, cid, num_classes, geometry, coeffs = detector._candidate_fields(candidate)
        mask = detector._decode_mask(coeffs, prototypes, geometry, x2-x1, y2-y1)
        decoded.append(detector._candidate_to_box(candidate, mask))
    return detector._nms_box_group(
        decoded,
        score_threshold=detector.conf_threshold,
        iou_threshold=0.30,
    )


def signature(boxes):
    return [
        (
            box.x1, box.y1, box.x2, box.y2,
            round(float(box.confidence), 6),
            None if box.mask is None else box.mask.tobytes(),
        )
        for box in boxes
    ]


def run():
    detector = make_detector()
    candidates, prototypes = synthetic_case()
    old = reference_old(detector, candidates, prototypes)
    new = detector._nms(candidates, prototypes)
    check(signature(old) == signature(new), "optimized masks/geometry differ from reference")

    kept, buckets = detector._plan_candidate_buckets(candidates)
    full_elements = len(candidates) * prototypes.shape[1] * prototypes.shape[2]
    roi_elements = 0
    peak_batch_elements = 0
    for kept_index in kept:
        members = buckets[kept_index]
        bounds = [detector._prototype_crop_bounds(candidates[index][7], prototypes) for index in members]
        bounds = [item for item in bounds if item is not None]
        ux1 = min(item[0] for item in bounds)
        uy1 = min(item[1] for item in bounds)
        ux2 = max(item[2] for item in bounds)
        uy2 = max(item[3] for item in bounds)
        roi = (ux2-ux1) * (uy2-uy1)
        roi_elements += len(members) * roi
        peak_batch_elements = max(peak_batch_elements, min(8, len(members)) * roi)
    check(roi_elements < full_elements, "ROI plan did not reduce prototype work")
    check(peak_batch_elements < 2 * prototypes.shape[1] * prototypes.shape[2], "ROI batch temporary exceeds old logits+probability footprint")

    # Warm both paths, then compare best-of-two to reduce CI noise. Dense overlap
    # makes the reference perform one full prototype product per candidate.
    reference_old(detector, candidates, prototypes)
    detector._nms(candidates, prototypes)
    old_times = []
    new_times = []
    for _ in range(2):
        start = time.perf_counter(); reference_old(detector, candidates, prototypes); old_times.append(time.perf_counter()-start)
        start = time.perf_counter(); detector._nms(candidates, prototypes); new_times.append(time.perf_counter()-start)
    old_ms = min(old_times) * 1000.0
    new_ms = min(new_times) * 1000.0
    check(new_ms < old_ms, f"dense postprocess did not improve: old={old_ms:.2f}ms new={new_ms:.2f}ms")
    print(f"phase5 dense postprocess reference={old_ms:.2f}ms optimized={new_ms:.2f}ms work={full_elements}->{roi_elements} peak_elements<={peak_batch_elements}")
    print("backend detector postprocess sanity: phase 5 PASS")


run()
