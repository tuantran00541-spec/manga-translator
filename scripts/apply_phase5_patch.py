from pathlib import Path
import re


def replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return updated


path = Path("app/detector/bubble_detector.py")
source = path.read_text(encoding="utf-8")
replacement = r'''    @staticmethod
    def _candidate_iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        bb = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / max(1.0, aa + bb - inter)

    def _plan_candidate_buckets(
        self,
        subset: list[tuple],
    ) -> tuple[list[int], dict[int, list[int]]]:
        """Plan class-local suppression before allocating any mask pixels."""
        if not subset:
            return [], {}
        rects = [
            [
                int(candidate[0]),
                int(candidate[1]),
                max(0, int(candidate[2]) - int(candidate[0])),
                max(0, int(candidate[3]) - int(candidate[1])),
            ]
            for candidate in subset
        ]
        scores = [
            max(self.conf_threshold, float(self._candidate_fields(candidate)[0]))
            for candidate in subset
        ]
        raw_indices = cv2.dnn.NMSBoxes(
            rects,
            scores,
            self.conf_threshold,
            BUBBLE_IOU_THRESHOLD,
        )
        kept = [int(index) for index in np.asarray(raw_indices).reshape(-1)]
        if not kept:
            return [], {}
        buckets = {index: [index] for index in kept}
        kept_set = set(kept)
        for index, candidate in enumerate(subset):
            if index in kept_set:
                continue
            target = max(
                kept,
                key=lambda kept_index: self._candidate_iou(
                    candidate, subset[kept_index]
                ),
            )
            if self._candidate_iou(candidate, subset[target]) > BUBBLE_IOU_THRESHOLD:
                buckets[target].append(index)
        return kept, buckets

    @staticmethod
    def _prototype_crop_bounds(
        geometry,
        prototypes: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        if geometry is None or prototypes is None or prototypes.ndim != 3:
            return None
        _num_proto, mh, mw = prototypes.shape
        if isinstance(geometry, MaskDecodeGeometry):
            canvas_box = geometry.transform.canvas_box_from_page(geometry.source_box)
            input_w = geometry.transform.input_w
            input_h = geometry.transform.input_h
        else:
            canvas_box = geometry
            input_w = INPUT_SIZE
            input_h = INPUT_SIZE
        if canvas_box is None:
            return None
        cx1, cy1, cx2, cy2 = (float(v) for v in canvas_box)
        scale_x = mw / float(input_w)
        scale_y = mh / float(input_h)
        px1 = max(0, min(mw - 1, int(np.floor(cx1 * scale_x))))
        py1 = max(0, min(mh - 1, int(np.floor(cy1 * scale_y))))
        px2 = max(px1 + 1, min(mw, int(np.ceil(cx2 * scale_x))))
        py2 = max(py1 + 1, min(mh, int(np.ceil(cy2 * scale_y))))
        return px1, py1, px2, py2

    def _decode_candidate_masks_roi_batch(
        self,
        subset: list[tuple],
        member_indices: list[int],
        prototypes: np.ndarray,
        *,
        batch_size: int = 8,
    ) -> dict[int, np.ndarray | None]:
        """Batch coefficient products inside the union prototype ROI only."""
        decoded: dict[int, np.ndarray | None] = {index: None for index in member_indices}
        if prototypes is None or prototypes.ndim != 3:
            return decoded
        num_proto, _mh, _mw = prototypes.shape
        prepared = []
        for index in member_indices:
            candidate = subset[index]
            _score, _cid, _classes, geometry, coeffs = self._candidate_fields(candidate)
            bounds = self._prototype_crop_bounds(geometry, prototypes)
            if coeffs is None or bounds is None:
                continue
            coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
            if coeffs.size != num_proto:
                continue
            prepared.append((index, coeffs, bounds))
        if not prepared:
            return decoded

        ux1 = min(item[2][0] for item in prepared)
        uy1 = min(item[2][1] for item in prepared)
        ux2 = max(item[2][2] for item in prepared)
        uy2 = max(item[2][3] for item in prepared)
        proto_roi = prototypes[:, uy1:uy2, ux1:ux2].reshape(num_proto, -1)
        if proto_roi.size == 0:
            return decoded

        batch_size = max(1, int(batch_size))
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            coeff_matrix = np.stack([item[1] for item in chunk], axis=0)
            logits_batch = np.clip(coeff_matrix @ proto_roi, -88.0, 88.0)
            roi_h, roi_w = uy2 - uy1, ux2 - ux1
            logits_batch = logits_batch.reshape(len(chunk), roi_h, roi_w)
            for row, (index, _coeffs, bounds) in enumerate(chunk):
                px1, py1, px2, py2 = bounds
                crop_logits = logits_batch[
                    row,
                    py1 - uy1:py2 - uy1,
                    px1 - ux1:px2 - ux1,
                ]
                if crop_logits.size == 0:
                    continue
                probabilities = 1.0 / (1.0 + np.exp(-crop_logits))
                x1, y1, x2, y2 = map(int, subset[index][:4])
                box_w, box_h = x2 - x1, y2 - y1
                if box_w < 1 or box_h < 1:
                    continue
                resized = cv2.resize(
                    probabilities,
                    (box_w, box_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                decoded[index] = (
                    (resized > DETECTOR_MASK_THRESHOLD).astype(np.uint8) * 255
                )
        return decoded

    def _candidate_to_box(
        self,
        candidate: tuple,
        mask: np.ndarray | None,
    ) -> BubbleBox:
        x1, y1, x2, y2 = map(int, candidate[:4])
        score, cid, num_classes, _geometry, _coeffs = self._candidate_fields(candidate)
        class_name = self._class_name(int(cid), int(num_classes))
        return BubbleBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=min(float(score), DETECTOR_CONFIDENCE_MAX),
            mask=mask,
            source_model=getattr(self, "source_model", "unknown"),
            class_id=int(cid),
            class_name=class_name,
            semantic_type=self._semantic_type(class_name),
            source_role=self.model_role,
        )

    def _nms(self, candidates: list[tuple], prototypes=None) -> list[BubbleBox]:
        if not candidates:
            return []
        result: list[BubbleBox] = []
        by_class: dict[int, list[int]] = {}
        for idx, candidate in enumerate(candidates):
            _, class_id, _, _, _ = self._candidate_fields(candidate)
            by_class.setdefault(class_id, []).append(idx)

        for _class_id, member_indices in by_class.items():
            subset = [candidates[index] for index in member_indices]
            kept, buckets = self._plan_candidate_buckets(subset)
            if not kept:
                continue

            has_mask_model = bool(
                prototypes is not None
                and getattr(self, "model_role", "unknown") == "text_segmenter"
            )
            if not has_mask_model:
                result.extend(
                    self._candidate_to_box(subset[index], None)
                    for index in kept
                )
                continue

            for kept_index in kept:
                contributors = buckets.get(kept_index, [kept_index])
                masks = self._decode_candidate_masks_roi_batch(
                    subset,
                    contributors,
                    prototypes,
                )
                decoded = [
                    self._candidate_to_box(subset[index], masks.get(index))
                    for index in contributors
                ]
                result.append(self._merge_text_mask_evidence(decoded))

        result.sort(key=lambda box: box.confidence, reverse=True)
        return result
'''
source = replace_once(
    source,
    r"    def _nms\(self, candidates: list\[tuple\], prototypes=None\) -> list\[BubbleBox\]:.*?\n    def _nms_boxes\(",
    replacement + "\n    def _nms_boxes(",
    "optimized candidate nms",
)
path.write_text(source, encoding="utf-8")

sanity = r'''from __future__ import annotations

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
'''
Path("scripts/backend_phase5_sanity.py").write_text(sanity, encoding="utf-8")
print("phase5 patch applied")
