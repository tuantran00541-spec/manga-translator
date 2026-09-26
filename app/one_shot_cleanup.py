from __future__ import annotations

from dataclasses import replace
import threading
import time

import numpy as np

from app.config import TEXT_SEGMENTER_MODEL
import cv2

from app.detector.bubble_detector import BubbleBox, LetterboxTransform, YoloDetector, apply_final_nms
from app.parameters import (
    DETECTOR_COLLAGE,
    DETECTOR_COLLAGE_GAP,
    DETECTOR_COLLAGE_MIN_OVERLAP,
    DETECTOR_LETTERBOX_VALUE,
    DETECTOR_FINAL_NMS_IOU,
    DETECTOR_INPUT_SIZE,
    DETECTOR_TILE_OVERLAP,
    DETECTOR_TILE_SCALE,
    DETECTOR_RESIDUE_VERIFY_MAX_ROIS,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_RESIDUE_VERIFY_PAD,
    TEXT_CONF_THRESHOLD,
)


class OneShotTextMaskDetector:
    RESCUE_CONF_THRESHOLD = 0.12

    def __init__(self, detector: YoloDetector | None = None):
        self.detector = detector or YoloDetector(
            TEXT_SEGMENTER_MODEL,
            TEXT_CONF_THRESHOLD,
            model_role="text_segmenter",
        )
        self._decode_lock = threading.Lock()

    @staticmethod
    def _accept(box: BubbleBox) -> BubbleBox | None:
        if box.source_role != "text_segmenter" or not box.verified_mask:
            return None
        return replace(
            box,
            semantic_type="free_text",
            mask_source="text_segmenter",
            safe_to_inpaint=True,
            ocr_eligible=True,
            needs_review=False,
            deferred_reason=None,
        )

    def _accept_many(self, raw_boxes: list[BubbleBox]) -> list[BubbleBox]:
        accepted: list[BubbleBox] = []
        for raw in raw_boxes:
            box = self._accept(raw)
            if box is not None:
                accepted.append(box)
        return accepted

    def _run_session(self, blob):
        return self.detector.session.run(
            None,
            {self.detector.input_name: blob},
        )

    def _single_forward_outputs(self, image: np.ndarray):
        blob, transform = self.detector._preprocess(image, offset_x=0, offset_y=0)
        if blob is None or transform is None:
            return None, None
        outputs = self._run_session(blob)
        return outputs, transform

    def _postprocess_at_threshold(
        self,
        outputs,
        transform,
        threshold: float,
    ) -> list[BubbleBox]:
        with self._decode_lock:
            original_conf = self.detector.conf_threshold
            try:
                self.detector.conf_threshold = float(threshold)
                return self.detector._postprocess(outputs, transform)
            finally:
                self.detector.conf_threshold = original_conf

    tile_scale = DETECTOR_TILE_SCALE
    collage = DETECTOR_COLLAGE

    @staticmethod
    def collage_plan(height: int, width: int) -> tuple[float, list[tuple[int, int]]] | None:
        """Scale and the two (y1, y2) halves to lay side by side, or None.

        The halves overlap so text on the cut is whole in one of them; any
        vertical room the width leaves free goes into a bigger overlap.
        """
        size = DETECTOR_INPUT_SIZE
        if height <= 0 or width <= 0:
            return None
        single = min(size / width, size / height)
        scale = min(1.0, (size - DETECTOR_COLLAGE_GAP) / (2.0 * width))
        overlap = int(2 * size / scale) - height
        if overlap < DETECTOR_COLLAGE_MIN_OVERLAP:
            overlap = min(height, DETECTOR_COLLAGE_MIN_OVERLAP)
            scale = min(scale, 2.0 * size / (height + overlap))
        overlap = min(overlap, height)
        if scale < single * 1.2:
            return None
        half = (height + overlap + 1) // 2
        return scale, [(0, half), (height - half, height)]

    def _detect_collage(self, image: np.ndarray, scale: float, halves: list[tuple[int, int]]):
        size = DETECTOR_INPUT_SIZE
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image
        canvas = np.full((size, size, 3), DETECTOR_LETTERBOX_VALUE, dtype=np.uint8)
        width = image.shape[1]
        transforms = []
        x = 0
        for y0, y1 in halves:
            resized_w = max(1, min(size - x, int(width * scale)))
            resized_h = max(1, min(size, int((y1 - y0) * scale)))
            canvas[:resized_h, x:x + resized_w] = cv2.resize(rgb[y0:y1], (resized_w, resized_h))
            transforms.append(LetterboxTransform(
                src_w=width, src_h=y1 - y0, input_w=size, input_h=size,
                resized_w=resized_w, resized_h=resized_h, pad_x=x, pad_y=0,
                scale_x=resized_w / width, scale_y=resized_h / (y1 - y0),
                offset_x=0, offset_y=y0,
            ))
            x += resized_w + DETECTOR_COLLAGE_GAP
        outputs = self._run_session((canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        found: list[tuple[BubbleBox, int, bool]] = []
        raw_count = 0
        for index, ((y0, y1), transform) in enumerate(zip(halves, transforms)):
            # Each half's transform clamps boxes of the other half to nothing.
            raw = self._postprocess_at_threshold(outputs, transform, TEXT_CONF_THRESHOLD)
            raw_count += len(raw)
            for box in self._accept_many(raw):
                cut = (index > 0 and box.y1 <= y0 + 2) or (index < len(halves) - 1 and box.y2 >= y1 - 2)
                found.append((box, index, cut))
        return self.merge_tiles(found), raw_count

    @staticmethod
    def tile_windows(height: int, width: int, scale: float) -> list[tuple[int, int]]:
        """Vertical windows (y1, y2) that reach the model at about ``scale``.

        One window covers the whole slice when tiling is off or the slice is
        short enough to reach the model at that scale anyway.
        """
        if scale <= 0 or height <= 0 or width <= 0:
            return [(0, max(0, height))]
        scale = min(scale, DETECTOR_INPUT_SIZE / float(width))
        window = max(1, int(DETECTOR_INPUT_SIZE / scale))
        if height <= window * 1.1:
            return [(0, height)]
        overlap = max(64, int(window * DETECTOR_TILE_OVERLAP))
        count = max(2, int(np.ceil((height - overlap) / max(1, window - overlap))))
        # Spread the windows evenly: every overlap is at least ``overlap``.
        starts = [round(i * (height - window) / (count - 1)) for i in range(count)]
        return [(y0, y0 + window) for y0 in starts]

    @staticmethod
    def merge_tiles(found: list[tuple[BubbleBox, int, bool]]) -> list[BubbleBox]:
        """One box per text block from overlapping windows.

        ``found`` holds (box, window index, cut) where cut means the box touches
        an inner window edge. A box seen by two windows keeps the uncut, then the
        larger version; boxes from the same window never replace each other, and
        a block cut by every window keeps its pieces so its mask stays whole.
        """
        def area(b: BubbleBox) -> int:
            return max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)

        def overlap(a: BubbleBox, b: BubbleBox) -> int:
            return max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))

        def union(a: BubbleBox, b: BubbleBox) -> BubbleBox:
            x1, y1, x2, y2 = min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2)
            mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            for part in (a, b):
                if part.mask is not None and part.mask.shape == (part.y2 - part.y1, part.x2 - part.x1):
                    view = mask[part.y1 - y1:part.y2 - y1, part.x1 - x1:part.x2 - x1]
                    np.maximum(view, part.mask, out=view)
            return replace(a, x1=x1, y1=y1, x2=x2, y2=y2, mask=mask, confidence=max(a.confidence, b.confidence))

        def inside(a: BubbleBox, b: BubbleBox, slack: int = 2) -> bool:
            return a.x1 >= b.x1 - slack and a.y1 >= b.y1 - slack and a.x2 <= b.x2 + slack and a.y2 <= b.y2 + slack

        ranked = sorted(found, key=lambda item: (item[2], -area(item[0]), -item[0].confidence))
        kept: list[list] = []
        for box, tile, _cut in ranked:
            match = next((entry for entry in kept if entry[1] != tile
                          and overlap(box, entry[0]) >= 0.5 * max(1, min(area(box), area(entry[0])))), None)
            if match is None:
                kept.append([box, tile])
            elif not inside(box, match[0]):
                # Both windows cut this block: keep all of its mask.
                match[0] = union(match[0], box)
        return sorted((entry[0] for entry in kept), key=lambda b: (b.y1, b.x1))

    def _detect_tiled(self, image: np.ndarray, windows: list[tuple[int, int]]) -> tuple[list[BubbleBox], int]:
        found: list[tuple[BubbleBox, int, bool]] = []
        raw_count = 0
        for tile, (y0, y1) in enumerate(windows):
            blob, transform = self.detector._preprocess(image[y0:y1], offset_x=0, offset_y=y0)
            if blob is None or transform is None:
                continue
            raw = self._postprocess_at_threshold(self._run_session(blob), transform, TEXT_CONF_THRESHOLD)
            raw_count += len(raw)
            for box in self._accept_many(raw):
                cut = (tile > 0 and box.y1 <= y0 + 2) or (tile < len(windows) - 1 and box.y2 >= y1 - 2)
                found.append((box, tile, cut))
        return self.merge_tiles(found), raw_count

    def detect(self, image: np.ndarray) -> tuple[list[BubbleBox], dict[str, float | int]]:
        started = time.perf_counter()

        windows = self.tile_windows(image.shape[0], image.shape[1], float(self.tile_scale))
        plan = self.collage_plan(image.shape[0], image.shape[1]) if self.collage and len(windows) == 1 else None
        if len(windows) > 1 or plan is not None:
            if plan is not None:
                boxes, raw_count = self._detect_collage(image, *plan)
            else:
                boxes, raw_count = self._detect_tiled(image, windows)
            return boxes, {
                "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "detector_forward_calls": 1 if plan is not None else len(windows),
                "detector_boxes": int(raw_count),
                "accepted_mask_boxes": int(len(boxes)),
                "normal_conf_boxes": int(len(boxes)),
                "low_conf_rescue": 0,
                "low_conf_rescue_boxes": 0,
                "rescue_conf_threshold": float(self.RESCUE_CONF_THRESHOLD),
            }

        outputs, transform = self._single_forward_outputs(image)
        if outputs is None or transform is None:
            return [], {
                "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "detector_forward_calls": 0,
                "detector_boxes": 0,
                "accepted_mask_boxes": 0,
                "normal_conf_boxes": 0,
                "low_conf_rescue": 0,
                "low_conf_rescue_boxes": 0,
                "rescue_conf_threshold": float(self.RESCUE_CONF_THRESHOLD),
            }

        raw_boxes = self._postprocess_at_threshold(
            outputs,
            transform,
            TEXT_CONF_THRESHOLD,
        )
        boxes = self._accept_many(raw_boxes)

        return boxes, {
            "detector_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "detector_forward_calls": 1,
            "detector_boxes": int(len(raw_boxes)),
            "accepted_mask_boxes": int(len(boxes)),
            "normal_conf_boxes": int(len(boxes)),
            "low_conf_rescue": 0,
            "low_conf_rescue_boxes": 0,
            "rescue_conf_threshold": float(self.RESCUE_CONF_THRESHOLD),
        }

class OneShotProductionDetector:
    _MERGE_GAP = max(4, int(DETECTOR_RESIDUE_VERIFY_PAD))
    _UNION_SLACK_RATIO = 0.20
    _UNION_SLACK_PIXELS = 4096

    _FLAT_NEGATIVE_MIN_PIXELS = 64
    _FLAT_NEGATIVE_CHANNEL_SPAN_MAX = 12
    _FLAT_NEGATIVE_GRAY_STD_MAX = 3.5
    _FLAT_NEGATIVE_EDGE_DENSITY_MAX = 0.0015

    def __init__(self):
        self.core = OneShotTextMaskDetector()
        self.text_detector = self.core.detector
        self._metrics_local = threading.local()
        self._residue_metrics_local = threading.local()
        self._residue_metrics_lock = threading.Lock()
        self._residue_totals: dict[str, int] = {}
        self._residue_flat_gate_enabled = True

    def _set_residue_metrics(self, **values: int) -> None:
        snapshot = {str(name): int(value) for name, value in values.items()}
        self._residue_metrics_local.value = snapshot
        with self._residue_metrics_lock:
            self._residue_totals["verification_calls"] = (
                int(self._residue_totals.get("verification_calls", 0)) + 1
            )
            for name, value in snapshot.items():
                self._residue_totals[name] = int(
                    self._residue_totals.get(name, 0)
                ) + int(value)

    def last_residue_metrics(self) -> dict[str, int]:
        return dict(getattr(self._residue_metrics_local, "value", {}) or {})

    def residue_metrics_snapshot(self, *, reset: bool = False) -> dict[str, int]:
        with self._residue_metrics_lock:
            snapshot = {
                str(name): int(value)
                for name, value in self._residue_totals.items()
            }
            if reset:
                self._residue_totals.clear()
        return snapshot

    @staticmethod
    def _roi_gap(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
        gap_y = max(0, max(ay1, by1) - min(ay2, by2))
        return int(gap_x), int(gap_y)

    @classmethod
    def _can_merge_roi(
        cls,
        current: tuple[int, int, int, int],
        candidate: tuple[int, int, int, int],
    ) -> tuple[bool, tuple[int, int, int, int]]:
        ux1 = min(current[0], candidate[0])
        uy1 = min(current[1], candidate[1])
        ux2 = max(current[2], candidate[2])
        uy2 = max(current[3], candidate[3])
        union = (ux1, uy1, ux2, uy2)
        union_w = ux2 - ux1
        union_h = uy2 - uy1
        if union_w <= 0 or union_h <= 0:
            return False, union
        if max(union_w, union_h) > int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE):
            return False, union

        gap_x, gap_y = cls._roi_gap(current, candidate)
        if gap_x > cls._MERGE_GAP or gap_y > cls._MERGE_GAP:
            return False, union

        current_area = max(
            1,
            (current[2] - current[0]) * (current[3] - current[1]),
        )
        candidate_area = max(
            1,
            (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
        )
        union_area = union_w * union_h
        source_area = current_area + candidate_area
        allowed = source_area + max(
            cls._UNION_SLACK_PIXELS,
            int(round(source_area * cls._UNION_SLACK_RATIO)),
        )
        return union_area <= allowed, union

    @classmethod
    def _plan_residue_groups(
        cls,
        items: list[tuple[BubbleBox, tuple[int, int, int, int]]],
    ) -> list[dict]:
        groups: list[dict] = []
        for source, roi in items:
            best_index = None
            best_union = None
            best_area = None
            for index, group in enumerate(groups):
                ok, union = cls._can_merge_roi(group["roi"], roi)
                if not ok:
                    continue
                area = max(
                    1,
                    (union[2] - union[0]) * (union[3] - union[1]),
                )
                if best_area is None or area < best_area:
                    best_index = index
                    best_union = union
                    best_area = area
            if best_index is None:
                groups.append({"roi": roi, "sources": [source]})
                continue
            groups[best_index]["roi"] = best_union
            groups[best_index]["sources"].append(source)
        return groups

    def _is_flat_negative_residue_source(
        self,
        image: np.ndarray,
        source: BubbleBox,
    ) -> bool:
        if not bool(getattr(self, "_residue_flat_gate_enabled", True)):
            return False
        if image is None or image.size == 0:
            return False

        h, w = image.shape[:2]
        x1 = max(0, min(w, int(source.x1)))
        y1 = max(0, min(h, int(source.y1)))
        x2 = max(x1, min(w, int(source.x2)))
        y2 = max(y1, min(h, int(source.y2)))
        if x2 <= x1 or y2 <= y1:
            return False

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        pixel_count = int(crop.shape[0] * crop.shape[1])
        if pixel_count < self._FLAT_NEGATIVE_MIN_PIXELS:
            return False

        if crop.ndim == 3 and crop.shape[2] >= 3:
            pixels = crop[:, :, :3].reshape(-1, 3).astype(np.int16, copy=False)
            channel_span = pixels.max(axis=0) - pixels.min(axis=0)
            if int(channel_span.max()) > self._FLAT_NEGATIVE_CHANNEL_SPAN_MAX:
                return False
            gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = crop if crop.ndim == 2 else np.squeeze(crop)
            if gray.ndim != 2:
                return False
            gray_span = int(gray.max()) - int(gray.min())
            if gray_span > self._FLAT_NEGATIVE_CHANNEL_SPAN_MAX:
                return False

        if float(gray.std()) > self._FLAT_NEGATIVE_GRAY_STD_MAX:
            return False
        edges = cv2.Canny(gray, 32, 96, L2gradient=True)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, gray.size))
        return edge_density <= self._FLAT_NEGATIVE_EDGE_DENSITY_MAX

    @staticmethod
    def _tight_verified_mask_roi(
        image_shape: tuple[int, ...],
        source: BubbleBox,
    ) -> tuple[int, int, int, int] | None:
        h, w = int(image_shape[0]), int(image_shape[1])
        box_w = max(0, int(source.x2) - int(source.x1))
        box_h = max(0, int(source.y2) - int(source.y1))
        if box_w <= 0 or box_h <= 0:
            return None

        mask = source.mask
        if mask is not None and getattr(mask, "ndim", 0) >= 2:
            support = mask
            if support.shape[:2] != (box_h, box_w):
                support = cv2.resize(
                    support,
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            ys, xs = np.nonzero(support > 127)
            if xs.size and ys.size:
                pad = int(DETECTOR_RESIDUE_VERIFY_PAD)
                x1 = max(0, int(source.x1) + int(xs.min()) - pad)
                y1 = max(0, int(source.y1) + int(ys.min()) - pad)
                x2 = min(w, int(source.x1) + int(xs.max()) + 1 + pad)
                y2 = min(h, int(source.y1) + int(ys.max()) + 1 + pad)
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2

        pad = int(DETECTOR_RESIDUE_VERIFY_PAD)
        x1 = max(0, int(source.x1) - pad)
        y1 = max(0, int(source.y1) - pad)
        x2 = min(w, int(source.x2) + pad)
        y2 = min(h, int(source.y2) + pad)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def verify_post_inpaint_residue(
        self,
        image,
        authorized_boxes: list[BubbleBox],
    ) -> list[BubbleBox]:
        if image is None or image.size == 0:
            self._set_residue_metrics(
                source_candidates=0,
                scheduled_sources=0,
                neural_sources=0,
                flat_negative_sources=0,
                deferred_budget=0,
                deferred_size=0,
                groups=0,
                model_calls=0,
                merged_sources=0,
                source_pixels=0,
                grouped_pixels=0,
                residue_hits=0,
            )
            return []

        h, w = image.shape[:2]
        candidates = [
            box
            for box in authorized_boxes
            if box.verified_mask or bool(getattr(box, "verify_region_only", False))
        ]
        candidates.sort(
            key=lambda box: (
                (box.x2 - box.x1) * (box.y2 - box.y1)
            ),
            reverse=True,
        )

        residue: list[BubbleBox] = []
        scheduled: list[tuple[BubbleBox, tuple[int, int, int, int]]] = []
        deferred_budget = 0
        deferred_size = 0
        flat_negative_sources = 0
        source_pixels = 0

        neural_budget_used = 0
        for source in candidates:
            roi = self._tight_verified_mask_roi(image.shape, source)
            if roi is None:
                continue
            x1, y1, x2, y2 = roi
            if max(x2 - x1, y2 - y1) > int(
                DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE
            ):
                deferred_size += 1
                residue.append(
                    replace(
                        source,
                        safe_to_inpaint=False,
                        ocr_eligible=True,
                        needs_review=True,
                        deferred_reason="post_inpaint_verification_size",
                    )
                )
                continue

            if self._is_flat_negative_residue_source(image, source):
                flat_negative_sources += 1
                continue

            if neural_budget_used >= int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS):
                deferred_budget += 1
                residue.append(
                    replace(
                        source,
                        safe_to_inpaint=False,
                        ocr_eligible=True,
                        needs_review=True,
                        deferred_reason="post_inpaint_verification_budget",
                    )
                )
                continue

            neural_budget_used += 1
            source_pixels += (x2 - x1) * (y2 - y1)
            scheduled.append((source, roi))

        groups = self._plan_residue_groups(scheduled)
        grouped_pixels = sum(
            max(0, group["roi"][2] - group["roi"][0])
            * max(0, group["roi"][3] - group["roi"][1])
            for group in groups
        )

        model_calls = 0
        for group in groups:
            x1, y1, x2, y2 = group["roi"]
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            model_calls += 1
            verified_boxes = [
                self.text_detector._with_semantics(box)
                for box in self.text_detector._detect_single_plain(crop, x1, y1)
            ]
            for verified in verified_boxes:
                if not verified.verified_mask:
                    continue
                center_x = (verified.x1 + verified.x2) * 0.5
                center_y = (verified.y1 + verified.y2) * 0.5
                for source in group["sources"]:
                    if not (
                        source.x1 <= center_x <= source.x2
                        and source.y1 <= center_y <= source.y2
                    ):
                        continue
                    residue.append(
                        replace(
                            verified,
                            safe_to_inpaint=False,
                            ocr_eligible=True,
                            needs_review=True,
                            deferred_reason="post_inpaint_text_residue",
                        )
                    )
                    break

        result = apply_final_nms(
            residue,
            iou_threshold=DETECTOR_FINAL_NMS_IOU,
        )
        self._set_residue_metrics(
            source_candidates=len(candidates),
            scheduled_sources=len(scheduled),
            neural_sources=len(scheduled),
            flat_negative_sources=flat_negative_sources,
            deferred_budget=deferred_budget,
            deferred_size=deferred_size,
            groups=len(groups),
            model_calls=model_calls,
            merged_sources=max(0, len(scheduled) - len(groups)),
            source_pixels=source_pixels,
            grouped_pixels=grouped_pixels,
            residue_hits=sum(
                1
                for box in result
                if box.deferred_reason == "post_inpaint_text_residue"
            ),
        )
        return result

    def detect(self, image: np.ndarray, *, parallel: bool = False) -> list[BubbleBox]:
        boxes, metrics = self.core.detect(image)
        detector_ms = float(metrics.get("detector_ms", 0.0))
        self._metrics_local.value = {
            "bubble_model_ms": 0.0,
            "text_model_ms": detector_ms,
            "mser_ms": 0.0,
            "text_grayscale_fallback_runs": 0,
            "text_grayscale_fallback_calls": 0,
            "text_grayscale_fallback_source_pixels": 0,
            "text_grayscale_fallback_deferred_regions": 0,
            "text_grayscale_fallback_ms": 0.0,
            "text_grayscale_fallback_proposals": 0,
            "mser_segmenter_promotion_calls": 0,
            "mser_segmenter_promotions": 0,
            "mser_segmenter_promotion_deferred_regions": 0,
            "result_boxes": int(len(boxes)),
            "review_boxes": int(sum(1 for box in boxes if box.needs_review)),
            "detector_forward_calls": int(metrics.get("detector_forward_calls", 1)),
            "normal_conf_boxes": int(metrics.get("normal_conf_boxes", 0)),
            "low_conf_rescue": int(metrics.get("low_conf_rescue", 0)),
            "low_conf_rescue_boxes": int(metrics.get("low_conf_rescue_boxes", 0)),
            "total_ms": detector_ms,
        }
        return boxes

    def last_metrics(self) -> dict[str, float | int]:
        metrics = getattr(self._metrics_local, "value", {})
        return {
            str(name): float(value) if str(name).endswith("_ms") else int(value)
            for name, value in metrics.items()
        }
