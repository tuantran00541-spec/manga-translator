from __future__ import annotations

from dataclasses import replace
import threading

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.parallel_focus_detector import (
    ParallelAdaptiveFocusCombinedTextDetector,
)
from app.parameters import (
    DETECTOR_FINAL_NMS_IOU,
    DETECTOR_RESIDUE_VERIFY_MAX_ROIS,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_RESIDUE_VERIFY_PAD,
)


class FastResidueAdaptiveFocusCombinedTextDetector(
    ParallelAdaptiveFocusCombinedTextDetector
):
    """Adaptive detector with parallel coarse prefetch and fast residue checks.

    Two optimizations remain conservative and independently fail-safe:

    * When the chapter pipeline has only one page worker, the parent detector may
      overlap bubble inference with the independent coarse/full text pass. Focus
      retries remain sequential because their geometry depends on bubble/MSER
      proposals. Multi-page processing keeps the validated sequential path.
    * Residue verification coalesces nearby windows, then skips neural inference
      only when an already-cleaned source bbox is almost perfectly flat. Any
      meaningful luminance/colour span or edge signal falls back to the existing
      1024px text segmenter verifier.

    MSER extraction/base-candidate work is already cached inside
    ``SecondaryTextRecovery`` before ``existing`` filtering, so the pre-focus and
    final recovery passes reuse the expensive primitives while retaining their
    distinct filtering semantics.
    """

    # Shared-ROI coalescing is experimental. A real chapter probe found it can
    # change residue evidence at ROI boundaries, so it stays disabled until it
    # passes the confirmed hard/holdout set.
    _MERGE_GAP = max(4, int(DETECTOR_RESIDUE_VERIFY_PAD))
    _UNION_SLACK_RATIO = 0.20
    _UNION_SLACK_PIXELS = 4096
    _RESIDUE_COALESCING_DEFAULT = False

    # Extremely conservative negative gate. It remains disabled by default
    # until a residue-positive hard set proves that it does not skip real text.
    # When explicitly enabled, one clearly contrasting residual stroke is
    # enough to force the neural verifier.
    _FLAT_NEGATIVE_MIN_PIXELS = 64
    _FLAT_NEGATIVE_CHANNEL_SPAN_MAX = 12
    _FLAT_NEGATIVE_GRAY_STD_MAX = 3.5
    _FLAT_NEGATIVE_EDGE_DENSITY_MAX = 0.0015
    _FLAT_NEGATIVE_GATE_DEFAULT = False

    def __init__(self):
        super().__init__()
        self._residue_metrics_local = threading.local()
        self._residue_metrics_lock = threading.Lock()
        self._residue_totals: dict[str, int] = {}
        self._residue_flat_gate_enabled = self._FLAT_NEGATIVE_GATE_DEFAULT
        self._residue_coalescing_enabled = self._RESIDUE_COALESCING_DEFAULT

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
        """Return cross-worker residue counters for profiling/health telemetry."""
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
        """Return True only for an almost perfectly flat cleaned source bbox.

        This is a one-sided gate: ``False`` means "run the neural verifier" and
        is always safe. ``True`` is deliberately hard to reach so a residual
        glyph, outline, colour edge, gradient or artwork texture keeps the old
        verification path.
        """
        if not bool(getattr(self, "_residue_flat_gate_enabled", False)):
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
        candidates = [box for box in authorized_boxes if box.verified_mask]
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

        for index, source in enumerate(candidates):
            # Preserve the historical source-budget/order semantics. A cheap
            # negative inside the first N candidates does not silently make a
            # later source eligible when it used to be review-deferred.
            if index >= int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS):
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

            x1 = max(0, int(source.x1) - int(DETECTOR_RESIDUE_VERIFY_PAD))
            y1 = max(0, int(source.y1) - int(DETECTOR_RESIDUE_VERIFY_PAD))
            x2 = min(w, int(source.x2) + int(DETECTOR_RESIDUE_VERIFY_PAD))
            y2 = min(h, int(source.y2) + int(DETECTOR_RESIDUE_VERIFY_PAD))
            if x2 <= x1 or y2 <= y1:
                continue
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

            roi = (x1, y1, x2, y2)
            source_pixels += (x2 - x1) * (y2 - y1)
            scheduled.append((source, roi))

        if bool(getattr(self, "_residue_coalescing_enabled", False)):
            groups = self._plan_residue_groups(scheduled)
        else:
            groups = [
                {"roi": roi, "sources": [source]}
                for source, roi in scheduled
            ]
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
                    # One verified text detection is enough evidence for this
                    # source region. NMS below removes cross-source duplicates.
                    break

        result = self._apply_final_nms(
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
