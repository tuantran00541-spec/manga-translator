from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.detector.independent_residue_detector import (
    IndependentRegionResidueSequentialTextDetector,
)
from app.image_io import encode_mask, read_image
from app.mask_store import decode_mask_value
from app.optimized_pipeline import OptimizedChapterPipeline


class MaskRecallOptimizedChapterPipeline(OptimizedChapterPipeline):
    """ Recover clipped glyphs with independent residue verification. """

    _FLAT_DELTA_MAX = 8
    _FLAT_RATIO_MIN = 0.98
    _INK_STRONG_DELTA = 32
    _INK_WEAK_DELTA = 12
    _INK_FRACTION_MAX = 0.08
    _INK_MIN_COMPONENT_AREA = 3
    _INK_MAX_COMPONENT_SPAN_FRACTION = 0.65
    _INK_GROW_STEPS = 3
    _INK_COMMIT_MARGIN = 2
    _MAX_REPAIR_PASSES = 2

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = IndependentRegionResidueSequentialTextDetector()
        return self._detector

    @staticmethod
    def _residue_repair_effective_boxes(records: list[dict] | None):
        filtered_records: list[dict] = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            if (
                record.get("overlap_context_only")
                and not record.get("geometry_overridden")
            ):
                continue
            filtered_records.append(record)

        boxes = OptimizedChapterPipeline._residue_repair_effective_boxes(
            filtered_records
        )
        for box in boxes:
            if box.source_role != "text_segmenter":
                continue
            box_h = int(box.y2) - int(box.y1)
            box_w = int(box.x2) - int(box.x1)
            if box_h <= 0 or box_w <= 0:
                continue
            # The full detector envelope is candidate repair authority only.
            # Actual writes are still intersected with a residue scope and with
            # preserve-region subtraction in the parent repair path.
            box.mask = np.full((box_h, box_w), 255, dtype=np.uint8)
        return boxes

    @classmethod
    def _flat_residual_ink_mask(cls, crop: np.ndarray) -> np.ndarray | None:
        """ Return bounded residual-ink support for an overwhelmingly flat crop. """
        if crop is None or crop.size == 0:
            return None
        if crop.ndim == 2:
            work = crop[:, :, None]
        elif crop.ndim == 3 and crop.shape[2] >= 3:
            work = crop[:, :, :3]
        else:
            return None

        height, width = work.shape[:2]
        if height < 8 or width < 8:
            return None

        pixels = work.reshape(-1, work.shape[2]).astype(np.int16, copy=False)
        background = np.median(pixels, axis=0).astype(np.int16)
        delta = np.max(
            np.abs(work.astype(np.int16, copy=False) - background),
            axis=2,
        )
        flat_ratio = float(np.mean(delta <= cls._FLAT_DELTA_MAX))
        if flat_ratio < cls._FLAT_RATIO_MIN:
            return None

        strong = delta >= cls._INK_STRONG_DELTA
        strong_fraction = float(np.mean(strong))
        if strong_fraction <= 0.0 or strong_fraction > cls._INK_FRACTION_MAX:
            return None

        binary = strong.astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        seeds = np.zeros((height, width), dtype=np.uint8)
        max_w = max(1, int(round(width * cls._INK_MAX_COMPONENT_SPAN_FRACTION)))
        max_h = max(1, int(round(height * cls._INK_MAX_COMPONENT_SPAN_FRACTION)))
        for label in range(1, count):
            _, _, comp_w, comp_h, area = (int(v) for v in stats[label])
            if area < cls._INK_MIN_COMPONENT_AREA:
                continue
            if comp_w > max_w or comp_h > max_h:
                continue
            seeds[labels == label] = 255

        if not np.any(seeds):
            return None

        weak = delta >= cls._INK_WEAK_DELTA
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        support = seeds > 0
        for _ in range(max(0, int(cls._INK_GROW_STEPS))):
            grown = cv2.dilate(support.astype(np.uint8), kernel) > 0
            support = support | (grown & weak)

        # Return stroke support here; the caller applies the explicit commit
        # margin after neural and contrast evidence have been merged.
        return support.astype(np.uint8) * 255

    @staticmethod
    def _record_is_flat_repair_source(record: dict) -> bool:
        if not isinstance(record, dict):
            return False
        if record.get("removed") or record.get("deferred_reason"):
            return False
        if (
            record.get("overlap_context_only")
            and not record.get("geometry_overridden")
        ):
            return False
        if not (record.get("safe_to_inpaint") or record.get("geometry_overridden")):
            return False
        if str(record.get("source_role") or "") != "text_segmenter":
            return False
        return str(record.get("semantic_type") or "") in {
            "speech_bubble",
            "text",
            "free_text",
        }

    @classmethod
    def _merge_verified_hit_into_source_mask(
        cls,
        source: dict,
        region: dict,
        source_mask: np.ndarray,
    ) -> np.ndarray:
        merged = source_mask.copy()
        sx1, sy1 = int(source["x1"]), int(source["y1"])
        sx2, sy2 = int(source["x2"]), int(source["y2"])
        rx1, ry1 = int(region["x1"]), int(region["y1"])
        rx2, ry2 = int(region["x2"]), int(region["y2"])
        ix1, iy1 = max(sx1, rx1), max(sy1, ry1)
        ix2, iy2 = min(sx2, rx2), min(sy2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            return merged

        hit_mask = decode_mask_value(region.get("mask"))
        if hit_mask is None:
            merged[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = 255
            return merged

        target_w, target_h = rx2 - rx1, ry2 - ry1
        if target_w <= 0 or target_h <= 0:
            return merged
        if hit_mask.shape[:2] != (target_h, target_w):
            hit_mask = cv2.resize(
                hit_mask,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        hx1, hy1 = ix1 - rx1, iy1 - ry1
        hx2, hy2 = ix2 - rx1, iy2 - ry1
        local = hit_mask[hy1:hy2, hx1:hx2]
        target = merged[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1]
        target[local > 127] = 255
        return merged

    @staticmethod
    def _region_center_inside_record(region: dict, record: dict) -> bool:
        try:
            center_x = (int(region["x1"]) + int(region["x2"])) * 0.5
            center_y = (int(region["y1"]) + int(region["y2"])) * 0.5
            return (
                int(record["x1"]) <= center_x <= int(record["x2"])
                and int(record["y1"]) <= center_y <= int(record["y2"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    @classmethod
    def _augment_flat_residue_regions(
        cls,
        clean_before: np.ndarray,
        result: dict,
    ) -> int:
        """Add independent flat-region residue evidence, with or without neural hits."""
        regions = list(result.get("residue_regions") or [])
        records = [
            record
            for record in list(result.get("boxes") or [])
            if cls._record_is_flat_repair_source(record)
        ]
        if not records:
            return 0

        height, width = clean_before.shape[:2]
        matched_region_indexes: set[int] = set()
        generated: list[dict] = []

        for source in records:
            try:
                sx1, sy1 = int(source["x1"]), int(source["y1"])
                sx2, sy2 = int(source["x2"]), int(source["y2"])
            except (KeyError, TypeError, ValueError):
                continue

            cx1, cy1 = max(0, sx1), max(0, sy1)
            cx2, cy2 = min(width, sx2), min(height, sy2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = clean_before[cy1:cy2, cx1:cx2]
            residual_mask = cls._flat_residual_ink_mask(crop)
            if residual_mask is None:
                continue

            source_for_merge = dict(source)
            source_for_merge.update(
                {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2}
            )
            neural_matches: list[tuple[int, dict]] = []
            for index, region in enumerate(regions):
                if (
                    isinstance(region, dict)
                    and region.get("deferred_reason")
                    == "post_inpaint_text_residue"
                    and cls._region_center_inside_record(region, source_for_merge)
                ):
                    neural_matches.append((index, region))

            for index, region in neural_matches:
                residual_mask = cls._merge_verified_hit_into_source_mask(
                    source_for_merge,
                    region,
                    residual_mask,
                )
                matched_region_indexes.add(index)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            residual_mask = cv2.dilate(
                (residual_mask > 127).astype(np.uint8) * 255,
                kernel,
                iterations=max(0, int(cls._INK_COMMIT_MARGIN)),
            )
            if not np.any(residual_mask > 127):
                continue

            if neural_matches:
                replacement = dict(neural_matches[0][1])
            else:
                replacement = {
                    key: source.get(key)
                    for key in (
                        "confidence",
                        "source_model",
                        "source_role",
                        "class_name",
                        "semantic_type",
                    )
                }
                replacement["deferred_reason"] = "post_inpaint_text_residue"
                replacement["needs_review"] = True
                replacement["safe_to_inpaint"] = False
                replacement["ocr_eligible"] = True

            replacement.update(
                {
                    "x1": cx1,
                    "y1": cy1,
                    "x2": cx2,
                    "y2": cy2,
                    "mask": encode_mask(residual_mask),
                    "repair_scope_source": "flat_independent_ink",
                }
            )
            generated.append(replacement)

        if not generated:
            return 0

        output = [
            region
            for index, region in enumerate(regions)
            if index not in matched_region_indexes
        ]
        output.extend(generated)
        result["residue_regions"] = output
        return len(generated)

    @staticmethod
    def _mark_residue_review_state(result: dict) -> None:
        issues = [
            issue
            for issue in list(result.get("detection_issues") or [])
            if issue != "post_inpaint_text_residue"
        ]
        issues.append("post_inpaint_text_residue")
        result["detection_issues"] = issues
        result["detection_state"] = "needs_review"
        result["cleanup_verified"] = False
        result["needs_review"] = True

    def _repair_post_inpaint_result(
        self,
        img_path: Path,
        result: dict,
        preserve_regions: list[dict] | None,
    ) -> dict:
        """Run bounded independent flat-residue repair and final flat verification."""
        if result.get("manual_mask") or result.get("manual_lama_mask"):
            return super()._repair_post_inpaint_result(
                img_path,
                result,
                preserve_regions,
            )

        tmp_clean_value = result.get("tmp_clean")
        tmp_clean_path = Path(tmp_clean_value) if tmp_clean_value else None
        if tmp_clean_path is None or not tmp_clean_path.exists():
            return super()._repair_post_inpaint_result(
                img_path,
                result,
                preserve_regions,
            )

        metrics = result.setdefault("processing_metrics", {})
        recall_metrics = metrics.setdefault("mask_recall_repair", {})
        clean_before = read_image(tmp_clean_path)
        pre_regions = self._augment_flat_residue_regions(clean_before, result)
        recall_metrics["pre_repair_flat_regions"] = int(pre_regions)

        updated = super()._repair_post_inpaint_result(
            img_path,
            result,
            preserve_regions,
        )
        repair_passes = 1 if (
            (updated.get("processing_metrics") or {})
            .get("residue_repair", {})
            .get("attempted")
        ) else 0

        if self._MAX_REPAIR_PASSES > 1 and tmp_clean_path.exists():
            clean_after = read_image(tmp_clean_path)
            post_regions = self._augment_flat_residue_regions(clean_after, updated)
            recall_metrics = updated.setdefault("processing_metrics", {}).setdefault(
                "mask_recall_repair",
                {},
            )
            recall_metrics["post_repair_flat_regions"] = int(post_regions)
            if post_regions:
                self._mark_residue_review_state(updated)
                updated = super()._repair_post_inpaint_result(
                    img_path,
                    updated,
                    preserve_regions,
                )
                repair_passes += 1

        # Never let a neural false-negative certify a flat detector-owned text
        # region that still contains strong residual ink after the bounded repair.
        final_flat_regions = 0
        if tmp_clean_path.exists():
            clean_final = read_image(tmp_clean_path)
            final_flat_regions = self._augment_flat_residue_regions(
                clean_final,
                updated,
            )
            if final_flat_regions:
                self._mark_residue_review_state(updated)

        recall_metrics = updated.setdefault("processing_metrics", {}).setdefault(
            "mask_recall_repair",
            {},
        )
        recall_metrics.update(
            {
                "repair_passes": int(repair_passes),
                "final_flat_residue_regions": int(final_flat_regions),
                "independent_verifier": "full_detector_region",
            }
        )
        return updated
