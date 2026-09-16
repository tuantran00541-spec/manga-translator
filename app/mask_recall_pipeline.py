from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.image_io import encode_mask, read_image
from app.mask_store import decode_mask_value
from app.optimized_pipeline import OptimizedChapterPipeline


class MaskRecallOptimizedChapterPipeline(OptimizedChapterPipeline):
    """Recover verifier-confirmed clipped glyphs without blind bbox repaint.

    The first automatic inpaint pass still uses the production segmenter mask
    unchanged. During the residue pass, text-segmenter boxes expose their full
    detector envelope as *candidate* authority, but actual writes remain limited
    by the post-inpaint residue scope and preserve regions stay hard-locked by the
    parent pipeline.

    A second conservative fallback is used only for flat speech/narration boxes
    after the neural verifier has already confirmed at least one residue hit. In
    that case we recover small, strongly contrasting ink components elsewhere in
    the same flat box so isolated first/last glyphs cannot disappear from the
    repair scope merely because the verifier missed them.
    """

    _FLAT_DELTA_MAX = 8
    _FLAT_RATIO_MIN = 0.98
    _INK_STRONG_DELTA = 32
    _INK_WEAK_DELTA = 12
    _INK_FRACTION_MAX = 0.08
    _INK_MIN_COMPONENT_AREA = 3
    _INK_MAX_COMPONENT_SPAN_FRACTION = 0.65
    _INK_GROW_STEPS = 3

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
            # This envelope does not itself authorize a repaint. The parent repair
            # path intersects it with a post-inpaint residue scope first.
            box.mask = np.full((box_h, box_w), 255, dtype=np.uint8)
        return boxes

    @classmethod
    def _flat_residual_ink_mask(cls, crop: np.ndarray) -> np.ndarray | None:
        """Return residual ink support only for an overwhelmingly flat crop.

        The gate is intentionally one-sided: uncertain/textured crops return None
        and fall back to the neural residue mask. Components are allowed to touch
        the detector-box edge because the real failure mode is clipped first/last
        glyphs there. Long frame/box strokes are still rejected by their span.
        """
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

        # Include one background pixel around the recovered stroke. LaMa receives
        # a larger hidden inference mask internally, while final writes remain
        # clipped to this bounded support by the parent repair path.
        support = cv2.dilate(support.astype(np.uint8), kernel) > 0
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
        return str(record.get("semantic_type") or "") == "speech_bubble"

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

    @classmethod
    def _augment_flat_residue_regions(
        cls,
        clean_before: np.ndarray,
        result: dict,
    ) -> int:
        """Expand confirmed residue scope with flat-box residual ink evidence."""
        regions = list(result.get("residue_regions") or [])
        records = [
            record
            for record in list(result.get("boxes") or [])
            if cls._record_is_flat_repair_source(record)
        ]
        if not regions or not records:
            return 0

        height, width = clean_before.shape[:2]
        augmented = 0
        used_sources: set[tuple[int, int, int, int]] = set()
        output: list[dict] = []

        for region in regions:
            if (
                not isinstance(region, dict)
                or region.get("deferred_reason") != "post_inpaint_text_residue"
            ):
                output.append(region)
                continue
            try:
                rx1, ry1 = int(region["x1"]), int(region["y1"])
                rx2, ry2 = int(region["x2"]), int(region["y2"])
            except (KeyError, TypeError, ValueError):
                output.append(region)
                continue
            center_x = (rx1 + rx2) * 0.5
            center_y = (ry1 + ry2) * 0.5

            matches: list[tuple[int, dict]] = []
            for record in records:
                try:
                    sx1, sy1 = int(record["x1"]), int(record["y1"])
                    sx2, sy2 = int(record["x2"]), int(record["y2"])
                except (KeyError, TypeError, ValueError):
                    continue
                if sx2 <= sx1 or sy2 <= sy1:
                    continue
                if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                    matches.append(((sx2 - sx1) * (sy2 - sy1), record))
            if not matches:
                output.append(region)
                continue

            _, source = min(matches, key=lambda item: item[0])
            sx1, sy1 = int(source["x1"]), int(source["y1"])
            sx2, sy2 = int(source["x2"]), int(source["y2"])
            key = (sx1, sy1, sx2, sy2)
            if key in used_sources:
                output.append(region)
                continue
            used_sources.add(key)

            cx1, cy1 = max(0, sx1), max(0, sy1)
            cx2, cy2 = min(width, sx2), min(height, sy2)
            if cx2 <= cx1 or cy2 <= cy1:
                output.append(region)
                continue
            crop = clean_before[cy1:cy2, cx1:cx2]
            residual_mask = cls._flat_residual_ink_mask(crop)
            if residual_mask is None:
                output.append(region)
                continue

            # Detector boxes are normally in-bounds; keep coordinate/mask geometry
            # exact even when a malformed/legacy record is clipped to the page.
            if (cx1, cy1, cx2, cy2) != key:
                source_for_merge = dict(source)
                source_for_merge.update({"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2})
            else:
                source_for_merge = source
            residual_mask = cls._merge_verified_hit_into_source_mask(
                source_for_merge,
                region,
                residual_mask,
            )
            replacement = dict(region)
            replacement.update(
                {
                    "x1": cx1,
                    "y1": cy1,
                    "x2": cx2,
                    "y2": cy2,
                    "mask": encode_mask(residual_mask),
                    "repair_scope_source": "flat_residual_ink",
                }
            )
            output.append(replacement)
            augmented += 1

        if augmented:
            result["residue_regions"] = output
        return augmented

    def _repair_post_inpaint_result(
        self,
        img_path: Path,
        result: dict,
        preserve_regions: list[dict] | None,
    ) -> dict:
        """Augment only verifier-confirmed flat-box residue before parent repair."""
        if not result.get("manual_mask") and not result.get("manual_lama_mask"):
            tmp_clean_value = result.get("tmp_clean")
            if tmp_clean_value:
                tmp_clean_path = Path(tmp_clean_value)
                if tmp_clean_path.exists():
                    clean_before = read_image(tmp_clean_path)
                    augmented = self._augment_flat_residue_regions(
                        clean_before,
                        result,
                    )
                    if augmented:
                        metrics = result.setdefault("processing_metrics", {})
                        metrics.setdefault("residue_repair_scope", {})[
                            "flat_residual_ink_regions"
                        ] = int(augmented)
        return super()._repair_post_inpaint_result(
            img_path,
            result,
            preserve_regions,
        )
