from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.image_io import encode_mask
from app.mask_recall_pipeline import MaskRecallOptimizedChapterPipeline


class FlatEnvelopeMaskRecallPipeline(MaskRecallOptimizedChapterPipeline):
    """Recover clipped edge glyphs outside a text detector bbox safely.

    The first-pass detector geometry remains unchanged. During residue repair,
    overwhelmingly flat text containers may search a small envelope around the
    detector bbox. Only small dark/bright components surrounded mostly by the
    same flat background become repair scope. This lets clipped first/last glyphs
    cross the detector edge without turning a bubble/frame outline into authority.
    """

    _FLAT_SEARCH_PAD_X = 96
    _FLAT_SEARCH_PAD_Y = 32
    _FLAT_ENVELOPE_MIN_SOURCE_SIDE = 160
    _INK_MAX_COMPONENT_AREA_FRACTION = 0.005
    _INK_MAX_COMPONENT_AREA_MIN = 256
    _INK_RING_RADIUS = 4
    _INK_RING_FLAT_RATIO_MIN = 0.62
    _EXPAND_MARKER = "_flat_residue_envelope_authorized"

    @classmethod
    def _residue_repair_effective_boxes(cls, records: list[dict] | None):
        """Expand repair authority only for sources that passed the flat gate."""
        expandable: set[tuple[int, int, int, int]] = set()
        for record in records or []:
            if not isinstance(record, dict) or not record.get(cls._EXPAND_MARKER):
                continue
            try:
                expandable.add(
                    (
                        int(record["x1"]),
                        int(record["y1"]),
                        int(record["x2"]),
                        int(record["y2"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        boxes = super()._residue_repair_effective_boxes(records)
        if not expandable:
            return boxes

        for box in boxes:
            key = (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
            if key not in expandable or box.source_role != "text_segmenter":
                continue
            source_w = int(box.x2) - int(box.x1)
            source_h = int(box.y2) - int(box.y1)
            if max(source_w, source_h) < int(cls._FLAT_ENVELOPE_MIN_SOURCE_SIDE):
                continue
            box.x1 = int(box.x1) - int(cls._FLAT_SEARCH_PAD_X)
            box.x2 = int(box.x2) + int(cls._FLAT_SEARCH_PAD_X)
            box.y1 = int(box.y1) - int(cls._FLAT_SEARCH_PAD_Y)
            box.y2 = int(box.y2) + int(cls._FLAT_SEARCH_PAD_Y)
            box_h = int(box.y2) - int(box.y1)
            box_w = int(box.x2) - int(box.x1)
            if box_h > 0 and box_w > 0:
                box.mask = np.full((box_h, box_w), 255, dtype=np.uint8)
        return boxes

    @classmethod
    def _flat_residual_ink_envelope(
        cls,
        image: np.ndarray,
        source: dict,
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        try:
            sx1, sy1 = int(source["x1"]), int(source["y1"])
            sx2, sy2 = int(source["x2"]), int(source["y2"])
        except (KeyError, TypeError, ValueError):
            return None

        ox1, oy1 = max(0, sx1), max(0, sy1)
        ox2, oy2 = min(width, sx2), min(height, sy2)
        if ox2 <= ox1 or oy2 <= oy1:
            return None
        original = image[oy1:oy2, ox1:ox2]
        if original.ndim == 2:
            original_work = original[:, :, None]
        elif original.ndim == 3 and original.shape[2] >= 3:
            original_work = original[:, :, :3]
        else:
            return None

        original_pixels = original_work.reshape(-1, original_work.shape[2]).astype(
            np.int16, copy=False
        )
        background = np.median(original_pixels, axis=0).astype(np.int16)
        original_delta = np.max(
            np.abs(original_work.astype(np.int16, copy=False) - background),
            axis=2,
        )
        if float(np.mean(original_delta <= cls._FLAT_DELTA_MAX)) < cls._FLAT_RATIO_MIN:
            return None

        source_w, source_h = max(1, ox2 - ox1), max(1, oy2 - oy1)
        if max(source_w, source_h) < int(cls._FLAT_ENVELOPE_MIN_SOURCE_SIDE):
            local_mask = cls._flat_residual_ink_mask(original)
            if local_mask is None:
                return None
            return (ox1, oy1, ox2, oy2), local_mask

        ex1 = max(0, sx1 - int(cls._FLAT_SEARCH_PAD_X))
        ey1 = max(0, sy1 - int(cls._FLAT_SEARCH_PAD_Y))
        ex2 = min(width, sx2 + int(cls._FLAT_SEARCH_PAD_X))
        ey2 = min(height, sy2 + int(cls._FLAT_SEARCH_PAD_Y))
        if ex2 <= ex1 or ey2 <= ey1:
            return None

        crop = image[ey1:ey2, ex1:ex2]
        work = crop[:, :, None] if crop.ndim == 2 else crop[:, :, :3]
        delta = np.max(
            np.abs(work.astype(np.int16, copy=False) - background),
            axis=2,
        )

        flat = (delta <= cls._FLAT_DELTA_MAX).astype(np.uint8)
        count, labels, _, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
        if count <= 1:
            return None
        local_x1, local_y1 = max(0, ox1 - ex1), max(0, oy1 - ey1)
        local_x2 = min(flat.shape[1], ox2 - ex1)
        local_y2 = min(flat.shape[0], oy2 - ey1)
        best_label, best_overlap = 0, 0
        for label in range(1, count):
            overlap = int(
                np.count_nonzero(
                    labels[local_y1:local_y2, local_x1:local_x2] == label
                )
            )
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
        if best_label <= 0 or best_overlap <= 0:
            return None
        container = labels == best_label

        strong = (delta >= cls._INK_STRONG_DELTA).astype(np.uint8) * 255
        strong_fraction = float(np.mean(strong > 127))
        if strong_fraction <= 0.0 or strong_fraction > cls._INK_FRACTION_MAX:
            return None
        comp_count, comp_labels, stats, _ = cv2.connectedComponentsWithStats(
            strong, connectivity=8
        )
        source_area = max(1, source_w * source_h)
        max_w = max(1, int(round(source_w * cls._INK_MAX_COMPONENT_SPAN_FRACTION)))
        max_h = max(1, int(round(source_h * cls._INK_MAX_COMPONENT_SPAN_FRACTION)))
        max_area = max(
            int(cls._INK_MAX_COMPONENT_AREA_MIN),
            int(round(source_area * cls._INK_MAX_COMPONENT_AREA_FRACTION)),
        )
        ring_radius = max(1, int(cls._INK_RING_RADIUS))
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ring_radius * 2 + 1, ring_radius * 2 + 1),
        )

        seeds = np.zeros(strong.shape, dtype=np.uint8)
        for label in range(1, comp_count):
            _, _, comp_w, comp_h, area = (int(v) for v in stats[label])
            if area < cls._INK_MIN_COMPONENT_AREA or area > max_area:
                continue
            if comp_w > max_w or comp_h > max_h:
                continue
            component = comp_labels == label
            ring = (cv2.dilate(component.astype(np.uint8), ring_kernel) > 0) & (~component)
            if not np.any(ring):
                continue
            if float(np.mean(container[ring])) < cls._INK_RING_FLAT_RATIO_MIN:
                continue
            seeds[component] = 255
        if not np.any(seeds):
            return None

        weak = delta >= cls._INK_WEAK_DELTA
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        support = seeds > 0
        for _ in range(max(0, int(cls._INK_GROW_STEPS))):
            grown = cv2.dilate(support.astype(np.uint8), kernel) > 0
            support = support | (grown & weak)
        return (ex1, ey1, ex2, ey2), support.astype(np.uint8) * 255

    @classmethod
    def _augment_flat_residue_regions(cls, clean_before: np.ndarray, result: dict) -> int:
        regions = list(result.get("residue_regions") or [])
        records = [
            record
            for record in list(result.get("boxes") or [])
            if cls._record_is_flat_repair_source(record)
        ]
        if not records:
            return 0

        matched: set[int] = set()
        generated: list[dict] = []
        for source in records:
            envelope = cls._flat_residual_ink_envelope(clean_before, source)
            if envelope is None:
                source.pop(cls._EXPAND_MARKER, None)
                continue
            (cx1, cy1, cx2, cy2), residual_mask = envelope
            expanded = (
                cx1 < max(0, int(source["x1"]))
                or cy1 < max(0, int(source["y1"]))
                or cx2 > min(clean_before.shape[1], int(source["x2"]))
                or cy2 > min(clean_before.shape[0], int(source["y2"]))
            )
            if expanded:
                source[cls._EXPAND_MARKER] = True

            source_for_merge = dict(source)
            source_for_merge.update({"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2})
            neural_matches: list[tuple[int, dict]] = []
            for index, region in enumerate(regions):
                if (
                    isinstance(region, dict)
                    and region.get("deferred_reason") == "post_inpaint_text_residue"
                    and cls._region_center_inside_record(region, source_for_merge)
                ):
                    neural_matches.append((index, region))
            for index, region in neural_matches:
                residual_mask = cls._merge_verified_hit_into_source_mask(
                    source_for_merge, region, residual_mask
                )
                matched.add(index)

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
                replacement.update(
                    {
                        "deferred_reason": "post_inpaint_text_residue",
                        "needs_review": True,
                        "safe_to_inpaint": False,
                        "ocr_eligible": True,
                    }
                )
            replacement.update(
                {
                    "x1": cx1,
                    "y1": cy1,
                    "x2": cx2,
                    "y2": cy2,
                    "mask": encode_mask(residual_mask),
                    "repair_scope_source": "flat_container_envelope",
                }
            )
            generated.append(replacement)

        if not generated:
            return 0
        result["residue_regions"] = [
            region for index, region in enumerate(regions) if index not in matched
        ] + generated
        return len(generated)

    def _repair_post_inpaint_result(
        self,
        img_path: Path,
        result: dict,
        preserve_regions: list[dict] | None,
    ) -> dict:
        try:
            return super()._repair_post_inpaint_result(
                img_path, result, preserve_regions
            )
        finally:
            for record in list(result.get("boxes") or []):
                if isinstance(record, dict):
                    record.pop(self._EXPAND_MARKER, None)
