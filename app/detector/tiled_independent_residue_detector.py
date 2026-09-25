from __future__ import annotations

from dataclasses import replace

from app.detector.independent_residue_detector import (
    IndependentRegionResidueSequentialTextDetector,
)
from app.parameters import (
    DETECTOR_FINAL_NMS_IOU,
    DETECTOR_RESIDUE_VERIFY_MAX_ROIS,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
)


class TiledIndependentRegionResidueSequentialTextDetector(
    IndependentRegionResidueSequentialTextDetector
):
    _RESIDUE_TILE_SIDE = 1024
    _RESIDUE_TILE_OVERLAP = 160
    _RESIDUE_MAX_TILES_PER_SOURCE = 4

    @classmethod
    def _axis_windows(
        cls,
        start: int,
        end: int,
        *,
        side: int | None = None,
        overlap: int | None = None,
    ) -> list[tuple[int, int]]:
        start = int(start)
        end = int(end)
        if end <= start:
            return []
        side = max(1, int(side or cls._RESIDUE_TILE_SIDE))
        overlap = max(0, min(side - 1, int(
            cls._RESIDUE_TILE_OVERLAP if overlap is None else overlap
        )))
        if end - start <= side:
            return [(start, end)]

        step = max(1, side - overlap)
        windows: list[tuple[int, int]] = []
        cursor = start
        while cursor + side < end:
            windows.append((cursor, cursor + side))
            cursor += step
        final_start = max(start, end - side)
        final = (final_start, end)
        if not windows or windows[-1] != final:
            windows.append(final)
        return windows

    @classmethod
    def _tiles_for_roi(
        cls,
        roi: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = (int(v) for v in roi)
        side = min(
            int(cls._RESIDUE_TILE_SIDE),
            int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE),
        )
        xs = cls._axis_windows(x1, x2, side=side)
        ys = cls._axis_windows(y1, y2, side=side)
        return [
            (tx1, ty1, tx2, ty2)
            for ty1, ty2 in ys
            for tx1, tx2 in xs
        ]

    def _publish_combined_metrics(
        self,
        base: dict[str, int],
        combined: dict[str, int],
    ) -> None:
        self._residue_metrics_local.value = {
            str(name): int(value) for name, value in combined.items()
        }
        with self._residue_metrics_lock:
            for name, value in combined.items():
                delta = int(value) - int(base.get(name, 0))
                if delta:
                    self._residue_totals[str(name)] = int(
                        self._residue_totals.get(str(name), 0)
                    ) + delta

    def verify_post_inpaint_residue(
        self,
        image,
        authorized_boxes,
    ):
        if image is None or image.size == 0:
            return super().verify_post_inpaint_residue(image, authorized_boxes)

        normal = []
        oversized: list[tuple[object, tuple[int, int, int, int]]] = []
        for source in authorized_boxes or []:
            if not getattr(source, "verified_mask", False):
                normal.append(source)
                continue
            roi = self._tight_verified_mask_roi(image.shape, source)
            if roi is None:
                normal.append(source)
                continue
            if max(roi[2] - roi[0], roi[3] - roi[1]) > int(
                DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE
            ):
                oversized.append((source, roi))
            else:
                normal.append(source)

        result = list(super().verify_post_inpaint_residue(image, normal))
        base = self.last_residue_metrics()
        combined = dict(base)
        combined["source_candidates"] = int(base.get("source_candidates", 0)) + len(oversized)
        combined.setdefault("tiled_sources", 0)
        combined.setdefault("tiled_model_calls", 0)
        combined.setdefault("tiled_source_pixels", 0)
        combined.setdefault("tiled_pixels", 0)

        remaining_budget = max(
            0,
            int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS)
            - int(base.get("neural_sources", 0)),
        )

        for source, roi in oversized:
            if self._is_flat_negative_residue_source(image, source):
                combined["flat_negative_sources"] = int(
                    combined.get("flat_negative_sources", 0)
                ) + 1
                continue

            tiles = self._tiles_for_roi(roi)
            if not tiles or len(tiles) > int(self._RESIDUE_MAX_TILES_PER_SOURCE):
                combined["deferred_size"] = int(combined.get("deferred_size", 0)) + 1
                result.append(
                    replace(
                        source,
                        safe_to_inpaint=False,
                        ocr_eligible=True,
                        needs_review=True,
                        deferred_reason="post_inpaint_verification_size",
                    )
                )
                continue

            if remaining_budget <= 0:
                combined["deferred_budget"] = int(combined.get("deferred_budget", 0)) + 1
                result.append(
                    replace(
                        source,
                        safe_to_inpaint=False,
                        ocr_eligible=True,
                        needs_review=True,
                        deferred_reason="post_inpaint_verification_budget",
                    )
                )
                continue

            remaining_budget -= 1
            combined["scheduled_sources"] = int(combined.get("scheduled_sources", 0)) + 1
            combined["neural_sources"] = int(combined.get("neural_sources", 0)) + 1
            combined["tiled_sources"] = int(combined.get("tiled_sources", 0)) + 1
            source_area = max(0, roi[2] - roi[0]) * max(0, roi[3] - roi[1])
            combined["source_pixels"] = int(combined.get("source_pixels", 0)) + source_area
            combined["tiled_source_pixels"] = int(
                combined.get("tiled_source_pixels", 0)
            ) + source_area

            source_hit = False
            for x1, y1, x2, y2 in tiles:
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                combined["model_calls"] = int(combined.get("model_calls", 0)) + 1
                combined["tiled_model_calls"] = int(
                    combined.get("tiled_model_calls", 0)
                ) + 1
                tile_pixels = max(0, x2 - x1) * max(0, y2 - y1)
                combined["grouped_pixels"] = int(
                    combined.get("grouped_pixels", 0)
                ) + tile_pixels
                combined["tiled_pixels"] = int(combined.get("tiled_pixels", 0)) + tile_pixels

                verified_boxes = [
                    self.text_detector._with_semantics(box)
                    for box in self.text_detector._detect_single_plain(crop, x1, y1)
                ]
                for verified in verified_boxes:
                    if not verified.verified_mask:
                        continue
                    center_x = (verified.x1 + verified.x2) * 0.5
                    center_y = (verified.y1 + verified.y2) * 0.5
                    if not (
                        source.x1 <= center_x <= source.x2
                        and source.y1 <= center_y <= source.y2
                    ):
                        continue
                    result.append(
                        replace(
                            verified,
                            safe_to_inpaint=False,
                            ocr_eligible=True,
                            needs_review=True,
                            deferred_reason="post_inpaint_text_residue",
                        )
                    )
                    source_hit = True

            if source_hit:
                pass

        result = self._apply_final_nms(
            result,
            iou_threshold=DETECTOR_FINAL_NMS_IOU,
        )
        combined["residue_hits"] = sum(
            1
            for box in result
            if box.deferred_reason == "post_inpaint_text_residue"
        )
        self._publish_combined_metrics(base, combined)
        return result
