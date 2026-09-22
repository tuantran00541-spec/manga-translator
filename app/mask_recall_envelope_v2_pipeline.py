from __future__ import annotations

from pathlib import Path

from app.detector.adaptive_tiled_residue_detector import (
    AdaptiveTiledIndependentRegionResidueSequentialTextDetector,
)
from app.image_io import read_image
from app.inpaint.overlap_aware_adaptive_inpainter import (
    OverlapAwareAdaptiveFastInpainter,
)
from app.mask_recall_envelope_pipeline import FlatEnvelopeMaskRecallPipeline
from app.optimized_pipeline import OptimizedChapterPipeline
from app.parameters import (
    DETECTOR_INPUT_SIZE,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_TEXT_MASK_DECODE_PAD,
)


class CompleteFlatEnvelopeMaskRecallPipeline(FlatEnvelopeMaskRecallPipeline):
    """ Adaptive mask recall with bounded evidence-driven repair follow-ups. Geometry derives from detector scale instead of chapter-specific pixels. Near-duplicate destructive authorities are unioned before LaMa grouping, and post-inpaint repair follows fresh verifier evidence even on textured regions. Hard caps remain safety rails; they do not decide which pixels are writable. """

    _ADAPTIVE_GEOMETRY_SIDE = max(
        1,
        min(
            int(DETECTOR_INPUT_SIZE),
            int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE),
        ),
    )
    _FLAT_SEARCH_PAD_X = min(
        max(1, int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE) // 6),
        max(
            int(DETECTOR_TEXT_MASK_DECODE_PAD) * 4,
            int(round(_ADAPTIVE_GEOMETRY_SIDE * 0.125)),
        ),
    )
    _FLAT_SEARCH_PAD_Y = min(
        max(1, int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE) // 16),
        max(
            int(DETECTOR_TEXT_MASK_DECODE_PAD) * 2,
            int(round(_ADAPTIVE_GEOMETRY_SIDE * 0.03125)),
        ),
    )
    _REPAIR_PASS_HARD_CAP = 3

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = (
                        AdaptiveTiledIndependentRegionResidueSequentialTextDetector()
                    )
        return self._detector

    @property
    def inpainter(self):
        if self._inpainter is None:
            with self._inpainter_init_lock:
                if self._inpainter is None:
                    self._inpainter = OverlapAwareAdaptiveFastInpainter()
        return self._inpainter

    @staticmethod
    def _text_residue_regions(result: dict) -> list[dict]:
        return [
            region
            for region in list(result.get("residue_regions") or [])
            if isinstance(region, dict)
            and region.get("deferred_reason") == "post_inpaint_text_residue"
        ]

    @classmethod
    def _repair_pass_budget(cls, result: dict) -> int:
        """Scale the bounded retry budget with independent text authorities."""
        owned_sources = 0
        for record in list(result.get("boxes") or []):
            if not isinstance(record, dict):
                continue
            if record.get("removed") or record.get("deferred_reason"):
                continue
            if (
                record.get("overlap_context_only")
                and not record.get("geometry_overridden")
            ):
                continue
            if not (record.get("safe_to_inpaint") or record.get("geometry_overridden")):
                continue
            if str(record.get("source_role") or "") != "text_segmenter":
                continue
            owned_sources += 1
        if owned_sources <= 0:
            return 1
        return min(int(cls._REPAIR_PASS_HARD_CAP), owned_sources + 1)

    @classmethod
    def _clear_expand_markers(cls, result: dict) -> None:
        marker = getattr(cls, "_EXPAND_MARKER", None)
        if not marker:
            return
        for record in list(result.get("boxes") or []):
            if isinstance(record, dict):
                record.pop(marker, None)

    def _repair_post_inpaint_result(
        self,
        img_path: Path,
        result: dict,
        preserve_regions: list[dict] | None,
    ) -> dict:
        updated = super()._repair_post_inpaint_result(
            img_path,
            result,
            preserve_regions,
        )
        if updated.get("manual_mask") or updated.get("manual_lama_mask"):
            return updated

        tmp_clean_value = updated.get("tmp_clean")
        tmp_clean_path = Path(tmp_clean_value) if tmp_clean_value else None
        if tmp_clean_path is None or not tmp_clean_path.exists():
            return updated

        recall_metrics = updated.setdefault("processing_metrics", {}).setdefault(
            "mask_recall_repair", {}
        )
        repair_passes = int(recall_metrics.get("repair_passes", 0) or 0)
        pass_budget = self._repair_pass_budget(updated)
        followup_passes = 0
        followup_pixels = 0
        followup_outside = 0

        # The base recall path already handled the first repair and any flat-only
        # follow-up. Continue only while fresh verifier/ink evidence still exists.
        while repair_passes < pass_budget:
            clean_now = read_image(tmp_clean_path)
            added_flat = self._augment_flat_residue_regions(clean_now, updated)
            if added_flat:
                self._mark_residue_review_state(updated)

            if not self._text_residue_regions(updated):
                break

            self._mark_residue_review_state(updated)
            updated = OptimizedChapterPipeline._repair_post_inpaint_result(
                self,
                img_path,
                updated,
                preserve_regions,
            )
            self._clear_expand_markers(updated)

            repair_metrics = (
                updated.get("processing_metrics") or {}
            ).get("residue_repair") or {}
            if not int(repair_metrics.get("attempted", 0) or 0):
                break

            repair_passes += 1
            followup_passes += 1
            pass_pixels = int(repair_metrics.get("repair_mask_pixels", 0) or 0)
            followup_pixels += pass_pixels
            followup_outside += int(
                repair_metrics.get("outside_authority_changed_channel_values", 0)
                or 0
            )
            if pass_pixels <= 0:
                break

        # Re-run the independent flat audit after neural follow-ups. If the hard
        # cap is reached, keep review state rather than silently certifying clean.
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
            "mask_recall_repair", {}
        )
        recall_metrics.update(
            {
                "repair_passes": int(repair_passes),
                "repair_pass_budget": int(pass_budget),
                "neural_followup_passes": int(followup_passes),
                "neural_followup_repair_pixels": int(followup_pixels),
                "neural_followup_outside_changed_channel_values": int(
                    followup_outside
                ),
                "final_flat_residue_regions": int(final_flat_regions),
            }
        )
        self._clear_expand_markers(updated)
        return updated
