from __future__ import annotations

import numpy as np

from app.optimized_pipeline import OptimizedChapterPipeline


class MaskRecallOptimizedChapterPipeline(OptimizedChapterPipeline):
    """Repair verifier-confirmed missed glyphs without blind global mask growth.

    Automatic inpaint still uses the production segmenter mask unchanged. Only the
    post-inpaint residue pass gets a larger *candidate* authority: the original
    detector box envelope. The repair mask is still intersected with the residue
    verifier scope by ``OptimizedChapterPipeline._repair_post_inpaint_result`` and
    preserve regions remain hard-locked there.

    This targets the failure mode where a text-segmenter box is correct but its
    mask clips the first/last glyphs of a line. It also mirrors the initial page
    processor by excluding overlap-only slice context from destructive sources.
    """

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
            # path intersects it with a fresh post-inpaint text-residue mask first.
            box.mask = np.full((box_h, box_w), 255, dtype=np.uint8)
        return boxes
