from __future__ import annotations

from app.detector.tiled_independent_residue_detector import (
    TiledIndependentRegionResidueSequentialTextDetector,
)
from app.parameters import (
    DETECTOR_INPUT_SIZE,
    DETECTOR_RESIDUE_VERIFY_MAX_ROIS,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_TEXT_MASK_DECODE_PAD,
)


class AdaptiveTiledIndependentRegionResidueSequentialTextDetector(
    TiledIndependentRegionResidueSequentialTextDetector
):
    """Scale residue tiling from detector geometry instead of fixed pixels.

    The model input side is the natural tile scale. Overlap follows that scale,
    with the existing text-mask decode pad as a lower bound so seams keep enough
    context. The per-source tile cap follows the configured neural ROI budget.
    """

    _RESIDUE_TILE_SIDE = max(
        1,
        min(
            int(DETECTOR_INPUT_SIZE),
            int(DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE),
        ),
    )
    _RESIDUE_TILE_OVERLAP = min(
        max(1, _RESIDUE_TILE_SIDE // 3),
        max(
            int(DETECTOR_TEXT_MASK_DECODE_PAD) * 2,
            int(round(_RESIDUE_TILE_SIDE * 0.16)),
        ),
    )
    _RESIDUE_MAX_TILES_PER_SOURCE = max(
        2,
        min(
            int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS),
            (int(DETECTOR_RESIDUE_VERIFY_MAX_ROIS) + 1) // 2,
        ),
    )
