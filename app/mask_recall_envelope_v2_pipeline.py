from __future__ import annotations

from app.detector.adaptive_tiled_residue_detector import (
    AdaptiveTiledIndependentRegionResidueSequentialTextDetector,
)
from app.mask_recall_envelope_pipeline import FlatEnvelopeMaskRecallPipeline
from app.parameters import (
    DETECTOR_INPUT_SIZE,
    DETECTOR_RESIDUE_VERIFY_MAX_SOURCE_SIDE,
    DETECTOR_TEXT_MASK_DECODE_PAD,
)


class CompleteFlatEnvelopeMaskRecallPipeline(FlatEnvelopeMaskRecallPipeline):
    """Adapt edge-glyph recovery and oversized verification to detector geometry.

    The envelope is derived from the configured detector scale instead of a
    chapter-specific pixel constant. Hard bounds remain only as safety rails;
    flat-container/ring/component gates still decide whether any expanded pixels
    may become repair scope. Oversized verification uses adaptive tiling.
    """

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

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = (
                        AdaptiveTiledIndependentRegionResidueSequentialTextDetector()
                    )
        return self._detector
