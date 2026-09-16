from __future__ import annotations

from app.detector.tiled_independent_residue_detector import (
    TiledIndependentRegionResidueSequentialTextDetector,
)
from app.mask_recall_envelope_pipeline import FlatEnvelopeMaskRecallPipeline


class CompleteFlatEnvelopeMaskRecallPipeline(FlatEnvelopeMaskRecallPipeline):
    """Complete clipped edge glyphs and verify oversized text regions by tiles.

    Real-page audit on Shadow Slave p117 showed the source detector box ending at
    x=1266, while the final surviving glyph fragment occupied roughly x=1362-1365.
    The old 96 px envelope therefore ended exactly through the glyph. Keep all
    flat-container/ring/component safety gates and add only enough horizontal
    slack to contain that failure mode. Oversized residue verification is handled
    separately by a tiled detector, so large regions are not blindly downscaled
    or marked clean without inspection.
    """

    _FLAT_SEARCH_PAD_X = 128
    _FLAT_SEARCH_PAD_Y = 32

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = (
                        TiledIndependentRegionResidueSequentialTextDetector()
                    )
        return self._detector
