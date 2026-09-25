from __future__ import annotations

from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.parameters import DETECTOR_RESIDUE_VERIFY_PAD


class IndependentRegionResidueSequentialTextDetector(
    SequentialFastResidueAdaptiveFocusCombinedTextDetector
):
    @staticmethod
    def _tight_verified_mask_roi(
        image_shape: tuple[int, ...],
        source,
    ) -> tuple[int, int, int, int] | None:
        h, w = int(image_shape[0]), int(image_shape[1])
        pad = int(DETECTOR_RESIDUE_VERIFY_PAD)
        x1 = max(0, int(source.x1) - pad)
        y1 = max(0, int(source.y1) - pad)
        x2 = min(w, int(source.x2) + pad)
        y2 = min(h, int(source.y2) + pad)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
