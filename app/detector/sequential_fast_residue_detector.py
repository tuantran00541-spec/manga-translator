from __future__ import annotations

from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.detector.fast_residue_detector import FastResidueAdaptiveFocusCombinedTextDetector


class SequentialFastResidueAdaptiveFocusCombinedTextDetector(
    FastResidueAdaptiveFocusCombinedTextDetector
):
    def detect(self, image, *, parallel: bool = False):
        return AdaptiveFocusCombinedTextDetector.detect(
            self,
            image,
            parallel=False,
        )
