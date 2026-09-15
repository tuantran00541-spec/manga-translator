from __future__ import annotations

from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.detector.fast_residue_detector import FastResidueAdaptiveFocusCombinedTextDetector


class SequentialFastResidueAdaptiveFocusCombinedTextDetector(
    FastResidueAdaptiveFocusCombinedTextDetector
):
    """Use the validated sequential detector schedule with the fast residue gate.

    Real-chapter A/B on the four-core CPU runner showed that overlapping bubble
    and full-text OpenVINO requests created substantial compute overlap but did
    not reduce detector wall time. Keep the parallel implementation available as
    an experiment, while production-candidate scheduling stays sequential and
    retains the MSER reuse and residue-verification optimizations.
    """

    def detect(self, image, *, parallel: bool = False):
        return AdaptiveFocusCombinedTextDetector.detect(
            self,
            image,
            parallel=False,
        )
