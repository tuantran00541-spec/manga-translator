from __future__ import annotations

from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.pipeline import ChapterPipeline


class OptimizedChapterPipeline(ChapterPipeline):
    """Chapter pipeline using the validated adaptive-focus production detector."""

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = AdaptiveFocusCombinedTextDetector()
        return self._detector
