from __future__ import annotations

from app.detector.adaptive_focus_detector import AdaptiveFocusCombinedTextDetector
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.runtime_responsiveness import responsive_process_workers


class OptimizedChapterPipeline(ChapterPipeline):
    """Chapter pipeline using the validated adaptive-focus production detector."""

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = AdaptiveFocusCombinedTextDetector()
        return self._detector

    def process_pages(
        self,
        chapter_id: str,
        page_indices: list[int],
        workers: int = PIPELINE_DEFAULT_WORKERS,
    ) -> dict:
        """Process pages without starving the local browser of CPU time."""
        return super().process_pages(
            chapter_id,
            page_indices,
            workers=responsive_process_workers(workers),
        )
