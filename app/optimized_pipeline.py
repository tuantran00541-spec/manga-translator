from __future__ import annotations

from app.detector.fast_residue_detector import (
    FastResidueAdaptiveFocusCombinedTextDetector,
)
from app.inpaint.fast_lama_inpainter import FastInpainter
from app.parameters import PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.runtime_responsiveness import responsive_process_workers


class OptimizedChapterPipeline(ChapterPipeline):
    """Chapter pipeline using adaptive detector and CPU fast cleanup candidates."""

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = FastResidueAdaptiveFocusCombinedTextDetector()
        return self._detector

    @property
    def inpainter(self):
        if self._inpainter is None:
            with self._inpainter_init_lock:
                if self._inpainter is None:
                    self._inpainter = FastInpainter()
        return self._inpainter

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
