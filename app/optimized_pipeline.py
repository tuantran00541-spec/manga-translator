from __future__ import annotations

from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
from app.parameters import INPAINT_PRELOAD_ENABLED, PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.runtime_responsiveness import responsive_process_workers


class OptimizedChapterPipeline(ChapterPipeline):
    """Chapter pipeline using validated CPU detector and cleanup candidates."""

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = (
                        SequentialFastResidueAdaptiveFocusCombinedTextDetector()
                    )
        return self._detector

    @property
    def inpainter(self):
        if self._inpainter is None:
            with self._inpainter_init_lock:
                if self._inpainter is None:
                    self._inpainter = AdaptiveFastInpainter()
        return self._inpainter

    def process_pages(
        self,
        chapter_id: str,
        page_indices: list[int],
        workers: int = PIPELINE_DEFAULT_WORKERS,
    ) -> dict:
        """Process pages without oversubscribing LaMa or starving the browser."""
        effective_workers = responsive_process_workers(workers)
        inpainter = self.inpainter
        prepare = getattr(inpainter, "prepare_for_page_workers", None)
        if callable(prepare):
            prepare(effective_workers)
        # A session-affecting runtime profile must be selected before loading
        # LaMa.  Loading lazily from the first page worker otherwise makes that
        # worker pay the cold-start cost and can race with its sibling.  Keep
        # the existing opt-out for installations that deliberately disable
        # preload to conserve memory.
        if INPAINT_PRELOAD_ENABLED:
            preload = getattr(inpainter, "preload", None)
            if callable(preload):
                preload()
        return super().process_pages(
            chapter_id,
            page_indices,
            workers=effective_workers,
        )
