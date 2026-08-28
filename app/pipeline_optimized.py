from __future__ import annotations

import cv2
import numpy as np

from app.inpaint.lama_inpainter import Inpainter
from app.pipeline_artwork_safe import ArtworkSafeChapterPipeline


class ConcurrentDynamicInpainter(Inpainter):
    """Keep fixed-LaMa safety while allowing the dynamic model to overlap runs.

    The preferred dynamic model is created without the global ORT serialization
    wrapper.  The base inpainter still takes a per-instance lock around every
    ``session.run`` though, which accidentally removes that concurrency.  ORT
    sessions support concurrent ``run`` calls, so only the fixed compatibility
    path needs the lock because it may recycle the session between calls.
    """

    def _run_lama(self, canvas: np.ndarray, mask_canvas: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_blob = np.ascontiguousarray(
            (crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )
        mask_blob = np.ascontiguousarray(
            (mask_canvas > 127).astype(np.float32)[None, None]
        )
        feed = {self.image_input: img_blob, self.mask_input: mask_blob}

        if self.dynamic_lama:
            output = self.session.run(None, feed)[0]
        else:
            with self._session_lock:
                self._recycle_fixed_session_if_needed()
                output = self.session.run(None, feed)[0]
                self._session_run_count += 1

        painted_rgb = output[0].transpose(1, 2, 0)
        if painted_rgb.max() <= 1.0:
            painted_rgb = painted_rgb * 255.0
        painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)


class OptimizedArtworkSafeChapterPipeline(ArtworkSafeChapterPipeline):
    """Runtime pipeline tuned for bounded CPU throughput on desktop machines."""

    @property
    def inpainter(self):
        if self._inpainter is None:
            self._inpainter = ConcurrentDynamicInpainter()
        return self._inpainter

    def process_pages(
        self,
        chapter_id: str,
        page_indices: list[int],
        workers: int = 2,
    ) -> dict:
        # Avoid scheduling the same expensive detector/inpaint job twice when a
        # caller accidentally sends duplicate indices.  Keep request order so
        # manifest workflow/page ordering remains deterministic.
        unique_indices = list(dict.fromkeys(page_indices))

        # Two heavy page workers are enough to overlap detector and dynamic-LaMa
        # work while keeping ORT thread pools bounded.  Download/slicing retain
        # their separate higher worker limit in the base pipeline.
        bounded_workers = max(1, min(int(workers or 2), 2))
        return super().process_pages(
            chapter_id,
            unique_indices,
            workers=bounded_workers,
        )
