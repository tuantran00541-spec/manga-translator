from __future__ import annotations

import cv2
import numpy as np

from app.inpaint.lama_inpainter import Inpainter
from app.pipeline_artwork_safe import ArtworkSafeChapterPipeline


class ConcurrentDynamicInpainter(Inpainter):
    """Keep fixed-LaMa safety while allowing the dynamic model to overlap runs."""

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
    """Artwork-safe pipeline with concurrent dynamic-LaMa inference."""

    @property
    def inpainter(self):
        if self._inpainter is None:
            self._inpainter = ConcurrentDynamicInpainter()
        return self._inpainter
