from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from app.detector.sequential_fast_residue_detector import (
    SequentialFastResidueAdaptiveFocusCombinedTextDetector,
)
from app.image_io import read_image, write_image
from app.inpaint.adaptive_fast_inpainter import AdaptiveFastInpainter
from app.manifest_utils import atomic_replace
from app.parameters import MANUAL_MASK_THRESHOLD, PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.region_policy import subtract_regions_from_mask
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
        return super().process_pages(
            chapter_id,
            page_indices,
            workers=effective_workers,
        )

    @staticmethod
    def _restore_preserve_pixels(
        clean_image: np.ndarray,
        original_image: np.ndarray,
        preserve_regions: list[dict] | None,
    ) -> None:
        """Hard-lock user preserve rectangles without growing their geometry."""
        if not preserve_regions:
            return
        h, w = clean_image.shape[:2]
        for region in preserve_regions:
            if not isinstance(region, dict):
                continue
            try:
                x1 = max(0, min(w, int(region["x1"])))
                y1 = max(0, min(h, int(region["y1"])))
                x2 = max(0, min(w, int(region["x2"])))
                y2 = max(0, min(h, int(region["y2"])))
            except (KeyError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            clean_image[y1:y2, x1:x2] = original_image[y1:y2, x1:x2]

    def _do_reinpaint(
        self,
        processed_dir: Path,
        img_path: Path,
        image: np.ndarray,
        boxes: list[dict],
        manual_mask_posix: str | None = None,
        manual_lama_mask_posix: str | None = None,
        *,
        reuse_auto_clean: bool = False,
        apply_manual_mask: bool = True,
        preserve_regions: list[dict] | None = None,
    ) -> str:
        """Use original pixels as LaMa context, then commit only user authority.

        Automatic cleanup and standard repaint keep the existing production path.
        Explicit LaMa repaint is evaluated from the raw/original slice so an
        already-damaged clean result cannot become model context.  The LaMa
        candidate is copied back only where the persisted manual LaMa mask grants
        authority. Preserve rectangles are finally restored from the original
        image exactly as drawn; they are never dilated or feathered.
        """
        if not apply_manual_mask:
            return super()._do_reinpaint(
                processed_dir,
                img_path,
                image,
                boxes,
                manual_mask_posix=manual_mask_posix,
                manual_lama_mask_posix=manual_lama_mask_posix,
                reuse_auto_clean=reuse_auto_clean,
                apply_manual_mask=False,
                preserve_regions=preserve_regions,
            )

        lama_mask_path = (
            Path(manual_lama_mask_posix)
            if manual_lama_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=True)
        )
        lama_mask = self._read_manual_mask(lama_mask_path, image.shape[:2])
        lama_mask = subtract_regions_from_mask(lama_mask, preserve_regions)

        # The parent implementation discovers a default manual-LaMa sidecar when
        # None is supplied. Point it at a guaranteed-nonexistent path so this
        # subclass can run the LaMa pass exactly once from original context.
        suppressed_lama_path = processed_dir / (
            f".original-context-lama-{uuid.uuid4().hex}.png"
        )
        clean_path_posix = super()._do_reinpaint(
            processed_dir,
            img_path,
            image,
            boxes,
            manual_mask_posix=manual_mask_posix,
            manual_lama_mask_posix=suppressed_lama_path.as_posix(),
            reuse_auto_clean=reuse_auto_clean,
            apply_manual_mask=True,
            preserve_regions=preserve_regions,
        )

        needs_rewrite = bool(preserve_regions)
        clean_image = None
        if lama_mask is not None and np.any(lama_mask > MANUAL_MASK_THRESHOLD):
            clean_image = read_image(Path(clean_path_posix))
            lama_candidate = self.inpainter.inpaint_mask(
                image.copy(),
                lama_mask,
                force_lama=True,
            )
            authority = lama_mask > MANUAL_MASK_THRESHOLD
            clean_image[authority] = lama_candidate[authority]
            needs_rewrite = True

        if preserve_regions:
            if clean_image is None:
                clean_image = read_image(Path(clean_path_posix))
            self._restore_preserve_pixels(clean_image, image, preserve_regions)

        if needs_rewrite and clean_image is not None:
            clean_path = Path(clean_path_posix)
            tmp_clean_path = processed_dir / (
                f"{clean_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
            )
            write_image(tmp_clean_path, clean_image)
            atomic_replace(tmp_clean_path, clean_path)

        return clean_path_posix
