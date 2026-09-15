from __future__ import annotations

import uuid
from pathlib import Path

import cv2
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

    def _exact_manual_repaint_candidate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        force_lama: bool,
    ) -> np.ndarray:
        """Repaint with the exact user mask: no dilation and no feather outside it.

        The crop may include surrounding image context for the model, but the mask
        passed to the inpaint operation is exactly the persisted manual mask.
        Each connected component is inferred from the same source image so one
        manual region cannot become context for another.
        """
        binary = (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
        if not np.any(binary):
            return image.copy()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        result = image.copy()
        h, w = image.shape[:2]
        used_components = 0

        for label in range(1, num_labels):
            x, y, bbox_w, bbox_h, area = (int(v) for v in stats[label])
            if area <= 0 or bbox_w <= 0 or bbox_h <= 0:
                continue

            # x2/y2 are exclusive here. The crop grows only to provide source
            # context; the local model mask itself remains exact.
            crop_box = self.inpainter._compute_manual_crop_region(
                x,
                y,
                x + bbox_w,
                y + bbox_h,
                w,
                h,
            )
            cx1, cy1, cx2, cy2 = crop_box
            local_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
            component = labels[cy1:cy2, cx1:cx2] == label
            local_mask[component] = 255

            component_candidate = self.inpainter._smart_paint_region(
                image.copy(),
                local_mask,
                crop_box,
                feather=False,
                force_lama=force_lama,
            )
            authority = labels == label
            result[authority] = component_candidate[authority]
            used_components += 1

        if force_lama:
            user_pixels = int(np.count_nonzero(binary > MANUAL_MASK_THRESHOLD))
            self._last_repaint_detector_stats = {
                "mask_mode": "exact_user_mask",
                "detector_source": "not_used",
                "mask_components": int(used_components),
                "user_mask_pixels": user_pixels,
                "inference_mask_pixels": user_pixels,
                "detector_added_pixels": 0,
                "dilation_pixels": 0,
                "feather_outside_mask": False,
            }

        return result

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
        """Replay manual repaint with exact masks while keeping RAW LaMa context."""
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

        standard_mask_path = (
            Path(manual_mask_posix)
            if manual_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=False)
        )
        lama_mask_path = (
            Path(manual_lama_mask_posix)
            if manual_lama_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=True)
        )

        standard_mask = self._read_manual_mask(
            standard_mask_path,
            image.shape[:2],
        )
        lama_mask = self._read_manual_mask(
            lama_mask_path,
            image.shape[:2],
        )
        standard_mask = subtract_regions_from_mask(
            standard_mask,
            preserve_regions,
        )
        lama_mask = subtract_regions_from_mask(
            lama_mask,
            preserve_regions,
        )

        # Suppress both parent manual passes. They use Inpainter.inpaint_mask(),
        # whose production default intentionally dilates manual masks. Repaint
        # policy here replays both modes below with exact user geometry instead.
        suppressed_standard_path = processed_dir / (
            f".exact-standard-{uuid.uuid4().hex}.png"
        )
        suppressed_lama_path = processed_dir / (
            f".exact-lama-{uuid.uuid4().hex}.png"
        )
        clean_path_posix = super()._do_reinpaint(
            processed_dir,
            img_path,
            image,
            boxes,
            manual_mask_posix=suppressed_standard_path.as_posix(),
            manual_lama_mask_posix=suppressed_lama_path.as_posix(),
            reuse_auto_clean=reuse_auto_clean,
            apply_manual_mask=True,
            preserve_regions=preserve_regions,
        )

        needs_rewrite = bool(preserve_regions)
        clean_image = None

        if (
            standard_mask is not None
            and np.any(standard_mask > MANUAL_MASK_THRESHOLD)
        ):
            clean_image = read_image(Path(clean_path_posix))
            standard_candidate = self._exact_manual_repaint_candidate(
                clean_image.copy(),
                standard_mask,
                force_lama=False,
            )
            authority = standard_mask > MANUAL_MASK_THRESHOLD
            clean_image[authority] = standard_candidate[authority]
            needs_rewrite = True

        if lama_mask is not None and np.any(lama_mask > MANUAL_MASK_THRESHOLD):
            if clean_image is None:
                clean_image = read_image(Path(clean_path_posix))
            # Explicit LaMa repaint keeps RAW/original pixels as source context.
            lama_candidate = self._exact_manual_repaint_candidate(
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
