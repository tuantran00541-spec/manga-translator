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
from app.logging_config import logger
from app.manifest_utils import atomic_replace
from app.parameters import MANUAL_MASK_THRESHOLD, PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.region_policy import subtract_regions_from_mask
from app.runtime_responsiveness import responsive_process_workers


REPAINT_DETECTOR_CONTEXT_PAD = 96
REPAINT_DETECTOR_NEAR_PAD = 24


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

    def _detector_assisted_lama_mask(
        self,
        image: np.ndarray,
        user_mask: np.ndarray,
        preserve_regions: list[dict] | None,
    ) -> np.ndarray:
        """Expand only LaMa inference mask with nearby text-segmenter masks.

        The user's mask remains the only output/composite authority. The focused
        text segmenter is run on a small RAW crop so adjacent source glyphs are
        hidden from LaMa instead of becoming reconstruction context.
        """
        base = (user_mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
        base = subtract_regions_from_mask(base, preserve_regions)
        if base is None or not np.any(base):
            self._last_repaint_detector_stats = {
                "detector_source": "none",
                "detector_boxes_total": 0,
                "detector_boxes_used": 0,
                "user_mask_pixels": 0,
                "inference_mask_pixels": 0,
                "detector_added_pixels": 0,
            }
            return base

        ys, xs = np.nonzero(base > MANUAL_MASK_THRESHOLD)
        h, w = image.shape[:2]
        ux1 = int(xs.min())
        uy1 = int(ys.min())
        ux2 = int(xs.max()) + 1
        uy2 = int(ys.max()) + 1

        cx1 = max(0, ux1 - REPAINT_DETECTOR_CONTEXT_PAD)
        cy1 = max(0, uy1 - REPAINT_DETECTOR_CONTEXT_PAD)
        cx2 = min(w, ux2 + REPAINT_DETECTOR_CONTEXT_PAD)
        cy2 = min(h, uy2 + REPAINT_DETECTOR_CONTEXT_PAD)
        crop = image[cy1:cy2, cx1:cx2]

        near_x1 = max(0, ux1 - cx1 - REPAINT_DETECTOR_NEAR_PAD)
        near_y1 = max(0, uy1 - cy1 - REPAINT_DETECTOR_NEAR_PAD)
        near_x2 = min(cx2 - cx1, ux2 - cx1 + REPAINT_DETECTOR_NEAR_PAD)
        near_y2 = min(cy2 - cy1, uy2 - cy1 + REPAINT_DETECTOR_NEAR_PAD)

        inference_mask = base.copy()
        detected = []
        error = None
        detector_source = "combined"
        try:
            detector = self.detector
            focused = getattr(detector, "text_detector", None)
            if focused is not None and callable(getattr(focused, "detect", None)):
                detector_source = "text_segmenter"
                detected = focused.detect(crop)
            else:
                detected = detector.detect(crop)
        except Exception as exc:
            error = str(exc)
            logger.warning("Detector-assisted repaint fallback to user mask: {}", exc)

        used = 0
        for box in detected:
            if not bool(getattr(box, "verified_mask", False)):
                continue
            if not bool(getattr(box, "safe_to_inpaint", False)):
                continue

            bx1 = max(0, min(crop.shape[1], int(box.x1)))
            by1 = max(0, min(crop.shape[0], int(box.y1)))
            bx2 = max(0, min(crop.shape[1], int(box.x2)))
            by2 = max(0, min(crop.shape[0], int(box.y2)))
            if bx2 <= bx1 or by2 <= by1:
                continue
            if (
                bx2 <= near_x1
                or bx1 >= near_x2
                or by2 <= near_y1
                or by1 >= near_y2
            ):
                continue

            local = getattr(box, "mask", None)
            expected = (by2 - by1, bx2 - bx1)
            if local is None:
                continue
            if local.shape[:2] != expected:
                try:
                    local = cv2.resize(
                        local,
                        (expected[1], expected[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except Exception:
                    continue
            local = (local > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
            if not np.any(local):
                continue

            gy1, gy2 = cy1 + by1, cy1 + by2
            gx1, gx2 = cx1 + bx1, cx1 + bx2
            target = inference_mask[gy1:gy2, gx1:gx2]
            np.maximum(target, local, out=target)
            used += 1

        inference_mask = subtract_regions_from_mask(inference_mask, preserve_regions)
        user_pixels = int(np.count_nonzero(base > MANUAL_MASK_THRESHOLD))
        inference_pixels = int(
            np.count_nonzero(inference_mask > MANUAL_MASK_THRESHOLD)
        )
        stats = {
            "detector_source": detector_source,
            "detector_boxes_total": int(len(detected)),
            "detector_boxes_used": int(used),
            "user_mask_pixels": user_pixels,
            "inference_mask_pixels": inference_pixels,
            "detector_added_pixels": max(0, inference_pixels - user_pixels),
            "crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
        }
        if error is not None:
            stats["detector_error"] = error
        self._last_repaint_detector_stats = stats
        logger.info("Detector-assisted repaint stats: {}", stats)
        return inference_mask

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
        """Use detector-assisted original context, then commit only user authority.

        Automatic cleanup and standard repaint keep the existing production path.
        Explicit LaMa repaint uses the raw/original slice as model context. The
        detector may expand only the LaMa inference mask around the user's region
        to hide nearby source text. The candidate is copied back only where the
        persisted manual LaMa mask grants user authority. Preserve rectangles are
        finally restored exactly as drawn and are never grown or feathered.
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
            inference_mask = self._detector_assisted_lama_mask(
                image,
                lama_mask,
                preserve_regions,
            )
            lama_candidate = self.inpainter.inpaint_mask(
                image.copy(),
                inference_mask,
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
