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
from app.mask_store import decode_mask_value
from app.parameters import MANUAL_MASK_THRESHOLD, PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.region_policy import subtract_regions_from_mask
from app.runtime_responsiveness import responsive_process_workers


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

    @staticmethod
    def _near_repaint(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        user_bbox: tuple[int, int, int, int],
    ) -> bool:
        ux1, uy1, ux2, uy2 = user_bbox
        pad = REPAINT_DETECTOR_NEAR_PAD
        return not (
            x2 <= ux1 - pad
            or x1 >= ux2 + pad
            or y2 <= uy1 - pad
            or y1 >= uy2 + pad
        )

    @staticmethod
    def _union_local_mask(
        inference_mask: np.ndarray,
        local_mask: np.ndarray | None,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> bool:
        h, w = inference_mask.shape[:2]
        x1 = max(0, min(w, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1 or local_mask is None:
            return False
        expected = (y2 - y1, x2 - x1)
        if local_mask.shape[:2] != expected:
            try:
                local_mask = cv2.resize(
                    local_mask,
                    (expected[1], expected[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            except Exception:
                return False
        local_mask = (
            (local_mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
        )
        if not np.any(local_mask):
            return False
        target = inference_mask[y1:y2, x1:x2]
        before = int(np.count_nonzero(target > MANUAL_MASK_THRESHOLD))
        np.maximum(target, local_mask, out=target)
        after = int(np.count_nonzero(target > MANUAL_MASK_THRESHOLD))
        return after > before

    def _detector_assisted_lama_mask(
        self,
        image: np.ndarray,
        user_mask: np.ndarray,
        preserve_regions: list[dict] | None,
        boxes: list[dict] | None = None,
    ) -> np.ndarray:
        """Expand only LaMa inference mask using text-segmenter evidence.

        First reuse persisted detector masks from the processed page. This keeps
        repaint cheap and preserves the detector geometry that already succeeded
        at page-processing scale. If no persisted segmenter mask adds context,
        fall back to a full-slice text-segmenter pass; tight crops are avoided
        because the detector can become scale/context sensitive on them.
        """
        base = (user_mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
        base = subtract_regions_from_mask(base, preserve_regions)
        if base is None or not np.any(base):
            self._last_repaint_detector_stats = {
                "detector_source": "none",
                "persisted_boxes_total": 0,
                "persisted_boxes_used": 0,
                "fallback_boxes_total": 0,
                "fallback_boxes_used": 0,
                "user_mask_pixels": 0,
                "inference_mask_pixels": 0,
                "detector_added_pixels": 0,
            }
            return base

        ys, xs = np.nonzero(base > MANUAL_MASK_THRESHOLD)
        user_bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        inference_mask = base.copy()
        user_pixels = int(np.count_nonzero(base > MANUAL_MASK_THRESHOLD))

        persisted_total = 0
        persisted_used = 0
        for box in boxes or []:
            if not isinstance(box, dict) or box.get("removed"):
                continue
            segmenter_evidence = bool(
                box.get("source_role") == "text_segmenter"
                or box.get("mask_source") == "text_segmenter"
                or str(box.get("source_model") or "") == "text_segmenter.onnx"
            )
            if not segmenter_evidence or not bool(box.get("safe_to_inpaint")):
                continue
            try:
                x1, y1, x2, y2 = (
                    int(box["x1"]),
                    int(box["y1"]),
                    int(box["x2"]),
                    int(box["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not self._near_repaint(x1, y1, x2, y2, user_bbox):
                continue
            persisted_total += 1
            local_mask = decode_mask_value(box.get("mask"))
            if self._union_local_mask(
                inference_mask, local_mask, x1, y1, x2, y2
            ):
                persisted_used += 1

        inference_mask = subtract_regions_from_mask(inference_mask, preserve_regions)
        after_persisted = int(
            np.count_nonzero(inference_mask > MANUAL_MASK_THRESHOLD)
        )

        detector_source = "persisted_text_segmenter"
        fallback_total = 0
        fallback_used = 0
        error = None
        if after_persisted <= user_pixels:
            detector_source = "text_segmenter_full"
            detected = []
            try:
                detector = self.detector
                focused = getattr(detector, "text_detector", None)
                if focused is not None and callable(getattr(focused, "detect", None)):
                    detected = focused.detect(image)
                else:
                    detected = detector.detect(image)
            except Exception as exc:
                error = str(exc)
                logger.warning(
                    "Detector-assisted repaint fallback kept user mask: {}", exc
                )

            fallback_total = int(len(detected))
            for box in detected:
                if not bool(getattr(box, "verified_mask", False)):
                    continue
                if not bool(getattr(box, "safe_to_inpaint", False)):
                    continue
                x1, y1, x2, y2 = (
                    int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                )
                if not self._near_repaint(x1, y1, x2, y2, user_bbox):
                    continue
                if self._union_local_mask(
                    inference_mask,
                    getattr(box, "mask", None),
                    x1,
                    y1,
                    x2,
                    y2,
                ):
                    fallback_used += 1
            inference_mask = subtract_regions_from_mask(
                inference_mask, preserve_regions
            )

        inference_pixels = int(
            np.count_nonzero(inference_mask > MANUAL_MASK_THRESHOLD)
        )
        stats = {
            "detector_source": detector_source,
            "persisted_boxes_total": int(persisted_total),
            "persisted_boxes_used": int(persisted_used),
            "fallback_boxes_total": int(fallback_total),
            "fallback_boxes_used": int(fallback_used),
            "user_mask_pixels": user_pixels,
            "inference_mask_pixels": inference_pixels,
            "detector_added_pixels": max(0, inference_pixels - user_pixels),
            "user_bbox": [int(v) for v in user_bbox],
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
        """Use detector-assisted original context, then commit only user authority."""
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

        # Suppress the parent's manual-LaMa pass so it runs exactly once below
        # with RAW context and the detector-assisted inference mask.
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
                boxes,
            )
            lama_candidate = self.inpainter.inpaint_mask(
                image.copy(),
                inference_mask,
                force_lama=True,
            )
            # User mask is the hard output authority. Detector-only pixels affect
            # model context/inference but never get committed to the clean page.
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
