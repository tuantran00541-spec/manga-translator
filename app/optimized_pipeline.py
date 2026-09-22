from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.detector.bubble_detector import BubbleBox
from app.detector.mask_builder import build_mask
from app.one_shot_cleanup import OneShotProductionDetector
from app.image_io import read_image, write_image
from app.inpaint.lama_inpainter import Inpainter
from app.manifest_utils import atomic_replace
from app.mask_store import decode_mask_value
from app.parameters import MANUAL_MASK_THRESHOLD, PIPELINE_DEFAULT_WORKERS
from app.pipeline import ChapterPipeline
from app.region_policy import geometry_center_in_regions, subtract_regions_from_mask


class OptimizedChapterPipeline(ChapterPipeline):
    """Chapter pipeline using validated CPU detector and cleanup candidates."""

    @property
    def detector(self):
        if self._detector is None:
            with self._detector_init_lock:
                if self._detector is None:
                    self._detector = OneShotProductionDetector()
        return self._detector

    @property
    def inpainter(self):
        if self._inpainter is None:
            with self._inpainter_init_lock:
                if self._inpainter is None:
                    self._inpainter = Inpainter()
        return self._inpainter

    def process_pages(
        self,
        chapter_id: str,
        page_indices: list[int],
        workers: int = PIPELINE_DEFAULT_WORKERS,
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict:
        """Process pages with exactly the user-requested page concurrency.

        The UI/API contract is 1..8 workers. Runtime tuning belongs inside the
        detector/inpainter sessions; it must not silently rewrite page
        concurrency behind the user's back.
        """
        requested_workers = max(1, min(8, int(workers or PIPELINE_DEFAULT_WORKERS)))
        inpainter = self.inpainter
        prepare = getattr(inpainter, "prepare_for_page_workers", None)
        if callable(prepare):
            prepare(requested_workers)
        return super().process_pages(
            chapter_id,
            page_indices,
            workers=requested_workers,
            progress_callback=progress_callback,
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

    @staticmethod
    def _residue_repair_effective_boxes(records: list[dict] | None) -> list[BubbleBox]:
        """Rebuild destructive detector authorities persisted by _process_page."""
        boxes: list[BubbleBox] = []
        for record in records or []:
            if (
                not isinstance(record, dict)
                or record.get("removed")
                or record.get("deferred_reason")
            ):
                continue
            geometry_overridden = bool(record.get("geometry_overridden"))
            if not (record.get("safe_to_inpaint") or geometry_overridden):
                continue
            try:
                x1, y1 = int(record["x1"]), int(record["y1"])
                x2, y2 = int(record["x2"]), int(record["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue
            mask = decode_mask_value(record.get("mask"))
            if mask is not None and mask.shape != (box_h, box_w):
                mask = cv2.resize(
                    mask,
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            box = BubbleBox(
                x1,
                y1,
                x2,
                y2,
                float(record.get("confidence", 1.0)),
                mask,
                source_model=str(record.get("source_model") or "unknown"),
                class_id=int(record.get("class_id") or 0),
                class_name=str(record.get("class_name") or "unknown"),
                semantic_type=str(record.get("semantic_type") or "unknown"),
                mask_source=str(record.get("mask_source") or "none"),
                safe_to_inpaint=True,
                ocr_eligible=bool(record.get("ocr_eligible")),
                needs_review=bool(record.get("needs_review")),
                source_role=str(record.get("source_role") or "unknown"),
                deferred_reason=None,
            )
            if geometry_overridden:
                box.allow_rectangle_fallback = True
            boxes.append(box)
        return boxes

    @classmethod
    def _residue_verification_boxes(cls, records: list[dict] | None) -> list[BubbleBox]:
        """Return destructive authorities plus review-only verification sources."""
        boxes = list(cls._residue_repair_effective_boxes(records))
        boxes.extend(cls._review_only_residue_sources(records))
        return boxes

    def _repair_post_inpaint_result(
        self,
        img_path: Path,
        result: dict,
        preserve_regions: list[dict] | None,
    ) -> dict:
        """Repair text that a focused post-inpaint verification proves remains.

        Existing automatic authority stays valid. A freshly verified
        text-segmenter residue mask may also add stroke-shaped authority outside
        the original mask; raw rectangles and review-only geometry never gain
        destructive permission. LaMa can use context internally, but final writes
        remain clipped to verified masks and preserve rectangles.
        """
        initial_regions = list(result.get("residue_regions") or [])
        actual_hits = [
            region
            for region in initial_regions
            if isinstance(region, dict)
            and region.get("deferred_reason") == "post_inpaint_text_residue"
        ]
        metrics = result.setdefault("processing_metrics", {})
        repair_metrics = {
            "attempted": 0,
            "initial_regions": len(initial_regions),
            "initial_text_hits": len(actual_hits),
            "repair_mask_pixels": 0,
            "verified_extension_pixels": 0,
            "outside_authority_changed_channel_values": 0,
            "final_regions": len(initial_regions),
            "skipped_manual_state": 0,
        }
        metrics["residue_repair"] = repair_metrics
        if not actual_hits:
            return result
        if result.get("manual_mask") or result.get("manual_lama_mask"):
            repair_metrics["skipped_manual_state"] = 1
            return result

        tmp_clean_value = result.get("tmp_clean")
        if not tmp_clean_value:
            return result
        tmp_clean_path = Path(tmp_clean_value)
        if not tmp_clean_path.exists():
            return result

        authorized_boxes = self._residue_repair_effective_boxes(result.get("boxes"))

        started_at = time.perf_counter()
        original = read_image(img_path)
        clean_before = read_image(tmp_clean_path)
        h, w = clean_before.shape[:2]
        full_authority = np.zeros((h, w), dtype=np.uint8)
        if authorized_boxes:
            full_authority = build_mask(
                clean_before.shape[:2],
                authorized_boxes,
                original,
            )
            full_authority = subtract_regions_from_mask(
                full_authority,
                preserve_regions,
            )
            if full_authority is None:
                full_authority = np.zeros((h, w), dtype=np.uint8)

        verified_scope = np.zeros((h, w), dtype=np.uint8)
        legacy_scope = np.zeros((h, w), dtype=np.uint8)
        fallback_pad = 6
        for region in actual_hits:
            try:
                raw_x1, raw_y1 = int(region["x1"]), int(region["y1"])
                raw_x2, raw_y2 = int(region["x2"]), int(region["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            x1, y1 = max(0, raw_x1), max(0, raw_y1)
            x2, y2 = min(w, raw_x2), min(h, raw_y2)
            if x2 <= x1 or y2 <= y1:
                continue

            residue_mask = decode_mask_value(region.get("mask"))
            verified_segmenter = (
                residue_mask is not None
                and str(region.get("source_role") or "") == "text_segmenter"
            )
            if verified_segmenter:
                target_w, target_h = x2 - x1, y2 - y1
                if residue_mask.shape[:2] != (target_h, target_w):
                    residue_mask = cv2.resize(
                        residue_mask,
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                local_scope = verified_scope[y1:y2, x1:x2]
                local_scope[residue_mask > MANUAL_MASK_THRESHOLD] = 255
                continue

            # Backward-compatible geometry can only narrow an already-authorized
            # mask. It never creates new destructive pixels.
            px1 = max(0, raw_x1 - fallback_pad)
            py1 = max(0, raw_y1 - fallback_pad)
            px2 = min(w, raw_x2 + fallback_pad)
            py2 = min(h, raw_y2 + fallback_pad)
            if px2 > px1 and py2 > py1:
                legacy_scope[py1:py2, px1:px2] = 255

        legacy_repair = cv2.bitwise_and(full_authority, legacy_scope)
        repair_mask = cv2.bitwise_or(verified_scope, legacy_repair)
        repair_mask = subtract_regions_from_mask(repair_mask, preserve_regions)
        if repair_mask is None:
            repair_mask = np.zeros((h, w), dtype=np.uint8)
        extension = (
            (verified_scope > MANUAL_MASK_THRESHOLD)
            & (full_authority <= MANUAL_MASK_THRESHOLD)
        )
        repair_metrics["verified_extension_pixels"] = int(
            np.count_nonzero(extension)
        )
        repair_pixels = int(
            np.count_nonzero(repair_mask > MANUAL_MASK_THRESHOLD)
        )
        repair_metrics["repair_mask_pixels"] = repair_pixels
        if repair_pixels <= 0:
            return result

        candidate = self.inpainter.inpaint_mask(
            clean_before.copy(),
            repair_mask,
            force_lama=True,
        )
        model_metrics = self.inpainter.last_metrics()
        authority = repair_mask > MANUAL_MASK_THRESHOLD
        repaired = clean_before.copy()
        repaired[authority] = candidate[authority]
        outside = ~authority
        repair_metrics["outside_authority_changed_channel_values"] = int(
            np.count_nonzero(repaired[outside] != clean_before[outside])
        )
        self._restore_preserve_pixels(repaired, original, preserve_regions)

        write_image(tmp_clean_path, repaired)
        tmp_auto_value = result.get("tmp_auto_clean")
        if tmp_auto_value:
            tmp_auto_path = Path(tmp_auto_value)
            if tmp_auto_path.exists():
                write_image(tmp_auto_path, repaired)

        verification_boxes = self._residue_verification_boxes(result.get("boxes"))
        final_boxes = self.detector.verify_post_inpaint_residue(
            repaired,
            verification_boxes,
        )
        final_boxes = [
            box
            for box in final_boxes
            if not geometry_center_in_regions(
                {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                },
                preserve_regions,
            )
        ]
        decision_fields = (
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "source_model",
            "source_role",
            "class_name",
            "semantic_type",
            "deferred_reason",
        )
        final_regions = [
            {key: getattr(box, key) for key in decision_fields}
            for box in final_boxes
        ]
        result["residue_regions"] = final_regions
        issues = [
            issue
            for issue in list(result.get("detection_issues") or [])
            if issue != "post_inpaint_text_residue"
        ]
        if final_regions:
            issues.append("post_inpaint_text_residue")
        result["detection_issues"] = issues
        result["detection_state"] = "needs_review" if issues else "verified"
        result["cleanup_verified"] = not bool(issues)
        result["needs_review"] = bool(issues)

        detector_metrics = metrics.setdefault("detector", {})
        detector_metrics["post_inpaint_residue_initial"] = int(
            detector_metrics.get(
                "post_inpaint_residue",
                len(initial_regions),
            )
            or 0
        )
        detector_metrics["post_inpaint_residue"] = len(final_regions)
        repair_metrics.update(
            {
                "attempted": 1,
                "final_regions": len(final_regions),
                "lama_model_runs": int(
                    model_metrics.get("lama_model_runs", 0) or 0
                ),
                "lama_model_ms": int(
                    model_metrics.get("lama_model_ms", 0) or 0
                ),
                "final_verify": self.detector.last_residue_metrics(),
            }
        )
        repair_ms = (time.perf_counter() - started_at) * 1000.0
        timing = metrics.setdefault("timing_ms", {})
        timing["residue_repair"] = round(repair_ms, 3)
        timing["total"] = round(
            float(timing.get("total", 0.0)) + repair_ms,
            3,
        )
        return result

    def _process_page(
        self,
        img_path: Path,
        processed_dir: Path,
        preserve_regions: list[dict] | None = None,
        existing_boxes: list[dict] | None = None,
        stitch_core: dict | None = None,
        supplemental_detections: list[BubbleBox] | None = None,
        seam_context_unavailable: bool = False,
        *,
        parallel_detectors: bool = False,
    ) -> dict:
        return super()._process_page(
            img_path,
            processed_dir,
            preserve_regions=preserve_regions,
            existing_boxes=existing_boxes,
            stitch_core=stitch_core,
            supplemental_detections=supplemental_detections,
            seam_context_unavailable=seam_context_unavailable,
            parallel_detectors=parallel_detectors,
        )

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
