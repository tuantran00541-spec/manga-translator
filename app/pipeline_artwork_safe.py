from __future__ import annotations

import copy
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from app.config import PROCESSED_DIR
from app.detector.bubble_detector import BubbleBox
from app.downloader.asura import is_asura_chapter_page
from app.inpaint.mask_geometry import geometry_dict, remap_local_mask_page_space
from app.logging_config import logger
from app.manifest_utils import (
    assign_stable_detector_box_ids,
    bump_page_revision,
    capture_processing_state,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    is_processing_state_current,
    load_manifest_raw,
    save_manifest_raw,
)
from app.mask_store import decode_mask_value, externalize_page_masks
from app.pipeline import ChapterPipeline, _encode_mask, read_image, write_image


class ArtworkSafeChapterPipeline(ChapterPipeline):
    """Chapter pipeline with safe detector masks and durable page commits."""

    @staticmethod
    def _apply_box_geometry(box: dict, new_geometry: dict[str, int]) -> None:
        """Geometry-safe edit that understands legacy and sidecar detector masks."""
        source_geometry = geometry_dict(box)
        source_mask = decode_mask_value(box.get("mask"))
        detector_origin = box.get("origin") == "detector" and not box.get("manual")

        if detector_origin:
            if not box.get("geometry_overridden"):
                box["detector_anchor"] = copy.deepcopy(source_geometry)
            box["geometry_overridden"] = True

        remapped = (
            remap_local_mask_page_space(source_mask, source_geometry, new_geometry)
            if source_mask is not None
            else None
        )

        box.update(new_geometry)
        if remapped is None or not np.any(remapped > 127):
            box["mask"] = None
        else:
            box["mask"] = _encode_mask(remapped)

    def _do_reinpaint(
        self,
        processed_dir: Path,
        img_path: Path,
        image: np.ndarray,
        boxes: list[dict],
        manual_mask_posix: str | None = None,
        *,
        reuse_auto_clean: bool = False,
        apply_manual_mask: bool = True,
    ) -> str:
        """Repaint using masks stored inline or in managed PNG sidecars."""
        boxes_objects = []
        for box in boxes:
            if box.get("removed"):
                continue
            box_h = int(box["y2"]) - int(box["y1"])
            box_w = int(box["x2"]) - int(box["x1"])
            mask_arr = decode_mask_value(box.get("mask"))
            if mask_arr is not None and mask_arr.shape != (box_h, box_w):
                try:
                    mask_arr = cv2.resize(
                        mask_arr,
                        (box_w, box_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except Exception:
                    mask_arr = None
            boxes_objects.append(
                BubbleBox(
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    box.get("confidence", 1.0),
                    mask_arr,
                )
            )

        clean_image = None
        if reuse_auto_clean:
            clean_image = self._load_auto_clean_cache(
                processed_dir, img_path, image.shape[:2]
            )

        if clean_image is None:
            clean_image = self.inpainter.inpaint(image, boxes_objects)
            self._write_auto_clean_cache(processed_dir, img_path, clean_image)
        else:
            clean_image = clean_image.copy()

        manual_mask_path = (
            Path(manual_mask_posix)
            if manual_mask_posix
            else processed_dir / f"manual_mask_{img_path.name}"
        )
        if apply_manual_mask and manual_mask_path.exists():
            raw = np.fromfile(str(manual_mask_path), dtype=np.uint8)
            manual_mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if manual_mask is not None and np.any(manual_mask > 10):
                h, w = clean_image.shape[:2]
                if manual_mask.shape[:2] != (h, w):
                    try:
                        manual_mask = cv2.resize(
                            manual_mask,
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    except Exception:
                        manual_mask = None
                if manual_mask is not None and np.any(manual_mask > 10):
                    manual_mask = (manual_mask > 10).astype(np.uint8) * 255
                    clean_image = self.inpainter.inpaint_mask(
                        clean_image, manual_mask
                    )

        clean_path = processed_dir / f"clean_{img_path.name}"
        tmp_clean_path = (
            processed_dir
            / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        )
        write_image(tmp_clean_path, clean_image)
        os.replace(tmp_clean_path, clean_path)
        return clean_path.as_posix()

    def _commit_processed_page(
        self,
        chapter_id: str,
        processed_dir: Path,
        page_index: int,
        page_data: dict,
        snapshot: dict | None,
    ) -> bool:
        """Atomically publish one completed page as soon as its worker finishes."""
        tmp_clean_value = page_data.get("tmp_clean")
        tmp_clean_path = Path(tmp_clean_value) if tmp_clean_value else None
        tmp_auto_clean_value = page_data.get("tmp_auto_clean")
        tmp_auto_clean_path = Path(tmp_auto_clean_value) if tmp_auto_clean_value else None

        try:
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                pages = manifest.get("pages", [])
                if not (
                    0 <= page_index < len(pages)
                    and is_processing_state_current(
                        manifest, page_index, snapshot, processed_dir
                    )
                ):
                    logger.warning(
                        "Chapter %s page %s: page state changed during processing, discarding stale output",
                        chapter_id,
                        page_index,
                    )
                    return False

                target_page = pages[page_index]
                orig_path = Path(target_page["original"])

                if tmp_clean_path and tmp_clean_path.exists():
                    final_clean_path = processed_dir / f"clean_{orig_path.name}"
                    os.replace(tmp_clean_path, final_clean_path)
                    target_page["clean"] = final_clean_path.as_posix()

                if tmp_auto_clean_path and tmp_auto_clean_path.exists():
                    os.replace(
                        tmp_auto_clean_path,
                        self._auto_clean_path(processed_dir, orig_path),
                    )

                existing_boxes = target_page.get("boxes", [])
                detected_boxes = assign_stable_detector_box_ids(
                    page_data["boxes"], existing_boxes
                )
                manual_boxes = [b for b in existing_boxes if b.get("manual")]
                committed_boxes = detected_boxes + manual_boxes
                externalized = externalize_page_masks(
                    processed_dir, page_index, committed_boxes
                )
                if externalized:
                    logger.debug(
                        "Chapter {} page {} externalized {} detector mask(s)",
                        chapter_id,
                        page_index,
                        externalized,
                    )
                target_page["boxes"] = committed_boxes
                target_page["detection_state"] = page_data.get(
                    "detection_state", "verified"
                )
                target_page["detection_issues"] = list(
                    page_data.get("detection_issues") or []
                )
                target_page["unverified_regions"] = list(
                    page_data.get("unverified_regions") or []
                )
                target_page["needs_review"] = bool(page_data.get("needs_review"))
                bump_page_revision(target_page, "process_revision")
                bump_page_revision(target_page, "clean_revision")

                if "manual_mask" in page_data:
                    target_page["manual_mask"] = page_data["manual_mask"]

                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
                return True
        finally:
            for tmp_path in (tmp_clean_path, tmp_auto_clean_path):
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    def process_pages(
        self,
        chapter_id: str,
        page_indices: list[int],
        workers: int = 2,
    ) -> dict:
        """Process pages with shared seams and durable per-page progress."""
        processed_dir = PROCESSED_DIR / chapter_id
        unique_indices = list(dict.fromkeys(page_indices))

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            manifest_pages = manifest.get("pages", [])
            source_page_ids = [
                int(page.get("source_page", -1))
                for page in manifest_pages
                if isinstance(page, dict)
            ]
            last_source_page = max(source_page_ids, default=-1)
            protect_asura_tail_credits = is_asura_chapter_page(
                str(manifest.get("source_url") or "")
            )
            work_items = []
            for idx in unique_indices:
                if 0 <= idx < len(manifest_pages):
                    page = manifest_pages[idx]
                    if page.get("skipped", False):
                        continue
                    state_snapshot = capture_processing_state(
                        manifest, idx, processed_dir
                    )
                    stitch_core = copy.deepcopy(page.get("stitch_core"))
                    if isinstance(stitch_core, dict):
                        stitch_core["_source_page"] = int(
                            page.get("source_page", -1)
                        )
                    is_final_source_tail = (
                        protect_asura_tail_credits
                        and int(page.get("source_page", -1)) == last_source_page
                    )
                    if is_final_source_tail and isinstance(stitch_core, dict):
                        try:
                            is_final_source_tail = (
                                int(stitch_core.get("core_source_y2", -1))
                                >= int(stitch_core.get("source_height", 0))
                            )
                        except (TypeError, ValueError):
                            is_final_source_tail = False
                    work_items.append(
                        (
                            idx,
                            Path(page["original"]),
                            copy.deepcopy(page.get("excluded_regions", [])),
                            copy.deepcopy(page.get("boxes", [])),
                            stitch_core,
                            is_final_source_tail,
                            state_snapshot,
                        )
                    )

        if not work_items:
            return manifest

        _ = self.detector
        _ = self.inpainter

        max_workers = max(1, min(int(workers or 2), 2, len(work_items)))
        shared_seam_detections, seam_context_unavailable = (
            self._shared_seam_detections(chapter_id, work_items)
        )
        committed_indices: list[int] = []
        errors: list[tuple[int, Exception]] = []
        parallel_detectors = max_workers == 1

        def _process_one(item) -> tuple[int, dict, dict | None]:
            (
                idx,
                img_path,
                excluded,
                existing_boxes,
                stitch_core,
                is_final_source_tail,
                snapshot,
            ) = item
            return (
                idx,
                self._process_page(
                    img_path,
                    processed_dir,
                    excluded_regions=excluded,
                    existing_boxes=existing_boxes,
                    stitch_core=stitch_core,
                    supplemental_detections=shared_seam_detections.get(idx),
                    seam_context_unavailable=idx in seam_context_unavailable,
                    protect_tail_credits=is_final_source_tail,
                    parallel_detectors=parallel_detectors,
                ),
                snapshot,
            )

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="page-process"
        ) as pool:
            futures = {
                pool.submit(_process_one, item): item[0] for item in work_items
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    page_idx, page_data, snapshot = future.result()
                    if self._commit_processed_page(
                        chapter_id,
                        processed_dir,
                        page_idx,
                        page_data,
                        snapshot,
                    ):
                        committed_indices.append(page_idx)
                except Exception as exc:
                    logger.opt(exception=True).error(
                        "Chapter {} page {} operation 'process_page' failed: {}",
                        chapter_id,
                        idx,
                        exc,
                    )
                    errors.append((idx, exc))

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if committed_indices:
                manifest["workflow"] = {
                    "stage": "review",
                    "page_index": min(committed_indices),
                }
                save_manifest_raw(chapter_id, manifest)

        if errors:
            failed_indices = [item[0] for item in errors]
            first_exc = errors[0][1]
            if not committed_indices:
                raise RuntimeError(
                    f"Chapter {chapter_id}: All requested pages failed to process. "
                    f"First error (page {failed_indices[0]}): {first_exc}"
                ) from first_exc
            raise RuntimeError(
                f"Chapter {chapter_id}: Failed to process {len(errors)} page(s) "
                f"(indices: {failed_indices}). First error: {first_exc}"
            ) from first_exc

        return manifest

    def update_box(
        self,
        chapter_id: str,
        page_index: int,
        box_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                page = manifest["pages"][page_index]
                boxes = page.get("boxes", [])
                if box_index < 0 or box_index >= len(boxes):
                    raise ValueError(
                        f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}"
                    )
                img_path = Path(page["original"])
                expected_box_id = boxes[box_index].get("id")

            image = read_image(img_path)
            height, width = image.shape[:2]
            nx1, nx2 = sorted((int(x1), int(x2)))
            ny1, ny2 = sorted((int(y1), int(y2)))
            if (
                nx1 < 0
                or ny1 < 0
                or nx2 > width
                or ny2 > height
                or (nx2 - nx1) < 10
                or (ny2 - ny1) < 10
            ):
                raise ValueError(
                    f"Chapter {chapter_id} page {page_index}: Box coordinates out of bounds or smaller than 10px"
                )

            new_geometry = {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2}
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                target_boxes = target_page.get("boxes", [])
                if box_index < 0 or box_index >= len(target_boxes):
                    raise ValueError(
                        f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}"
                    )
                if expected_box_id and target_boxes[box_index].get("id") != expected_box_id:
                    raise RuntimeError(
                        f"Chapter {chapter_id} page {page_index}: Box changed while geometry edit was pending"
                    )

                boxes_snapshot = copy.deepcopy(target_boxes)
                self._apply_box_geometry(boxes_snapshot[box_index], new_geometry)
                manual_mask_posix = target_page.get("manual_mask")

            clean_path_posix = self._do_reinpaint(
                processed_dir, img_path, image, boxes_snapshot, manual_mask_posix
            )

            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                target_boxes = target_page.get("boxes", [])
                if box_index < 0 or box_index >= len(target_boxes):
                    raise ValueError(
                        f"Chapter {chapter_id} page {page_index}: Invalid box_index {box_index}"
                    )
                if expected_box_id and target_boxes[box_index].get("id") != expected_box_id:
                    raise RuntimeError(
                        f"Chapter {chapter_id} page {page_index}: Box changed during geometry repaint"
                    )

                self._apply_box_geometry(target_boxes[box_index], new_geometry)
                target_page["clean"] = clean_path_posix
                bump_page_revision(target_page, "clean_revision")
                invalidate_page_render(manifest, page_index)
                save_manifest_raw(chapter_id, manifest)
                self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest
