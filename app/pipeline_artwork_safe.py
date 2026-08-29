from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import PROCESSED_DIR
from app.detector.bubble_detector import BubbleBox
from app.inpaint.mask_geometry import geometry_dict, remap_local_mask_page_space
from app.manifest_utils import (
    bump_page_revision,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
)
from app.mask_store import decode_mask_value
from app.pipeline import ChapterPipeline, _encode_mask, read_image, write_image


class ArtworkSafeChapterPipeline(ChapterPipeline):
    """Chapter pipeline with geometry-safe detector-mask editing and repainting."""

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
            # Missing/non-overlapping detector masks are intentionally left empty.
            # The artwork-safe mask builder will no-op instead of erasing a full
            # rectangle, and a later detector pass can recover a fresh mask.
            box["mask"] = None
        else:
            # Geometry edits are comparatively rare. Keep the freshly remapped
            # mask inline for this transaction; the next processing commit moves
            # it back to a sidecar.
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
