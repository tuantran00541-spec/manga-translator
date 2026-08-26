from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from app.config import PROCESSED_DIR
from app.inpaint.mask_geometry import geometry_dict, remap_local_mask_page_space
from app.manifest_utils import (
    bump_page_revision,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
)
from app.pipeline import ChapterPipeline, _decode_mask, _encode_mask, read_image


class ArtworkSafeChapterPipeline(ChapterPipeline):
    """Chapter pipeline with geometry-safe detector-mask editing.

    The base pipeline historically dropped detector segmentation whenever a user
    edited the detector rectangle. That forced later cleanup into a destructive
    rectangle fallback. This override changes only ``update_box``: the existing
    mask is kept in page coordinates and cropped into the edited geometry.
    Re-detection/reprocessing is handled by the shared stable-ID reconciliation in
    ``manifest_utils`` so fresh detector masks follow the same rule.
    """

    @staticmethod
    def _apply_box_geometry(box: dict, new_geometry: dict[str, int]) -> None:
        source_geometry = geometry_dict(box)
        source_mask = _decode_mask(box.get("mask"))
        detector_origin = box.get("origin") == "detector" and not box.get("manual")

        if detector_origin:
            if not box.get("geometry_overridden"):
                box["detector_anchor"] = copy.deepcopy(source_geometry)
            box["geometry_overridden"] = True

        remapped = remap_local_mask_page_space(
            source_mask, source_geometry, new_geometry
        ) if source_mask is not None else None

        box.update(new_geometry)
        if remapped is None or not np.any(remapped > 127):
            # Missing/non-overlapping detector masks are intentionally left empty.
            # The artwork-safe mask builder will no-op instead of erasing a full
            # rectangle, and a later detector pass can recover a fresh mask.
            box["mask"] = None
        else:
            box["mask"] = _encode_mask(remapped)

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
