from __future__ import annotations

import copy
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import PROCESSED_DIR
from app.detector.bubble_detector import BubbleBox
from app.image_io import read_image, write_image
from app.logging_config import logger
from app.manifest_utils import (
    PageArtifactTransaction,
    atomic_replace,
    bump_page_revision,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    load_manifest_raw,
    new_box_id,
    save_manifest_raw,
)
from app.mask_store import decode_mask_value
from app.parameters import MANUAL_MASK_THRESHOLD
from app.region_policy import subtract_regions_from_mask


_TEXT_OBJECT_STYLE_KEYS = {
    "color", "font", "fontSize", "bold",
    "strokeWidth", "strokeColor", "bgColor", "cornerRadius",
    "horizontalAlign", "verticalAlign",
}

DEFAULT_TEXT_OBJECT_STYLE = {
    "color": "auto",
    "font": "default",
    "fontSize": "auto",
    "bold": False,
    "strokeWidth": "auto",
    "strokeColor": "auto",
    "bgColor": "transparent",
    "cornerRadius": "0",
    "horizontalAlign": "center",
    "verticalAlign": "middle",
}


def _normalize_region(region: dict, w: int, h: int) -> tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = (
            int(region["x1"]), int(region["y1"]),
            int(region["x2"]), int(region["y2"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid region payload") from exc
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class PipelineEditingMixin:
    def add_manual_box(self, chapter_id: str, page_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                page = manifest["pages"][page_index]
                img_path = Path(page["original"])

            image = read_image(img_path)
            h, w = image.shape[:2]

            nx1, nx2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
            ny1, ny2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
            if nx2 <= nx1 or ny2 <= ny1:
                return manifest

            new_box = {
                "id": new_box_id(), "origin": "manual",
                "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                "confidence": 1.0, "mask": None, "manual": True,
            }
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                preserve_regions = copy.deepcopy(target_page.get("preserve_regions", []))
                boxes_snapshot = copy.deepcopy(target_page.get("boxes", []))
                boxes_snapshot.append(copy.deepcopy(new_box))
                manual_mask_posix = target_page.get("manual_mask")
                manual_lama_mask_posix = target_page.get("manual_lama_mask")
                target_clean_revision = int(
                    target_page.get("clean_revision") or 0
                ) + 1

            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
                    preserve_regions=preserve_regions,
                )

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                    target_page = manifest["pages"][page_index]
                    target_page.setdefault("boxes", []).append(new_box)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
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
                preserve_regions = copy.deepcopy(target_page.get("preserve_regions", []))
                manual_mask_posix = target_page.get("manual_mask")
                manual_lama_mask_posix = target_page.get("manual_lama_mask")
                target_clean_revision = int(
                    target_page.get("clean_revision") or 0
                ) + 1

            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
                    preserve_regions=preserve_regions,
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
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def preserve_and_reinpaint(self, chapter_id: str, page_index: int, regions: list[dict]) -> dict:
        """Add preserve regions and re-inpaint with the page's current boxes.

        Unlike saving preserve regions from the editor, this does not send the
        page back through detection, so its boxes, text objects and
        translations stay; only the pixels inside the new regions return to
        the original, and the objects there drop out of translation and render.
        """
        processed_dir = PROCESSED_DIR / chapter_id
        added = [
            {key: int(region[key]) for key in ("x1", "y1", "x2", "y2")}
            for region in regions
            if int(region["x2"]) > int(region["x1"]) and int(region["y2"]) > int(region["y1"])
        ]
        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                page = manifest["pages"][page_index]
                if page.get("skipped") or page.get("process_required") or not page.get("clean"):
                    raise ValueError("Page must be active and processed before preserving regions")
                img_path = Path(page["original"])
                preserve_regions = copy.deepcopy(page.get("preserve_regions", [])) + added
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                manual_mask_posix = page.get("manual_mask")
                manual_lama_mask_posix = page.get("manual_lama_mask")
                target_clean_revision = int(page.get("clean_revision") or 0) + 1
            if not added:
                return manifest

            image = read_image(img_path)
            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
                    preserve_regions=preserve_regions,
                )
                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    target_page = manifest["pages"][page_index]
                    target_page["preserve_regions"] = preserve_regions
                    target_page["clean"] = clean_path_posix
                    if bump_page_revision(target_page, "clean_revision") != target_clean_revision:
                        raise RuntimeError("Page clean revision changed while preserving regions")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def remove_box(self, chapter_id: str, page_index: int, box_index: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                page = manifest["pages"][page_index]
                boxes = page.get("boxes", [])
                if box_index < 0 or box_index >= len(boxes):
                    raise ValueError("Invalid box index")

                img_path = Path(page["original"])
                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))
                manual_mask_posix = page.get("manual_mask")
                manual_lama_mask_posix = page.get("manual_lama_mask")
                boxes_snapshot = copy.deepcopy(boxes)
                boxes_snapshot[box_index]["removed"] = True
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            image = read_image(img_path)
            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=manual_mask_posix,
                    manual_lama_mask_posix=manual_lama_mask_posix,
                    preserve_regions=preserve_regions,
                )

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError("Invalid page index")
                    target_page = manifest["pages"][page_index]
                    target_boxes = target_page.get("boxes", [])
                    if box_index < 0 or box_index >= len(target_boxes):
                        raise ValueError("Invalid box index")

                    target_boxes[box_index]["removed"] = True
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def repaint_mask(
        self,
        chapter_id: str,
        page_index: int,
        mask: np.ndarray,
        *,
        force_lama: bool = False,
    ) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                page = manifest["pages"][page_index]
                if page.get("skipped"):
                    raise ValueError("Cannot repaint a skipped page; unskip it first")
                if page.get("process_required"):
                    raise ValueError("Cannot repaint a page that requires processing")
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1
                manual_mask_posix = page.get("manual_mask")
                manual_lama_mask_posix = page.get("manual_lama_mask")
                clean_posix = page.get("clean")

            image = read_image(img_path)
            img_h, img_w = image.shape[:2]

            manual_mask_path = (
                Path(manual_mask_posix)
                if manual_mask_posix
                else self._manual_mask_path(processed_dir, img_path)
            )
            manual_lama_mask_path = (
                Path(manual_lama_mask_posix)
                if manual_lama_mask_posix
                else self._manual_mask_path(
                    processed_dir, img_path, force_lama=True
                )
            )

            bin_mask = (
                (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
                if mask is not None
                else None
            )
            if bin_mask is not None and bin_mask.shape[:2] != (img_h, img_w):
                raise ValueError(
                    "Repaint mask dimensions "
                    f"{bin_mask.shape[:2]} must exactly match page dimensions "
                    f"{(img_h, img_w)}"
                )

            target_field = "manual_lama_mask" if force_lama else "manual_mask"
            target_path = manual_lama_mask_path if force_lama else manual_mask_path
            target_prefix = "manual_lama_mask" if force_lama else "manual_mask"
            existing_mask = self._read_manual_mask(target_path, (img_h, img_w))

            existing_mask = subtract_regions_from_mask(existing_mask, preserve_regions)
            bin_mask = subtract_regions_from_mask(bin_mask, preserve_regions)
            if bin_mask is not None and not np.any(bin_mask):
                raise ValueError("Repaint mask falls entirely inside a preserve region")
            if existing_mask is not None and bin_mask is not None:
                accumulated_mask = np.maximum(existing_mask, bin_mask)
            elif bin_mask is not None:
                accumulated_mask = bin_mask
            elif existing_mask is not None:
                accumulated_mask = existing_mask
            else:
                accumulated_mask = None

            tmp_mask_path = processed_dir / (
                f"{target_prefix}_{img_path.name}.{uuid.uuid4().hex}.tmp.png"
            )
            if accumulated_mask is not None:
                write_image(tmp_mask_path, accumulated_mask)

            final_mask_path = self._manual_mask_path(
                processed_dir, img_path, force_lama=force_lama
            )
            with self._page_artifact_transaction(
                processed_dir,
                img_path,
                page_index,
                target_clean_revision,
                [final_mask_path],
            ) as artifact_tx:
                auto_clean_path = self._auto_clean_path(processed_dir, img_path)
                if (
                    not auto_clean_path.exists()
                    and not manual_mask_path.exists()
                    and not manual_lama_mask_path.exists()
                ):
                    clean_path = Path(clean_posix) if clean_posix else None
                    if clean_path is not None and clean_path.exists():
                        try:
                            cached = read_image(clean_path)
                            if cached.shape[:2] == (img_h, img_w):
                                self._write_auto_clean_cache(
                                    processed_dir, img_path, cached
                                )
                        except Exception as exc:
                            logger.warning(
                                "Could not seed auto-clean cache from {}: {}",
                                clean_path,
                                exc,
                            )

                try:
                    standard_mask_for_repaint = (
                        tmp_mask_path.as_posix()
                        if accumulated_mask is not None and not force_lama
                        else manual_mask_posix
                    )
                    lama_mask_for_repaint = (
                        tmp_mask_path.as_posix()
                        if accumulated_mask is not None and force_lama
                        else manual_lama_mask_posix
                    )
                    clean_path_posix = self._do_reinpaint(
                        processed_dir,
                        img_path,
                        image,
                        boxes_snapshot,
                        manual_mask_posix=standard_mask_for_repaint,
                        manual_lama_mask_posix=lama_mask_for_repaint,
                        reuse_auto_clean=True,
                        preserve_regions=preserve_regions,
                    )
                    if accumulated_mask is not None:
                        atomic_replace(tmp_mask_path, final_mask_path)
                        target_mask_posix = final_mask_path.as_posix()
                    else:
                        target_mask_posix = None
                finally:
                    if tmp_mask_path.exists():
                        try:
                            tmp_mask_path.unlink()
                        except OSError:
                            pass

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                    target_page = manifest["pages"][page_index]
                    if target_mask_posix:
                        target_page[target_field] = target_mask_posix
                    else:
                        target_page.pop(target_field, None)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def reset_manual_mask(self, chapter_id: str, page_index: int) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id

        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                page = manifest["pages"][page_index]
                if page.get("skipped"):
                    raise ValueError("Cannot reset repaint state on a skipped page; unskip it first")
                if page.get("process_required"):
                    raise ValueError("Cannot reset repaint state before processing")
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            manual_mask_path = self._manual_mask_path(processed_dir, img_path)
            manual_lama_mask_path = self._manual_mask_path(
                processed_dir, img_path, force_lama=True
            )

            image = read_image(img_path)
            with self._page_artifact_transaction(
                processed_dir,
                img_path,
                page_index,
                target_clean_revision,
                [manual_mask_path, manual_lama_mask_path],
            ) as artifact_tx:
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    manual_mask_posix=None,
                    manual_lama_mask_posix=None,
                    reuse_auto_clean=True,
                    apply_manual_mask=False,
                    preserve_regions=preserve_regions,
                )

                for mask_path in (manual_mask_path, manual_lama_mask_path):
                    if mask_path.exists():
                        try:
                            mask_path.unlink()
                        except OSError as exc:
                            raise RuntimeError(
                                f"Cannot remove manual mask: {exc}"
                            ) from exc

                with get_manifest_lock(chapter_id):
                    manifest = load_manifest_raw(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError("Invalid page index")
                    target_page = manifest["pages"][page_index]
                    target_page.pop("manual_mask", None)
                    target_page.pop("manual_lama_mask", None)
                    target_page["clean"] = clean_path_posix
                    clean_revision = bump_page_revision(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def create_text_object(
        self, chapter_id: str, page_index: int, shape: str, region: dict
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            img_path = Path(pages[page_index]["original"])

        image = read_image(img_path)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = _normalize_region(region, w, h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            raise ValueError(f"Chapter {chapter_id}: Region too small (min 10px)")

        obj = {
            "id": uuid.uuid4().hex,
            "shape": shape,
            "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "source_boxes": [],
            "ocr_text": "",
            "translation": "",
            "style": dict(DEFAULT_TEXT_OBJECT_STYLE),
        }

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target = pages[page_index]
            target.setdefault("text_objects", []).append(obj)
            target.setdefault("width", w)
            target.setdefault("height", h)
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def update_text_object(
        self, chapter_id: str, page_index: int, text_object_id: str, changes: dict
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target = pages[page_index]
            objs = target.setdefault("text_objects", [])
            obj = next((o for o in objs if o.get("id") == text_object_id), None)
            if obj is None:
                raise ValueError(f"Chapter {chapter_id}: Text object not found {text_object_id!r}")

            if "shape" in changes and changes["shape"] is not None:
                if changes["shape"] not in ("rectangle", "ellipse"):
                    raise ValueError("Invalid shape")
                obj["shape"] = changes["shape"]

            if "region" in changes and changes["region"] is not None:
                region = changes["region"]
                try:
                    x1, y1, x2, y2 = (
                        int(region["x1"]), int(region["y1"]),
                        int(region["x2"]), int(region["y2"]),
                    )
                except (KeyError, TypeError, ValueError):
                    raise ValueError("Invalid region payload")
                if x1 < 0 or y1 < 0:
                    raise ValueError("Region coordinates must be non-negative")
                if x1 >= x2 or y1 >= y2:
                    raise ValueError("Region must have x1 < x2 and y1 < y2")
                if target.get("width") is not None and target.get("height") is not None:
                    pw, ph = int(target["width"]), int(target["height"])
                    x1 = max(0, min(x1, pw))
                    x2 = max(0, min(x2, pw))
                    y1 = max(0, min(y1, ph))
                    y2 = max(0, min(y2, ph))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    raise ValueError("Region too small (min 10px)")
                obj["region"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            if "ocr_text" in changes:
                obj["ocr_text"] = changes["ocr_text"] or ""
            if "translation" in changes:
                obj["translation"] = changes["translation"] or ""
            if "style" in changes and changes["style"] is not None:
                style = changes["style"]
                if not isinstance(style, dict):
                    raise ValueError("Invalid style payload")
                cleaned: dict = {}
                for k, v in style.items():
                    if k not in _TEXT_OBJECT_STYLE_KEYS:
                        raise ValueError(f"Invalid style key {k!r}")
                    if k == "bold":
                        cleaned[k] = bool(v)
                    elif k == "horizontalAlign":
                        if v not in ("left", "center", "right"):
                            raise ValueError("Invalid horizontalAlign value")
                        cleaned[k] = v
                    elif k == "verticalAlign":
                        if v not in ("top", "middle", "bottom"):
                            raise ValueError("Invalid verticalAlign value")
                        cleaned[k] = v
                    else:
                        cleaned[k] = v
                merged = dict(DEFAULT_TEXT_OBJECT_STYLE)
                merged.update(cleaned)
                obj["style"] = merged

            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
        return manifest

    def delete_text_object(
        self, chapter_id: str, page_index: int, text_object_id: str
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
            target = pages[page_index]
            objs = target.get("text_objects") or []
            idx = next((i for i, o in enumerate(objs) if o.get("id") == text_object_id), None)
            if idx is None:
                raise ValueError(f"Chapter {chapter_id}: Text object not found {text_object_id!r}")
            del objs[idx]
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
        return manifest

    @staticmethod
    def _auto_clean_path(processed_dir: Path, img_path: Path) -> Path:
        return processed_dir / f"auto_clean_{img_path.name}"

    def _page_artifact_transaction(
        self,
        processed_dir: Path,
        img_path: Path,
        page_index: int,
        target_clean_revision: int,
        extra_paths: list[Path] | None = None,
    ) -> PageArtifactTransaction:
        paths = [
            processed_dir / f"clean_{img_path.name}",
            self._auto_clean_path(processed_dir, img_path),
        ]
        paths.extend(extra_paths or [])
        return PageArtifactTransaction(
            processed_dir,
            page_index,
            paths,
            target_clean_revision,
        )

    @staticmethod
    def _manual_mask_path(
        processed_dir: Path, img_path: Path, *, force_lama: bool = False
    ) -> Path:
        prefix = "manual_lama_mask" if force_lama else "manual_mask"
        return processed_dir / f"{prefix}_{img_path.name}"

    @staticmethod
    def _read_manual_mask(
        mask_path: Path, expected_shape: tuple[int, int]
    ) -> np.ndarray | None:
        if not mask_path.exists():
            return None
        try:
            raw = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read manual mask at {}: {}", mask_path, exc)
            return None
        if mask is None or not np.any(mask > MANUAL_MASK_THRESHOLD):
            return None
        if mask.shape[:2] != expected_shape:
            try:
                mask = cv2.resize(
                    mask,
                    (expected_shape[1], expected_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            except Exception as exc:
                logger.warning("Could not resize manual mask at {}: {}", mask_path, exc)
                return None
        if not np.any(mask > MANUAL_MASK_THRESHOLD):
            return None
        return (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255

    def _write_auto_clean_cache(self, processed_dir: Path, img_path: Path, image: np.ndarray) -> Path:
        auto_path = self._auto_clean_path(processed_dir, img_path)
        tmp_path = processed_dir / f"auto_clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        write_image(tmp_path, image)
        atomic_replace(tmp_path, auto_path)
        return auto_path

    def _load_auto_clean_cache(self, processed_dir: Path, img_path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
        auto_path = self._auto_clean_path(processed_dir, img_path)
        if not auto_path.exists():
            return None
        try:
            cached = read_image(auto_path)
        except Exception as exc:
            logger.warning("Could not read auto-clean cache at {}: {}", auto_path, exc)
            return None
        if cached.shape[:2] != expected_shape:
            logger.warning(
                "Ignoring stale auto-clean cache at {}: expected shape {}, got {}",
                auto_path, expected_shape, cached.shape[:2],
            )
            return None
        return cached

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
        boxes_objects = []
        for box in boxes:
            if box.get("removed"):
                continue

            confidence = float(box.get("confidence", 1.0))
            geometry_overridden = bool(box.get("geometry_overridden"))
            explicit_manual = bool(
                box.get("manual")
                or box.get("origin") == "manual"
                or confidence >= 1.0
            )
            safe_to_inpaint = bool(box.get("safe_to_inpaint"))
            overlap_context_only = bool(box.get("overlap_context_only"))

            if overlap_context_only and not geometry_overridden:
                continue
            if not (safe_to_inpaint or geometry_overridden or explicit_manual):
                continue

            box_h = int(box["y2"]) - int(box["y1"])
            box_w = int(box["x2"]) - int(box["x1"])
            if box_h <= 0 or box_w <= 0:
                continue
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

            box_object = BubbleBox(
                int(box["x1"]),
                int(box["y1"]),
                int(box["x2"]),
                int(box["y2"]),
                confidence,
                mask_arr,
                source_model=str(box.get("source_model") or "unknown"),
                class_id=int(box.get("class_id") or 0),
                class_name=str(box.get("class_name") or "unknown"),
                semantic_type=str(box.get("semantic_type") or "unknown"),
                mask_source=str(box.get("mask_source") or "none"),
                safe_to_inpaint=safe_to_inpaint,
                ocr_eligible=bool(box.get("ocr_eligible")),
                needs_review=bool(box.get("needs_review")),
                source_role=str(box.get("source_role") or "unknown"),
                deferred_reason=box.get("deferred_reason"),
            )
            if geometry_overridden or explicit_manual:
                box_object.allow_rectangle_fallback = True
            boxes_objects.append(box_object)

        clean_image = None
        if reuse_auto_clean:
            clean_image = self._load_auto_clean_cache(
                processed_dir, img_path, image.shape[:2]
            )

        if clean_image is None:
            clean_image = self.inpainter.inpaint(image, boxes_objects, protected_regions=preserve_regions)
            self._write_auto_clean_cache(processed_dir, img_path, clean_image)
        else:
            clean_image = clean_image.copy()

        manual_mask_path = (
            Path(manual_mask_posix)
            if manual_mask_posix
            else self._manual_mask_path(processed_dir, img_path)
        )
        manual_lama_mask_path = (
            Path(manual_lama_mask_posix)
            if manual_lama_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=True)
        )
        mask_passes = (
            (manual_mask_path, False),
            (manual_lama_mask_path, True),
        )
        if apply_manual_mask:
            for mask_path, force_lama in mask_passes:
                manual_mask = self._read_manual_mask(
                    mask_path, clean_image.shape[:2]
                )
                manual_mask = subtract_regions_from_mask(manual_mask, preserve_regions)
                if manual_mask is not None and np.any(manual_mask):
                    clean_image = self.inpainter.inpaint_mask(
                        clean_image, manual_mask, force_lama=force_lama
                    )

        clean_path = processed_dir / f"clean_{img_path.name}"
        tmp_clean_path = (
            processed_dir
            / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        )
        write_image(tmp_clean_path, clean_image)
        atomic_replace(tmp_clean_path, clean_path)
        return clean_path.as_posix()
