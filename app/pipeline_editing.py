from __future__ import annotations

import copy
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.config import PROCESSED_DIR
from app.text_objects import ensure_page_text_objects
from app.image_io import read_image, write_image
from app.manifest_utils import (
    atomic_replace,
    bump_page_revision,
    get_manifest_lock,
    get_page_lock,
    invalidate_page_render,
    load_manifest_raw,
    new_box_id,
    save_manifest_raw,
)
from app.pipeline_masks import PipelineMasksMixin
from app.region_policy import overlap_share, seam_preserve_plan, subtract_regions_from_mask

STROKE_BOX_MAX_SHARE = 0.4

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


_RESIDUE_GHOST_MARGIN = 60
_RESIDUE_FILL_SIGMA = 8.0
_RESIDUE_INK_DROP = 15
_RESIDUE_MIN_COMPONENT_PX = 8


def _paint_residue(
    mask: np.ndarray,
    clean: np.ndarray | None,
    rect: tuple[int, int, int, int],
    original: np.ndarray | None = None,
) -> None:
    from app.detector.kiuyha_detector import letter_mask

    if clean is not None and clean.shape[:2] == mask.shape:
        (x1, y1, x2, y2), strokes = letter_mask(clean, rect)
        if strokes.any():
            view = mask[y1:y2, x1:x2]
            view[strokes] = 255
            return
        if original is not None and original.shape[:2] == mask.shape:
            (gx1, gy1, gx2, gy2), _grown, seed = letter_mask(original, rect, with_letters=True)
            if seed.any():
                orig_gray = (
                    cv2.cvtColor(original[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
                    if original.ndim == 3
                    else original[gy1:gy2, gx1:gx2]
                )
                clean_gray = (
                    cv2.cvtColor(clean[gy1:gy2, gx1:gx2], cv2.COLOR_BGR2GRAY)
                    if clean.ndim == 3
                    else clean[gy1:gy2, gx1:gx2]
                )
                orig_fill = cv2.GaussianBlur(orig_gray, (0, 0), _RESIDUE_FILL_SIGMA)
                was_ink = orig_gray.astype(np.int32) < orig_fill.astype(np.int32) - _RESIDUE_INK_DROP
                ghost = (
                    seed
                    & was_ink
                    & (clean_gray.astype(np.int32) <= orig_gray.astype(np.int32) + _RESIDUE_GHOST_MARGIN)
                )
                if ghost.any():
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(
                        ghost.astype(np.uint8), connectivity=8
                    )
                    keep = np.zeros(count, bool)
                    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= _RESIDUE_MIN_COMPONENT_PX
                    ghost = keep[labels]
                if ghost.any():
                    view = mask[gy1:gy2, gx1:gx2]
                    view[ghost] = 255
                    return
            return
    mask[rect[1]:rect[3], rect[0]:rect[2]] = 255


COVERED_SHARE = 0.3


def _stroke_box(image: np.ndarray, rect: tuple[int, int, int, int]) -> dict:
    from app.detector.kiuyha_detector import letter_mask
    from app.image_io import encode_mask

    grown, mask = letter_mask(image, rect)
    too_big = (grown[2] - grown[0]) * (grown[3] - grown[1]) > STROKE_BOX_MAX_SHARE * image.shape[0] * image.shape[1]
    if too_big or not mask.any():
        grown, mask = rect, None
    return {"id": new_box_id(), "origin": "manual", "x1": int(grown[0]), "y1": int(grown[1]),
            "x2": int(grown[2]), "y2": int(grown[3]), "confidence": 1.0, "manual": True,
            "mask": encode_mask(mask.astype(np.uint8) * 255) if mask is not None else None}

class PipelineEditingMixin(PipelineMasksMixin):
    """Box, text-object, and manual mask editing operations for ChapterPipeline."""

    def add_manual_box(self, chapter_id: str, page_index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        return self.add_manual_boxes(chapter_id, page_index, [(x1, y1, x2, y2)])

    def add_manual_boxes(self, chapter_id: str, page_index: int, rects: list[tuple[int, int, int, int]]) -> dict:
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

            new_boxes = []
            for x1, y1, x2, y2 in rects:
                nx1, nx2 = sorted((max(0, min(int(x1), w)), max(0, min(int(x2), w))))
                ny1, ny2 = sorted((max(0, min(int(y1), h)), max(0, min(int(y2), h))))
                if nx2 <= nx1 or ny2 <= ny1:
                    continue
                new_boxes.append({
                    "id": new_box_id(), "origin": "manual",
                    "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2,
                    "confidence": 1.0, "mask": None, "manual": True,
                })
            if not new_boxes:
                return manifest
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page_index {page_index}")
                target_page = manifest["pages"][page_index]
                preserve_regions = copy.deepcopy(target_page.get("preserve_regions", []))
                boxes_snapshot = copy.deepcopy(target_page.get("boxes", []))
                boxes_snapshot.extend(copy.deepcopy(new_boxes))
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
                    target_page.setdefault("boxes", []).extend(new_boxes)
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
        added = [
            {key: int(region[key]) for key in ("x1", "y1", "x2", "y2")}
            for region in regions
            if int(region["x2"]) > int(region["x1"]) and int(region["y2"]) > int(region["y1"])
        ]
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                raise ValueError("Invalid page index")
            page = manifest["pages"][page_index]
            if page.get("skipped") or page.get("process_required") or not page.get("clean"):
                raise ValueError("Page must be active and processed before preserving regions")
            plan = seam_preserve_plan(manifest, page_index, added)
        if not plan:
            return manifest
        for idx, regs in plan.items():
            manifest = self._preserve_one_page(chapter_id, idx, regs)
        return manifest

    def _preserve_one_page(self, chapter_id: str, page_index: int, regions: list[dict]) -> dict:
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
                    ensure_page_text_objects(target_page)
                    if bump_page_revision(target_page, "clean_revision") != target_clean_revision:
                        raise RuntimeError("Page clean revision changed while preserving regions")
                    invalidate_page_render(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def apply_review_fixes(
        self,
        chapter_id: str,
        page_index: int,
        *,
        preserve: list[tuple[int, int, int, int]] = (),
        boxes: list[tuple[int, int, int, int]] = (),
        repaint: list[tuple[int, int, int, int]] = (),
        skip_covered: bool = False,
    ) -> dict:
        processed_dir = PROCESSED_DIR / chapter_id
        with get_page_lock(chapter_id, page_index):
            with get_manifest_lock(chapter_id):
                manifest = load_manifest_raw(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError("Invalid page index")
                page = manifest["pages"][page_index]
                if page.get("skipped") or page.get("process_required") or not page.get("clean"):
                    raise ValueError("Page must be active and processed before review fixes")
                img_path = Path(page["original"])
                clean_now = Path(page["clean"])
                old_preserve = copy.deepcopy(page.get("preserve_regions", []))
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                manual_mask_posix = page.get("manual_mask")
                manual_lama_mask_posix = page.get("manual_lama_mask")
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            image = read_image(img_path)
            h, w = image.shape[:2]

            def clip(rect):
                x1, x2 = sorted((max(0, min(int(rect[0]), w)), max(0, min(int(rect[2]), w))))
                y1, y2 = sorted((max(0, min(int(rect[1]), h)), max(0, min(int(rect[3]), h))))
                return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

            added_preserve = [dict(zip(("x1", "y1", "x2", "y2"), r)) for r in map(clip, preserve) if r]
            new_boxes = [_stroke_box(image, r) for r in map(clip, boxes) if r]
            if skip_covered:
                active = [b for b in boxes_snapshot if not b.get("removed")]
                new_boxes = [b for b in new_boxes if not any(overlap_share(b, other) >= COVERED_SHARE for other in active)]
            preserve_regions = old_preserve + added_preserve
            boxes_snapshot.extend(copy.deepcopy(new_boxes))

            final_mask_path = self._manual_mask_path(processed_dir, img_path)
            mask = None
            repaint_rects = [r for r in map(clip, repaint) if r]
            clean = read_image(clean_now) if repaint_rects else None
            for rect in repaint_rects:
                if mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                _paint_residue(mask, clean, rect, original=image)
            if mask is not None:
                existing = self._read_manual_mask(
                    Path(manual_mask_posix) if manual_mask_posix else final_mask_path, (h, w))
                if existing is not None:
                    mask = np.maximum(existing, mask)
                mask = subtract_regions_from_mask(mask, preserve_regions)
                if mask is not None and not np.any(mask):
                    mask = None
            if not added_preserve and not new_boxes and mask is None:
                return manifest

            tmp_mask_path = processed_dir / f"manual_mask_{img_path.name}.{uuid.uuid4().hex}.tmp.png"
            try:
                if mask is not None:
                    write_image(tmp_mask_path, mask)
                with self._page_artifact_transaction(
                    processed_dir, img_path, page_index, target_clean_revision, [final_mask_path]
                ) as artifact_tx:
                    clean_path_posix = self._do_reinpaint(
                        processed_dir,
                        img_path,
                        image,
                        boxes_snapshot,
                        manual_mask_posix=tmp_mask_path.as_posix() if mask is not None else manual_mask_posix,
                        manual_lama_mask_posix=manual_lama_mask_posix,
                        reuse_auto_clean=not added_preserve and not new_boxes,
                        preserve_regions=preserve_regions,
                    )
                    if mask is not None:
                        atomic_replace(tmp_mask_path, final_mask_path)
                    with get_manifest_lock(chapter_id):
                        manifest = load_manifest_raw(chapter_id)
                        target_page = manifest["pages"][page_index]
                        target_page.setdefault("boxes", []).extend(new_boxes)
                        target_page["preserve_regions"] = preserve_regions
                        if mask is not None:
                            target_page["manual_mask"] = final_mask_path.as_posix()
                        target_page["clean"] = clean_path_posix
                        ensure_page_text_objects(target_page)
                        if bump_page_revision(target_page, "clean_revision") != target_clean_revision:
                            raise RuntimeError("Page clean revision changed during review fixes")
                        invalidate_page_render(manifest, page_index)
                        artifact_tx.mark_manifest_commit(target_page)
                        save_manifest_raw(chapter_id, manifest)
                        artifact_tx.commit()
                        self._sync_output_dir(chapter_id, manifest, [page_index])
            finally:
                tmp_mask_path.unlink(missing_ok=True)
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

            for key in ("font_selection_mode", "font_match", "font_ai_id"):
                if key in changes:
                    obj[key] = changes[key]

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
