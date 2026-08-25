from __future__ import annotations

import copy
import threading
from pathlib import Path

import cv2
import numpy as np

from app.logging_config import logger
from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
)
from app.ocr.identity import (
    engine_identity,
    file_revision,
    geometry_signature,
    machine_cache_valid,
    stamp_machine_cache,
)
from app.pipeline import _decode_mask, read_image
from app.security import validate_chapter_id


class OCRResultStale(RuntimeError):
    pass


class OCRCancelled(RuntimeError):
    pass


def _region_overlaps_box(region: dict, box: dict) -> bool:
    if not box:
        return False
    bx = (box.get("x1", 0) + box.get("x2", 0)) / 2
    by = (box.get("y1", 0) + box.get("y2", 0)) / 2
    rx1, ry1 = region.get("x1", 0), region.get("y1", 0)
    rx2, ry2 = region.get("x2", 0), region.get("y2", 0)
    if rx1 <= bx <= rx2 and ry1 <= by <= ry2:
        return True
    ix1 = max(rx1, box.get("x1", 0))
    iy1 = max(ry1, box.get("y1", 0))
    ix2 = min(rx2, box.get("x2", 0))
    iy2 = min(ry2, box.get("y2", 0))
    return ix2 > ix1 and iy2 > iy1


def ocr_crop_from_box(image: np.ndarray, box: dict) -> np.ndarray:
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = map(
        int, (box["x1"], box["y1"], box["x2"], box["y2"])
    )
    bx1, by1 = max(0, min(w, bx1)), max(0, min(h, by1))
    bx2, by2 = max(bx1, min(w, bx2)), max(by1, min(h, by2))
    if bx2 <= bx1 or by2 <= by1:
        return image[0:0, 0:0]
    mask = _decode_mask(box.get("mask"))
    expected_shape = (by2 - by1, bx2 - bx1)
    if mask is not None and mask.shape == expected_shape:
        ys, xs = np.where(mask > 127)
        if xs.size and ys.size:
            pad = 12
            x1 = max(bx1, bx1 + int(xs.min()) - pad)
            y1 = max(by1, by1 + int(ys.min()) - pad)
            x2 = min(bx2, bx1 + int(xs.max()) + 1 + pad)
            y2 = min(by2, by1 + int(ys.max()) + 1 + pad)
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]
    pad = 20
    return image[
        max(0, by1 - pad) : min(h, by2 + pad),
        max(0, bx1 - pad) : min(w, bx2 + pad),
    ]


def _active_overlap_signatures(page: dict, region: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
    matches = []
    for box in page.get("boxes", []) or []:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        box_id = box.get("id")
        if not box_id or not _region_overlaps_box(region, box):
            continue
        matches.append((str(box_id), geometry_signature(box)))
    matches.sort(key=lambda item: item[0])
    return matches


class OCRService:
    def __init__(self, ocr_engine, pipeline):
        self.ocr = ocr_engine
        self.pipeline = pipeline

    def plan_chapter(self, chapter_id: str) -> list[tuple[int, str]]:
        validate_chapter_id(chapter_id)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            items: list[tuple[int, str]] = []
            for page_index, page in enumerate(manifest.get("pages", [])):
                if page.get("skipped"):
                    continue
                for box in page.get("boxes", []) or []:
                    if not isinstance(box, dict) or box.get("removed"):
                        continue
                    box_id = box.get("id")
                    if box_id:
                        items.append((page_index, str(box_id)))
            return items

    def box_id_at_index(self, chapter_id: str, page_index: int, box_index: int) -> str:
        validate_chapter_id(chapter_id)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page_index: {page_index}")
            boxes = pages[page_index].get("boxes", []) or []
            if box_index < 0 or box_index >= len(boxes):
                raise ValueError(f"Invalid box_index: {box_index}")
            box_id = boxes[box_index].get("id")
            if not box_id:
                raise ValueError("OCR target box has no stable id")
            return str(box_id)

    def inspect_box_index(
        self,
        chapter_id: str,
        page_index: int,
        box_index: int,
        lang: str,
        *,
        force: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        box_id = self.box_id_at_index(chapter_id, page_index, box_index)
        return self.inspect_box_id(
            chapter_id,
            page_index,
            box_id,
            lang,
            force=force,
            cancel_event=cancel_event,
        )

    def inspect_box_id(
        self,
        chapter_id: str,
        page_index: int,
        box_id: str,
        lang: str,
        *,
        force: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> dict:
        validate_chapter_id(chapter_id)
        engine = engine_identity(lang)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page_index: {page_index}")
            page = pages[page_index]
            if page.get("skipped"):
                raise ValueError("Cannot OCR a skipped page")
            box = next(
                (
                    item
                    for item in (page.get("boxes", []) or [])
                    if isinstance(item, dict) and str(item.get("id")) == str(box_id)
                ),
                None,
            )
            if box is None or box.get("removed"):
                raise ValueError(f"OCR target box not found: {box_id}")
            box_snapshot = copy.deepcopy(box)
            original_value = page.get("original")
            if not original_value:
                raise FileNotFoundError("Original page image is not configured")
            original_path = Path(original_value)
            source_revision = int(page.get("source_revision") or 0)

        if not original_path.is_file():
            raise FileNotFoundError("Original page image not found")
        original_revision = file_revision(original_path)
        if not force and machine_cache_valid(
            box_snapshot,
            lang=lang,
            engine=engine,
            source_revision=source_revision,
            original_revision=original_revision,
        ):
            return {
                "page_index": page_index,
                "box_id": str(box_id),
                "text": str(box_snapshot.get("ocr_text") or ""),
                "lang": lang,
                "engine": engine,
                "cached": True,
                "committed": True,
                "stale": False,
            }

        if cancel_event is not None and cancel_event.is_set():
            raise OCRCancelled("OCR job was cancelled")

        image = read_image(original_path)
        crop = ocr_crop_from_box(image, box_snapshot)
        if crop.size:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            text = self.ocr.read(rgb, lang)
        else:
            text = ""

        if cancel_event is not None and cancel_event.is_set():
            raise OCRCancelled("OCR job was cancelled")

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise OCRResultStale("Page disappeared while OCR was running")
            page = pages[page_index]
            current_original = page.get("original")
            current_source_revision = int(page.get("source_revision") or 0)
            if current_original != original_value or current_source_revision != source_revision:
                raise OCRResultStale("Original page changed while OCR was running")
            current_path = Path(str(current_original))
            try:
                current_file_revision = file_revision(current_path)
            except OSError as exc:
                raise OCRResultStale("Original page changed while OCR was running") from exc
            if current_file_revision != original_revision:
                raise OCRResultStale("Original page file changed while OCR was running")

            target = next(
                (
                    item
                    for item in (page.get("boxes", []) or [])
                    if isinstance(item, dict) and str(item.get("id")) == str(box_id)
                ),
                None,
            )
            if (
                target is None
                or target.get("removed")
                or geometry_signature(target) != geometry_signature(box_snapshot)
            ):
                raise OCRResultStale("OCR target box changed while OCR was running")
            if cancel_event is not None and cancel_event.is_set():
                raise OCRCancelled("OCR job was cancelled")

            stamp_machine_cache(
                target,
                text=text,
                lang=lang,
                engine=engine,
                source_revision=source_revision,
                original_revision=original_revision,
            )
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])

        return {
            "page_index": page_index,
            "box_id": str(box_id),
            "text": text or "",
            "lang": lang,
            "engine": engine,
            "cached": False,
            "committed": True,
            "stale": False,
        }

    def group_text_object(
        self,
        chapter_id: str,
        page_index: int,
        text_object_id: str,
        lang: str,
    ) -> dict:
        validate_chapter_id(chapter_id)
        engine = engine_identity(lang)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page index {page_index}")
            page = pages[page_index]
            obj = next(
                (
                    item
                    for item in (page.get("text_objects", []) or [])
                    if isinstance(item, dict) and item.get("id") == text_object_id
                ),
                None,
            )
            if obj is None:
                raise ValueError(f"Text object not found {text_object_id!r}")
            region = copy.deepcopy(obj.get("region") or {})
            text_snapshot = obj.get("ocr_text") or ""
            original_value = page.get("original")
            source_revision = int(page.get("source_revision") or 0)
            signatures = _active_overlap_signatures(page, region)

        if not original_value:
            raise FileNotFoundError("Original page image is not configured")
        original_path = Path(original_value)
        if not original_path.is_file():
            raise FileNotFoundError("Original page image not found")
        original_revision = file_revision(original_path)

        texts: list[str] = []
        source_box_ids: list[str] = []
        for box_id, _geometry in signatures:
            try:
                result = self.inspect_box_id(
                    chapter_id,
                    page_index,
                    box_id,
                    lang,
                    force=False,
                )
            except OCRResultStale:
                logger.warning(
                    "Chapter {} page {} box {} became stale during grouped OCR",
                    chapter_id,
                    page_index,
                    box_id,
                )
                continue
            source_box_ids.append(box_id)
            text = str(result.get("text") or "")
            if text:
                texts.append(text)
        combined = "\n".join(texts)

        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                return manifest
            page = pages[page_index]
            obj = next(
                (
                    item
                    for item in (page.get("text_objects", []) or [])
                    if isinstance(item, dict) and item.get("id") == text_object_id
                ),
                None,
            )
            if obj is None:
                return manifest
            current_original = page.get("original")
            current_source_revision = int(page.get("source_revision") or 0)
            try:
                current_file_revision = file_revision(Path(str(current_original)))
            except OSError:
                current_file_revision = None
            stale = (
                current_original != original_value
                or current_source_revision != source_revision
                or current_file_revision != original_revision
                or obj.get("region") != region
                or (obj.get("ocr_text") or "") != text_snapshot
                or _active_overlap_signatures(page, region) != signatures
            )
            if stale:
                logger.warning(
                    "Chapter {} page {} object {}: OCR result became stale; keeping newer state",
                    chapter_id,
                    page_index,
                    text_object_id,
                )
                return manifest

            obj["source_boxes"] = source_box_ids
            obj["ocr_text"] = combined
            obj["ocr_source"] = "machine"
            obj["ocr_lang"] = lang
            obj["ocr_engine"] = engine
            obj["ocr_source_revision"] = source_revision
            obj["ocr_file_revision"] = list(original_revision)
            obj["ocr_region"] = [
                int(region.get(key, 0)) for key in ("x1", "y1", "x2", "y2")
            ]
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest
