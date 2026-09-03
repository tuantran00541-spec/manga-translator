from __future__ import annotations

from collections import OrderedDict
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
from app.mask_store import decode_mask_value
from app.ocr.identity import (
    engine_identity,
    file_revision,
    machine_cache_valid,
    ocr_crop_signature,
    stamp_machine_cache,
)
from app.ocr.quality import classify_ocr_quality
from app.parameters import (
    OCR_BOX_CROP_PADDING,
    OCR_IMAGE_CACHE_MB,
    OCR_MASK_CROP_PADDING,
)
from app.pipeline import read_image
from app.security import validate_chapter_id
from app.text_objects import (
    invalidate_stale_machine_translation,
    sync_existing_auto_text_object,
)

OCR_CANCELLED_MESSAGE = "OCR job was cancelled"


class OCRResultStale(RuntimeError):
    pass


class OCRCancelled(RuntimeError):
    pass


def _cache_budget_bytes() -> int:
    return int(OCR_IMAGE_CACHE_MB) * 1024 * 1024


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
    mask = decode_mask_value(box.get("mask"))
    expected_shape = (by2 - by1, bx2 - bx1)
    if mask is not None and mask.shape == expected_shape:
        ys, xs = np.nonzero(mask > 127)
        if xs.size and ys.size:
            pad = OCR_MASK_CROP_PADDING
            x1 = max(bx1, bx1 + int(xs.min()) - pad)
            y1 = max(by1, by1 + int(ys.min()) - pad)
            x2 = min(bx2, bx1 + int(xs.max()) + 1 + pad)
            y2 = min(by2, by1 + int(ys.max()) + 1 + pad)
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]
    pad = OCR_BOX_CROP_PADDING
    return image[
        max(0, by1 - pad) : min(h, by2 + pad),
        max(0, bx1 - pad) : min(w, bx2 + pad),
    ]


def _active_overlap_signatures(
    page: dict, region: dict
) -> list[tuple[str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    for box in page.get("boxes", []) or []:
        if not isinstance(box, dict) or box.get("removed"):
            continue
        box_id = box.get("id")
        if not box_id or not _region_overlaps_box(region, box):
            continue
        matches.append(
            (
                int(box.get("y1", 0)),
                int(box.get("x1", 0)),
                str(box_id),
                ocr_crop_signature(box),
            )
        )
    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(box_id, geometry) for _, _, box_id, geometry in matches]


def _find_box(page: dict, box_id: str) -> dict | None:
    return next(
        (
            item
            for item in (page.get("boxes", []) or [])
            if isinstance(item, dict) and str(item.get("id")) == str(box_id)
        ),
        None,
    )


def _find_text_object(page: dict, text_object_id: str) -> dict | None:
    return next(
        (
            item
            for item in (page.get("text_objects", []) or [])
            if isinstance(item, dict) and item.get("id") == text_object_id
        ),
        None,
    )


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OCRCancelled(OCR_CANCELLED_MESSAGE)


class OCRService:
    def __init__(self, ocr_engine, pipeline):
        self.ocr = ocr_engine
        self.pipeline = pipeline
        self._image_cache: OrderedDict[
            tuple[str, tuple[int, int, int]], np.ndarray
        ] = OrderedDict()
        self._image_cache_bytes = 0
        self._image_cache_budget = _cache_budget_bytes()
        self._image_cache_lock = threading.RLock()
        # Detailed OCR metadata is transient per call. One OCRService instance can
        # serve concurrent jobs, so it must never live directly on the instance.
        self._result_local = threading.local()

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
                    if box.get("ocr_eligible") is False:
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
        self._result_local.metadata = None
        try:
            validate_chapter_id(chapter_id)
            engine = engine_identity(lang)
            box_snapshot, original_value, source_revision = self._snapshot_box(
                chapter_id, page_index, box_id
            )
            original_path, original_revision = self._source_identity(original_value)

            if not force and machine_cache_valid(
                box_snapshot,
                lang=lang,
                engine=engine,
                source_revision=source_revision,
                original_revision=original_revision,
            ):
                return self._cached_box_result(
                    page_index, box_id, box_snapshot, lang, engine
                )

            _check_cancelled(cancel_event)
            text = self._read_box_text(original_path, box_snapshot, lang)
            _check_cancelled(cancel_event)
            self._commit_box_result(
                chapter_id,
                page_index,
                box_id,
                box_snapshot=box_snapshot,
                original_value=original_value,
                source_revision=source_revision,
                original_revision=original_revision,
                text=text,
                lang=lang,
                engine=engine,
                cancel_event=cancel_event,
            )
            result = {
                "page_index": page_index,
                "box_id": str(box_id),
                "text": text or "",
                "lang": lang,
                "engine": engine,
                "cached": False,
                "committed": True,
                "stale": False,
            }
            metadata = getattr(self._result_local, "metadata", None)
            if metadata:
                result.update(metadata)
            return result
        finally:
            self._result_local.metadata = None

    def _snapshot_box(
        self, chapter_id: str, page_index: int, box_id: str
    ) -> tuple[dict, str, int]:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page_index: {page_index}")
            page = pages[page_index]
            if page.get("skipped"):
                raise ValueError("Cannot OCR a skipped page")
            box = _find_box(page, box_id)
            if box is None or box.get("removed"):
                raise ValueError(f"OCR target box not found: {box_id}")
            if box.get("ocr_eligible") is False:
                raise ValueError(f"OCR target box is not eligible: {box_id}")
            original_value = page.get("original")
            if not original_value:
                raise FileNotFoundError("Original page image is not configured")
            box_snapshot = copy.deepcopy(box)
            box_snapshot["_ocr_snapshot_crop_signature"] = ocr_crop_signature(box)
            return (
                box_snapshot,
                str(original_value),
                int(page.get("source_revision") or 0),
            )

    @staticmethod
    def _source_identity(original_value: str) -> tuple[Path, tuple[int, int, int]]:
        original_path = Path(original_value)
        if not original_path.is_file():
            raise FileNotFoundError("Original page image not found")
        return original_path, file_revision(original_path)

    @staticmethod
    def _cached_box_result(
        page_index: int, box_id: str, box_snapshot: dict, lang: str, engine: str
    ) -> dict:
        return {
            "page_index": page_index,
            "box_id": str(box_id),
            "text": str(box_snapshot.get("ocr_text") or ""),
            "lang": lang,
            "engine": engine,
            "cached": True,
            "committed": True,
            "stale": False,
            "confidence": box_snapshot.get("ocr_confidence"),
            "model": str(box_snapshot.get("ocr_model") or ""),
            "orientation": str(box_snapshot.get("ocr_orientation") or "unknown"),
            "region_count": int(box_snapshot.get("ocr_region_count") or 0),
            "quality": str(box_snapshot.get("ocr_quality") or "unknown"),
            "quality_reason": box_snapshot.get("ocr_quality_reason"),
        }

    def _cached_source_image(self, original_path: Path) -> np.ndarray:
        revision = file_revision(original_path)
        key = (str(original_path), revision)
        with self._image_cache_lock:
            cached = self._image_cache.pop(key, None)
            if cached is not None:
                self._image_cache[key] = cached
                return cached

        image = read_image(original_path)
        image_bytes = int(image.nbytes)
        if self._image_cache_budget <= 0 or image_bytes > self._image_cache_budget:
            return image

        with self._image_cache_lock:
            cached = self._image_cache.pop(key, None)
            if cached is not None:
                self._image_cache[key] = cached
                return cached

            while (
                self._image_cache
                and self._image_cache_bytes + image_bytes > self._image_cache_budget
            ):
                _old_key, old_image = self._image_cache.popitem(last=False)
                self._image_cache_bytes -= int(old_image.nbytes)

            self._image_cache[key] = image
            self._image_cache_bytes += image_bytes
        return image

    def _read_box_text(self, original_path: Path, box_snapshot: dict, lang: str) -> str:
        image = self._cached_source_image(original_path)
        crop = ocr_crop_from_box(image, box_snapshot)
        if not crop.size:
            self._result_local.metadata = {
                "confidence": None,
                "model": "none",
                "orientation": "unknown",
                "region_count": 0,
                "quality": "reject",
                "quality_reason": "empty-crop",
            }
            return ""

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        detailed_reader = getattr(self.ocr, "read_detailed", None)
        if callable(detailed_reader):
            result = detailed_reader(rgb, lang)
            text = str(getattr(result, "text", "") or "").strip()
            self._result_local.metadata = {
                "confidence": getattr(result, "confidence", None),
                "model": str(getattr(result, "model", "") or ""),
                "orientation": str(
                    getattr(result, "orientation", "unknown") or "unknown"
                ),
                "region_count": int(getattr(result, "region_count", 0) or 0),
                "quality": str(getattr(result, "quality", "unknown") or "unknown"),
                "quality_reason": getattr(result, "quality_reason", None),
            }
            return text

        text = str(self.ocr.read(rgb, lang) or "").strip()
        quality = classify_ocr_quality(text, lang, confidence=None)
        self._result_local.metadata = {
            "confidence": None,
            "model": "legacy-reader",
            "orientation": "unknown",
            "region_count": 1 if text else 0,
            "quality": quality.status,
            "quality_reason": quality.reason,
        }
        return text

    def _commit_box_result(
        self,
        chapter_id: str,
        page_index: int,
        box_id: str,
        *,
        box_snapshot: dict,
        original_value: str,
        source_revision: int,
        original_revision: tuple[int, int, int],
        text: str,
        lang: str,
        engine: str,
        cancel_event: threading.Event | None,
    ) -> None:
        metadata = getattr(self._result_local, "metadata", None)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            page = self._current_box_page(
                manifest,
                page_index,
                original_value=original_value,
                source_revision=source_revision,
                original_revision=original_revision,
            )
            target = _find_box(page, box_id)
            if self._box_changed(target, box_snapshot):
                raise OCRResultStale("OCR target box changed while OCR was running")
            _check_cancelled(cancel_event)
            stamp_machine_cache(
                target,
                text=text,
                lang=lang,
                engine=engine,
                source_revision=source_revision,
                original_revision=original_revision,
                metadata=metadata,
            )
            # Existing auto-generated objects track committed machine OCR in the
            # same manifest transaction, including translation ownership rules.
            sync_existing_auto_text_object(page, target)
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])

    @staticmethod
    def _current_box_page(
        manifest: dict,
        page_index: int,
        *,
        original_value: str,
        source_revision: int,
        original_revision: tuple[int, int, int],
    ) -> dict:
        pages = manifest.get("pages", [])
        if page_index < 0 or page_index >= len(pages):
            raise OCRResultStale("Page disappeared while OCR was running")
        page = pages[page_index]
        current_original = page.get("original")
        current_source_revision = int(page.get("source_revision") or 0)
        if current_original != original_value or current_source_revision != source_revision:
            raise OCRResultStale("Original page changed while OCR was running")
        try:
            current_file_revision = file_revision(Path(str(current_original)))
        except OSError as exc:
            raise OCRResultStale("Original page changed while OCR was running") from exc
        if current_file_revision != original_revision:
            raise OCRResultStale("Original page file changed while OCR was running")
        return page

    @staticmethod
    def _box_changed(target: dict | None, box_snapshot: dict) -> bool:
        return (
            target is None
            or bool(target.get("removed"))
            or target.get("ocr_eligible") is False
            or ocr_crop_signature(target)
            != box_snapshot.get(
                "_ocr_snapshot_crop_signature",
                ocr_crop_signature(box_snapshot),
            )
        )

    def group_text_object(
        self,
        chapter_id: str,
        page_index: int,
        text_object_id: str,
        lang: str,
    ) -> dict:
        validate_chapter_id(chapter_id)
        engine = engine_identity(lang)
        snapshot = self._snapshot_group(chapter_id, page_index, text_object_id)
        original_path, original_revision = self._source_identity(snapshot["original_value"])
        del original_path

        source_box_ids, combined = self._collect_group_text(
            chapter_id,
            page_index,
            snapshot["signatures"],
            lang,
        )
        return self._commit_group_text(
            chapter_id,
            page_index,
            text_object_id,
            lang=lang,
            engine=engine,
            source_box_ids=source_box_ids,
            combined=combined,
            snapshot=snapshot,
            original_revision=original_revision,
        )

    def _snapshot_group(
        self, chapter_id: str, page_index: int, text_object_id: str
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                raise ValueError(f"Invalid page index {page_index}")
            page = pages[page_index]
            obj = _find_text_object(page, text_object_id)
            if obj is None:
                raise ValueError(f"Text object not found {text_object_id!r}")
            original_value = page.get("original")
            if not original_value:
                raise FileNotFoundError("Original page image is not configured")
            region = copy.deepcopy(obj.get("region") or {})
            return {
                "region": region,
                "text": obj.get("ocr_text") or "",
                "original_value": str(original_value),
                "source_revision": int(page.get("source_revision") or 0),
                "signatures": _active_overlap_signatures(page, region),
            }

    def _collect_group_text(
        self,
        chapter_id: str,
        page_index: int,
        signatures: list[tuple[str, str]],
        lang: str,
    ) -> tuple[list[str], str]:
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
        return source_box_ids, "\n".join(texts)

    def _commit_group_text(
        self,
        chapter_id: str,
        page_index: int,
        text_object_id: str,
        *,
        lang: str,
        engine: str,
        source_box_ids: list[str],
        combined: str,
        snapshot: dict,
        original_revision: tuple[int, int, int],
    ) -> dict:
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            pages = manifest.get("pages", [])
            if page_index < 0 or page_index >= len(pages):
                return manifest
            page = pages[page_index]
            obj = _find_text_object(page, text_object_id)
            if obj is None:
                return manifest
            if self._group_result_stale(page, obj, snapshot, original_revision):
                logger.warning(
                    "Chapter {} page {} object {}: OCR result became stale; keeping newer state",
                    chapter_id,
                    page_index,
                    text_object_id,
                )
                return manifest

            self._stamp_group_object(
                obj,
                source_box_ids=source_box_ids,
                combined=combined,
                lang=lang,
                engine=engine,
                source_revision=snapshot["source_revision"],
                original_revision=original_revision,
                region=snapshot["region"],
            )
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    @staticmethod
    def _group_result_stale(
        page: dict,
        obj: dict,
        snapshot: dict,
        original_revision: tuple[int, int, int],
    ) -> bool:
        current_original = page.get("original")
        try:
            current_file_revision = file_revision(Path(str(current_original)))
        except OSError:
            current_file_revision = None
        return (
            current_original != snapshot["original_value"]
            or int(page.get("source_revision") or 0) != snapshot["source_revision"]
            or current_file_revision != original_revision
            or obj.get("region") != snapshot["region"]
            or (obj.get("ocr_text") or "") != snapshot["text"]
            or _active_overlap_signatures(page, snapshot["region"])
            != snapshot["signatures"]
        )

    @staticmethod
    def _stamp_group_object(
        obj: dict,
        *,
        source_box_ids: list[str],
        combined: str,
        lang: str,
        engine: str,
        source_revision: int,
        original_revision: tuple[int, int, int],
        region: dict,
    ) -> None:
        # Grouped OCR writes directly to a text object, so apply the same
        # translation-ownership rule before replacing its machine-owned source.
        invalidate_stale_machine_translation(obj, combined)
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
        obj["ocr_quality"] = "review"
        obj["ocr_quality_reason"] = "grouped-machine-ocr"
