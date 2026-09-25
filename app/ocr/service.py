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
    OCR_IMAGE_CACHE_MB,
)
from app.image_io import read_image
from app.region_policy import geometry_center_in_regions, page_preserve_regions, text_object_in_preserve_region
from app.security import validate_chapter_id
from app.text_objects import (
    invalidate_stale_machine_translation,
    sync_existing_auto_text_object,
)
from app.ocr.crop_geometry import active_overlap_signatures, edge_recrop_bounds_sequence, expanded_context_crop_bounds, ocr_crop_bounds
from app.ocr.visual_metadata import sync_group_visual_metadata, visual_cache_complete, visual_text_metadata
from app.ocr.targeting import is_story_retry_candidate, prefer_context_retry, prefer_edge_recrop, ocr_target_mode_for_box, ocr_target_skip_reason

OCR_CANCELLED_MESSAGE = "OCR job was cancelled"


class OCRResultStale(RuntimeError):
    pass


class OCRCancelled(RuntimeError):
    pass


def _cache_budget_bytes() -> int:
    return int(OCR_IMAGE_CACHE_MB) * 1024 * 1024


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
        self._result_local = threading.local()

    def plan_chapter(self, chapter_id: str) -> list[tuple[int, str]]:
        validate_chapter_id(chapter_id)
        with get_manifest_lock(chapter_id):
            manifest = load_manifest_raw(chapter_id)
            items: list[tuple[int, str]] = []
            for page_index, page in enumerate(manifest.get("pages", [])):
                if page.get("skipped") or page.get("process_required"):
                    continue
                preserve_regions = page_preserve_regions(page)
                for box in page.get("boxes", []) or []:
                    if not isinstance(box, dict) or box.get("removed"):
                        continue
                    if box.get("ocr_eligible") is False:
                        continue
                    if geometry_center_in_regions(box, preserve_regions):
                        continue
                    if ocr_target_skip_reason(box):
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

            if (
                not force
                and machine_cache_valid(
                    box_snapshot,
                    lang=lang,
                    engine=engine,
                    source_revision=source_revision,
                    original_revision=original_revision,
                )
                and visual_cache_complete(box_snapshot)
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
            if page.get("process_required"):
                raise ValueError("Cannot OCR a page that requires processing")
            box = _find_box(page, box_id)
            if box is None or box.get("removed"):
                raise ValueError(f"OCR target box not found: {box_id}")
            if box.get("ocr_eligible") is False:
                raise ValueError(f"OCR target box is not eligible: {box_id}")
            if geometry_center_in_regions(box, page_preserve_regions(page)):
                raise ValueError(f"OCR target box is inside a preserve region: {box_id}")
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
            "coverage": box_snapshot.get("ocr_coverage"),
            "target_mode": str(box_snapshot.get("ocr_target_mode") or "all"),
            "retry_applied": bool(box_snapshot.get("ocr_retry_applied")),
            "text_color": box_snapshot.get("ocr_text_color"),
            "font_size": box_snapshot.get("ocr_font_size"),
            "text_region": copy.deepcopy(box_snapshot.get("ocr_text_region")),
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
        crop_bounds = ocr_crop_bounds(image.shape, box_snapshot)
        x1, y1, x2, y2 = crop_bounds
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            self._result_local.metadata = {
                "confidence": None,
                "model": "none",
                "orientation": "unknown",
                "region_count": 0,
                "quality": "reject",
                "quality_reason": "empty-crop",
                "context_retry_applied": False,
            }
            return ""

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        detailed_reader = getattr(self.ocr, "read_detailed", None)
        if callable(detailed_reader):
            target_mode = ocr_target_mode_for_box(box_snapshot)

            def read_candidate(
                bounds: tuple[int, int, int, int],
            ) -> tuple[str, dict, object | None]:
                cx1, cy1, cx2, cy2 = bounds
                candidate_crop = image[cy1:cy2, cx1:cx2]
                if not candidate_crop.size:
                    return "", {
                        "confidence": None,
                        "model": "none",
                        "orientation": "unknown",
                        "region_count": 0,
                        "quality": "reject",
                        "quality_reason": "empty-crop",
                        "coverage": None,
                        "target_mode": target_mode,
                        "reader_retry_applied": False,
                    }, None
                candidate_rgb = cv2.cvtColor(candidate_crop, cv2.COLOR_BGR2RGB)
                try:
                    candidate_result = detailed_reader(
                        candidate_rgb, lang, target_mode=target_mode
                    )
                except TypeError:
                    candidate_result = detailed_reader(candidate_rgb, lang)
                candidate_text = str(
                    getattr(candidate_result, "text", "") or ""
                ).strip()
                candidate_coverage = self._mask_text_coverage(
                    box_snapshot, bounds, candidate_result
                )
                checked = classify_ocr_quality(
                    candidate_text,
                    lang,
                    confidence=getattr(candidate_result, "confidence", None),
                    coverage=candidate_coverage,
                )
                candidate_quality, candidate_reason = self._conservative_quality(
                    str(
                        getattr(candidate_result, "quality", "unknown")
                        or "unknown"
                    ),
                    getattr(candidate_result, "quality_reason", None),
                    checked.status,
                    checked.reason,
                )
                return candidate_text, {
                    "confidence": getattr(candidate_result, "confidence", None),
                    "model": str(getattr(candidate_result, "model", "") or ""),
                    "orientation": str(
                        getattr(candidate_result, "orientation", "unknown")
                        or "unknown"
                    ),
                    "region_count": int(
                        getattr(candidate_result, "region_count", 0) or 0
                    ),
                    "quality": candidate_quality,
                    "quality_reason": candidate_reason,
                    "coverage": (
                        candidate_coverage
                        if candidate_coverage is not None
                        else getattr(candidate_result, "coverage", None)
                    ),
                    "target_mode": str(
                        getattr(candidate_result, "target_mode", target_mode)
                        or target_mode
                    ),
                    "reader_retry_applied": bool(
                        getattr(candidate_result, "retry_applied", False)
                    ),
                }, candidate_result

            text, metadata, result = read_candidate(crop_bounds)
            recrop_attempted = False
            context_retry_applied = False
            no_recrop = metadata.get("target_mode") in {
                "recognition-only",
                "hybrid-fallback",
            }

            if not no_recrop and metadata["quality_reason"] == "crop-edge-text":
                initial_crop_bounds = crop_bounds
                for expanded_bounds in edge_recrop_bounds_sequence(
                    image.shape, initial_crop_bounds
                ):
                    recrop_attempted = True
                    expanded_text, expanded_metadata, expanded_result = read_candidate(
                        expanded_bounds
                    )
                    if expanded_result is None:
                        continue
                    if not prefer_edge_recrop(
                        base_text=text,
                        base_quality=metadata["quality"],
                        base_reason=metadata["quality_reason"],
                        base_confidence=metadata["confidence"],
                        base_region_count=metadata["region_count"],
                        expanded_text=expanded_text,
                        expanded_quality=expanded_metadata["quality"],
                        expanded_reason=expanded_metadata["quality_reason"],
                        expanded_confidence=expanded_metadata["confidence"],
                        expanded_region_count=expanded_metadata["region_count"],
                    ):
                        continue
                    text = expanded_text
                    metadata = expanded_metadata
                    result = expanded_result
                    crop_bounds = expanded_bounds
                    if metadata["quality_reason"] != "crop-edge-text":
                        break
            elif not no_recrop and (
                metadata["quality_reason"] == "incomplete-coverage"
                or (
                    not text
                    and is_story_retry_candidate(box_snapshot)
                )
            ):
                expanded_bounds = expanded_context_crop_bounds(
                    image.shape, box_snapshot
                )
                if expanded_bounds != crop_bounds:
                    recrop_attempted = True
                    context_retry_applied = True
                    expanded_text, expanded_metadata, expanded_result = read_candidate(
                        expanded_bounds
                    )
                    if (
                        expanded_result is not None
                        and prefer_context_retry(
                            base_text=text,
                            base_quality=metadata["quality"],
                            base_reason=metadata["quality_reason"],
                            base_coverage=metadata["coverage"],
                            base_confidence=metadata["confidence"],
                            expanded_text=expanded_text,
                            expanded_quality=expanded_metadata["quality"],
                            expanded_reason=expanded_metadata["quality_reason"],
                            expanded_coverage=expanded_metadata["coverage"],
                            expanded_confidence=expanded_metadata["confidence"],
                            expanded_result=expanded_result,
                            expanded_bounds=expanded_bounds,
                            box=box_snapshot,
                        )
                    ):
                        text = expanded_text
                        metadata = expanded_metadata
                        result = expanded_result
                        crop_bounds = expanded_bounds

            reader_retry_applied = bool(
                metadata.pop("reader_retry_applied", False)
            )
            metadata["retry_applied"] = bool(
                reader_retry_applied or recrop_attempted
            )
            metadata["context_retry_applied"] = context_retry_applied
            metadata.update(
                visual_text_metadata(
                    image,
                    box_snapshot,
                    crop_bounds,
                    result,
                    text=text,
                    region_count=int(metadata.get("region_count") or 0),
                )
            )
            self._result_local.metadata = metadata
            return text

        text = str(self.ocr.read(rgb, lang) or "").strip()
        quality = classify_ocr_quality(text, lang, confidence=None)
        metadata = {
            "confidence": None,
            "model": "legacy-reader",
            "orientation": "unknown",
            "region_count": 1 if text else 0,
            "quality": quality.status,
            "quality_reason": quality.reason,
            "context_retry_applied": False,
        }
        metadata.update(
            visual_text_metadata(
                image,
                box_snapshot,
                crop_bounds,
                None,
                text=text,
                region_count=1 if text else 0,
            )
        )
        self._result_local.metadata = metadata
        return text

    @staticmethod
    def _conservative_quality(
        base_status: str,
        base_reason: str | None,
        checked_status: str,
        checked_reason: str | None,
    ) -> tuple[str, str | None]:
        rank = {"reject": 2, "review": 1, "good": 0, "unknown": 0}
        base = base_status if base_status in rank else "unknown"
        checked = checked_status if checked_status in rank else "unknown"
        if rank[base] >= rank[checked]:
            return base, base_reason
        return checked, checked_reason

    @staticmethod
    def _mask_text_coverage(
        box: dict,
        crop_bounds: tuple[int, int, int, int],
        result: object,
    ) -> float | None:
        raw_mask = box.get("mask")
        mask = raw_mask if isinstance(raw_mask, np.ndarray) else decode_mask_value(raw_mask)
        if mask is None:
            return None
        try:
            bx1, by1, bx2, by2 = map(
                int, (box["x1"], box["y1"], box["x2"], box["y2"])
            )
        except (KeyError, TypeError, ValueError):
            return None
        if mask.shape != (max(0, by2 - by1), max(0, bx2 - bx1)):
            return None
        ys, xs = np.nonzero(mask > 127)
        text_bounds = getattr(result, "text_bounds", None)
        input_shape = getattr(result, "input_shape", None)
        if not xs.size or not ys.size or not text_bounds or not input_shape:
            return None
        try:
            prepared_h, prepared_w = float(input_shape[0]), float(input_shape[1])
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
            predicted_x1, predicted_y1, predicted_x2, predicted_y2 = (
                float(value) for value in text_bounds
            )
        except (TypeError, ValueError, IndexError):
            return None
        crop_w, crop_h = max(1, crop_x2 - crop_x1), max(1, crop_y2 - crop_y1)
        expected_x1, expected_x2 = bx1 + int(xs.min()), bx1 + int(xs.max()) + 1
        expected_y1, expected_y2 = by1 + int(ys.min()), by1 + int(ys.max()) + 1
        observed_x1 = crop_x1 + predicted_x1 * crop_w / max(1.0, prepared_w)
        observed_x2 = crop_x1 + predicted_x2 * crop_w / max(1.0, prepared_w)
        observed_y1 = crop_y1 + predicted_y1 * crop_h / max(1.0, prepared_h)
        observed_y2 = crop_y1 + predicted_y2 * crop_h / max(1.0, prepared_h)
        orientation = str(getattr(result, "orientation", "horizontal") or "horizontal")
        if orientation == "vertical":
            expected_start, expected_end = expected_x1, expected_x2
            observed_start, observed_end = observed_x1, observed_x2
        else:
            expected_start, expected_end = expected_y1, expected_y2
            observed_start, observed_end = observed_y1, observed_y2
        expected_span = max(1.0, float(expected_end - expected_start))
        overlap = max(
            0.0,
            min(float(expected_end), observed_end) - max(float(expected_start), observed_start),
        )
        return max(0.0, min(1.0, overlap / expected_span))

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
            if page.get("skipped") or page.get("process_required"):
                raise OCRResultStale("Page became unavailable while OCR was running")
            target = _find_box(page, box_id)
            if target is not None and geometry_center_in_regions(target, page_preserve_regions(page)):
                raise OCRResultStale("OCR target entered a preserve region while OCR was running")
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

        source_box_ids, combined = self._collect_group_text(
            chapter_id,
            page_index,
            snapshot["signatures"],
            lang,
        )
        if not combined:
            combined = self._read_box_text(
                original_path,
                snapshot["region"],
                lang,
            )
            self._result_local.metadata = None
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
            if page.get("skipped") or page.get("process_required"):
                raise ValueError("Cannot OCR this page before processing")
            if text_object_in_preserve_region(page, obj):
                raise ValueError("Cannot OCR a text object inside a preserve region")
            original_value = page.get("original")
            if not original_value:
                raise FileNotFoundError("Original page image is not configured")
            region = copy.deepcopy(obj.get("region") or {})
            return {
                "region": region,
                "text": obj.get("ocr_text") or "",
                "original_value": str(original_value),
                "source_revision": int(page.get("source_revision") or 0),
                "signatures": active_overlap_signatures(page, region),
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
            if page.get("skipped") or page.get("process_required") or text_object_in_preserve_region(page, obj):
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
            sync_group_visual_metadata(obj, page, source_box_ids)
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
            or active_overlap_signatures(page, snapshot["region"])
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
