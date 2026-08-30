from __future__ import annotations

from pathlib import Path
import threading

import cv2

from app.ocr.identity import stamp_machine_cache
from app.ocr.quality import classify_ocr_quality
from app.ocr.service import OCRService, _check_cancelled, _find_box, ocr_crop_from_box
from app.text_objects import (
    invalidate_stale_machine_translation,
    sync_existing_auto_text_object,
)


class HybridOCRService(OCRService):
    """OCRService that persists detailed hybrid-runtime metadata.

    Snapshot, cache, cancellation and stale-result orchestration stay in the
    base service. This subclass only captures detailed reader metadata and
    includes it in the same atomic manifest commit as the OCR text.
    """

    def __init__(self, ocr_engine, pipeline):
        super().__init__(ocr_engine, pipeline)
        self._result_local = threading.local()

    def inspect_box_id(self, *args, **kwargs) -> dict:
        # One OCRService instance can serve concurrent jobs, so never keep the
        # transient metadata on the instance itself.
        self._result_local.metadata = None
        try:
            result = super().inspect_box_id(*args, **kwargs)
            if result.get("cached"):
                # Base orchestration dispatches to our detailed cached-result
                # formatter, so the metadata is already present.
                return result
            metadata = getattr(self._result_local, "metadata", None)
            if metadata:
                return {**result, **metadata}
            return result
        finally:
            self._result_local.metadata = None

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

    def _read_box_text(self, original_path: Path, box_snapshot: dict, lang: str) -> str:
        image = self._cached_source_image(original_path)
        crop = ocr_crop_from_box(image, box_snapshot)
        if not crop.size:
            metadata = {
                "confidence": None,
                "model": "none",
                "orientation": "unknown",
                "region_count": 0,
                "quality": "reject",
                "quality_reason": "empty-crop",
            }
            self._result_local.metadata = metadata
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
        from app.manifest_utils import (
            get_manifest_lock,
            invalidate_page_render,
            load_manifest_raw,
            save_manifest_raw,
        )

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
                from app.ocr.service import OCRResultStale

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
            # Keep an already-created auto text object in the same transaction as
            # the box OCR commit. This also invalidates an untouched generated
            # translation if its OCR source changed, without creating unrelated
            # text objects elsewhere on the page.
            sync_existing_auto_text_object(page, target)
            invalidate_page_render(manifest, page_index)
            save_manifest_raw(chapter_id, manifest)
            self.pipeline._sync_output_dir(chapter_id, manifest, [page_index])

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
        # Grouped OCR writes directly to a text object rather than a detector box,
        # so apply the same translation-ownership rule before replacing its source.
        invalidate_stale_machine_translation(obj, combined)
        OCRService._stamp_group_object(
            obj,
            source_box_ids=source_box_ids,
            combined=combined,
            lang=lang,
            engine=engine,
            source_revision=source_revision,
            original_revision=original_revision,
            region=region,
        )
        obj["ocr_quality"] = "review"
        obj["ocr_quality_reason"] = "grouped-machine-ocr"
