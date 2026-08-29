from __future__ import annotations

from pathlib import Path
import threading

import cv2

from app.ocr.identity import engine_identity, machine_cache_valid, stamp_machine_cache
from app.ocr.quality import classify_ocr_quality
from app.ocr.service import OCRService, _check_cancelled, _find_box


class HybridOCRService(OCRService):
    """OCRService variant that persists detailed hybrid-runtime metadata.

    The base service remains untouched while the migration branch is being
    validated. Once the hybrid runtime is accepted this small override can be
    folded back into OCRService.
    """

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
        from app.security import validate_chapter_id

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
        text, metadata = self._read_box_result(original_path, box_snapshot, lang)
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
            metadata=metadata,
            lang=lang,
            engine=engine,
            cancel_event=cancel_event,
        )
        return {
            "page_index": page_index,
            "box_id": str(box_id),
            "text": text,
            "lang": lang,
            "engine": engine,
            "cached": False,
            "committed": True,
            "stale": False,
            **metadata,
        }

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

    def _read_box_result(
        self, original_path: Path, box_snapshot: dict, lang: str
    ) -> tuple[str, dict]:
        image = self._cached_source_image(original_path)
        crop = self._crop(image, box_snapshot)
        if not crop.size:
            return "", {
                "confidence": None,
                "model": "none",
                "orientation": "unknown",
                "region_count": 0,
                "quality": "reject",
                "quality_reason": "empty-crop",
            }

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        detailed_reader = getattr(self.ocr, "read_detailed", None)
        if callable(detailed_reader):
            result = detailed_reader(rgb, lang)
            return str(getattr(result, "text", "") or "").strip(), {
                "confidence": getattr(result, "confidence", None),
                "model": str(getattr(result, "model", "") or ""),
                "orientation": str(getattr(result, "orientation", "unknown") or "unknown"),
                "region_count": int(getattr(result, "region_count", 0) or 0),
                "quality": str(getattr(result, "quality", "unknown") or "unknown"),
                "quality_reason": getattr(result, "quality_reason", None),
            }

        text = str(self.ocr.read(rgb, lang) or "").strip()
        quality = classify_ocr_quality(text, lang, confidence=None)
        return text, {
            "confidence": None,
            "model": "legacy-reader",
            "orientation": "unknown",
            "region_count": 1 if text else 0,
            "quality": quality.status,
            "quality_reason": quality.reason,
        }

    @staticmethod
    def _crop(image, box_snapshot):
        from app.ocr.service import ocr_crop_from_box

        return ocr_crop_from_box(image, box_snapshot)

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
        metadata: dict,
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
