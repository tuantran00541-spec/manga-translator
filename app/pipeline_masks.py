from __future__ import annotations

import copy
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.detector.boxes import BubbleBox
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
    save_manifest_raw,
)
from app.mask_store import decode_mask_value
from app.parameters import MANUAL_MASK_THRESHOLD
from app.region_policy import subtract_regions_from_mask


def _ed(name: str, fallback):
    editing_mod = sys.modules.get("app.pipeline_editing")
    if editing_mod is not None and hasattr(editing_mod, name):
        return getattr(editing_mod, name)
    return fallback


def _processed_dir_root() -> Path:
    editing_mod = sys.modules.get("app.pipeline_editing")
    if editing_mod is not None and hasattr(editing_mod, "PROCESSED_DIR"):
        return Path(editing_mod.PROCESSED_DIR)
    from app.config import PROCESSED_DIR
    return Path(PROCESSED_DIR)


class PipelineMasksMixin:
    """Manual repaint/restore mask and clean-cache operations for ChapterPipeline."""

    def repaint_mask(
        self,
        chapter_id: str,
        page_index: int,
        mask: np.ndarray,
        *,
        force_lama: bool = False,
    ) -> dict:
        processed_dir = _processed_dir_root() / chapter_id

        with _ed("get_page_lock", get_page_lock)(chapter_id, page_index):
            with _ed("get_manifest_lock", get_manifest_lock)(chapter_id):
                manifest = _ed("load_manifest_raw", load_manifest_raw)(chapter_id)
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

            image = _ed("read_image", read_image)(img_path)
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

            existing_mask = _ed("subtract_regions_from_mask", subtract_regions_from_mask)(existing_mask, preserve_regions)
            bin_mask = _ed("subtract_regions_from_mask", subtract_regions_from_mask)(bin_mask, preserve_regions)
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
                _ed("write_image", write_image)(tmp_mask_path, accumulated_mask)

            final_mask_path = self._manual_mask_path(
                processed_dir, img_path, force_lama=force_lama
            )
            restore_path = self._restore_mask_path(processed_dir, img_path)
            with self._page_artifact_transaction(
                processed_dir,
                img_path,
                page_index,
                target_clean_revision,
                [final_mask_path, restore_path],
            ) as artifact_tx:
                restore = self._read_manual_mask(restore_path, (img_h, img_w))
                if restore is not None and bin_mask is not None:
                    restore[bin_mask > MANUAL_MASK_THRESHOLD] = 0
                    if np.any(restore):
                        _ed("write_image", write_image)(restore_path, restore)
                    else:
                        restore_path.unlink()
                auto_clean_path = self._auto_clean_path(processed_dir, img_path)
                if (
                    not auto_clean_path.exists()
                    and not manual_mask_path.exists()
                    and not manual_lama_mask_path.exists()
                ):
                    clean_path = Path(clean_posix) if clean_posix else None
                    if clean_path is not None and clean_path.exists():
                        try:
                            cached = _ed("read_image", read_image)(clean_path)
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
                        _ed("atomic_replace", atomic_replace)(tmp_mask_path, final_mask_path)
                        target_mask_posix = final_mask_path.as_posix()
                    else:
                        target_mask_posix = None
                finally:
                    if tmp_mask_path.exists():
                        try:
                            tmp_mask_path.unlink()
                        except OSError:
                            pass

                with _ed("get_manifest_lock", get_manifest_lock)(chapter_id):
                    manifest = _ed("load_manifest_raw", load_manifest_raw)(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                    target_page = manifest["pages"][page_index]
                    if target_mask_posix:
                        target_page[target_field] = target_mask_posix
                    else:
                        target_page.pop(target_field, None)
                    target_page["clean"] = clean_path_posix
                    clean_revision = _ed("bump_page_revision", bump_page_revision)(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    _ed("invalidate_page_render", invalidate_page_render)(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    _ed("save_manifest_raw", save_manifest_raw)(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def restore_mask(self, chapter_id: str, page_index: int, mask: np.ndarray) -> dict:
        processed_dir = _processed_dir_root() / chapter_id

        with _ed("get_page_lock", get_page_lock)(chapter_id, page_index):
            with _ed("get_manifest_lock", get_manifest_lock)(chapter_id):
                manifest = _ed("load_manifest_raw", load_manifest_raw)(chapter_id)
                if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                    raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                page = manifest["pages"][page_index]
                if page.get("skipped"):
                    raise ValueError("Cannot restore a skipped page; unskip it first")
                if page.get("process_required"):
                    raise ValueError("Cannot restore a page that requires processing")
                img_path = Path(page["original"])
                boxes_snapshot = copy.deepcopy(page.get("boxes", []))
                preserve_regions = copy.deepcopy(page.get("preserve_regions", []))
                target_clean_revision = int(page.get("clean_revision") or 0) + 1

            image = _ed("read_image", read_image)(img_path)
            img_h, img_w = image.shape[:2]
            painted = (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
            if painted.shape[:2] != (img_h, img_w):
                raise ValueError(
                    f"Restore mask dimensions {painted.shape[:2]} must exactly match page dimensions {(img_h, img_w)}"
                )
            if not np.any(painted):
                raise ValueError("Restore mask is empty")
            restore_path = self._restore_mask_path(processed_dir, img_path)
            existing = self._read_manual_mask(restore_path, (img_h, img_w))
            if existing is not None:
                painted = np.maximum(existing, painted)

            with self._page_artifact_transaction(
                processed_dir, img_path, page_index, target_clean_revision, [restore_path],
            ) as artifact_tx:
                _ed("write_image", write_image)(restore_path, painted)
                clean_path_posix = self._do_reinpaint(
                    processed_dir,
                    img_path,
                    image,
                    boxes_snapshot,
                    reuse_auto_clean=True,
                    preserve_regions=preserve_regions,
                )
                with _ed("get_manifest_lock", get_manifest_lock)(chapter_id):
                    manifest = _ed("load_manifest_raw", load_manifest_raw)(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError(f"Chapter {chapter_id}: Invalid page index {page_index}")
                    target_page = manifest["pages"][page_index]
                    target_page["restore_mask"] = restore_path.as_posix()
                    target_page["clean"] = clean_path_posix
                    if _ed("bump_page_revision", bump_page_revision)(target_page, "clean_revision") != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during restore")
                    _ed("invalidate_page_render", invalidate_page_render)(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    _ed("save_manifest_raw", save_manifest_raw)(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
            return manifest

    def reset_manual_mask(self, chapter_id: str, page_index: int) -> dict:
        processed_dir = _processed_dir_root() / chapter_id

        with _ed("get_page_lock", get_page_lock)(chapter_id, page_index):
            with _ed("get_manifest_lock", get_manifest_lock)(chapter_id):
                manifest = _ed("load_manifest_raw", load_manifest_raw)(chapter_id)
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
            restore_path = self._restore_mask_path(processed_dir, img_path)

            image = _ed("read_image", read_image)(img_path)
            with self._page_artifact_transaction(
                processed_dir,
                img_path,
                page_index,
                target_clean_revision,
                [manual_mask_path, manual_lama_mask_path, restore_path],
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

                for mask_path in (manual_mask_path, manual_lama_mask_path, restore_path):
                    if mask_path.exists():
                        try:
                            mask_path.unlink()
                        except OSError as exc:
                            raise RuntimeError(
                                f"Cannot remove manual mask: {exc}"
                            ) from exc

                with _ed("get_manifest_lock", get_manifest_lock)(chapter_id):
                    manifest = _ed("load_manifest_raw", load_manifest_raw)(chapter_id)
                    if page_index < 0 or page_index >= len(manifest.get("pages", [])):
                        raise ValueError("Invalid page index")
                    target_page = manifest["pages"][page_index]
                    target_page.pop("manual_mask", None)
                    target_page.pop("manual_lama_mask", None)
                    target_page.pop("restore_mask", None)
                    target_page["clean"] = clean_path_posix
                    clean_revision = _ed("bump_page_revision", bump_page_revision)(
                        target_page, "clean_revision"
                    )
                    if clean_revision != target_clean_revision:
                        raise RuntimeError("Page clean revision changed during repaint")
                    _ed("invalidate_page_render", invalidate_page_render)(manifest, page_index)
                    artifact_tx.mark_manifest_commit(target_page)
                    _ed("save_manifest_raw", save_manifest_raw)(chapter_id, manifest)
                    artifact_tx.commit()
                    self._sync_output_dir(chapter_id, manifest, [page_index])
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
        return _ed("PageArtifactTransaction", PageArtifactTransaction)(
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
    def _restore_mask_path(processed_dir: Path, img_path: Path) -> Path:
        return processed_dir / f"restore_mask_{img_path.name}"

    def _apply_restore_mask(self, clean: np.ndarray, original: np.ndarray, processed_dir: Path, img_path: Path) -> bool:
        mask = self._read_manual_mask(self._restore_mask_path(processed_dir, img_path), clean.shape[:2])
        if mask is None:
            return False
        keep = mask > MANUAL_MASK_THRESHOLD
        clean[keep] = original[keep]
        return True

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
        _ed("write_image", write_image)(tmp_path, image)
        _ed("atomic_replace", atomic_replace)(tmp_path, auto_path)
        return auto_path

    def _load_auto_clean_cache(self, processed_dir: Path, img_path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
        auto_path = self._auto_clean_path(processed_dir, img_path)
        if not auto_path.exists():
            return None
        try:
            cached = _ed("read_image", read_image)(auto_path)
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
            mask_arr = _ed("decode_mask_value", decode_mask_value)(box.get("mask"))
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
                manual_mask = _ed("subtract_regions_from_mask", subtract_regions_from_mask)(manual_mask, preserve_regions)
                if manual_mask is not None and np.any(manual_mask):
                    clean_image = self.inpainter.inpaint_mask(
                        clean_image, manual_mask, force_lama=force_lama
                    )

        clean_path = processed_dir / f"clean_{img_path.name}"
        tmp_clean_path = (
            processed_dir
            / f"clean_{img_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        )
        _ed("write_image", write_image)(tmp_clean_path, clean_image)
        _ed("atomic_replace", atomic_replace)(tmp_clean_path, clean_path)
        return clean_path.as_posix()
