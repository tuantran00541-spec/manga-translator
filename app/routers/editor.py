from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.dependencies import pipeline
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, invalidate_page_render, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.pipeline import read_image
from app.schemas import (
    AddBoxRequest,
    CreateTextObjectRequest,
    DeleteTextObjectRequest,
    RemoveBoxRequest,
    ResetManualMaskRequest,
    SaveDraftRequest,
    UpdateBoxRequest,
    UpdateTextObjectRequest,
)
from app.security import validate_chapter_id
from app.text_objects import invalidate_stale_machine_translation

router = APIRouter(prefix="/api", tags=["editor"])


def _reconcile_translation_after_ocr_edit(req: UpdateTextObjectRequest) -> dict:
    """Invalidate an untouched generated translation after a source-only OCR edit.

    ``pipeline.update_text_object`` owns the editor mutation itself. This follow-up
    transaction is optimistic and only applies if the object still contains the
    exact OCR text from this request; a concurrent newer edit therefore wins. If
    the request also supplied a translation, that is explicit user intent and the
    caller skips this reconciliation entirely.
    """
    expected_source = str(req.ocr_text or "")
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_index < 0 or req.page_index >= len(pages):
            return manifest
        page = pages[req.page_index]
        obj = next(
            (
                item
                for item in (page.get("text_objects") or [])
                if isinstance(item, dict) and str(item.get("id")) == str(req.id)
            ),
            None,
        )
        if obj is None or str(obj.get("ocr_text") or "") != expected_source:
            return manifest
        if invalidate_stale_machine_translation(obj, expected_source):
            invalidate_page_render(manifest, req.page_index)
            save_manifest_raw(req.chapter_id, manifest)
            pipeline._sync_output_dir(req.chapter_id, manifest, [req.page_index])
        return manifest


@router.post("/add_box")
def add_box(req: AddBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    pages = manifest_raw.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    if req.x2 <= req.x1 or req.y2 <= req.y1:
        raise HTTPException(400, "Invalid box coordinates: x1 must be < x2 and y1 must be < y2")
    if req.x2 - req.x1 < 10 or req.y2 - req.y1 < 10:
        raise HTTPException(400, "Invalid box dimensions: width and height must be at least 10px")

    img_path = Path(pages[req.page_index]["original"])
    if not img_path.is_file():
        raise HTTPException(404, f"Original page image not found: page_{req.page_index:03d}")
    try:
        image = read_image(img_path)
        h, w = image.shape[:2]
        if req.x1 < 0 or req.y1 < 0 or req.x2 > w or req.y2 > h:
            raise HTTPException(400, f"Box coordinates ({req.x1},{req.y1})-({req.x2},{req.y2}) exceed image dimensions ({w}x{h})")
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} operation 'add_box' cannot read image: {}", req.chapter_id, req.page_index, exc)
        raise HTTPException(500, f"Cannot read page image: {exc}") from exc

    try:
        manifest = pipeline.add_manual_box(req.chapter_id, req.page_index, req.x1, req.y1, req.x2, req.y2)
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error("Chapter {} page {} operation 'add_box' invalid value: {}", req.chapter_id, req.page_index, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} operation 'add_box' failed: {}", req.chapter_id, req.page_index, exc)
        raise HTTPException(500, f"Add box failed: {exc}") from exc


@router.post("/update_box")
def update_box(req: UpdateBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    pages = manifest_raw.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    boxes = pages[req.page_index].get("boxes", [])
    if req.box_index < 0 or req.box_index >= len(boxes):
        raise HTTPException(400, f"Invalid box_index: {req.box_index}")
    if req.x2 <= req.x1 or req.y2 <= req.y1:
        raise HTTPException(400, "Invalid box coordinates: x1 must be < x2 and y1 must be < y2")
    if req.x2 - req.x1 < 10 or req.y2 - req.y1 < 10:
        raise HTTPException(400, "Invalid box dimensions: width and height must be at least 10px")

    img_path = Path(pages[req.page_index]["original"])
    if not img_path.is_file():
        raise HTTPException(404, f"Original page image not found: page_{req.page_index:03d}")
    try:
        image = read_image(img_path)
        h, w = image.shape[:2]
        if req.x1 < 0 or req.y1 < 0 or req.x2 > w or req.y2 > h:
            raise HTTPException(400, f"Box coordinates ({req.x1},{req.y1})-({req.x2},{req.y2}) exceed image dimensions ({w}x{h})")
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} box {} operation 'update_box' cannot read image: {}", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(500, f"Cannot read page image: {exc}") from exc

    try:
        manifest = pipeline.update_box(
            req.chapter_id, req.page_index, req.box_index, req.x1, req.y1, req.x2, req.y2
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error("Chapter {} page {} box {} operation 'update_box' invalid value: {}", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} box {} operation 'update_box' failed: {}", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(500, f"Update box failed: {exc}") from exc


@router.post("/remove_box")
def remove_box(req: RemoveBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    pages = manifest_raw.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    boxes = pages[req.page_index].get("boxes", [])
    if req.box_index < 0 or req.box_index >= len(boxes):
        raise HTTPException(400, f"Invalid box_index: {req.box_index}")

    try:
        manifest = pipeline.remove_box(req.chapter_id, req.page_index, req.box_index)
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error("Chapter {} page {} box {} operation 'remove_box' invalid value: {}", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} box {} operation 'remove_box' failed: {}", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(500, f"Remove box failed: {exc}") from exc


@router.post("/repaint_mask")
async def repaint_mask(
    chapter_id: str = Form(...),
    page_index: int = Form(...),
    mode: str = Form("standard"),
    mask: UploadFile = File(...),
) -> dict:
    validate_chapter_id(chapter_id)
    mode = str(mode).strip().lower()
    if mode not in {"standard", "lama"}:
        raise HTTPException(400, "Invalid repaint mode")
    manifest_raw = load_manifest_raw(chapter_id)
    pages = manifest_raw.get("pages", [])
    if page_index < 0 or page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {page_index}")

    img_path = Path(pages[page_index]["original"])
    if not img_path.is_file():
        raise HTTPException(404, f"Original page image not found: page_{page_index:03d}")

    try:
        image = read_image(img_path)
        img_h, img_w = image.shape[:2]
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} operation 'repaint_mask' cannot read base image: {}", chapter_id, page_index, exc)
        raise HTTPException(500, f"Cannot read base page image: {exc}") from exc

    mask_bytes = await mask.read()
    if not mask_bytes:
        raise HTTPException(400, "Empty mask payload")
    logger.info(
        "Chapter {} page {}: repaint mask ({} bytes, mode={})",
        chapter_id,
        page_index,
        len(mask_bytes),
        mode,
    )

    encoded = np.frombuffer(mask_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(400, "Invalid repaint mask image: failed to decode")

    if decoded.ndim == 3 and decoded.shape[2] == 4:
        mask_array = decoded[:, :, 3]
    elif decoded.ndim == 3:
        mask_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    else:
        mask_array = decoded

    if mask_array.shape[:2] != (img_h, img_w):
        try:
            mask_array = cv2.resize(mask_array, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        except Exception as exc:
            raise HTTPException(400, f"Mask dimensions {mask_array.shape[:2]} cannot be matched to page dimensions {(img_h, img_w)}") from exc

    if not np.any(mask_array > 0):
        raise HTTPException(400, "Repaint mask is empty")

    try:
        manifest = await run_in_threadpool(
            pipeline.repaint_mask,
            chapter_id,
            page_index,
            mask_array,
            force_lama=mode == "lama",
        )
        return urlify_manifest(manifest)
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} operation 'repaint_mask' failed: {}",
            chapter_id,
            page_index,
            exc,
        )
        raise HTTPException(500, f"Repaint mask failed: {exc}") from exc


@router.post("/reset_manual_mask")
def reset_manual_mask(req: ResetManualMaskRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    pages = manifest_raw.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")

    try:
        manifest = pipeline.reset_manual_mask(req.chapter_id, req.page_index)
        return urlify_manifest(manifest)
    except Exception as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} operation 'reset_manual_mask' failed: {}",
            req.chapter_id, req.page_index, exc,
        )
        raise HTTPException(500, f"Reset manual mask failed: {exc}") from exc


@router.post("/text_object/create")
def create_text_object(req: CreateTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        manifest = pipeline.create_text_object(
            req.chapter_id, req.page_index, req.shape, req.region.model_dump()
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} operation 'create_text_object' invalid value: {}",
            req.chapter_id, req.page_index, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter {} page {} operation 'create_text_object' failed: {}",
            req.chapter_id, req.page_index, exc,
        )
        raise HTTPException(500, f"Create text object failed: {exc}") from exc


@router.post("/text_object/update")
def update_text_object(req: UpdateTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    changes: dict = {}
    if req.shape is not None:
        changes["shape"] = req.shape
    if req.region is not None:
        changes["region"] = req.region.model_dump()
    if req.ocr_text is not None:
        changes["ocr_text"] = req.ocr_text
    if req.translation is not None:
        changes["translation"] = req.translation
    if req.style is not None:
        changes["style"] = req.style.model_dump()
    try:
        manifest = pipeline.update_text_object(
            req.chapter_id, req.page_index, req.id, changes
        )
        if req.ocr_text is not None and req.translation is None:
            manifest = _reconcile_translation_after_ocr_edit(req)
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} object {} operation 'update_text_object' invalid value: {}",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter {} page {} object {} operation 'update_text_object' failed: {}",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(500, f"Update text object failed: {exc}") from exc


@router.post("/text_object/delete")
def delete_text_object(req: DeleteTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        manifest = pipeline.delete_text_object(
            req.chapter_id, req.page_index, req.id
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.opt(exception=True).error(
            "Chapter {} page {} object {} operation 'delete_text_object' invalid value: {}",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter {} page {} object {} operation 'delete_text_object' failed: {}",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(500, f"Delete text object failed: {exc}") from exc


@router.post("/save_draft")
def save_draft(req: SaveDraftRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        with get_manifest_lock(req.chapter_id):
            manifest = load_manifest_raw(req.chapter_id)
            manifest.setdefault("drafts", {}).update(req.drafts)
            affected_pages = set()
            for key in req.drafts.keys():
                parts = str(key).split("_")
                if parts and parts[0].isdigit():
                    page_idx = int(parts[0])
                    if 0 <= page_idx < len(manifest.get("pages", [])):
                        invalidate_page_render(manifest, page_idx)
                        affected_pages.add(page_idx)
            save_manifest_raw(req.chapter_id, manifest)
            if affected_pages:
                pipeline._sync_output_dir(req.chapter_id, manifest, list(affected_pages))
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} operation 'save_draft' failed: {}", req.chapter_id, exc)
        raise HTTPException(500, f"Save draft failed: {exc}") from exc
