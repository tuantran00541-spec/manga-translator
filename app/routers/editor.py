"""API Router for interactive box editing, OCR, repainting, and draft saving."""

from pathlib import Path
import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.dependencies import ocr, pipeline
from app.logging_config import logger
from app.manifest_utils import (
    get_manifest_lock,
    load_manifest_raw,
    save_manifest_raw,
    urlify_manifest,
)
from app.pipeline import read_image
from app.schemas import (
    AddBoxRequest,
    OcrBoxRequest,
    RemoveBoxRequest,
    SaveDraftRequest,
)
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["editor"])


@router.post("/add_box")
def add_box(req: AddBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.add_manual_box(
        req.chapter_id, req.page_index, req.x1, req.y1, req.x2, req.y2
    )
    return urlify_manifest(manifest)


@router.post("/remove_box")
def remove_box(req: RemoveBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.remove_box(req.chapter_id, req.page_index, req.box_index)
    return urlify_manifest(manifest)


@router.post("/repaint_mask")
async def repaint_mask(
    chapter_id: str = Form(...),
    page_index: int = Form(...),
    mask: UploadFile = File(...),
) -> dict:
    validate_chapter_id(chapter_id)
    mask_bytes = await mask.read()
    logger.info(f"Chapter {chapter_id} page {page_index}: repaint mask ({len(mask_bytes)} bytes)")
    manifest = await run_in_threadpool(pipeline.repaint_mask, chapter_id, page_index, mask_bytes)
    return urlify_manifest(manifest)


@router.post("/ocr_box")
def ocr_box(req: OcrBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = load_manifest_raw(req.chapter_id)
    pages = manifest.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    page = pages[req.page_index]
    boxes = page.get("boxes", [])
    if req.box_index < 0 or req.box_index >= len(boxes):
        raise HTTPException(400, f"Invalid box_index: {req.box_index}")
    box = boxes[req.box_index]

    try:
        image = read_image(Path(page["original"]))
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}") from e

    h, w = image.shape[:2]

    pad = 20
    y1 = max(0, box["y1"] - pad)
    y2 = min(h, box["y2"] + pad)
    x1 = max(0, box["x1"] - pad)
    x2 = min(w, box["x2"] + pad)

    crop = image[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    text = ocr.read(crop_rgb, req.lang)

    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if 0 <= req.page_index < len(manifest["pages"]):
            if 0 <= req.box_index < len(manifest["pages"][req.page_index]["boxes"]):
                manifest["pages"][req.page_index]["boxes"][req.box_index]["ocr_text"] = text
                save_manifest_raw(req.chapter_id, manifest)

    return {"text": text}


@router.post("/save_draft")
def save_draft(req: SaveDraftRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if "drafts" not in manifest:
            manifest["drafts"] = {}
        manifest["drafts"].update(req.drafts)
        save_manifest_raw(req.chapter_id, manifest)
    return {"ok": True}
