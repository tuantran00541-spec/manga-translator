"""API Router for interactive box editing, OCR, repainting, and draft saving."""

from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.dependencies import ocr, pipeline
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.pipeline import _decode_mask, read_image
from app.schemas import AddBoxRequest, OcrBoxRequest, RemoveBoxRequest, ResetManualMaskRequest, SaveDraftRequest, UpdateBoxRequest
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["editor"])

@router.post("/add_box")
def add_box(req: AddBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.add_manual_box(req.chapter_id, req.page_index, req.x1, req.y1, req.x2, req.y2)
    return urlify_manifest(manifest)

@router.post("/update_box")
def update_box(req: UpdateBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    if req.x2 <= req.x1 or req.y2 <= req.y1:
        raise HTTPException(400, "Invalid box dimensions")
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_index < 0 or req.page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = pages[req.page_index]
        boxes = page.get("boxes", [])
        if req.box_index < 0 or req.box_index >= len(boxes):
            raise HTTPException(400, f"Invalid box_index: {req.box_index}")
        try:
            image = read_image(Path(page["original"]))
            height, width = image.shape[:2]
        except Exception as exc:
            raise HTTPException(400, f"Cannot read image: {exc}") from exc
        if req.x1 < 0 or req.y1 < 0 or req.x2 > width or req.y2 > height:
            raise HTTPException(400, "Box is outside image bounds")
        boxes[req.box_index].update({"x1": req.x1, "y1": req.y1, "x2": req.x2, "y2": req.y2})
        save_manifest_raw(req.chapter_id, manifest)
        return urlify_manifest(manifest)

@router.post("/remove_box")
def remove_box(req: RemoveBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.remove_box(req.chapter_id, req.page_index, req.box_index)
    return urlify_manifest(manifest)

@router.post("/repaint_mask")
async def repaint_mask(chapter_id: str = Form(...), page_index: int = Form(...), mask: UploadFile = File(...)) -> dict:
    validate_chapter_id(chapter_id)
    mask_bytes = await mask.read()
    logger.info(f"Chapter {chapter_id} page {page_index}: repaint mask ({len(mask_bytes)} bytes)")

    # The brush canvas is transparent except for painted strokes. Decode the
    # alpha channel rather than grayscale RGB: red paint has grayscale value
    # ~76, which is below the pipeline's >127 mask threshold and therefore
    # turns a perfectly valid brush stroke into an empty mask.
    encoded = np.frombuffer(mask_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(400, "Invalid repaint mask image")

    if decoded.ndim == 3 and decoded.shape[2] == 4:
        mask_array = decoded[:, :, 3]
    elif decoded.ndim == 3:
        # Fallback for opaque RGB images: derive a mask from non-black pixels.
        mask_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    else:
        mask_array = decoded

    if not np.any(mask_array > 0):
        raise HTTPException(400, "Repaint mask is empty")

    manifest = await run_in_threadpool(pipeline.repaint_mask, chapter_id, page_index, mask_array)
    return urlify_manifest(manifest)

@router.post("/reset_manual_mask")
def reset_manual_mask(req: ResetManualMaskRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.reset_manual_mask(req.chapter_id, req.page_index)
    return urlify_manifest(manifest)

def _ocr_crop_from_box(image: np.ndarray, box: dict) -> np.ndarray:
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
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
    return image[max(0, by1 - pad):min(h, by2 + pad), max(0, bx1 - pad):min(w, bx2 + pad)]

@router.post("/ocr_box")
def ocr_box(req: OcrBoxRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = load_manifest_raw(req.chapter_id)
    pages = manifest.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages): raise HTTPException(400, f"Invalid page_index: {req.page_index}")
    page = pages[req.page_index]
    boxes = page.get("boxes", [])
    if req.box_index < 0 or req.box_index >= len(boxes): raise HTTPException(400, f"Invalid box_index: {req.box_index}")
    box = boxes[req.box_index]
    try: image = read_image(Path(page["original"]))
    except Exception as e: raise HTTPException(400, f"Cannot read image: {e}") from e
    text = ocr.read(cv2.cvtColor(_ocr_crop_from_box(image, box), cv2.COLOR_BGR2RGB), req.lang)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if 0 <= req.page_index < len(manifest["pages"]) and 0 <= req.box_index < len(manifest["pages"][req.page_index]["boxes"]):
            target = manifest["pages"][req.page_index]["boxes"][req.box_index]
            target["ocr_text"], target["ocr_lang"] = text, req.lang
            save_manifest_raw(req.chapter_id, manifest)
    return {"text": text, "lang": req.lang}

@router.post("/save_draft")
def save_draft(req: SaveDraftRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        manifest.setdefault("drafts", {}).update(req.drafts)
        save_manifest_raw(req.chapter_id, manifest)
    return {"ok": True}
