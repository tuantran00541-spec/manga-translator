"""API Router for interactive box editing, OCR, repainting, and draft saving."""

from pathlib import Path
import cv2
import numpy as np
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
from app.pipeline import _decode_mask, read_image
from app.schemas import (
    AddBoxRequest,
    OcrBoxRequest,
    RemoveBoxRequest,
    ResetManualMaskRequest,
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


@router.post("/reset_manual_mask")
def reset_manual_mask(req: ResetManualMaskRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest = pipeline.reset_manual_mask(req.chapter_id, req.page_index)
    return urlify_manifest(manifest)


def _ocr_crop_from_box(image: np.ndarray, box: dict) -> np.ndarray:
    """Crop the original RGB content around the detector mask when it is valid.

    The detector mask is segmentation-derived and is already stored in box-local
    coordinates. Use it to tighten the OCR region without painting or thresholding
    the source pixels themselves. This preserves glyph color, anti-aliasing,
    outlines, and shadows while reducing unrelated artwork/background.
    """
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
    bx1 = max(0, min(w, bx1))
    by1 = max(0, min(h, by1))
    bx2 = max(bx1, min(w, bx2))
    by2 = max(by1, min(h, by2))
    if bx2 <= bx1 or by2 <= by1:
        return image[0:0, 0:0]

    mask = _decode_mask(box.get("mask"))
    expected_shape = (by2 - by1, bx2 - bx1)
    if mask is not None and mask.shape == expected_shape:
        ys, xs = np.where(mask > 127)
        if xs.size and ys.size:
            # Keep a modest margin around the segmentation so glyph outlines,
            # anti-aliasing, and detector edge errors are not clipped.
            mask_pad = 12
            x1 = max(bx1, bx1 + int(xs.min()) - mask_pad)
            y1 = max(by1, by1 + int(ys.min()) - mask_pad)
            x2 = min(bx2, bx1 + int(xs.max()) + 1 + mask_pad)
            y2 = min(by2, by1 + int(ys.max()) + 1 + mask_pad)
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]

    # Invalid/missing masks retain the previous bbox-based behavior.
    pad = 20
    x1 = max(0, bx1 - pad)
    y1 = max(0, by1 - pad)
    x2 = min(w, bx2 + pad)
    y2 = min(h, by2 + pad)
    return image[y1:y2, x1:x2]


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

    crop = _ocr_crop_from_box(image, box)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    text = ocr.read(crop_rgb, req.lang)

    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if 0 <= req.page_index < len(manifest["pages"]):
            if 0 <= req.box_index < len(manifest["pages"][req.page_index]["boxes"]):
                target = manifest["pages"][req.page_index]["boxes"][req.box_index]
                target["ocr_text"] = text
                target["ocr_lang"] = req.lang
                save_manifest_raw(req.chapter_id, manifest)

    return {"text": text, "lang": req.lang}


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
