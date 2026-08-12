"""API Router for interactive box editing, OCR, repainting, and draft saving."""

from pathlib import Path
import copy
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.dependencies import ocr, pipeline
from app.logging_config import logger
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw, urlify_manifest
from app.pipeline import _decode_mask, read_image
from app.schemas import (
    AddBoxRequest,
    CreateTextObjectRequest,
    DeleteTextObjectRequest,
    OcrBoxRequest,
    OcrTextObjectRequest,
    RemoveBoxRequest,
    ResetManualMaskRequest,
    SaveDraftRequest,
    UpdateBoxRequest,
    UpdateTextObjectRequest,
)
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["editor"])


@router.post("/text_object/create")
def create_text_object(req: CreateTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    pages = manifest_raw.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")

    img_path = Path(pages[req.page_index]["original"])
    if not img_path.is_file():
        raise HTTPException(404, f"Original page image not found: page_{req.page_index:03d}")
    try:
        image = read_image(img_path)
        h, w = image.shape[:2]
        r = req.region
        if r.x1 < 0 or r.y1 < 0 or r.x2 > w or r.y2 > h:
            raise HTTPException(
                400,
                f"Region coordinates ({r.x1},{r.y1})-({r.x2},{r.y2}) exceed image dimensions ({w}x{h})",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Chapter %s page %s operation 'create_text_object' cannot read image: %s",
            req.chapter_id, req.page_index, exc, exc_info=True,
        )
        raise HTTPException(500, f"Cannot read page image: {exc}") from exc

    try:
        manifest = pipeline.create_text_object(
            req.chapter_id,
            req.page_index,
            req.shape,
            req.region.model_dump(),
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error(
            "Chapter %s page %s operation 'create_text_object' invalid value: %s",
            req.chapter_id, req.page_index, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter %s page %s operation 'create_text_object' failed: %s",
            req.chapter_id, req.page_index, exc, exc_info=True,
        )
        raise HTTPException(500, f"Create text object failed: {exc}") from exc


@router.post("/text_object/update")
def update_text_object(req: UpdateTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    manifest_raw = load_manifest_raw(req.chapter_id)
    pages = manifest_raw.get("pages", [])
    if req.page_index < 0 or req.page_index >= len(pages):
        raise HTTPException(400, f"Invalid page_index: {req.page_index}")

    if req.region is not None:
        img_path = Path(pages[req.page_index]["original"])
        if not img_path.is_file():
            raise HTTPException(404, f"Original page image not found: page_{req.page_index:03d}")
        try:
            image = read_image(img_path)
            h, w = image.shape[:2]
            r = req.region
            if r.x1 < 0 or r.y1 < 0 or r.x2 > w or r.y2 > h:
                raise HTTPException(
                    400,
                    f"Region coordinates ({r.x1},{r.y1})-({r.x2},{r.y2}) exceed image dimensions ({w}x{h})",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Chapter %s page %s object %s operation 'update_text_object' cannot read image: %s",
                req.chapter_id, req.page_index, req.id, exc, exc_info=True,
            )
            raise HTTPException(500, f"Cannot read page image: {exc}") from exc

    try:
        manifest = pipeline.update_text_object(
            req.chapter_id,
            req.page_index,
            req.id,
            shape=req.shape,
            region=req.region.model_dump() if req.region is not None else None,
            ocr_text=req.ocr_text,
            translation=req.translation,
            style=req.style.model_dump() if req.style is not None else None,
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error(
            "Chapter %s page %s object %s operation 'update_text_object' invalid value: %s",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter %s page %s object %s operation 'update_text_object' failed: %s",
            req.chapter_id, req.page_index, req.id, exc, exc_info=True,
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
        logger.error(
            "Chapter %s page %s object %s operation 'delete_text_object' invalid value: %s",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter %s page %s object %s operation 'delete_text_object' failed: %s",
            req.chapter_id, req.page_index, req.id, exc, exc_info=True,
        )
        raise HTTPException(500, f"Delete text object failed: {exc}") from exc


@router.post("/text_object/ocr")
async def text_object_ocr(req: OcrTextObjectRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    try:
        manifest = await run_in_threadpool(
            _group_text_object_ocr, req.chapter_id, req.page_index, req.id, req.lang
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error(
            "Chapter %s page %s object %s operation 'text_object_ocr' invalid value: %s",
            req.chapter_id, req.page_index, req.id, exc,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Chapter %s page %s object %s operation 'text_object_ocr' failed: %s",
            req.chapter_id, req.page_index, req.id, exc, exc_info=True,
        )
        raise HTTPException(500, f"Text object OCR failed: {exc}") from exc


def _region_overlaps_box(region: dict, box: dict) -> bool:
    """True when a detection box intersects a text-object region."""
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


def _group_text_object_ocr(chapter_id: str, page_index: int, text_object_id: str, lang: str) -> dict:
    """Group OCR fragments from detection boxes overlapping a text-object region.

    Runs the heavy OCR work outside the manifest lock, then persists under the
    lock. Reuses a box's cached ocr_text when its language matches.
    """
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if page_index < 0 or page_index >= len(pages):
            raise ValueError(f"Invalid page index {page_index}")
        page = pages[page_index]
        obj = next(
            (o for o in (page.get("text_objects") or []) if o.get("id") == text_object_id),
            None,
        )
        if obj is None:
            raise ValueError(f"Text object not found {text_object_id!r}")
        region = obj.get("region") or {}
        boxes = page.get("boxes", [])
        img_path = Path(page["original"])

    image = read_image(img_path)
    overlap = [
        (idx, b) for idx, b in enumerate(boxes)
        if not b.get("removed") and _region_overlaps_box(region, b)
    ]

    fragments = []
    source_boxes = []

    for b_idx, box in overlap:
        source_boxes.append(b_idx)
        cached_text = box.get("ocr_text")
        cached_lang = box.get("ocr_lang")
        if cached_text and cached_lang == lang:
            fragments.append(cached_text.strip())
            continue

        try:

            from app.routers.editor import _ocr_crop_from_box
            crop = _ocr_crop_from_box(image, box)
            text = ocr.read(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), lang)
            if text and text.strip():
                fragments.append(text.strip())
                box["ocr_text"] = text.strip()
                box["ocr_lang"] = lang
        except Exception as exc:
            logger.warning(
                "Chapter %s page %s box %s OCR during text_object group failed: %s",
                chapter_id, page_index, b_idx, exc,
            )

    combined_ocr = "\n".join(f for f in fragments if f)

    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        pages = manifest.get("pages", [])
        if 0 <= page_index < len(pages):
            p = pages[page_index]
            cur_obj = next(
                (o for o in (p.get("text_objects") or []) if o.get("id") == text_object_id),
                None,
            )
            if cur_obj is not None:
                cur_obj["ocr_text"] = combined_ocr
                cur_obj["source_boxes"] = source_boxes
                p["rendered"] = False
                save_manifest_raw(chapter_id, manifest)
                pipeline._sync_output_dir(chapter_id, manifest, [page_index])
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
        logger.error("Chapter %s page %s operation 'add_box' cannot read image: %s", req.chapter_id, req.page_index, exc, exc_info=True)
        raise HTTPException(500, f"Cannot read page image: {exc}") from exc

    try:
        manifest = pipeline.add_manual_box(req.chapter_id, req.page_index, req.x1, req.y1, req.x2, req.y2)
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error("Chapter %s page %s operation 'add_box' invalid value: %s", req.chapter_id, req.page_index, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("Chapter %s page %s operation 'add_box' failed: %s", req.chapter_id, req.page_index, exc, exc_info=True)
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
        logger.error("Chapter %s page %s box %s operation 'update_box' cannot read image: %s", req.chapter_id, req.page_index, req.box_index, exc, exc_info=True)
        raise HTTPException(500, f"Cannot read page image: {exc}") from exc

    try:
        manifest = pipeline.update_box(
            req.chapter_id, req.page_index, req.box_index, req.x1, req.y1, req.x2, req.y2
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        logger.error("Chapter %s page %s box %s operation 'update_box' invalid value: %s", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("Chapter %s page %s box %s operation 'update_box' failed: %s", req.chapter_id, req.page_index, req.box_index, exc, exc_info=True)
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
        logger.error("Chapter %s page %s box %s operation 'remove_box' invalid value: %s", req.chapter_id, req.page_index, req.box_index, exc)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("Chapter %s page %s box %s operation 'remove_box' failed: %s", req.chapter_id, req.page_index, req.box_index, exc, exc_info=True)
        raise HTTPException(500, f"Remove box failed: {exc}") from exc


@router.post("/repaint_mask")
async def repaint_mask(chapter_id: str = Form(...), page_index: int = Form(...), mask: UploadFile = File(...)) -> dict:
    validate_chapter_id(chapter_id)
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
        logger.error("Chapter %s page %s operation 'repaint_mask' cannot read base image: %s", chapter_id, page_index, exc, exc_info=True)
        raise HTTPException(500, f"Cannot read base page image: {exc}") from exc

    mask_bytes = await mask.read()
    if not mask_bytes:
        raise HTTPException(400, "Empty mask payload")
    logger.info(f"Chapter {chapter_id} page {page_index}: repaint mask ({len(mask_bytes)} bytes)")

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
        manifest = await run_in_threadpool(pipeline.repaint_mask, chapter_id, page_index, mask_array)
        return urlify_manifest(manifest)
    except Exception as exc:
        logger.error(
            "Chapter %s page %s operation 'repaint_mask' failed: %s",
            chapter_id,
            page_index,
            exc,
            exc_info=True,
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
        logger.error(
            "Chapter %s page %s operation 'reset_manual_mask' failed: %s",
            req.chapter_id,
            req.page_index,
            exc,
            exc_info=True,
        )
        raise HTTPException(500, f"Reset manual mask failed: {exc}") from exc


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
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_index < 0 or req.page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = pages[req.page_index]
        boxes = page.get("boxes", [])
        if req.box_index < 0 or req.box_index >= len(boxes):
            raise HTTPException(400, f"Invalid box_index: {req.box_index}")
        box_snapshot = copy.deepcopy(boxes[req.box_index])
        img_path = Path(page["original"])

    if not img_path.is_file():
        raise HTTPException(404, f"Original page image not found: page_{req.page_index:03d}")

    try:
        image = read_image(img_path)
        crop = _ocr_crop_from_box(image, box_snapshot)
        text = ocr.read(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), req.lang)
    except Exception as e:
        logger.error(
            "Chapter %s page %s box %s operation 'ocr_box' failed: %s",
            req.chapter_id,
            req.page_index,
            req.box_index,
            e,
            exc_info=True,
        )
        raise HTTPException(500, f"OCR failed: {e}") from e

    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        if (
            0 <= req.page_index < len(manifest.get("pages", []))
            and 0 <= req.box_index < len(manifest["pages"][req.page_index].get("boxes", []))
        ):
            target = manifest["pages"][req.page_index]["boxes"][req.box_index]
            target["ocr_text"] = text
            target["ocr_lang"] = req.lang
            manifest["pages"][req.page_index]["rendered"] = False
            save_manifest_raw(req.chapter_id, manifest)
            pipeline._sync_output_dir(req.chapter_id, manifest, [req.page_index])

    return {"text": text, "lang": req.lang}


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
                        manifest["pages"][page_idx]["rendered"] = False
                        affected_pages.add(page_idx)
            save_manifest_raw(req.chapter_id, manifest)
            if affected_pages:
                pipeline._sync_output_dir(req.chapter_id, manifest, list(affected_pages))
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chapter %s operation 'save_draft' failed: %s", req.chapter_id, exc, exc_info=True)
        raise HTTPException(500, f"Save draft failed: {exc}") from exc
