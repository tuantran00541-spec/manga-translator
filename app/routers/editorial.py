from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.editorial_qc import (
    SCRIPT_STATUS_DRAFT,
    SCRIPT_STATUS_REVIEWED,
    SCRIPT_STATUS_SKIP,
    build_final_qc_report,
    page_editorial_issues,
    script_review_fingerprint,
)
from app.manifest_utils import get_manifest_lock, load_manifest_raw, save_manifest_raw
from app.security import validate_chapter_id

router = APIRouter(prefix="/api", tags=["editorial"])


class FinalQCPageRequest(BaseModel):
    chapter_id: str
    page_index: int
    approved: bool = True


class CleanReviewPageRequest(BaseModel):
    chapter_id: str
    page_index: int
    approved: bool = True


class ScriptReviewRequest(BaseModel):
    chapter_id: str
    page_index: int
    object_id: str
    status: str


@router.post("/script/review")
def set_script_review(req: ScriptReviewRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    status = str(req.status or "").strip().lower()
    if status not in {SCRIPT_STATUS_DRAFT, SCRIPT_STATUS_REVIEWED, SCRIPT_STATUS_SKIP}:
        raise HTTPException(400, "status must be draft, reviewed, or skip")
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages") or []
        if req.page_index < 0 or req.page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        obj = next(
            (item for item in (pages[req.page_index].get("text_objects") or [])
             if isinstance(item, dict) and str(item.get("id")) == req.object_id),
            None,
        )
        if obj is None:
            raise HTTPException(404, "Text object not found")
        obj["script_status"] = status
        if status == SCRIPT_STATUS_REVIEWED:
            obj["script_review_fingerprint"] = script_review_fingerprint(obj)
        else:
            obj.pop("script_review_fingerprint", None)
        save_manifest_raw(req.chapter_id, manifest)
        return {
            "chapter_id": req.chapter_id,
            "page_index": req.page_index,
            "object_id": req.object_id,
            "status": status,
            "script_review_fingerprint": obj.get("script_review_fingerprint"),
        }


@router.post("/clean_review/page")
def set_clean_review_page(req: CleanReviewPageRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages") or []
        if req.page_index < 0 or req.page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = pages[req.page_index]
        if page.get("skipped"):
            raise HTTPException(409, "Skipped pages do not require clean review approval")
        if req.approved:
            clean_revision = int(page.get("clean_revision") or 0)
            if clean_revision <= 0 or not page.get("clean"):
                raise HTTPException(409, "Trang chưa có ảnh clean để xác nhận")
            page["clean_review_approved_revision"] = clean_revision
        else:
            page.pop("clean_review_approved_revision", None)
        save_manifest_raw(req.chapter_id, manifest)
        return {
            "chapter_id": req.chapter_id,
            "page_index": req.page_index,
            "approved": bool(req.approved),
            "clean_revision": int(page.get("clean_revision") or 0),
        }


@router.get("/final_qc/{chapter_id}")
def final_qc_report(chapter_id: str) -> dict:
    validate_chapter_id(chapter_id)
    with get_manifest_lock(chapter_id):
        manifest = load_manifest_raw(chapter_id)
        return build_final_qc_report(manifest)


@router.post("/final_qc/page")
def set_final_qc_page(req: FinalQCPageRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages") or []
        if req.page_index < 0 or req.page_index >= len(pages):
            raise HTTPException(400, f"Invalid page_index: {req.page_index}")
        page = pages[req.page_index]
        if page.get("skipped"):
            raise HTTPException(409, "Skipped pages do not require final QC approval")

        if req.approved:
            issues = page_editorial_issues(manifest, req.page_index)
            if issues:
                raise HTTPException(
                    409,
                    {
                        "message": "Trang vẫn còn lỗi chặn xuất bản.",
                        "issues": issues,
                    },
                )
            render_revision = int(page.get("render_revision") or 0)
            if render_revision <= 0:
                raise HTTPException(409, "Trang chưa có bản kết xuất hiện hành")
            page["final_qc_approved_render_revision"] = render_revision
        else:
            page.pop("final_qc_approved_render_revision", None)
        save_manifest_raw(req.chapter_id, manifest)
        return build_final_qc_report(manifest)
