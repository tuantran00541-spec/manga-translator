from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.manifest_utils import (
    get_manifest_lock,
    invalidate_page_render,
    load_manifest_raw,
    save_manifest_raw,
    urlify_manifest,
)
from app.security import validate_chapter_id
from app.text_objects import ensure_page_text_objects


router = APIRouter(prefix="/api", tags=["automation"])


class EnsureTextObjectsRequest(BaseModel):
    chapter_id: str
    page_indices: list[int] | None = None

    @field_validator("page_indices")
    @classmethod
    def _validate_indices(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if len(value) > 300:
            raise ValueError("Too many page indices")
        if any(index < 0 for index in value):
            raise ValueError("page indices must be non-negative")
        return value


@router.post("/text_objects/ensure")
def ensure_text_objects(req: EnsureTextObjectsRequest) -> dict:
    validate_chapter_id(req.chapter_id)
    with get_manifest_lock(req.chapter_id):
        manifest = load_manifest_raw(req.chapter_id)
        pages = manifest.get("pages", [])
        if req.page_indices is None:
            indices = list(range(len(pages)))
        else:
            indices = sorted(set(req.page_indices))
            invalid = [index for index in indices if index >= len(pages)]
            if invalid:
                raise HTTPException(400, f"Invalid page_index: {invalid[0]}")

        created = 0
        changed_pages: list[int] = []
        for page_index in indices:
            page = pages[page_index]
            if page.get("skipped"):
                continue
            page_created, changed = ensure_page_text_objects(page)
            created += page_created
            if changed:
                invalidate_page_render(manifest, page_index)
                changed_pages.append(page_index)

        if changed_pages:
            save_manifest_raw(req.chapter_id, manifest)

    result = urlify_manifest(manifest)
    result["auto_text_objects"] = {
        "created": created,
        "changed_pages": changed_pages,
    }
    return result
