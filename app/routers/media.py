from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.media_downloader import (
    cancel_media_job,
    delete_media_job,
    get_media_job,
    inspect_media_url,
    list_media_jobs,
    resolve_media_job_file,
    start_media_download,
)

router = APIRouter(prefix="/api/media", tags=["media"])


class MediaInspectRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


class MediaDownloadRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    preset: str = Field(..., pattern="^(video_1080|video_720|audio_only)$")
    title: str = Field(default="", max_length=300)
    is_vertical: bool = False


@router.post("/inspect")
async def api_media_inspect(payload: MediaInspectRequest):
    return await asyncio.to_thread(inspect_media_url, payload.url)


@router.post("/download")
async def api_media_download(payload: MediaDownloadRequest):
    return await asyncio.to_thread(
        start_media_download,
        payload.url,
        payload.preset,
        title=payload.title,
        is_vertical=payload.is_vertical,
    )


@router.get("/jobs")
async def api_media_jobs():
    return {"jobs": list_media_jobs()}


@router.get("/jobs/{job_id}")
async def api_media_job_status(job_id: str):
    return get_media_job(job_id)


@router.post("/jobs/{job_id}/cancel")
async def api_media_job_cancel(job_id: str):
    return cancel_media_job(job_id)


@router.delete("/jobs/{job_id}")
async def api_media_job_delete(job_id: str):
    return delete_media_job(job_id)


@router.get("/files/{job_id}")
async def api_media_file(job_id: str):
    path, filename, media_type = resolve_media_job_file(job_id)
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type=media_type,
    )
