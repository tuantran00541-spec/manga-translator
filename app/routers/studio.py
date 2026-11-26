from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import STUDIO_DIR
from app.security import validate_managed_path
from app.video_studio import (
    VALID_SUB_EXTS,
    VALID_VIDEO_EXTS,
    attach_subtitle_to_project,
    cancel_studio_project,
    cancel_studio_queue,
    create_project_from_local_file,
    create_project_from_url,
    delete_studio_project,
    get_studio_project,
    get_studio_queue_status,
    list_studio_projects,
    resolve_studio_export,
    resolve_studio_source_video,
    resolve_studio_thumbnail,
    start_studio_prepare,
    start_studio_queue,
    start_studio_render,
    synthesize_studio_preview,
    update_studio_project,
    validate_project_id,
)

router = APIRouter(prefix="/api/studio", tags=["video-studio"])

_STREAM_CHUNK_BYTES = 1024 * 1024  # 1 MB streaming chunks for multi-GB movie uploads
_MAX_UPLOAD_VIDEO_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_MAX_SUBTITLE_BYTES = 10 * 1024 * 1024  # 10 MB


class StudioCreateFromUrlRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=240)
    preset: str = Field(default="video_720", pattern="^(video_1080|video_720)$")
    settings: dict[str, Any] | None = None


class StudioUpdateProjectRequest(BaseModel):
    title: str | None = Field(default=None, max_length=240)
    settings: dict[str, Any] | None = None
    speaker_voices: dict[str, str] | None = None
    story_bible: dict[str, Any] | None = None
    segments: list[dict[str, Any]] | None = None


class StudioPrepareRequest(BaseModel):
    auto_render: bool | None = None
    force_retranslate: bool = False


class StudioPreviewTtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    voice: str = Field(default="Puck", max_length=32)
    tts_provider_id: str = Field(default="", max_length=64)
    tts_model: str = Field(default="", max_length=200)
    target_lang: str = Field(default="", max_length=8)


class StudioQueueRequest(BaseModel):
    project_ids: list[str] | None = None
    auto_render: bool = True


async def _stream_upload_to_path(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    total = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await upload.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(413, "Tệp tải lên vượt quá dung lượng tối đa cho phép.")
                fh.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return total


@router.get("/projects")
async def api_studio_list_projects():
    projects = await asyncio.to_thread(list_studio_projects)
    return {"projects": projects, "queue": get_studio_queue_status()}


@router.post("/projects/upload")
async def api_studio_upload_project(
    video: UploadFile = File(...),
    subtitle: UploadFile | None = File(default=None),
    title: str = Form(default=""),
    settings_json: str = Form(default="{}"),
):
    orig_name = str(video.filename or "video.mp4")
    ext = Path(orig_name).suffix.lower()
    if ext not in VALID_VIDEO_EXTS:
        await video.close()
        if subtitle is not None:
            await subtitle.close()
        raise HTTPException(
            400,
            f"Định dạng video '{ext}' chưa hỗ trợ. Chấp nhận: {', '.join(sorted(VALID_VIDEO_EXTS))}",
        )

    parsed_settings: dict[str, Any] = {}
    if settings_json.strip():
        try:
            raw_obj = json.loads(settings_json)
            if isinstance(raw_obj, dict):
                parsed_settings = raw_obj
        except ValueError:
            pass

    pid = uuid.uuid4().hex[:12]
    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    pdir = validate_managed_path(STUDIO_DIR / pid, STUDIO_DIR)
    pdir.mkdir(parents=True, exist_ok=True)

    video_dest = validate_managed_path(pdir / f"source_video{ext}", pdir)
    sub_dest: Path | None = None
    try:
        await _stream_upload_to_path(video, video_dest, _MAX_UPLOAD_VIDEO_BYTES)
        if subtitle is not None and subtitle.filename:
            sub_ext = Path(subtitle.filename).suffix.lower()
            if sub_ext not in VALID_SUB_EXTS:
                await subtitle.close()
                raise HTTPException(
                    400,
                    f"Định dạng phụ đề '{sub_ext}' chưa hỗ trợ. Chấp nhận: {', '.join(sorted(VALID_SUB_EXTS))}",
                )
            sub_dest = validate_managed_path(pdir / f"imported_sub{sub_ext}", pdir)
            await _stream_upload_to_path(subtitle, sub_dest, _MAX_SUBTITLE_BYTES)

        return await asyncio.to_thread(
            create_project_from_local_file,
            video_dest,
            original_filename=orig_name,
            title=title,
            subtitle_path=sub_dest,
            settings=parsed_settings,
            project_id=pid,
        )
    except Exception:
        if not (pdir / "project.json").exists():
            import shutil

            shutil.rmtree(pdir, ignore_errors=True)
        raise


@router.post("/projects/from-url")
async def api_studio_create_from_url(payload: StudioCreateFromUrlRequest):
    return await asyncio.to_thread(
        create_project_from_url,
        payload.url,
        title=payload.title,
        preset=payload.preset,
        settings=payload.settings,
    )


@router.get("/projects/{project_id}")
async def api_studio_get_project(project_id: str):
    return await asyncio.to_thread(get_studio_project, project_id)


@router.patch("/projects/{project_id}")
async def api_studio_update_project(project_id: str, payload: StudioUpdateProjectRequest):
    return await asyncio.to_thread(
        update_studio_project,
        project_id,
        title=payload.title,
        settings=payload.settings,
        speaker_voices=payload.speaker_voices,
        story_bible=payload.story_bible,
        segments=payload.segments,
    )


@router.post("/projects/{project_id}/subtitle")
async def api_studio_upload_subtitle(project_id: str, subtitle: UploadFile = File(...)):
    pid = validate_project_id(project_id)
    pdir = validate_managed_path(STUDIO_DIR / pid, STUDIO_DIR)
    if not (pdir / "project.json").is_file():
        await subtitle.close()
        raise HTTPException(404, "Không tìm thấy dự án Video Studio.")
    sub_ext = Path(str(subtitle.filename or "sub.srt")).suffix.lower()
    if sub_ext not in VALID_SUB_EXTS:
        await subtitle.close()
        raise HTTPException(400, f"Định dạng phụ đề '{sub_ext}' chưa hỗ trợ.")
    sub_dest = validate_managed_path(pdir / f"imported_sub{sub_ext}", pdir)
    await _stream_upload_to_path(subtitle, sub_dest, _MAX_SUBTITLE_BYTES)
    return await asyncio.to_thread(attach_subtitle_to_project, pid, sub_dest)


@router.post("/projects/{project_id}/prepare")
async def api_studio_prepare(project_id: str, payload: StudioPrepareRequest | None = None):
    req = payload or StudioPrepareRequest()
    return await asyncio.to_thread(
        start_studio_prepare,
        project_id,
        auto_render=req.auto_render,
        force_retranslate=req.force_retranslate,
    )


@router.post("/projects/{project_id}/render")
async def api_studio_render(project_id: str):
    return await asyncio.to_thread(start_studio_render, project_id)


@router.post("/projects/{project_id}/preview-tts")
async def api_studio_preview_tts(
    project_id: str,
    payload: StudioPreviewTtsRequest,
    background_tasks: BackgroundTasks,
):
    wav_path = await asyncio.to_thread(
        synthesize_studio_preview,
        project_id,
        text=payload.text,
        voice=payload.voice,
        tts_provider_id=payload.tts_provider_id,
        tts_model=payload.tts_model,
        target_lang=payload.target_lang,
    )
    background_tasks.add_task(wav_path.unlink, True)
    return FileResponse(path=str(wav_path), media_type="audio/wav", filename="preview.wav")


@router.post("/projects/{project_id}/cancel")
async def api_studio_cancel(project_id: str):
    return await asyncio.to_thread(cancel_studio_project, project_id)


@router.delete("/projects/{project_id}")
async def api_studio_delete(project_id: str):
    return await asyncio.to_thread(delete_studio_project, project_id)


@router.get("/projects/{project_id}/source-video")
async def api_studio_source_video(project_id: str):
    path, filename = await asyncio.to_thread(resolve_studio_source_video, project_id)
    return FileResponse(path=str(path), filename=filename)


@router.get("/projects/{project_id}/thumbnail")
async def api_studio_thumbnail(project_id: str):
    path = await asyncio.to_thread(resolve_studio_thumbnail, project_id)
    return FileResponse(path=str(path), media_type="image/jpeg")


@router.get("/projects/{project_id}/export/{kind}")
async def api_studio_export(project_id: str, kind: str):
    path, filename, media_type = await asyncio.to_thread(resolve_studio_export, project_id, kind)
    return FileResponse(path=str(path), filename=filename, media_type=media_type)


@router.get("/queue")
async def api_studio_queue_status():
    return get_studio_queue_status()


@router.post("/queue/start")
async def api_studio_queue_start(payload: StudioQueueRequest | None = None):
    req = payload or StudioQueueRequest()
    return await asyncio.to_thread(
        start_studio_queue,
        req.project_ids,
        auto_render=req.auto_render,
    )


@router.post("/queue/cancel")
async def api_studio_queue_cancel():
    return await asyncio.to_thread(cancel_studio_queue)
