from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.media_ai_studio import (
    AIProcessOptions,
    TARGET_LANGUAGES,
    VOICE_CATALOG,
    any_ai_provider_configured,
    is_edge_tts_available,
    is_local_whisper_available,
    is_piper_tts_available,
    list_configured_ai_providers,
)
from app.media_downloader import (
    cancel_media_job,
    delete_media_job,
    find_ffmpeg_binary,
    get_media_job,
    inspect_media_url,
    list_media_jobs,
    resolve_media_job_file,
    resolve_media_job_subtitle,
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
    ai_mode: str = Field(default="none", pattern="^(none|hardsub|dub)$")
    provider_id: str = Field(default="gemini", min_length=1, max_length=64)
    tts_provider_id: str = Field(default="", max_length=64)
    target_lang: str = Field(default="vi", min_length=2, max_length=8)
    voice: str = Field(default="auto", min_length=1, max_length=32)
    burn_subtitles: bool = True
    bilingual: bool = False
    subtitle_style: str = Field(default="classic", pattern="^(classic|box|cinema)$")
    subtitle_animation: str = Field(default="reveal", pattern="^(reveal|karaoke|static)$")
    keep_original_audio: bool = True
    original_volume: float = Field(default=0.16, ge=0.0, le=0.5)
    transcribe_model: str = Field(default="", max_length=200)
    tts_model: str = Field(default="", max_length=200)
    context_hint: str = Field(default="", max_length=500)
    asr_engine: str = Field(default="auto", pattern="^(auto|local_whisper|cloud)$")
    whisper_model: str = Field(
        default="small",
        pattern=r"^(tiny|base|small|medium|large-v3-turbo|distil-large-v3\.5)$",
    )


@router.get("/capabilities")
async def api_media_capabilities():
    providers = list_configured_ai_providers()
    return {
        "ffmpeg_available": bool(find_ffmpeg_binary()),
        "ai_available": any_ai_provider_configured(),
        "local_whisper_available": is_local_whisper_available(),
        "piper_tts_available": is_piper_tts_available(),
        "edge_tts_available": is_edge_tts_available(),
        "ai_providers": providers,
        "ai_voices": list(VOICE_CATALOG),
        "ai_languages": [
            {"code": code, "label": label} for code, label in TARGET_LANGUAGES.items()
        ],
    }


@router.post("/inspect")
async def api_media_inspect(payload: MediaInspectRequest):
    return await asyncio.to_thread(inspect_media_url, payload.url)


@router.post("/download")
async def api_media_download(payload: MediaDownloadRequest):
    ai_opts = AIProcessOptions(
        mode=payload.ai_mode,
        provider_id=payload.provider_id,
        tts_provider_id=payload.tts_provider_id,
        target_lang=payload.target_lang,
        voice=payload.voice,
        burn_subtitles=payload.burn_subtitles,
        bilingual=payload.bilingual,
        subtitle_style=payload.subtitle_style,
        subtitle_animation=payload.subtitle_animation,
        keep_original_audio=payload.keep_original_audio,
        original_volume=payload.original_volume,
        transcribe_model=payload.transcribe_model,
        tts_model=payload.tts_model,
        context_hint=payload.context_hint,
        asr_engine=payload.asr_engine,
        whisper_model=payload.whisper_model,
        video_title=payload.title,
    )
    return await asyncio.to_thread(
        start_media_download,
        payload.url,
        payload.preset,
        title=payload.title,
        is_vertical=payload.is_vertical,
        ai_options=ai_opts,
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


@router.get("/subtitles/{job_id}")
async def api_media_subtitle(job_id: str):
    path, filename = resolve_media_job_subtitle(job_id)
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/x-subrip; charset=utf-8",
    )
