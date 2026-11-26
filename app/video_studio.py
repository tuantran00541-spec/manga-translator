from __future__ import annotations

from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
import uuid
import wave

from fastapi import HTTPException

from app.config import STUDIO_DIR
from app.logging_config import logger
from app.media_ai_studio import (
    AICanceledError,
    DUB_SAMPLE_RATE,
    EDGE_TTS_PROVIDER_ID,
    LOCAL_WHISPER_DEFAULT_SIZE,
    PIPER_TTS_PROVIDER_ID,
    SLIDING_CONTEXT_WINDOW,
    StoryBibleSchema,
    TARGET_LANGUAGES,
    TRANSLATION_BATCH_SIZE,
    TranslatedCueItemSchema,
    VALID_ASR_ENGINES,
    VALID_SUBTITLE_ANIMATIONS,
    VALID_SUBTITLE_STYLES,
    VALID_VOICE_IDS,
    VALID_WHISPER_MODELS,
    VOICE_CATALOG,
    _assign_speaker_voices,
    _build_story_bible,
    _detect_silence_intervals,
    _prepare_fonts_dir,
    _read_pcm_mono_24k,
    _run_ffmpeg,
    _synthesize_segment_wav,
    _transcribe_local_whisper_words,
    _transcribe_words_gemini_chunk,
    _translate_cue_batch,
    _trim_to_acoustic_onset,
    build_dub_track_wav,
    choose_silence_split_points,
    default_transcribe_model_for,
    default_tts_model_for,
    extract_speech_mp3,
    group_words_into_cues,
    is_edge_tts_available,
    is_local_whisper_available,
    is_piper_tts_available,
    list_configured_ai_providers,
    normalize_model_id,
    parse_platform_subtitle_file,
    probe_media_stream,
    resolve_configured_ai_provider,
    write_ass_file,
    write_srt_file,
)
from app.media_downloader import (
    _fetch_platform_subtitles,
    _find_downloaded_file,
    _format_duration,
    _format_size,
    find_ffmpeg_binary,
    inspect_link_safety,
    sanitize_download_filename,
)
from app.security import validate_managed_path

PROJECT_ID_RE = re.compile(r"^[a-f0-9]{12}$")
VALID_VIDEO_EXTS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts"})
VALID_SUB_EXTS = frozenset({".srt", ".vtt", ".ass", ".json3"})
VALID_OUTPUT_MODES = frozenset({"dub_hardsub", "dub_only", "hardsub_only", "assets_only"})
VALID_RUN_MODES = frozenset({"studio", "auto"})

MAX_STUDIO_DURATION_SEC = 3 * 3600 + 300  # Up to 3 hours 5 minutes
MAX_STUDIO_SEGMENTS = 6000
LONG_FORM_THRESHOLD_SEC = 240.0  # >4 min uses 3-min checkpointed macro chunks
MACRO_CHUNK_TARGET_SEC = 180.0

_ASS_DIALOGUE_RE = re.compile(
    r"^Dialogue:\s*\d+\s*,\s*(\d+):(\d{2}):(\d{2})\.(\d{2})\s*,\s*(\d+):(\d{2}):(\d{2})\.(\d{2})\s*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)$"
)
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")

_RUNTIME_LOCK = threading.RLock()
_ACTIVE_WORKERS: dict[str, dict[str, Any]] = {}  # project_id -> {thread, cancel_event}
_QUEUE_STATE: dict[str, Any] = {
    "running": False,
    "project_ids": [],
    "current_project_id": None,
    "completed_ids": [],
    "failed_ids": [],
    "cancel_event": None,
}


def validate_project_id(project_id: str) -> str:
    pid = str(project_id or "").strip()
    if not PROJECT_ID_RE.match(pid):
        raise HTTPException(400, "Mã dự án Video Studio không hợp lệ.")
    return pid


def _project_dir(project_id: str) -> Path:
    validate_project_id(project_id)
    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return validate_managed_path(STUDIO_DIR / project_id, STUDIO_DIR)


def _project_json_path(project_id: str) -> Path:
    return _project_dir(project_id) / "project.json"


def _default_settings() -> dict[str, Any]:
    providers = list_configured_ai_providers()
    configured = [p["id"] for p in providers if p.get("configured")]
    default_pid = configured[0] if configured else "gemini"
    default_tts = (
        PIPER_TTS_PROVIDER_ID
        if is_piper_tts_available()
        else (EDGE_TTS_PROVIDER_ID if is_edge_tts_available() else default_pid)
    )
    return {
        "provider_id": default_pid,
        "transcribe_model": "",
        "tts_provider_id": default_tts,
        "tts_model": "",
        "target_lang": "vi",
        "default_voice": "Puck",
        "asr_engine": "auto",
        "whisper_model": LOCAL_WHISPER_DEFAULT_SIZE,
        "output_mode": "dub_hardsub",
        "subtitle_style": "box",
        "subtitle_animation": "reveal",
        "bilingual": False,
        "keep_original_audio": True,
        "original_volume": 0.16,
        "context_hint": "",
        "run_mode": "studio",
    }


def _normalize_settings(raw: dict[str, Any] | None, base: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(base or _default_settings())
    if not isinstance(raw, dict):
        return merged

    if "provider_id" in raw and str(raw["provider_id"]).strip():
        merged["provider_id"] = str(raw["provider_id"]).strip()[:64]
    if "transcribe_model" in raw:
        merged["transcribe_model"] = str(raw["transcribe_model"] or "").strip()[:200]
    if "tts_provider_id" in raw:
        merged["tts_provider_id"] = str(raw["tts_provider_id"] or "").strip()[:64]
    if "tts_model" in raw:
        merged["tts_model"] = str(raw["tts_model"] or "").strip()[:200]
    if "target_lang" in raw and raw["target_lang"] in TARGET_LANGUAGES:
        merged["target_lang"] = str(raw["target_lang"])
    if "default_voice" in raw and raw["default_voice"] in VALID_VOICE_IDS:
        merged["default_voice"] = str(raw["default_voice"])
    if "asr_engine" in raw and raw["asr_engine"] in VALID_ASR_ENGINES:
        merged["asr_engine"] = str(raw["asr_engine"])
    if "whisper_model" in raw and raw["whisper_model"] in VALID_WHISPER_MODELS:
        merged["whisper_model"] = str(raw["whisper_model"])
    if "output_mode" in raw and raw["output_mode"] in VALID_OUTPUT_MODES:
        merged["output_mode"] = str(raw["output_mode"])
    if "subtitle_style" in raw and raw["subtitle_style"] in VALID_SUBTITLE_STYLES:
        merged["subtitle_style"] = str(raw["subtitle_style"])
    if "subtitle_animation" in raw and raw["subtitle_animation"] in VALID_SUBTITLE_ANIMATIONS:
        merged["subtitle_animation"] = str(raw["subtitle_animation"])
    if "bilingual" in raw:
        merged["bilingual"] = bool(raw["bilingual"])
    if "keep_original_audio" in raw:
        merged["keep_original_audio"] = bool(raw["keep_original_audio"])
    if "original_volume" in raw:
        try:
            merged["original_volume"] = max(0.0, min(0.5, float(raw["original_volume"])))
        except (TypeError, ValueError):
            pass
    if "context_hint" in raw:
        merged["context_hint"] = str(raw["context_hint"] or "").strip()[:500]
    if "run_mode" in raw and raw["run_mode"] in VALID_RUN_MODES:
        merged["run_mode"] = str(raw["run_mode"])
    return merged


def _load_project(project_id: str) -> dict[str, Any]:
    path = _project_json_path(project_id)
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy dự án Video Studio.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"Lỗi đọc tệp dự án: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(500, "Dữ liệu dự án không hợp lệ.")
    return data


def _save_project(project: dict[str, Any], *, cancel_event: threading.Event | None = None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng ngay lập tức.")
    project_id = project["project_id"]
    path = _project_json_path(project_id)
    project["updated_at"] = time.time()
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(project, ensure_ascii=False, indent=2)
    with _RUNTIME_LOCK:
        if cancel_event is not None and cancel_event.is_set():
            raise AICanceledError("Đã tạm dừng ngay lập tức.")
        if not path.parent.exists():
            raise AICanceledError("Dự án đã bị xóa.")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)


def _update_project_fields(
    project_id: str,
    *,
    cancel_event: threading.Event | None = None,
    **updates: Any,
) -> dict[str, Any]:
    with _RUNTIME_LOCK:
        if cancel_event is not None and cancel_event.is_set():
            raise AICanceledError("Đã tạm dừng ngay lập tức.")
        proj = _load_project(project_id)
        proj.update(updates)
        _save_project(proj, cancel_event=cancel_event)
        return proj


def _extract_thumbnail(ffmpeg_bin: str, video_path: Path, out_jpg: Path, duration_sec: float) -> bool:
    seek_t = min(max(0.5, duration_sec * 0.12), 15.0) if duration_sec > 2.0 else 0.0
    code, _ = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{seek_t:.2f}",
            "-i",
            str(video_path),
            "-vf",
            "scale=480:-2",
            "-frames:v",
            "1",
            "-q:v",
            "4",
            str(out_jpg),
        ],
        timeout=25.0,
    )
    return code == 0 and out_jpg.is_file() and out_jpg.stat().st_size > 256


def parse_studio_subtitle_file(sub_path: Path) -> list[dict[str, Any]]:
    """Parse .srt, .vtt, .json3, OR .ass subtitle files into Studio segment dicts."""
    if not sub_path or not sub_path.is_file():
        return []
    if sub_path.suffix.lower() != ".ass":
        return parse_platform_subtitle_file(sub_path)

    cues: list[dict[str, Any]] = []
    for raw_line in sub_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ASS_DIALOGUE_RE.match(raw_line.strip())
        if not m:
            continue
        st = int(m.group(1)) * 3600.0 + int(m.group(2)) * 60.0 + int(m.group(3)) + int(m.group(4)) / 100.0
        en = int(m.group(5)) * 3600.0 + int(m.group(6)) * 60.0 + int(m.group(7)) + int(m.group(8)) / 100.0
        body = _ASS_OVERRIDE_RE.sub("", m.group(9))
        body = body.replace(r"\N", " ").replace(r"\n", " ")
        body = re.sub(r"\s+", " ", body).strip()
        if not body or en <= st:
            continue
        cues.append(
            {
                "id": len(cues),
                "start_sec": round(st, 3),
                "end_sec": round(max(st + 0.25, en), 3),
                "speaker": "S1",
                "gender": "female",
                "tone": "natural and clear",
                "original_text": body,
                "translated_text": body,
                "voice_override": "",
                "words": [],
            }
        )
    cues.sort(key=lambda c: (c["start_sec"], c["end_sec"]))
    for i, c in enumerate(cues):
        c["id"] = i
        if i + 1 < len(cues) and c["end_sec"] > cues[i + 1]["start_sec"]:
            c["end_sec"] = round(max(c["start_sec"] + 0.15, cues[i + 1]["start_sec"] - 0.02), 3)
    return cues[:MAX_STUDIO_SEGMENTS]


def _normalize_studio_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for seg in raw_segments[:MAX_STUDIO_SEGMENTS]:
        if not isinstance(seg, dict):
            continue
        try:
            st = max(0.0, float(seg.get("start_sec", 0.0)))
            en = max(st + 0.15, float(seg.get("end_sec", st + 1.0)))
        except (TypeError, ValueError):
            continue
        orig = re.sub(r"\s+", " ", str(seg.get("original_text") or "")).strip()
        tr = re.sub(r"\s+", " ", str(seg.get("translated_text") or "")).strip()
        if not orig and not tr:
            continue
        spk = re.sub(r"[^A-Za-z0-9_-]", "", str(seg.get("speaker") or "S1"))[:16] or "S1"
        gender = "male" if str(seg.get("gender") or "").strip().lower().startswith("m") else "female"
        tone = re.sub(r"\s+", " ", str(seg.get("tone") or "natural and clear")).strip()[:80] or "natural and clear"
        voice_ov = str(seg.get("voice_override") or "").strip()
        if voice_ov not in VALID_VOICE_IDS:
            voice_ov = ""
        words = seg.get("words") if isinstance(seg.get("words"), list) else []
        cleaned.append(
            {
                "id": len(cleaned),
                "start_sec": round(st, 3),
                "end_sec": round(en, 3),
                "speaker": spk,
                "gender": gender,
                "tone": tone,
                "original_text": orig or tr,
                "translated_text": tr or orig,
                "voice_override": voice_ov,
                "words": words,
            }
        )
    cleaned.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    for i, c in enumerate(cleaned):
        c["id"] = i
        if i + 1 < len(cleaned) and c["end_sec"] > cleaned[i + 1]["start_sec"]:
            c["end_sec"] = round(max(c["start_sec"] + 0.15, cleaned[i + 1]["start_sec"] - 0.02), 3)
    return cleaned


def _project_summary(proj: dict[str, Any], *, include_segments: bool = False) -> dict[str, Any]:
    pid = proj["project_id"]
    pdir = _project_dir(pid)
    with _RUNTIME_LOCK:
        is_active = pid in _ACTIVE_WORKERS

    status = proj.get("status", "idle")
    # If app restarted mid-run, surface cleanly as paused so user can press Resume!
    if status in ("ingesting", "running_prepare", "running_render") and not is_active:
        status = "paused"

    ck = proj.get("checkpoints") or {}
    outs = ck.get("rendered_outputs") or {}

    def _export_url(kind: str, rel_name: str | None) -> str | None:
        if not rel_name:
            return None
        cand = pdir / rel_name
        return f"/api/studio/projects/{pid}/export/{kind}" if cand.is_file() else None

    video_rel = proj.get("video_file")
    has_video = bool(video_rel and (pdir / video_rel).is_file())
    thumb_rel = proj.get("thumbnail_file")
    has_thumb = bool(thumb_rel and (pdir / thumb_rel).is_file())

    segs = proj.get("segments") or []
    speakers = sorted({str(s.get("speaker") or "S1") for s in segs}) or ["S1"]

    summary: dict[str, Any] = {
        "project_id": pid,
        "title": proj.get("title") or "Dự án Video",
        "source_type": proj.get("source_type") or "upload",
        "source_url": proj.get("source_url") or "",
        "status": status,
        "stage": proj.get("stage") or "idle",
        "stage_detail": proj.get("stage_detail") or "",
        "percent": round(float(proj.get("percent") or 0.0), 1),
        "error": proj.get("error"),
        "duration_sec": round(float(proj.get("duration_sec") or 0.0), 2),
        "duration_text": _format_duration(proj.get("duration_sec")),
        "width": int(proj.get("width") or 0),
        "height": int(proj.get("height") or 0),
        "file_size": proj.get("file_size"),
        "file_size_text": _format_size(proj.get("file_size")),
        "has_source_video": has_video,
        "source_video_url": f"/api/studio/projects/{pid}/source-video" if has_video else None,
        "thumbnail_url": f"/api/studio/projects/{pid}/thumbnail" if has_thumb else None,
        "segment_count": len(segs),
        "speakers": speakers,
        "settings": proj.get("settings") or _default_settings(),
        "speaker_voices": proj.get("speaker_voices") or {},
        "story_bible": proj.get("story_bible") or {},
        "checkpoints": {
            "video_ready": has_video,
            "asr_done": bool(ck.get("asr_done")),
            "asr_completed_chunks": int(ck.get("asr_completed_chunks") or 0),
            "asr_total_chunks": int(ck.get("asr_total_chunks") or 0),
            "bible_done": bool(ck.get("bible_done")),
            "translated_batches": int(ck.get("translated_batches") or 0),
            "total_batches": int(ck.get("total_batches") or 0),
            "translation_done": bool(ck.get("translation_done")),
            "render_done": bool(outs.get("video_file") or outs.get("audio_file") or outs.get("srt_file")),
        },
        "exports": {
            "video_url": _export_url("video", outs.get("video_file")),
            "video_filename": outs.get("video_filename"),
            "srt_url": _export_url("srt", outs.get("srt_file")),
            "srt_filename": outs.get("srt_filename"),
            "ass_url": _export_url("ass", outs.get("ass_file")),
            "ass_filename": outs.get("ass_filename"),
            "audio_url": _export_url("audio", outs.get("audio_file")),
            "audio_filename": outs.get("audio_filename"),
        },
        "created_at": proj.get("created_at"),
        "updated_at": proj.get("updated_at"),
        "active": is_active,
    }
    if include_segments:
        # Strip word-level sub-arrays from UI payload to keep 3,000-line movies snappy in browser!
        summary["segments"] = [
            {k: v for k, v in s.items() if k != "words"}
            for s in segs
        ]
    return summary


def list_studio_projects() -> list[dict[str, Any]]:
    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for child in STUDIO_DIR.iterdir():
        if not child.is_dir() or child.is_symlink() or not PROJECT_ID_RE.match(child.name):
            continue
        pjson = child / "project.json"
        if not pjson.is_file():
            continue
        try:
            proj = json.loads(pjson.read_text(encoding="utf-8"))
            items.append(_project_summary(proj, include_segments=False))
        except Exception:
            continue
    items.sort(key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0), reverse=True)
    return items


def get_studio_project(project_id: str) -> dict[str, Any]:
    proj = _load_project(project_id)
    return _project_summary(proj, include_segments=True)


def create_project_from_local_file(
    video_path: Path,
    *,
    original_filename: str,
    title: str = "",
    subtitle_path: Path | None = None,
    settings: dict[str, Any] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Finalize a streamed local video upload into a persistent Studio project."""
    pid = validate_project_id(project_id) if project_id else uuid.uuid4().hex[:12]
    pdir = _project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    safe_video = validate_managed_path(video_path, pdir)
    if not safe_video.is_file() or safe_video.stat().st_size == 0:
        raise HTTPException(400, "Tệp video tải lên rỗng hoặc không hợp lệ.")

    ffmpeg_bin = find_ffmpeg_binary()
    if not ffmpeg_bin:
        raise HTTPException(503, "Thiếu FFmpeg trên máy chủ để phân tích video.")

    probe = probe_media_stream(ffmpeg_bin, safe_video)
    if probe.duration_sec > MAX_STUDIO_DURATION_SEC:
        safe_video.unlink(missing_ok=True)
        raise HTTPException(400, "Video vượt quá giới hạn tối đa 3 giờ của Xưởng Video AI.")

    thumb_path = pdir / "thumb.jpg"
    has_thumb = _extract_thumbnail(ffmpeg_bin, safe_video, thumb_path, probe.duration_sec)

    imported_cues: list[dict[str, Any]] = []
    if subtitle_path is not None and subtitle_path.is_file():
        imported_cues = _normalize_studio_segments(parse_studio_subtitle_file(subtitle_path))

    clean_title = str(title or "").strip() or Path(original_filename or "video").stem
    now = time.time()
    norm_settings = _normalize_settings(settings)

    project: dict[str, Any] = {
        "project_id": pid,
        "title": clean_title[:240],
        "source_type": "upload",
        "source_url": "",
        "video_file": safe_video.name,
        "thumbnail_file": thumb_path.name if has_thumb else None,
        "duration_sec": round(probe.duration_sec, 3),
        "width": probe.width,
        "height": probe.height,
        "has_audio": probe.has_audio,
        "file_size": safe_video.stat().st_size,
        "status": "ready_to_edit" if imported_cues else "idle",
        "stage": "asr_imported" if imported_cues else "ready",
        "stage_detail": (
            f"Đã nạp sẵn {len(imported_cues)} câu phụ đề — sẵn sàng dịch hoặc biên tập."
            if imported_cues
            else "Đã nạp video thành công — bấm Bắt đầu để bóc băng & dịch."
        ),
        "percent": 35.0 if imported_cues else 0.0,
        "error": None,
        "settings": norm_settings,
        "speaker_voices": {},
        "story_bible": {},
        "segments": imported_cues,
        "checkpoints": {
            "asr_done": bool(imported_cues),
            "asr_completed_chunks": 1 if imported_cues else 0,
            "asr_total_chunks": 1 if imported_cues else 0,
            "asr_chunk_words": {},
            "bible_done": False,
            "translated_ids": [],
            "translated_batches": 0,
            "total_batches": 0,
            "translation_done": False,
            "rendered_outputs": {},
        },
        "created_at": now,
        "updated_at": now,
    }
    _save_project(project)
    return _project_summary(project, include_segments=True)


def create_project_from_url(
    raw_url: str,
    *,
    title: str = "",
    preset: str = "video_720",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate public HTTPS link and create a Studio project (downloads on Prepare)."""
    safety = inspect_link_safety(raw_url, resolve_redirects=True)
    if not safety["safe"]:
        raise HTTPException(400, safety["summary"])

    pid = uuid.uuid4().hex[:12]
    pdir = _project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    target_url = safety["resolved_url"] or safety["normalized_url"]
    platform_label = str(safety.get("platform_label") or safety.get("hostname") or "Web Video")
    now = time.time()
    norm_settings = _normalize_settings(settings)
    norm_settings["url_preset"] = preset if preset in ("video_1080", "video_720") else "video_720"

    project: dict[str, Any] = {
        "project_id": pid,
        "title": (str(title or "").strip() or f"Video {platform_label}")[:240],
        "source_type": "url",
        "source_url": target_url,
        "video_file": None,
        "thumbnail_file": None,
        "duration_sec": 0.0,
        "width": 0,
        "height": 0,
        "has_audio": True,
        "file_size": None,
        "status": "idle",
        "stage": "pending_download",
        "stage_detail": f"Liên kết {platform_label} an toàn — bấm Bắt đầu để tải & xử lý.",
        "percent": 0.0,
        "error": None,
        "settings": norm_settings,
        "speaker_voices": {},
        "story_bible": {},
        "segments": [],
        "checkpoints": {
            "asr_done": False,
            "asr_completed_chunks": 0,
            "asr_total_chunks": 0,
            "asr_chunk_words": {},
            "bible_done": False,
            "translated_ids": [],
            "translated_batches": 0,
            "total_batches": 0,
            "translation_done": False,
            "rendered_outputs": {},
        },
        "created_at": now,
        "updated_at": now,
    }
    _save_project(project)
    return _project_summary(project, include_segments=True)


def attach_subtitle_to_project(project_id: str, sub_path: Path) -> dict[str, Any]:
    with _RUNTIME_LOCK:
        if project_id in _ACTIVE_WORKERS:
            raise HTTPException(409, "Dự án đang xử lý, hãy dừng trước khi nạp phụ đề mới.")
        proj = _load_project(project_id)
        cues = _normalize_studio_segments(parse_studio_subtitle_file(sub_path))
        if not cues:
            raise HTTPException(400, "Không đọc được câu phụ đề hợp lệ nào từ tệp đã tải lên.")
        proj["segments"] = cues
        ck = proj.setdefault("checkpoints", {})
        ck["asr_done"] = True
        ck["asr_completed_chunks"] = 1
        ck["asr_total_chunks"] = 1
        ck["translated_ids"] = []
        ck["translated_batches"] = 0
        ck["translation_done"] = False
        proj["status"] = "ready_to_edit"
        proj["stage"] = "asr_imported"
        proj["stage_detail"] = f"Đã nạp {len(cues)} câu phụ đề từ tệp — sẵn sàng dịch hoặc chỉnh sửa."
        proj["percent"] = 35.0
        proj["error"] = None
        _save_project(proj)
        return _project_summary(proj, include_segments=True)


def update_studio_project(
    project_id: str,
    *,
    title: str | None = None,
    settings: dict[str, Any] | None = None,
    speaker_voices: dict[str, str] | None = None,
    story_bible: dict[str, Any] | None = None,
    segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with _RUNTIME_LOCK:
        if project_id in _ACTIVE_WORKERS:
            raise HTTPException(409, "Dự án đang chạy nền, hãy tạm dừng trước khi lưu thay đổi.")
        proj = _load_project(project_id)
        if title is not None and title.strip():
            proj["title"] = title.strip()[:240]
        if settings is not None:
            proj["settings"] = _normalize_settings(settings, proj.get("settings"))
        if speaker_voices is not None:
            valid_map: dict[str, str] = {}
            for spk, vid in speaker_voices.items():
                clean_spk = re.sub(r"[^A-Za-z0-9_-]", "", str(spk or ""))[:16]
                if clean_spk and vid in VALID_VOICE_IDS:
                    valid_map[clean_spk] = vid
            proj["speaker_voices"] = valid_map
        if story_bible is not None and isinstance(story_bible, dict):
            proj["story_bible"] = {
                "source_language": str(story_bible.get("source_language") or "unknown")[:40],
                "genre": str(story_bible.get("genre") or "general")[:40],
                "synopsis": str(story_bible.get("synopsis") or "")[:600],
                "glossary": [str(x).strip()[:120] for x in (story_bible.get("glossary") or []) if str(x).strip()][:30],
                "pronoun_rules": [str(x).strip()[:160] for x in (story_bible.get("pronoun_rules") or []) if str(x).strip()][:20],
                "style_notes": str(story_bible.get("style_notes") or "")[:400],
            }
        if segments is not None:
            existing_words = {
                int(s["id"]): s.get("words") or []
                for s in (proj.get("segments") or [])
                if isinstance(s, dict) and "id" in s
            }
            merged_in: list[dict[str, Any]] = []
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                copy_seg = dict(seg)
                if not copy_seg.get("words") and copy_seg.get("id") in existing_words:
                    copy_seg["words"] = existing_words[copy_seg["id"]]
                merged_in.append(copy_seg)
            proj["segments"] = _normalize_studio_segments(merged_in)
            ck = proj.setdefault("checkpoints", {})
            if proj["segments"]:
                ck["asr_done"] = True
                if proj.get("status") == "idle":
                    proj["status"] = "ready_to_edit"
        _save_project(proj)
        return _project_summary(proj, include_segments=True)


def delete_studio_project(project_id: str) -> dict[str, Any]:
    validate_project_id(project_id)
    with _RUNTIME_LOCK:
        active = _ACTIVE_WORKERS.pop(project_id, None)
        if active is not None:
            active["cancel_event"].set()
    pdir = _project_dir(project_id)
    if pdir.exists():
        for _ in range(4):
            shutil.rmtree(pdir, ignore_errors=True)
            if not pdir.exists():
                break
            time.sleep(0.08)
    return {"ok": True, "project_id": project_id}


def cancel_studio_project(project_id: str) -> dict[str, Any]:
    """Immediately stop/detach the active worker, preserve last saved Checkpoint, and mark paused."""
    validate_project_id(project_id)
    with _RUNTIME_LOCK:
        active = _ACTIVE_WORKERS.pop(project_id, None)
        if active is not None:
            active["cancel_event"].set()
        proj = _load_project(project_id)
        if active is not None or proj.get("status") in ("ingesting", "running_prepare", "running_render"):
            has_segs = bool(proj.get("segments"))
            proj["status"] = "paused"
            proj["stage"] = "paused"
            proj["stage_detail"] = (
                "Đã tạm dừng ngay — Checkpoint đã được lưu (bấm Tiếp tục bất kỳ lúc nào)."
                if has_segs
                else "Đã tạm dừng ngay."
            )
            proj["error"] = None
            _save_project(proj)
        return _project_summary(proj, include_segments=True)


def _ingest_url_source(
    project_id: str,
    proj: dict[str, Any],
    pdir: Path,
    ffmpeg_bin: str,
    cancel_event: threading.Event,
) -> tuple[Path, Path | None]:
    """Download URL source into project folder with progress updates + optional platform captions."""
    import yt_dlp  # type: ignore

    target_url = proj["source_url"]
    preset = (proj.get("settings") or {}).get("url_preset", "video_720")
    max_h = 1080 if preset == "video_1080" else 720
    fmt = f"bestvideo[height<={max_h}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_h}]+bestaudio/best[height<={max_h}]/best"

    def _hook(d: dict[str, Any]) -> None:
        if cancel_event.is_set():
            raise AICanceledError("Đã tạm dừng tải video.")
        if d.get("status") == "downloading":
            dl = int(d.get("downloaded_bytes") or 0)
            tot = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
            pct = min(14.0, (dl / tot) * 14.0) if tot > 0 else 4.0
            _update_project_fields(
                project_id,
                cancel_event=cancel_event,
                stage="downloading",
                percent=round(pct, 1),
                stage_detail=f"Đang tải video nguồn ({_format_size(dl) or ''})…",
            )

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "playlist_items": "1",
        "socket_timeout": 25,
        "retries": 3,
        "fragment_retries": 3,
        "restrictfilenames": True,
        "enable_file_urls": False,
        "format": fmt,
        "ffmpeg_location": ffmpeg_bin,
        "merge_output_format": "mp4",
        "outtmpl": str(pdir / "source_video.%(ext)s"),
        "progress_hooks": [_hook],
    }
    _update_project_fields(project_id, cancel_event=cancel_event, status="ingesting", stage="downloading", percent=2.0, stage_detail="Đang tải video từ liên kết…")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target_url, download=True)
    if cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng tải video.")

    found = _find_downloaded_file(pdir)
    if found is None or found.suffix.lower() not in VALID_VIDEO_EXTS:
        # Look explicitly for source_video.*
        matches = [p for p in pdir.glob("source_video.*") if p.is_file() and p.suffix.lower() in VALID_VIDEO_EXTS]
        found = matches[0] if matches else found
    if found is None or not found.is_file():
        raise RuntimeError("Không tìm thấy tệp video sau khi tải từ liên kết.")

    probe = probe_media_stream(ffmpeg_bin, found)
    if probe.duration_sec > MAX_STUDIO_DURATION_SEC:
        found.unlink(missing_ok=True)
        raise RuntimeError("Video vượt quá giới hạn tối đa 3 giờ của Xưởng Video AI.")

    thumb_path = pdir / "thumb.jpg"
    has_thumb = _extract_thumbnail(ffmpeg_bin, found, thumb_path, probe.duration_sec)
    info_title = str(info.get("title") or "").strip() if isinstance(info, dict) else ""

    sub_path: Path | None = None
    if (proj.get("settings") or {}).get("asr_engine", "auto") == "auto":
        sub_path = _fetch_platform_subtitles(target_url, pdir, info)

    proj["video_file"] = found.name
    proj["thumbnail_file"] = thumb_path.name if has_thumb else None
    proj["duration_sec"] = round(probe.duration_sec, 3)
    proj["width"] = probe.width
    proj["height"] = probe.height
    proj["has_audio"] = probe.has_audio
    proj["file_size"] = found.stat().st_size
    if info_title and proj["title"].startswith("Video "):
        proj["title"] = info_title[:240]
    proj["percent"] = 15.0
    _save_project(proj, cancel_event=cancel_event)
    return found, sub_path


def _run_long_form_asr(
    project_id: str,
    proj: dict[str, Any],
    pdir: Path,
    video_path: Path,
    ffmpeg_bin: str,
    cancel_event: threading.Event,
    platform_sub_path: Path | None = None,
) -> None:
    """Resumable short/long-form ASR: 1-pass for <=4min; 3-min silence-aligned checkpointed macro chunks up to 3h."""
    ck = proj.setdefault("checkpoints", {})
    if ck.get("asr_done") and proj.get("segments"):
        return

    settings = proj.get("settings") or _default_settings()
    engine = settings.get("asr_engine", "auto")
    whisper_model = settings.get("whisper_model", LOCAL_WHISPER_DEFAULT_SIZE)
    duration_sec = float(proj.get("duration_sec") or 0.0)
    if duration_sec <= 0:
        probe = probe_media_stream(ffmpeg_bin, video_path)
        duration_sec = probe.duration_sec
        proj["duration_sec"] = round(duration_sec, 3)

    # 1. Free 0-quota platform captions if available in 'auto' mode
    if engine == "auto" and platform_sub_path is not None and platform_sub_path.is_file():
        caption_cues = _normalize_studio_segments(parse_studio_subtitle_file(platform_sub_path))
        if caption_cues:
            proj["segments"] = caption_cues
            ck["asr_done"] = True
            ck["asr_completed_chunks"] = 1
            ck["asr_total_chunks"] = 1
            proj["percent"] = 40.0
            proj["stage_detail"] = f"Đã tận dụng {len(caption_cues)} câu phụ đề gốc từ nguồn (0 quota ASR)."
            _save_project(proj, cancel_event=cancel_event)
            return

    use_local = engine == "local_whisper" or (engine == "auto" and is_local_whisper_available())
    if use_local and not is_local_whisper_available():
        raise RuntimeError("Chưa cài thư viện faster-whisper để chạy Local Whisper CPU.")

    def _set_status(pct: float, msg: str) -> None:
        proj["stage"] = "asr"
        proj["percent"] = round(pct, 1)
        proj["stage_detail"] = msg
        _save_project(proj, cancel_event=cancel_event)

    # Short / medium clip (<= 4 minutes): run directly in 1 blazing-fast pass!
    if duration_sec <= LONG_FORM_THRESHOLD_SEC and use_local:
        _set_status(20.0, f"Whisper ({whisper_model}) đang bóc tách từng từ & căn âm học <10ms…")
        src_lang, cues = _transcribe_local_whisper_words(
            ffmpeg_bin,
            video_path,
            model_size=whisper_model,
            cancel_event=cancel_event,
            on_status=lambda msg: _set_status(30.0, msg),
        )
        if not cues:
            raise RuntimeError("Không phát hiện được lời thoại nào trong video.")
        proj["segments"] = _normalize_studio_segments(cues)
        if src_lang and src_lang != "unknown":
            proj.setdefault("story_bible", {})["source_language"] = src_lang
        ck["asr_done"] = True
        ck["asr_completed_chunks"] = 1
        ck["asr_total_chunks"] = 1
        proj["percent"] = 45.0
        proj["stage_detail"] = f"Hoàn tất bóc băng {len(proj['segments'])} câu chuẩn <10ms."
        _save_project(proj, cancel_event=cancel_event)
        return

    # Long-form video (> 4 min, up to 3 hours!) OR Cloud ASR: silence-aligned checkpointed macro chunks!
    _set_status(17.0, "Đang quét khoảng lặng để chia phân đoạn an toàn cho video dài…")
    macro_plan = ck.get("asr_macro_plan")
    if not isinstance(macro_plan, list) or not macro_plan:
        target_c = MACRO_CHUNK_TARGET_SEC if use_local else 40.0
        min_c = 120.0 if use_local else 26.0
        max_c = 220.0 if use_local else 52.0
        silences = _detect_silence_intervals(ffmpeg_bin, video_path, cancel_event=cancel_event)
        ranges = choose_silence_split_points(
            duration_sec,
            silences,
            target_chunk_sec=target_c,
            min_chunk_sec=min_c,
            max_chunk_sec=max_c,
        )
        macro_plan = [[round(a, 3), round(b, 3)] for a, b in ranges]
        ck["asr_macro_plan"] = macro_plan
        ck["asr_total_chunks"] = len(macro_plan)
        ck.setdefault("asr_chunk_words", {})
        _save_project(proj, cancel_event=cancel_event)

    chunk_words_map: dict[str, list[dict[str, Any]]] = ck.setdefault("asr_chunk_words", {})
    total_chunks = len(macro_plan)
    chunks_dir = pdir / "asr_macro_tmp"
    chunks_dir.mkdir(exist_ok=True)
    detected_lang = str((proj.get("story_bible") or {}).get("source_language") or "unknown")

    provider = None
    api_key = ""
    if not use_local:
        provider, api_key = resolve_configured_ai_provider(settings.get("provider_id", "gemini"))

    try:
        for idx, (c_start, c_end) in enumerate(macro_plan):
            key = str(idx)
            if key in chunk_words_map:
                continue
            if cancel_event.is_set():
                raise AICanceledError(f"Đã tạm dừng ở phân đoạn ASR {idx}/{total_chunks} (Checkpoint đã lưu).")

            pct = 18.0 + (27.0 * idx / max(1, total_chunks))
            _set_status(
                pct,
                f"Đang bóc băng đoạn {idx + 1}/{total_chunks} ({_format_duration(c_start)} → {_format_duration(c_end)})…",
            )
            slice_ext = ".wav" if use_local else ".mp3"
            slice_path = chunks_dir / f"macro_{idx:04d}{slice_ext}"
            seg_dur = max(0.5, float(c_end) - float(c_start))
            ff_args = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{c_start:.3f}",
                "-t",
                f"{seg_dur:.3f}",
                "-i",
                str(video_path),
                "-vn",
            ]
            if use_local:
                ff_args += ["-ac", "2", "-ar", "16000", "-sample_fmt", "s16", str(slice_path)]
            else:
                ff_args += ["-ac", "1", "-ar", "16000", "-b:a", "48k", str(slice_path)]
            code, stderr = _run_ffmpeg(ff_args, cancel_event=cancel_event, timeout=120.0)
            if code != 0 or not slice_path.is_file():
                raise RuntimeError(f"Không tách được phân đoạn âm thanh {idx + 1}: {stderr[:160]}")

            words_for_chunk: list[dict[str, Any]] = []
            try:
                if use_local:
                    c_lang, c_cues = _transcribe_local_whisper_words(
                        ffmpeg_bin,
                        slice_path,
                        model_size=whisper_model,
                        cancel_event=cancel_event,
                    )
                    if detected_lang == "unknown" and c_lang and c_lang != "unknown":
                        detected_lang = c_lang
                    for cue in c_cues:
                        c_words = cue.get("words") or [
                            {"start_sec": cue["start_sec"], "end_sec": cue["end_sec"], "text": cue["original_text"]}
                        ]
                        for w in c_words:
                            words_for_chunk.append(
                                {
                                    "start_sec": round(float(c_start) + float(w["start_sec"]), 3),
                                    "end_sec": round(float(c_start) + float(w["end_sec"]), 3),
                                    "text": str(w["text"]),
                                    "speaker": str(cue.get("speaker") or "S1"),
                                }
                            )
                else:
                    assert provider is not None
                    if provider.protocol == "gemini":
                        words_for_chunk = _transcribe_words_gemini_chunk(
                            slice_path,
                            offset_sec=float(c_start),
                            chunk_dur=seg_dur,
                            api_key=api_key,
                        )
                    else:
                        raise RuntimeError(
                            f"{provider.label} không hỗ trợ Cloud ASR cho video dài — hãy chọn Local Whisper CPU (0 quota)."
                        )
            finally:
                slice_path.unlink(missing_ok=True)

            chunk_words_map[key] = words_for_chunk
            ck["asr_completed_chunks"] = len(chunk_words_map)
            _save_project(proj, cancel_event=cancel_event)
    finally:
        shutil.rmtree(chunks_dir, ignore_errors=True)

    all_words: list[dict[str, Any]] = []
    for idx in range(total_chunks):
        all_words.extend(chunk_words_map.get(str(idx), []))

    cues = _group_long_form_words(all_words)
    if not cues:
        raise RuntimeError("Không phát hiện lời thoại nào trong video.")

    proj["segments"] = _normalize_studio_segments(cues)
    if detected_lang and detected_lang != "unknown":
        proj.setdefault("story_bible", {})["source_language"] = detected_lang
    ck["asr_done"] = True
    ck["asr_chunk_words"] = {}  # Free disk space once grouped into segments!
    proj["percent"] = 45.0
    proj["stage_detail"] = f"Hoàn tất bóc băng {len(proj['segments'])} câu ({total_chunks} phân đoạn)."
    _save_project(proj, cancel_event=cancel_event)


def _group_long_form_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group unlimited words (>320 cues for long movies!) into clean semantic cues."""
    if not words:
        return []
    # Slice into 1500-word windows at natural pauses so group_words_into_cues never hits its 320-cue short cap
    ordered = sorted(
        (w for w in words if str(w.get("text") or "").strip()),
        key=lambda item: (float(item["start_sec"]), float(item["end_sec"])),
    )
    all_cues: list[dict[str, Any]] = []
    step = 1400
    pos = 0
    while pos < len(ordered) and len(all_cues) < MAX_STUDIO_SEGMENTS:
        end_pos = min(len(ordered), pos + step)
        if end_pos < len(ordered):
            # Look forward up to 80 words for a natural pause/sentence break
            for k in range(end_pos, min(len(ordered) - 1, end_pos + 80)):
                gap = float(ordered[k + 1]["start_sec"]) - float(ordered[k]["end_sec"])
                if gap >= 0.45 or str(ordered[k]["text"]).strip().endswith((".", "?", "!")):
                    end_pos = k + 1
                    break
        part_cues = group_words_into_cues(ordered[pos:end_pos])
        for c in part_cues:
            c["id"] = len(all_cues)
            all_cues.append(c)
        pos = end_pos
    return all_cues[:MAX_STUDIO_SEGMENTS]


def _run_resumable_translation(
    project_id: str,
    proj: dict[str, Any],
    cancel_event: threading.Event,
    *,
    force_retranslate: bool = False,
) -> None:
    """Resumable Story Bible + 45-cue sliding-window Translate->Reflect with per-batch checkpoints."""
    ck = proj.setdefault("checkpoints", {})
    segments = proj.get("segments") or []
    if not segments:
        raise RuntimeError("Chưa có lời thoại gốc để dịch.")
    if ck.get("translation_done") and not force_retranslate:
        return

    settings = proj.get("settings") or _default_settings()
    provider, api_key = resolve_configured_ai_provider(settings.get("provider_id", "gemini"))
    chosen_model = normalize_model_id(settings.get("transcribe_model"), default_transcribe_model_for(provider))
    target_lang = settings.get("target_lang", "vi")
    for_dubbing = settings.get("output_mode", "dub_hardsub") in ("dub_hardsub", "dub_only", "assets_only")
    context_hint = str(settings.get("context_hint") or "").strip()
    video_title = str(proj.get("title") or "").strip()

    if force_retranslate:
        ck["translated_ids"] = []
        ck["translated_batches"] = 0
        ck["translation_done"] = False
        ck["bible_done"] = False

    # Pass 0: Story Bible (saved in checkpoint — skipped for short 1-batch clips to save quota!)
    if len(segments) > TRANSLATION_BATCH_SIZE and not ck.get("bible_done"):
        if cancel_event.is_set():
            raise AICanceledError("Đã tạm dừng trước bước Story Bible.")
        proj["stage"] = "story_bible"
        proj["percent"] = 48.0
        proj["stage_detail"] = f"{provider.label} ({chosen_model}) đang lập Story Bible (nhân vật, thuật ngữ & xưng hô toàn phim)…"
        _save_project(proj, cancel_event=cancel_event)
        bible = _build_story_bible(
            segments,
            provider=provider,
            api_key=api_key,
            model_id=chosen_model,
            target_lang=target_lang,
            context_hint=f"{video_title}\n{context_hint}".strip(),
            cancel_event=cancel_event,
        )
        proj["story_bible"] = bible.model_dump()
        ck["bible_done"] = True
        _save_project(proj, cancel_event=cancel_event)
    else:
        existing_bible = proj.get("story_bible") or {}
        if existing_bible and any(existing_bible.get(k) for k in ("synopsis", "glossary", "pronoun_rules", "style_notes")):
            bible = StoryBibleSchema.model_validate(existing_bible)
        else:
            vi_note = (
                "Maintain consistent natural Vietnamese pronouns (avoid robotic tôi-bạn) and consistent proper nouns."
                if target_lang == "vi"
                else "Maintain consistent register, tone, and proper nouns."
            )
            bible = StoryBibleSchema(
                source_language=str(existing_bible.get("source_language") or "unknown"),
                style_notes=vi_note,
            )

    translated_id_set = {int(x) for x in (ck.get("translated_ids") or [])}
    total_batches = max(1, math.ceil(len(segments) / TRANSLATION_BATCH_SIZE))
    ck["total_batches"] = total_batches

    # Build sliding context from already-translated lines when resuming mid-movie!
    previous_context: list[dict[str, str]] = []
    for batch_idx, start_i in enumerate(range(0, len(segments), TRANSLATION_BATCH_SIZE), start=1):
        batch_cues = segments[start_i : start_i + TRANSLATION_BATCH_SIZE]
        pending_in_batch = [c for c in batch_cues if int(c["id"]) not in translated_id_set]

        if not pending_in_batch:
            previous_context = [
                {
                    "speaker": str(c.get("speaker") or "S1"),
                    "original": str(c.get("original_text") or ""),
                    "translated": str(c.get("translated_text") or ""),
                }
                for c in batch_cues[-SLIDING_CONTEXT_WINDOW:]
            ]
            continue

        if cancel_event.is_set():
            raise AICanceledError(
                f"Đã tạm dừng dịch ở lô {batch_idx - 1}/{total_batches} (Checkpoint đã lưu — bấm Tiếp tục bất kỳ lúc nào)."
            )

        pct = 50.0 + (45.0 * (batch_idx - 1) / max(1, total_batches))
        proj["stage"] = "translating"
        proj["percent"] = round(pct, 1)
        proj["stage_detail"] = (
            f"{provider.label} ({chosen_model}) đang dịch & chuốt lô {batch_idx}/{total_batches} ({len(segments)} câu)…"
        )
        _save_project(proj, cancel_event=cancel_event)

        batch_resp = _translate_cue_batch(
            pending_in_batch,
            previous_context=previous_context[-SLIDING_CONTEXT_WINDOW:],
            bible=bible,
            context_hint=context_hint,
            video_title=video_title,
            provider=provider,
            api_key=api_key,
            model_id=chosen_model,
            target_lang=target_lang,
            for_dubbing=for_dubbing,
            cancel_event=cancel_event,
        )
        by_id: dict[int, TranslatedCueItemSchema] = {item.id: item for item in batch_resp.items}

        # Retry missing IDs once inside this batch
        missing = [c for c in pending_in_batch if int(c["id"]) not in by_id]
        if missing and not cancel_event.is_set():
            try:
                retry_resp = _translate_cue_batch(
                    missing,
                    previous_context=previous_context[-SLIDING_CONTEXT_WINDOW:],
                    bible=bible,
                    context_hint=context_hint,
                    video_title=video_title,
                    provider=provider,
                    api_key=api_key,
                    model_id=chosen_model,
                    target_lang=target_lang,
                    for_dubbing=for_dubbing,
                    cancel_event=cancel_event,
                )
                for item in retry_resp.items:
                    by_id[item.id] = item
            except AICanceledError:
                raise
            except Exception as exc:
                logger.warning("Studio missing-ID retry failed: {}", str(exc)[:140])

        for cue in batch_cues:
            tr = by_id.get(int(cue["id"]))
            if tr is not None:
                if tr.corrected_original.strip():
                    cue["original_text"] = tr.corrected_original.strip()
                polished = tr.translated_text.strip() or tr.draft_translation.strip()
                if polished:
                    cue["translated_text"] = polished
                cue["gender"] = "male" if str(tr.gender or "").strip().lower().startswith("m") else "female"
                cue["tone"] = re.sub(r"\s+", " ", str(tr.tone or "natural and clear")).strip()[:80] or "natural and clear"
                translated_id_set.add(int(cue["id"]))

        if batch_resp.source_language and batch_resp.source_language != "unknown":
            proj.setdefault("story_bible", {})["source_language"] = str(batch_resp.source_language)

        # Checkpoint immediately after EVERY batch!
        ck["translated_ids"] = sorted(translated_id_set)
        ck["translated_batches"] = batch_idx
        _save_project(proj, cancel_event=cancel_event)

        previous_context = [
            {
                "speaker": str(c.get("speaker") or "S1"),
                "original": str(c.get("original_text") or ""),
                "translated": str(c.get("translated_text") or ""),
            }
            for c in batch_cues[-SLIDING_CONTEXT_WINDOW:]
        ]

    # Seed speaker voice map using user's chosen default voice
    default_voice = settings.get("default_voice", "Puck")
    existing_map = proj.get("speaker_voices") or {}
    auto_map = _assign_speaker_voices(segments, default_voice)
    for spk, vid in auto_map.items():
        existing_map.setdefault(spk, vid)
    proj["speaker_voices"] = existing_map
    ck["translation_done"] = True
    proj["percent"] = 98.0
    proj["stage_detail"] = f"Đã dịch xong {len(segments)} câu — sẵn sàng biên tập hoặc xuất bản!"
    _save_project(proj, cancel_event=cancel_event)


def _build_long_dub_track_wav(
    ffmpeg_bin: str,
    segments: list[dict[str, Any]],
    wav_paths: list[Path | None],
    total_duration_sec: float,
    out_wav: Path,
    work_dir: Path,
) -> Path:
    """Memory-safe P-center dub track assembly supporting up to 3-hour movies using mmap PCM canvas."""
    max_end = max([total_duration_sec] + [float(seg["end_sec"]) for seg in segments] + [1.0])
    total_samples = int(math.ceil((max_end + 1.0) * DUB_SAMPLE_RATE))
    total_bytes = total_samples * 2

    # Short videos (<20 min) stay blazing fast in RAM via existing helper; long films use disk-backed mmap!
    if total_duration_sec <= 1200.0:
        return build_dub_track_wav(ffmpeg_bin, segments, wav_paths, total_duration_sec, out_wav)

    import numpy as np  # type: ignore

    raw_canvas_path = work_dir / "dub_canvas.pcm"
    with raw_canvas_path.open("w+b") as fh:
        fh.truncate(total_bytes)
        mm = mmap.mmap(fh.fileno(), total_bytes)
        try:
            canvas_np = np.frombuffer(mm, dtype=np.int16)
            fade_len = int(DUB_SAMPLE_RATE * 0.020)
            head_fade = int(DUB_SAMPLE_RATE * 0.008)

            for idx, seg in enumerate(segments):
                wav_path = wav_paths[idx]
                if wav_path is None or not wav_path.is_file():
                    continue
                samples, _ = _read_pcm_mono_24k(ffmpeg_bin, wav_path, tempo=1.0)
                samples = _trim_to_acoustic_onset(samples)
                clip_dur = len(samples) / float(DUB_SAMPLE_RATE)
                if clip_dur <= 0.01:
                    continue

                next_start = float(segments[idx + 1]["start_sec"]) if idx + 1 < len(segments) else max_end + 0.5
                placed_start = min(
                    max(0.0, float(seg["start_sec"]) + 0.024),
                    max(0.0, next_start - 0.32),
                )
                avail_window = max(0.40, min(next_start - placed_start - 0.03, (float(seg["end_sec"]) - placed_start) + 0.40))
                if clip_dur > avail_window * 1.03:
                    needed_tempo = min(1.45, clip_dur / max(0.32, avail_window))
                    samples, _ = _read_pcm_mono_24k(ffmpeg_bin, wav_path, tempo=needed_tempo)
                    samples = _trim_to_acoustic_onset(samples)

                max_samples = max(
                    int(0.28 * DUB_SAMPLE_RATE),
                    int((next_start - placed_start - 0.015) * DUB_SAMPLE_RATE),
                )
                if len(samples) > max_samples:
                    samples = samples[:max_samples]
                    tail = min(fade_len, len(samples))
                    for k in range(tail):
                        pos = len(samples) - tail + k
                        samples[pos] = int(samples[pos] * ((tail - k) / float(tail)))

                lead = min(head_fade, len(samples))
                for k in range(lead):
                    samples[k] = int(samples[k] * (k / float(lead)))

                start_idx = max(0, int(round(placed_start * DUB_SAMPLE_RATE)))
                end_idx = min(total_samples, start_idx + len(samples))
                if end_idx > start_idx:
                    clip_np = np.frombuffer(samples, dtype=np.int16)[: end_idx - start_idx].astype(np.int32)
                    mixed = np.clip(canvas_np[start_idx:end_idx].astype(np.int32) + clip_np, -32768, 32767)
                    canvas_np[start_idx:end_idx] = mixed.astype(np.int16)
            mm.flush()
            del canvas_np
        finally:
            mm.close()

    # Wrap raw PCM into standard WAV via FFmpeg stream copy (instant!)
    code, stderr = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(DUB_SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            str(raw_canvas_path),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        timeout=180.0,
    )
    raw_canvas_path.unlink(missing_ok=True)
    if code != 0 or not out_wav.is_file():
        raise RuntimeError(f"Không đóng gói được track âm thanh lồng tiếng dài: {stderr[:180]}")
    return out_wav


def _run_studio_render(
    project_id: str,
    proj: dict[str, Any],
    pdir: Path,
    ffmpeg_bin: str,
    cancel_event: threading.Event,
) -> None:
    """Phase 2: Export SRT + ASS + optional per-speaker TTS Dubbing + optional Hardsub MP4."""
    segments = proj.get("segments") or []
    if not segments:
        raise RuntimeError("Dự án chưa có câu phụ đề nào để xuất bản.")

    video_rel = proj.get("video_file")
    video_path = (pdir / video_rel) if video_rel else None
    if video_path is None or not video_path.is_file():
        raise RuntimeError("Không tìm thấy tệp video nguồn của dự án.")

    settings = proj.get("settings") or _default_settings()
    output_mode = settings.get("output_mode", "dub_hardsub")
    target_lang = settings.get("target_lang", "vi")
    lang_name = TARGET_LANGUAGES.get(target_lang, "Vietnamese")
    lang_tag = target_lang.upper()
    bilingual = bool(settings.get("bilingual"))
    style = settings.get("subtitle_style", "box")
    anim = settings.get("subtitle_animation", "reveal")
    width = int(proj.get("width") or 1280)
    height = int(proj.get("height") or 720)
    duration_sec = float(proj.get("duration_sec") or 0.0)
    if duration_sec <= 0:
        probe = probe_media_stream(ffmpeg_bin, video_path)
        duration_sec = probe.duration_sec
        width, height = probe.width, probe.height

    exports_dir = pdir / "exports"
    exports_dir.mkdir(exist_ok=True)
    srt_path = exports_dir / "subtitles.srt"
    ass_path = exports_dir / "subtitles.ass"
    write_srt_file(segments, srt_path, bilingual=bilingual)
    fonts_rel, font_family = _prepare_fonts_dir(pdir)
    write_ass_file(
        segments,
        ass_path,
        width=width,
        height=height,
        style_preset=style,
        animation=anim,
        bilingual=bilingual,
        font_name=font_family,
    )

    needs_dub = output_mode in ("dub_hardsub", "dub_only", "assets_only")
    needs_video = output_mode in ("dub_hardsub", "dub_only", "hardsub_only")
    burn_sub = output_mode in ("dub_hardsub", "hardsub_only")

    dub_wav_path: Path | None = None
    dub_mp3_path: Path | None = None
    if needs_dub:
        tts_pid = str(settings.get("tts_provider_id") or "").strip() or PIPER_TTS_PROVIDER_ID
        tts_provider = None
        tts_backend = ""
        tts_api_key = ""
        tts_model = ""
        if tts_pid == PIPER_TTS_PROVIDER_ID:
            if not is_piper_tts_available():
                raise RuntimeError("Thiếu thư viện piper-tts trên máy chủ.")
            tts_backend = PIPER_TTS_PROVIDER_ID
            tts_label = "Piper ONNX Local"
        elif tts_pid == EDGE_TTS_PROVIDER_ID:
            if not is_edge_tts_available():
                raise RuntimeError("Thiếu thư viện edge-tts trên máy chủ.")
            tts_backend = EDGE_TTS_PROVIDER_ID
            tts_label = "Microsoft Edge TTS"
        else:
            tts_provider, tts_api_key = resolve_configured_ai_provider(tts_pid)
            tts_backend = tts_provider.id
            tts_model = normalize_model_id(settings.get("tts_model"), default_tts_model_for(tts_provider))
            tts_label = f"{tts_provider.label} ({tts_model})"

        default_voice = settings.get("default_voice", "Puck")
        speaker_voices = dict(proj.get("speaker_voices") or {})
        auto_voices = _assign_speaker_voices(segments, default_voice)
        for spk, vid in auto_voices.items():
            speaker_voices.setdefault(spk, vid)

        tts_dir = pdir / "tts_render_tmp"
        if tts_dir.exists():
            shutil.rmtree(tts_dir, ignore_errors=True)
        tts_dir.mkdir(exist_ok=True)
        wav_paths: list[Path | None] = [None] * len(segments)
        completed = 0
        errors: list[str] = []

        proj["stage"] = "synthesizing_tts"
        proj["percent"] = 10.0
        proj["stage_detail"] = f"{tts_label} đang lồng tiếng 0/{len(segments)} câu…"
        _save_project(proj, cancel_event=cancel_event)

        def _tts_worker(index: int, seg: dict[str, Any]) -> tuple[int, Path]:
            if cancel_event.is_set():
                raise AICanceledError("Đã dừng xuất bản.")
            clip_out = tts_dir / f"seg_{index:05d}.wav"
            v_choice = str(seg.get("voice_override") or "").strip()
            if not v_choice or v_choice not in VALID_VOICE_IDS:
                v_choice = speaker_voices.get(str(seg.get("speaker") or "S1"), default_voice)
            if v_choice == "auto":
                v_choice = "Puck" if seg.get("gender") == "male" else "Kore"
            _synthesize_segment_wav(
                str(seg.get("translated_text") or ""),
                provider=tts_provider,
                tts_backend=tts_backend,
                api_key=tts_api_key,
                tts_model=tts_model,
                voice=v_choice,
                tone=str(seg.get("tone") or "natural and clear"),
                lang_name=lang_name,
                target_lang=target_lang,
                ffmpeg_bin=ffmpeg_bin,
                out_wav=clip_out,
            )
            return index, clip_out

        tts_workers = 1 if tts_backend == EDGE_TTS_PROVIDER_ID else (2 if tts_backend == PIPER_TTS_PROVIDER_ID else 4)
        with ThreadPoolExecutor(max_workers=tts_workers) as pool:
            futs = {pool.submit(_tts_worker, i, seg): i for i, seg in enumerate(segments)}
            for fut in as_completed(futs):
                if cancel_event.is_set():
                    raise AICanceledError("Đã dừng lồng tiếng.")
                idx = futs[fut]
                try:
                    i_done, p_done = fut.result()
                    wav_paths[i_done] = p_done
                except AICanceledError:
                    raise
                except Exception as exc:
                    logger.warning("Studio TTS segment {} failed: {}", idx, exc)
                    errors.append(str(exc))
                completed += 1
                if completed % max(1, len(segments) // 25) == 0 or completed == len(segments):
                    proj["percent"] = round(10.0 + (55.0 * completed / max(1, len(segments))), 1)
                    proj["stage_detail"] = f"{tts_label} đang lồng tiếng {completed}/{len(segments)} câu…"
                    _save_project(proj, cancel_event=cancel_event)

        if not any(wav_paths):
            shutil.rmtree(tts_dir, ignore_errors=True)
            raise RuntimeError(f"Lồng tiếng thất bại: {errors[0][:180] if errors else 'không có âm thanh'}")

        proj["stage"] = "assembling_audio"
        proj["percent"] = 68.0
        proj["stage_detail"] = "Đang căn chỉnh nhịp thoại P-Center & ghép track lồng tiếng…"
        _save_project(proj, cancel_event=cancel_event)
        dub_wav_path = pdir / "dub_track.wav"
        _build_long_dub_track_wav(ffmpeg_bin, segments, wav_paths, duration_sec, dub_wav_path, pdir)
        shutil.rmtree(tts_dir, ignore_errors=True)

        # Always export standalone MP3 voiceover track for editors (CapCut / Premiere)
        dub_mp3_path = exports_dir / "dub_voiceover.mp3"
        _run_ffmpeg(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(dub_wav_path),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(dub_mp3_path),
            ],
            cancel_event=cancel_event,
            timeout=300.0,
        )

    final_video_path: Path | None = None
    if needs_video:
        proj["stage"] = "rendering_video"
        proj["percent"] = 76.0
        proj["stage_detail"] = (
            "FFmpeg đang ghép video (-c:v copy siêu tốc)…"
            if (output_mode == "dub_only")
            else "FFmpeg đang trộn âm thanh & gắn phụ đề cứng…"
        )
        _save_project(proj, cancel_event=cancel_event)

        final_video_path = exports_dir / "studio_output.mp4"
        ass_rel = f"exports/{ass_path.name}"
        out_rel = f"exports/{final_video_path.name}"
        cmd: list[str] = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", video_path.name]

        if output_mode == "hardsub_only":
            cmd += [
                "-vf",
                f"ass={ass_rel}:fontsdir={fonts_rel}",
                "-c:v",
                "libx264",
                "-preset",
                "superfast",
                "-crf",
                "21",
                "-threads",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                out_rel,
            ]
        else:
            assert dub_wav_path is not None
            cmd += ["-i", dub_wav_path.name]
            orig_vol = max(0.0, min(0.5, float(settings.get("original_volume", 0.16))))
            if bool(settings.get("keep_original_audio", True)) and orig_vol > 0.01 and proj.get("has_audio", True):
                af = (
                    f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={orig_vol:.2f}[bg];"
                    "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[dub][dub_sc];"
                    "[bg][dub_sc]sidechaincompress=threshold=0.02:ratio=6:attack=25:release=280[bg_duck];"
                    "[bg_duck][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
                )
            else:
                af = "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-16:TP=-1.5:LRA=11[aout]"

            if burn_sub:
                cmd += [
                    "-filter_complex",
                    f"[0:v]ass={ass_rel}:fontsdir={fonts_rel}[vout];{af}",
                    "-map",
                    "[vout]",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "superfast",
                    "-crf",
                    "21",
                    "-threads",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    out_rel,
                ]
            else:
                # Zero video re-encoding: blazing fast even for 3-hour 1080p movies!
                cmd += [
                    "-filter_complex",
                    af,
                    "-map",
                    "0:v:0",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    out_rel,
                ]

        render_timeout = max(600.0, duration_sec * 3.0)
        code, stderr = _run_ffmpeg(cmd, cwd=pdir, cancel_event=cancel_event, timeout=render_timeout)
        if code != 0 or not final_video_path.is_file() or final_video_path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg xuất video thất bại: {stderr.strip()[:260]}")

    if dub_wav_path and dub_wav_path.is_file():
        dub_wav_path.unlink(missing_ok=True)

    title_str = str(proj.get("title") or "video")
    mode_tag = {
        "dub_hardsub": f"Dub+Sub-{lang_tag}",
        "dub_only": f"Dub-{lang_tag}",
        "hardsub_only": f"Sub-{lang_tag}",
        "assets_only": f"Assets-{lang_tag}",
    }.get(output_mode, lang_tag)

    ck = proj.setdefault("checkpoints", {})
    ck["rendered_outputs"] = {
        "video_file": f"exports/{final_video_path.name}" if (final_video_path and final_video_path.is_file()) else None,
        "video_filename": sanitize_download_filename(title_str, mode_tag, "mp4") if final_video_path else None,
        "srt_file": f"exports/{srt_path.name}",
        "srt_filename": sanitize_download_filename(title_str, f"Sub-{lang_tag}", "srt"),
        "ass_file": f"exports/{ass_path.name}",
        "ass_filename": sanitize_download_filename(title_str, f"Sub-{lang_tag}", "ass"),
        "audio_file": f"exports/{dub_mp3_path.name}" if (dub_mp3_path and dub_mp3_path.is_file()) else None,
        "audio_filename": sanitize_download_filename(title_str, f"Voice-{lang_tag}", "mp3") if dub_mp3_path else None,
        "updated_at": time.time(),
    }
    proj["status"] = "completed"
    proj["stage"] = "completed"
    proj["percent"] = 100.0
    proj["stage_detail"] = "Xuất bản hoàn tất! Tệp thành phẩm đã sẵn sàng tải về."
    proj["error"] = None
    _save_project(proj, cancel_event=cancel_event)


def _execute_prepare_sync(
    project_id: str,
    cancel_event: threading.Event,
    *,
    auto_render: bool = False,
    force_retranslate: bool = False,
) -> None:
    pdir = _project_dir(project_id)
    ffmpeg_bin = find_ffmpeg_binary()
    if not ffmpeg_bin:
        raise RuntimeError("Thiếu FFmpeg trên máy chủ.")

    proj = _load_project(project_id)
    proj["status"] = "running_prepare"
    proj["error"] = None
    _save_project(proj, cancel_event=cancel_event)

    platform_sub_path: Path | None = None
    video_rel = proj.get("video_file")
    video_path = (pdir / video_rel) if video_rel else None
    if video_path is None or not video_path.is_file():
        if proj.get("source_type") == "url" and proj.get("source_url"):
            video_path, platform_sub_path = _ingest_url_source(project_id, proj, pdir, ffmpeg_bin, cancel_event)
        else:
            raise RuntimeError("Không tìm thấy tệp video nguồn của dự án.")

    _run_long_form_asr(project_id, proj, pdir, video_path, ffmpeg_bin, cancel_event, platform_sub_path)
    _run_resumable_translation(project_id, proj, cancel_event, force_retranslate=force_retranslate)

    if auto_render:
        proj["status"] = "running_render"
        _save_project(proj, cancel_event=cancel_event)
        _run_studio_render(project_id, proj, pdir, ffmpeg_bin, cancel_event)
    else:
        proj["status"] = "ready_to_edit"
        proj["stage"] = "ready_to_edit"
        proj["percent"] = 100.0
        proj["stage_detail"] = f"Sẵn sàng biên tập {len(proj.get('segments') or [])} câu — chỉnh sửa xong bấm Xuất bản."
        _save_project(proj, cancel_event=cancel_event)


def _clear_worker_if_same(project_id: str, cancel_event: threading.Event) -> None:
    with _RUNTIME_LOCK:
        cur = _ACTIVE_WORKERS.get(project_id)
        if cur is not None and cur.get("cancel_event") is cancel_event:
            _ACTIVE_WORKERS.pop(project_id, None)


def start_studio_prepare(
    project_id: str,
    *,
    auto_render: bool | None = None,
    force_retranslate: bool = False,
) -> dict[str, Any]:
    validate_project_id(project_id)
    with _RUNTIME_LOCK:
        if project_id in _ACTIVE_WORKERS:
            raise HTTPException(409, "Dự án này đang được xử lý.")
        proj = _load_project(project_id)
        settings = proj.get("settings") or _default_settings()
        should_auto = (settings.get("run_mode") == "auto") if auto_render is None else bool(auto_render)
        cancel_event = threading.Event()

        def _worker() -> None:
            try:
                _execute_prepare_sync(
                    project_id,
                    cancel_event,
                    auto_render=should_auto,
                    force_retranslate=force_retranslate,
                )
            except AICanceledError as exc:
                # If already detached/paused or deleted by cancel_studio_project, do not overwrite newer state
                with _RUNTIME_LOCK:
                    cur = _ACTIVE_WORKERS.get(project_id)
                    still_mine = cur is not None and cur.get("cancel_event") is cancel_event
                if still_mine and _project_json_path(project_id).is_file():
                    _update_project_fields(
                        project_id,
                        status="paused",
                        stage="paused",
                        stage_detail=str(exc) or "Đã tạm dừng — Checkpoint đã được lưu.",
                        error=None,
                    )
            except Exception as exc:
                if cancel_event.is_set():
                    return
                logger.warning("Studio prepare {} failed: {}", project_id, exc)
                if _project_json_path(project_id).is_file():
                    _update_project_fields(
                        project_id,
                        status="failed",
                        stage="failed",
                        stage_detail="",
                        error=str(exc).strip()[:300] or "Xử lý dự án thất bại.",
                    )
            finally:
                _clear_worker_if_same(project_id, cancel_event)

        thread = threading.Thread(target=_worker, name=f"studio-prep-{project_id}", daemon=True)
        _ACTIVE_WORKERS[project_id] = {"thread": thread, "cancel_event": cancel_event}
        proj["status"] = "running_prepare"
        proj["stage"] = "starting"
        proj["stage_detail"] = "Đang khởi chạy Xưởng Video AI…"
        proj["error"] = None
        _save_project(proj)
        thread.start()
        return _project_summary(proj, include_segments=False)


def start_studio_render(project_id: str) -> dict[str, Any]:
    validate_project_id(project_id)
    with _RUNTIME_LOCK:
        if project_id in _ACTIVE_WORKERS:
            raise HTTPException(409, "Dự án này đang được xử lý.")
        proj = _load_project(project_id)
        if not (proj.get("segments") or []):
            raise HTTPException(400, "Chưa có phụ đề nào để xuất bản — hãy chạy Bóc băng & Dịch trước.")
        cancel_event = threading.Event()

        def _worker() -> None:
            try:
                ffmpeg_bin = find_ffmpeg_binary()
                if not ffmpeg_bin:
                    raise RuntimeError("Thiếu FFmpeg trên máy chủ.")
                fresh = _load_project(project_id)
                _run_studio_render(project_id, fresh, _project_dir(project_id), ffmpeg_bin, cancel_event)
            except AICanceledError as exc:
                with _RUNTIME_LOCK:
                    cur = _ACTIVE_WORKERS.get(project_id)
                    still_mine = cur is not None and cur.get("cancel_event") is cancel_event
                if still_mine and _project_json_path(project_id).is_file():
                    _update_project_fields(
                        project_id,
                        status="ready_to_edit",
                        stage="paused",
                        stage_detail=str(exc) or "Đã dừng xuất bản.",
                        error=None,
                    )
            except Exception as exc:
                if cancel_event.is_set():
                    return
                logger.warning("Studio render {} failed: {}", project_id, exc)
                if _project_json_path(project_id).is_file():
                    _update_project_fields(
                        project_id,
                        status="failed",
                        stage="failed",
                        stage_detail="",
                        error=str(exc).strip()[:300] or "Xuất bản thất bại.",
                    )
            finally:
                _clear_worker_if_same(project_id, cancel_event)

        thread = threading.Thread(target=_worker, name=f"studio-render-{project_id}", daemon=True)
        _ACTIVE_WORKERS[project_id] = {"thread": thread, "cancel_event": cancel_event}
        proj["status"] = "running_render"
        proj["stage"] = "rendering_start"
        proj["percent"] = 5.0
        proj["stage_detail"] = "Đang khởi chạy xuất bản…"
        proj["error"] = None
        _save_project(proj)
        thread.start()
        return _project_summary(proj, include_segments=False)


def synthesize_studio_preview(
    project_id: str,
    *,
    text: str,
    voice: str = "Puck",
    tts_provider_id: str = "",
    tts_model: str = "",
    target_lang: str = "",
) -> Path:
    """Synthesize a single line for instant ~50ms preview playback inside the Studio Editor."""
    proj = _load_project(project_id)
    pdir = _project_dir(project_id)
    ffmpeg_bin = find_ffmpeg_binary()
    if not ffmpeg_bin:
        raise HTTPException(503, "Thiếu FFmpeg trên máy chủ.")

    clean_text = str(text or "").strip()[:500]
    if not clean_text:
        raise HTTPException(400, "Nội dung câu thoại trống.")

    settings = proj.get("settings") or _default_settings()
    lang = target_lang if target_lang in TARGET_LANGUAGES else settings.get("target_lang", "vi")
    lang_name = TARGET_LANGUAGES.get(lang, "Vietnamese")
    v_choice = voice if voice in VALID_VOICE_IDS and voice != "auto" else settings.get("default_voice", "Puck")
    if v_choice == "auto":
        v_choice = "Puck"

    tts_pid = str(tts_provider_id or settings.get("tts_provider_id") or PIPER_TTS_PROVIDER_ID).strip()
    tts_provider = None
    tts_backend = ""
    tts_api_key = ""
    chosen_tts_model = ""
    if tts_pid == PIPER_TTS_PROVIDER_ID:
        if not is_piper_tts_available():
            raise HTTPException(400, "Chưa cài thư viện piper-tts.")
        tts_backend = PIPER_TTS_PROVIDER_ID
    elif tts_pid == EDGE_TTS_PROVIDER_ID:
        if not is_edge_tts_available():
            raise HTTPException(400, "Chưa cài thư viện edge-tts.")
        tts_backend = EDGE_TTS_PROVIDER_ID
    else:
        try:
            tts_provider, tts_api_key = resolve_configured_ai_provider(tts_pid)
            tts_backend = tts_provider.id
            chosen_tts_model = normalize_model_id(
                tts_model or settings.get("tts_model"),
                default_tts_model_for(tts_provider),
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    out_wav = pdir / f"preview_{uuid.uuid4().hex[:8]}.wav"
    try:
        _synthesize_segment_wav(
            clean_text,
            provider=tts_provider,
            tts_backend=tts_backend,
            api_key=tts_api_key,
            tts_model=chosen_tts_model,
            voice=v_choice,
            tone="natural and clear",
            lang_name=lang_name,
            target_lang=lang,
            ffmpeg_bin=ffmpeg_bin,
            out_wav=out_wav,
        )
    except Exception as exc:
        out_wav.unlink(missing_ok=True)
        raise HTTPException(500, f"Không tạo được âm thanh nghe thử: {str(exc)[:220]}") from exc
    return out_wav


def resolve_studio_source_video(project_id: str) -> tuple[Path, str]:
    proj = _load_project(project_id)
    pdir = _project_dir(project_id)
    rel = proj.get("video_file")
    if not rel:
        raise HTTPException(404, "Dự án chưa tải video nguồn.")
    path = validate_managed_path(pdir / rel, pdir)
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy tệp video nguồn.")
    return path, path.name


def resolve_studio_thumbnail(project_id: str) -> Path:
    proj = _load_project(project_id)
    pdir = _project_dir(project_id)
    rel = proj.get("thumbnail_file")
    if not rel:
        raise HTTPException(404, "Dự án chưa có ảnh xem trước.")
    path = validate_managed_path(pdir / rel, pdir)
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy ảnh xem trước.")
    return path


def resolve_studio_export(project_id: str, kind: str) -> tuple[Path, str, str]:
    proj = _load_project(project_id)
    pdir = _project_dir(project_id)
    outs = (proj.get("checkpoints") or {}).get("rendered_outputs") or {}
    mapping = {
        "video": ("video_file", "video_filename", "video/mp4"),
        "srt": ("srt_file", "srt_filename", "application/x-subrip; charset=utf-8"),
        "ass": ("ass_file", "ass_filename", "text/plain; charset=utf-8"),
        "audio": ("audio_file", "audio_filename", "audio/mpeg"),
    }
    if kind not in mapping:
        raise HTTPException(400, "Định dạng xuất không hợp lệ.")
    file_key, name_key, mime = mapping[kind]
    rel = outs.get(file_key)
    if not rel:
        raise HTTPException(404, f"Dự án chưa xuất tệp {kind}.")
    path = validate_managed_path(pdir / rel, pdir)
    if not path.is_file():
        raise HTTPException(404, f"Không tìm thấy tệp xuất {kind}.")
    return path, str(outs.get(name_key) or path.name), mime


def get_studio_queue_status() -> dict[str, Any]:
    with _RUNTIME_LOCK:
        return {
            "running": bool(_QUEUE_STATE["running"]),
            "project_ids": list(_QUEUE_STATE["project_ids"]),
            "current_project_id": _QUEUE_STATE["current_project_id"],
            "completed_ids": list(_QUEUE_STATE["completed_ids"]),
            "failed_ids": list(_QUEUE_STATE["failed_ids"]),
        }


def start_studio_queue(project_ids: list[str] | None = None, *, auto_render: bool = True) -> dict[str, Any]:
    """Run a sequential batch queue over multiple Studio projects (with full checkpointing)."""
    with _RUNTIME_LOCK:
        if _QUEUE_STATE["running"]:
            raise HTTPException(409, "Hàng đợi Xưởng Video AI đang chạy.")
        if project_ids:
            targets = [validate_project_id(pid) for pid in project_ids]
        else:
            targets = [
                p["project_id"]
                for p in list_studio_projects()
                if p["status"] in ("idle", "paused", "failed", "ready_to_edit")
            ]
        if not targets:
            raise HTTPException(400, "Không có dự án nào đang chờ để chạy hàng đợi.")

        cancel_event = threading.Event()
        _QUEUE_STATE.update(
            {
                "running": True,
                "project_ids": list(targets),
                "current_project_id": None,
                "completed_ids": [],
                "failed_ids": [],
                "cancel_event": cancel_event,
            }
        )

        def _queue_worker() -> None:
            try:
                for pid in targets:
                    if cancel_event.is_set():
                        break
                    with _RUNTIME_LOCK:
                        _QUEUE_STATE["current_project_id"] = pid
                        proj_cancel = threading.Event()
                        _ACTIVE_WORKERS[pid] = {"thread": threading.current_thread(), "cancel_event": proj_cancel}
                    try:
                        # Respect both per-project pause and full queue cancellation
                        _execute_prepare_sync(pid, proj_cancel, auto_render=auto_render, force_retranslate=False)
                        with _RUNTIME_LOCK:
                            _QUEUE_STATE["completed_ids"].append(pid)
                    except AICanceledError as exc:
                        _update_project_fields(
                            pid,
                            status="paused",
                            stage="paused",
                            stage_detail=str(exc) or "Đã tạm dừng hàng đợi (Checkpoint đã lưu).",
                        )
                        if cancel_event.is_set():
                            break
                    except Exception as exc:
                        logger.warning("Studio queue item {} failed: {}", pid, exc)
                        _update_project_fields(
                            pid,
                            status="failed",
                            stage="failed",
                            error=str(exc).strip()[:280] or "Thất bại trong hàng đợi.",
                        )
                        with _RUNTIME_LOCK:
                            _QUEUE_STATE["failed_ids"].append(pid)
                    finally:
                        with _RUNTIME_LOCK:
                            _ACTIVE_WORKERS.pop(pid, None)
            finally:
                with _RUNTIME_LOCK:
                    _QUEUE_STATE["running"] = False
                    _QUEUE_STATE["current_project_id"] = None
                    _QUEUE_STATE["cancel_event"] = None

        threading.Thread(target=_queue_worker, name="studio-batch-queue", daemon=True).start()
        return get_studio_queue_status()


def cancel_studio_queue() -> dict[str, Any]:
    with _RUNTIME_LOCK:
        ev = _QUEUE_STATE.get("cancel_event")
        if ev is not None:
            ev.set()
        cur = _QUEUE_STATE.get("current_project_id")
        _QUEUE_STATE["running"] = False
        _QUEUE_STATE["current_project_id"] = None
        if cur:
            active = _ACTIVE_WORKERS.pop(cur, None)
            if active is not None:
                active["cancel_event"].set()
            if _project_json_path(cur).is_file():
                _update_project_fields(
                    cur,
                    status="paused",
                    stage="paused",
                    stage_detail="Đã dừng hàng đợi ngay lập tức — Checkpoint đã được lưu.",
                    error=None,
                )
        return get_studio_queue_status()
