from __future__ import annotations

from array import array
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable
import wave

from pydantic import BaseModel, Field
import requests

from app.ai_providers import (
    AIProvider,
    PROVIDERS,
    auth_headers,
    normalize_provider_id,
    resolve_provider,
)
from app.config import BASE_DIR
from app.downloader.http import public_request
from app.logging_config import logger
from app.secret_store import (
    SecretStoreUnavailable,
    get_provider_api_key,
    get_provider_config,
    list_provider_configs,
    provider_key_status,
)
from app.security import validate_url

DEFAULT_GEMINI_TRANSCRIBE_MODEL = "gemini-4-hydrogen"
DEFAULT_GEMINI_ASR_MODEL = "gemini-3.5-transcribe"
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_OPENAI_TRANSCRIBE_MODEL = "gpt-4o-mini-audio-preview"
DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{1,199}$")
_OFFSET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s?\s*$")

MAX_AI_DURATION_SECONDS = 15 * 60  # 15 minutes max for AI hardsub / dubbing
MAX_SEGMENTS = 320
INLINE_AUDIO_LIMIT_BYTES = 14 * 1024 * 1024
ASR_CHUNK_SECONDS = 40.0
TRANSLATION_BATCH_SIZE = 45
SLIDING_CONTEXT_WINDOW = 5
DUB_SAMPLE_RATE = 24000
DSP_ALIGN_HOP_MS = 4.0  # 4ms frame hop -> ±2ms acoustic onset precision (<10ms!)
EDGE_TTS_PROVIDER_ID = "edge_tts"
PIPER_TTS_PROVIDER_ID = "piper_tts"
LOCAL_WHISPER_DEFAULT_SIZE = "small"
VALID_WHISPER_MODELS = frozenset({"tiny", "base", "small", "medium", "large-v3-turbo", "distil-large-v3.5"})

_WHISPER_CACHE_LOCK = threading.Lock()
_WHISPER_CACHE: dict[str, Any] = {}
_PIPER_CACHE_LOCK = threading.Lock()
_PIPER_CACHE: dict[str, Any] = {}
_PIPER_VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_PIPER_CAKE_VI_URL = "https://huggingface.co/CakeByVPBank/piper-pgl-v4-vi_VN-version39_epoch39/resolve/main/vi_VN-csa-voice-piper-v3-medium"
_PIPER_VNHELLBOY_URL = "https://huggingface.co/vnhellboy/vietnamese-piper-voices/resolve/main"
_PIPER_VI_VOICE_SPECS: dict[str, tuple[str, str, int | None]] = {
    "Puck": (_PIPER_CAKE_VI_URL, "vi_VN-csa-voice-piper-v3-medium", 2),     # Quang Huy (Nam Bắc - Review Phim)
    "Orus": (_PIPER_CAKE_VI_URL, "vi_VN-csa-voice-piper-v3-medium", 2),     # Quang Huy
    "Kore": (_PIPER_CAKE_VI_URL, "vi_VN-csa-voice-piper-v3-medium", 0),     # Ngọc Lan (Nữ Bắc - Review Phim)
    "Aoede": (_PIPER_CAKE_VI_URL, "vi_VN-csa-voice-piper-v3-medium", 3),    # Thu Hà (Nữ Bắc - nhẹ nhàng)
    "Sulafat": (_PIPER_CAKE_VI_URL, "vi_VN-csa-voice-piper-v3-medium", 4),  # Yến Nhi (Nữ Nam - trẻ trung)
    "Leda": (_PIPER_CAKE_VI_URL, "vi_VN-csa-voice-piper-v3-medium", 1),     # Minh Anh (Nữ Bắc - dứt khoát)
    "Charon": (f"{_PIPER_VNHELLBOY_URL}/vits-piper-vi_VN-tramvang-medium/vi_VN-tramvang-medium", "vi_VN-tramvang-medium", None),
    "Fenrir": (f"{_PIPER_VNHELLBOY_URL}/vits-piper-vi_VN-diemdam-medium/vi_VN-diemdam-medium", "vi_VN-diemdam-medium", None),
}
_PIPER_VOICE_CATALOG: dict[str, tuple[str, str]] = {
    "en": (f"{_PIPER_VOICES_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium", "en_US-lessac-medium"),
    "fr": (f"{_PIPER_VOICES_BASE_URL}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium", "fr_FR-siwis-medium"),
    "es": (f"{_PIPER_VOICES_BASE_URL}/es/es_ES/davefx/medium/es_ES-davefx-medium", "es_ES-davefx-medium"),
    "zh": (f"{_PIPER_VOICES_BASE_URL}/zh/zh_CN/huayan/medium/zh_CN-huayan-medium", "zh_CN-huayan-medium"),
}
_RETRY_DELAY_RE = re.compile(r"(?:retry(?:Delay|_after|\s+in)?\s*[:=]?\s*['\"]?)(\d+(?:\.\d+)?)s?", re.IGNORECASE)
_VTT_TIME_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_STAGE_NOISE_RE = re.compile(r"^\s*[\[(][^\]\)]{1,30}[\])]\s*$")

_NO_STRAND_END_WORDS = frozenset(
    {
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "with", "by", "from",
        "and", "or", "but", "that", "my", "your", "our", "their", "his", "her", "its",
        "this", "these", "those", "if", "as", "into", "about", "like", "than",
        "cho", "của", "với", "và", "nhưng", "vì", "nếu", "khi", "mà", "để", "một",
        "những", "các", "trong", "trên", "từ", "về", "đang", "đã", "sẽ", "không",
    }
)
_CLAUSE_BREAK_BEFORE = frozenset(
    {
        "and", "but", "or", "so", "because", "cause", "'cause", "that", "which",
        "when", "while", "if", "until", "unless", "though", "although", "before",
        "after", "where", "who", "nhưng", "vì", "nếu", "khi", "rằng", "để", "mà",
    }
)

VALID_AI_MODES = frozenset({"none", "hardsub", "dub"})
VALID_ASR_ENGINES = frozenset({"auto", "local_whisper", "cloud"})
VALID_SUBTITLE_STYLES = frozenset({"classic", "box", "cinema"})
VALID_SUBTITLE_ANIMATIONS = frozenset({"reveal", "karaoke", "static"})

TARGET_LANGUAGES: dict[str, str] = {
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "fr": "French",
    "es": "Spanish",
}

VOICE_CATALOG: tuple[dict[str, str], ...] = (
    {"id": "Puck", "label": "Quang Huy / Puck — Nam Bắc, Review Phim & năng động", "gender": "male"},
    {"id": "Kore", "label": "Ngọc Lan / Kore — Nữ Bắc, rõ ràng & Review Phim", "gender": "female"},
    {"id": "Aoede", "label": "Thu Hà / Aoede — Nữ Bắc, nhẹ nhàng & tự nhiên", "gender": "female"},
    {"id": "Sulafat", "label": "Yến Nhi / Sulafat — Nữ Nam, ấm áp & trẻ trung", "gender": "female"},
    {"id": "Charon", "label": "Trầm Vang / Charon — Nam Bắc, trầm ấm & kể chuyện", "gender": "male"},
    {"id": "Fenrir", "label": "Điềm Đạm / Fenrir — Nam Bắc, thuyết minh & mạnh mẽ", "gender": "male"},
    {"id": "auto", "label": "Tự động theo giới tính & người nói", "gender": "auto"},
)

VALID_VOICE_IDS = frozenset(v["id"] for v in VOICE_CATALOG)
_FEMALE_VOICES = ("Kore", "Aoede", "Sulafat", "Leda")
_MALE_VOICES = ("Puck", "Charon", "Fenrir", "Orus")
_OPENAI_VOICE_MAP: dict[str, str] = {
    "Kore": "nova",
    "Aoede": "shimmer",
    "Sulafat": "sage",
    "Leda": "coral",
    "Puck": "echo",
    "Charon": "onyx",
    "Fenrir": "fable",
    "Orus": "ash",
}
_EDGE_TTS_VOICE_MAP: dict[str, dict[str, str]] = {
    "vi": {"female": "vi-VN-HoaiMyNeural", "male": "vi-VN-NamMinhNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    "ko": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
}


def is_local_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def is_edge_tts_available() -> bool:
    return importlib.util.find_spec("edge_tts") is not None


def is_piper_tts_available() -> bool:
    return importlib.util.find_spec("piper") is not None


def _resolve_edge_tts_voice(target_lang: str, voice_id: str) -> str:
    gender = "male" if voice_id in _MALE_VOICES else "female"
    lang_voices = _EDGE_TTS_VOICE_MAP.get(target_lang) or _EDGE_TTS_VOICE_MAP["vi"]
    return lang_voices[gender]

_BUNDLED_VI_FONT = BASE_DIR / "app" / "static" / "fonts" / "dialogue" / "be-vietnam-pro.ttf"
_BUNDLED_FONT_FAMILY = "Be Vietnam Pro"

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_DIM_RE = re.compile(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})(?:\s|\[|,)")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*:\s*Audio:")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(\d+(?:\.\d+)?)")


class AICanceledError(Exception):
    """Raised when the user cancels an active AI media processing job."""


class TranscriptSegmentSchema(BaseModel):
    start_sec: float = Field(description="Start time in seconds from 0.0.")
    end_sec: float = Field(description="End time in seconds, strictly greater than start_sec.")
    speaker: str = Field(default="S1", description="Speaker identifier such as S1, S2.")
    gender: str = Field(default="female", description="Estimated speaker vocal gender: 'female' or 'male'.")
    tone: str = Field(default="natural and clear", description="Short English speaking style/emotion prompt for TTS.")
    original_text: str = Field(description="Verbatim spoken text in the original language.")
    translated_text: str = Field(description="Natural translation in the target language matching the spoken duration.")


class TranscriptPlanSchema(BaseModel):
    source_language: str = Field(default="unknown", description="Detected primary spoken language name or ISO code.")
    segments: list[TranscriptSegmentSchema] = Field(default_factory=list)


class StoryBibleSchema(BaseModel):
    source_language: str = Field(default="unknown", description="Primary spoken/sung language.")
    genre: str = Field(default="general", description="Content genre: movie, anime, music_video, tutorial, vlog, news.")
    synopsis: str = Field(default="", description="1-2 sentence summary of topic, setting, or plot.")
    glossary: list[str] = Field(default_factory=list, description="Key proper nouns & terms as 'Source -> Target'.")
    pronoun_rules: list[str] = Field(
        default_factory=list,
        description="Consistent pronoun & register rules per speaker/relationship (crucial for Vietnamese Vietsub).",
    )
    style_notes: str = Field(default="", description="Concise tone/register instruction for natural subtitling.")


class TranslatedCueItemSchema(BaseModel):
    id: int = Field(description="Exact integer id of the input cue.")
    corrected_original: str = Field(default="", description="ASR-proofread original text fixing misheard words from context.")
    draft_translation: str = Field(default="", description="Step 1 faithful direct translation preserving full meaning.")
    translated_text: str = Field(
        description="Step 2 reflected, natural, concise subtitle translation obeying pronouns, glossary, and <=20 CPS."
    )
    gender: str = Field(default="female", description="Estimated speaker vocal gender: 'female' or 'male'.")
    tone: str = Field(default="natural and clear", description="Short English speaking style/emotion prompt for TTS.")


class TranslatedCueBatchSchema(BaseModel):
    source_language: str = Field(default="unknown", description="Detected primary spoken/sung language.")
    items: list[TranslatedCueItemSchema] = Field(default_factory=list)


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    duration_sec: float
    has_audio: bool


@dataclass(frozen=True)
class AIProcessOptions:
    mode: str = "none"  # "none" | "hardsub" | "dub"
    provider_id: str = "gemini"
    tts_provider_id: str = ""  # empty = same as provider_id
    target_lang: str = "vi"
    voice: str = "auto"
    burn_subtitles: bool = True
    bilingual: bool = False
    subtitle_style: str = "classic"  # "classic" | "box" | "cinema"
    subtitle_animation: str = "reveal"  # "reveal" | "karaoke" | "static"
    keep_original_audio: bool = True
    original_volume: float = 0.16
    transcribe_model: str = ""
    tts_model: str = ""
    context_hint: str = ""
    asr_engine: str = "auto"  # "auto" | "local_whisper" | "cloud"
    whisper_model: str = LOCAL_WHISPER_DEFAULT_SIZE  # "tiny" | "base" | "small" | "medium"
    video_title: str = ""


def normalize_model_id(raw: str | None, fallback: str) -> str:
    cleaned = str(raw or "").strip()
    if cleaned.startswith("models/"):
        cleaned = cleaned[len("models/") :]
    if not cleaned:
        return fallback
    if not _MODEL_ID_RE.match(cleaned):
        raise ValueError(f"Tên model không hợp lệ: {cleaned!r}")
    return cleaned


def resolve_configured_ai_provider(provider_id: str) -> tuple[AIProvider, str]:
    """Resolve any builtin or custom provider configured in Settings along with its API key."""
    normalized = normalize_provider_id(provider_id or "gemini")
    if normalized in PROVIDERS:
        provider = PROVIDERS[normalized]
    else:
        stored = get_provider_config(normalized)
        if not isinstance(stored, dict):
            raise ValueError(f"Nhà cung cấp AI '{normalized}' chưa được cấu hình trong Cài đặt.")
        provider = resolve_provider(
            normalized,
            label=stored.get("label"),
            protocol=stored.get("protocol"),
            api_base=stored.get("api_base"),
        )
        if provider.chat_url:
            validate_url(provider.chat_url)
    try:
        api_key = get_provider_api_key(provider.id, provider_label=provider.label)
    except SecretStoreUnavailable as exc:
        raise RuntimeError(f"Không đọc được khóa API cho {provider.label}: {exc}") from exc
    if not api_key:
        raise ValueError(
            f"Nhà cung cấp {provider.label} chưa có khóa API — hãy thêm khóa trong Cài đặt → Mô hình AI."
        )
    return provider, api_key


def list_configured_ai_providers() -> list[dict[str, Any]]:
    """Return summary of all builtin & custom providers from Settings."""
    items: list[dict[str, Any]] = []
    for provider in PROVIDERS.values():
        status = provider_key_status(provider.id)
        items.append(
            {
                "id": provider.id,
                "label": provider.label,
                "protocol": provider.protocol,
                "configured": bool(status.get("configured")),
                "builtin": True,
            }
        )
    try:
        custom_configs = list_provider_configs()
    except SecretStoreUnavailable:
        custom_configs = []
    for cfg in custom_configs:
        try:
            provider = resolve_provider(
                str(cfg.get("id") or ""),
                label=cfg.get("label"),
                protocol=cfg.get("protocol"),
                api_base=cfg.get("api_base"),
            )
        except ValueError:
            continue
        status = provider_key_status(provider.id, provider_label=provider.label)
        items.append(
            {
                "id": provider.id,
                "label": provider.label,
                "protocol": provider.protocol,
                "configured": bool(status.get("configured")),
                "builtin": False,
            }
        )
    return items


def any_ai_provider_configured() -> bool:
    return any(item["configured"] for item in list_configured_ai_providers())


def default_transcribe_model_for(provider: AIProvider) -> str:
    if provider.protocol == "gemini":
        return DEFAULT_GEMINI_TRANSCRIBE_MODEL
    if provider.default_translation_model:
        return provider.default_translation_model
    if provider.default_qc_model:
        return provider.default_qc_model
    return DEFAULT_OPENAI_TRANSCRIBE_MODEL


def default_tts_model_for(provider: AIProvider) -> str:
    if provider.protocol == "gemini":
        return DEFAULT_GEMINI_TTS_MODEL
    return DEFAULT_OPENAI_TTS_MODEL


def _run_ffmpeg(
    args: list[str],
    *,
    cwd: Path | None = None,
    cancel_event: threading.Event | None = None,
    timeout: float = 600.0,
) -> tuple[int, str]:
    proc = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        try:
            stderr_chunks.append(proc.stderr.read())
        except Exception:
            pass

    reader = threading.Thread(target=_drain_stderr, daemon=True)
    reader.start()

    elapsed = 0.0
    step = 0.25
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise AICanceledError("Đã hủy xử lý FFmpeg.")
            if elapsed >= timeout:
                proc.kill()
                raise RuntimeError("FFmpeg vượt quá thời gian xử lý cho phép.")
            try:
                proc.wait(timeout=step)
            except subprocess.TimeoutExpired:
                elapsed += step
    finally:
        reader.join(timeout=2.0)
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception:
                pass

    return int(proc.returncode or 0), "".join(stderr_chunks)


def probe_media_stream(ffmpeg_bin: str, media_path: Path) -> VideoProbe:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(media_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    stderr = proc.stderr or ""
    duration_sec = 0.0
    m_dur = _DURATION_RE.search(stderr)
    if m_dur:
        h, m, s = int(m_dur.group(1)), int(m_dur.group(2)), float(m_dur.group(3))
        duration_sec = h * 3600.0 + m * 60.0 + s

    width, height = 1280, 720
    m_dim = _VIDEO_DIM_RE.search(stderr)
    if m_dim:
        width, height = int(m_dim.group(1)), int(m_dim.group(2))

    has_audio = bool(_AUDIO_STREAM_RE.search(stderr))
    return VideoProbe(width=width, height=height, duration_sec=duration_sec, has_audio=has_audio)


def extract_speech_mp3(
    ffmpeg_bin: str,
    video_path: Path,
    out_mp3: Path,
    *,
    cancel_event: threading.Event | None = None,
) -> Path:
    code, stderr = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            str(out_mp3),
        ],
        cancel_event=cancel_event,
        timeout=180.0,
    )
    if code != 0 or not out_mp3.is_file() or out_mp3.stat().st_size == 0:
        raise RuntimeError(f"Không tách được âm thanh từ video: {stderr.strip()[:220]}")
    return out_mp3


def _detect_silence_intervals(
    ffmpeg_bin: str,
    audio_mp3: Path,
    *,
    noise_db: float = -32.0,
    min_silence_sec: float = 0.25,
    cancel_event: threading.Event | None = None,
) -> list[tuple[float, float]]:
    """Run lightweight CPU FFmpeg silencedetect to locate natural vocal pauses."""
    code, stderr = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(audio_mp3),
            "-af",
            f"silencedetect=noise={noise_db:.0f}dB:d={min_silence_sec:.2f}",
            "-f",
            "null",
            "-",
        ],
        cancel_event=cancel_event,
        timeout=90.0,
    )
    if code != 0 and not stderr:
        return []
    intervals: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in stderr.splitlines():
        m_start = _SILENCE_START_RE.search(line)
        if m_start:
            cur_start = float(m_start.group(1))
            continue
        m_end = _SILENCE_END_RE.search(line)
        if m_end:
            end_t = float(m_end.group(1))
            dur = float(m_end.group(2))
            start_t = cur_start if cur_start is not None else max(0.0, end_t - dur)
            if end_t > start_t:
                intervals.append((round(start_t, 3), round(end_t, 3)))
            cur_start = None
    return intervals


def choose_silence_split_points(
    duration_sec: float,
    silences: list[tuple[float, float]],
    *,
    target_chunk_sec: float = ASR_CHUNK_SECONDS,
    min_chunk_sec: float = 26.0,
    max_chunk_sec: float = 52.0,
) -> list[tuple[float, float]]:
    """Split audio into ~40s ranges preferring natural silence midpoints so speech is never chopped mid-word."""
    total = max(1.0, float(duration_sec))
    if total <= max_chunk_sec:
        return [(0.0, total)]

    ranges: list[tuple[float, float]] = []
    pos = 0.0
    while total - pos > max_chunk_sec:
        win_lo = pos + min_chunk_sec
        win_hi = min(total - 8.0, pos + max_chunk_sec)
        target = pos + target_chunk_sec

        best_mid: float | None = None
        best_score = float("inf")
        for s_start, s_end in silences:
            mid = (s_start + s_end) * 0.5
            if win_lo <= mid <= win_hi:
                sil_dur = max(0.0, s_end - s_start)
                # Prefer longer natural pauses near the ~40s target
                score = abs(mid - target) - min(sil_dur, 1.5) * 4.0
                if score < best_score:
                    best_score = score
                    best_mid = mid

        split_at = round(best_mid, 3) if best_mid is not None else round(min(target, win_hi), 3)
        if split_at <= pos + 5.0:
            split_at = round(min(total, pos + target_chunk_sec), 3)
        ranges.append((round(pos, 3), split_at))
        pos = split_at

    if total - pos > 0.5 or not ranges:
        ranges.append((round(pos, 3), round(total, 3)))
    return ranges


def _split_audio_chunks(
    ffmpeg_bin: str,
    audio_mp3: Path,
    duration_sec: float,
    work_dir: Path,
    *,
    chunk_sec: float = ASR_CHUNK_SECONDS,
    cancel_event: threading.Event | None = None,
) -> list[tuple[float, float, Path]]:
    """Split long audio at natural silence boundaries (~40s) so ASR never chops words or drifts."""
    if duration_sec <= chunk_sec + 8.0:
        return [(0.0, max(1.0, duration_sec), audio_mp3)]

    silences = _detect_silence_intervals(ffmpeg_bin, audio_mp3, cancel_event=cancel_event)
    ranges = choose_silence_split_points(
        duration_sec,
        silences,
        target_chunk_sec=chunk_sec,
        min_chunk_sec=max(18.0, chunk_sec - 14.0),
        max_chunk_sec=chunk_sec + 12.0,
    )

    chunks_dir = work_dir / "asr_chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunks: list[tuple[float, float, Path]] = []
    for idx, (seg_start, seg_end) in enumerate(ranges):
        seg_dur = max(0.5, seg_end - seg_start)
        out_chunk = chunks_dir / f"chunk_{idx:03d}.mp3"
        code, stderr = _run_ffmpeg(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{seg_start:.3f}",
                "-t",
                f"{seg_dur:.3f}",
                "-i",
                str(audio_mp3),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "48k",
                str(out_chunk),
            ],
            cancel_event=cancel_event,
            timeout=90.0,
        )
        if code == 0 and out_chunk.is_file() and out_chunk.stat().st_size > 256:
            chunks.append((seg_start, seg_end, out_chunk))
        else:
            logger.warning("Audio chunk {} failed: {}", idx, stderr[:160])
    return chunks or [(0.0, max(1.0, duration_sec), audio_mp3)]


def _parse_offset_seconds(raw: Any) -> float | None:
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) and val >= 0 else None
    m = _OFFSET_RE.match(str(raw or ""))
    if not m:
        return None
    val = float(m.group(1))
    return val if math.isfinite(val) and val >= 0 else None


def _transcribe_words_gemini_chunk(
    chunk_mp3: Path,
    *,
    offset_sec: float,
    chunk_dur: float,
    api_key: str,
) -> list[dict[str, Any]]:
    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    audio_b64 = base64.b64encode(chunk_mp3.read_bytes()).decode("ascii")
    resp = client.interactions.create(
        model=DEFAULT_GEMINI_ASR_MODEL,
        input=[{"type": "audio", "data": audio_b64, "mime_type": "audio/mp3"}],
        generation_config={
            "transcription_config": {
                "mode": {
                    "type": "verbatim",
                    "diarization_mode": "speaker",
                    "timestamp_granularities": ["word"],
                }
            }
        },
    )

    words: list[dict[str, Any]] = []
    for step in resp.steps or []:
        for part in getattr(step, "content", None) or []:
            for ann in getattr(part, "annotations", None) or []:
                data = ann.model_dump() if hasattr(ann, "model_dump") else (ann if isinstance(ann, dict) else {})
                txt = str(data.get("text") or "").strip()
                st = _parse_offset_seconds(data.get("start_offset"))
                en = _parse_offset_seconds(data.get("end_offset"))
                if not txt or st is None or en is None:
                    continue
                if st > chunk_dur + 1.0:
                    continue
                st_abs = round(offset_sec + max(0.0, min(st, chunk_dur)), 3)
                en_abs = round(offset_sec + max(st + 0.04, min(en, chunk_dur)), 3)
                raw_spk = str(data.get("speaker") or "S1")
                spk = re.sub(r"[^A-Za-z0-9_-]", "", raw_spk.replace(":", "_"))[:16] or "S1"
                words.append(
                    {
                        "start_sec": st_abs,
                        "end_sec": en_abs,
                        "text": txt,
                        "speaker": spk,
                    }
                )
    return words


def _clean_word_token(text: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", str(text or "").strip().lower())


def _pick_semantic_split_index(bucket: list[dict[str, Any]]) -> int:
    """Choose the cleanest clause/pause boundary inside an over-length word bucket."""
    n = len(bucket)
    if n <= 3:
        return n
    lo = max(3, n // 2)
    best_k = n
    best_score = -1e9
    for k in range(lo, n):
        left = bucket[k - 1]
        right = bucket[k]
        left_raw = str(left["text"]).strip()
        left_tok = _clean_word_token(left_raw)
        right_tok = _clean_word_token(right["text"])
        gap = max(0.0, float(right["start_sec"]) - float(left["end_sec"]))

        score = gap * 12.0
        if left_raw and left_raw[-1] in ",;،、:–—":
            score += 6.0
        if right_tok in _CLAUSE_BREAK_BEFORE:
            score += 3.5
        if left_tok in _NO_STRAND_END_WORDS:
            score -= 6.5
        # Prefer balanced halves rather than leaving a 1-word tail
        tail_len = n - k
        if tail_len == 1:
            score -= 4.0
        if score > best_score:
            best_score = score
            best_k = k

    if best_score < 0.5:
        # Even without strong punctuation/pause, avoid stranding a preposition/article at the end
        if _clean_word_token(bucket[-1]["text"]) in _NO_STRAND_END_WORDS and n >= 4:
            return n - 1
        return n
    return best_k


def group_words_into_cues(
    words: list[dict[str, Any]],
    *,
    pause_split_sec: float = 0.45,
    max_words_per_cue: int = 8,
    max_cue_duration: float = 3.6,
) -> list[dict[str, Any]]:
    """Group exact word-level timestamps into semantic, rhythmic subtitle cues without orphan splits."""
    ordered = sorted(
        (w for w in words if str(w.get("text") or "").strip()),
        key=lambda item: (float(item["start_sec"]), float(item["end_sec"])),
    )
    if not ordered:
        return []

    # Deduplicate near-identical boundary echoes from adjacent chunks
    deduped: list[dict[str, Any]] = []
    for w in ordered:
        if deduped:
            prev = deduped[-1]
            if (
                _clean_word_token(w["text"]) == _clean_word_token(prev["text"])
                and abs(float(w["start_sec"]) - float(prev["start_sec"])) < 0.08
            ):
                continue
        deduped.append(w)

    cues: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []

    def _emit(items: list[dict[str, Any]]) -> None:
        if not items:
            return
        text = " ".join(w["text"] for w in items).strip()
        start = float(items[0]["start_sec"])
        end = max(start + 0.25, float(items[-1]["end_sec"]))
        speaker = items[0].get("speaker") or "S1"
        cues.append(
            {
                "id": len(cues),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "speaker": speaker,
                "gender": "female",
                "tone": "natural and clear",
                "original_text": text,
                "translated_text": text,
                "words": [
                    {
                        "start_sec": round(float(w["start_sec"]), 3),
                        "end_sec": round(float(w["end_sec"]), 3),
                        "text": str(w["text"]),
                    }
                    for w in items
                ],
            }
        )

    def _flush_all() -> None:
        if bucket:
            _emit(list(bucket))
            bucket.clear()

    def _ends_soon(idx_cur: int, speaker_cur: str, start_cur: float) -> bool:
        """Look ahead 1-2 words: if sentence/pause finishes right away, keep together instead of creating a 1-word orphan."""
        max_look = min(len(deduped), idx_cur + 2)
        for j in range(idx_cur, max_look):
            cand = deduped[j]
            if (cand.get("speaker") or "S1") != speaker_cur:
                return False
            if float(cand["end_sec"]) - start_cur > max_cue_duration + 1.1:
                return False
            if str(cand["text"]).strip()[-1] in ".?!。？！":
                return True
            if j + 1 < len(deduped):
                nxt_gap = float(deduped[j + 1]["start_sec"]) - float(cand["end_sec"])
                if nxt_gap >= pause_split_sec:
                    return True
            else:
                return True
        return False

    hard_max_words = max_words_per_cue + 3
    hard_max_dur = max_cue_duration + 1.1

    for idx, w in enumerate(deduped):
        if not bucket:
            bucket.append(w)
            continue
        prev = bucket[-1]
        gap = float(w["start_sec"]) - float(prev["end_sec"])
        cue_start = float(bucket[0]["start_sec"])
        cue_span = float(w["end_sec"]) - cue_start
        prev_text = str(prev["text"]).strip()
        prev_ends_sentence = bool(prev_text) and prev_text[-1] in ".?!。？！"
        prev_ends_clause = bool(prev_text) and prev_text[-1] in ",;،、" and len(bucket) >= 4
        cur_speaker = bucket[0].get("speaker") or "S1"
        speaker_changed = (w.get("speaker") or "S1") != cur_speaker

        if gap >= pause_split_sec or speaker_changed or prev_ends_sentence or prev_ends_clause:
            _flush_all()
            bucket.append(w)
            continue

        soft_limit_hit = len(bucket) >= max_words_per_cue or cue_span >= max_cue_duration
        hard_limit_hit = len(bucket) >= hard_max_words or cue_span >= hard_max_dur
        if soft_limit_hit:
            if not hard_limit_hit and _ends_soon(idx, cur_speaker, cue_start):
                bucket.append(w)
                continue
            # Split cleanly at the best semantic/acoustic boundary inside the bucket
            split_k = _pick_semantic_split_index(bucket)
            _emit(bucket[:split_k])
            bucket = bucket[split_k:]
        bucket.append(w)
    _flush_all()

    # Smooth micro-gaps (< 120ms) and prevent overlap between consecutive cues (Netflix readability)
    for i in range(len(cues) - 1):
        nxt_start = cues[i + 1]["start_sec"]
        gap_between = nxt_start - cues[i]["end_sec"]
        if gap_between < 0:
            cues[i]["end_sec"] = round(max(cues[i]["start_sec"] + 0.15, nxt_start - 0.02), 3)
        elif 0.02 < gap_between < 0.12:
            cues[i]["end_sec"] = round(nxt_start - 0.02, 3)
    return cues[:MAX_SEGMENTS]


def _normalize_segments(
    raw_segments: list[TranscriptSegmentSchema],
    duration_sec: float,
    offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    max_t = duration_sec if duration_sec > 0 else 36000.0
    for seg in raw_segments:
        text_tr = re.sub(r"\s+", " ", str(seg.translated_text or "")).strip()
        text_orig = re.sub(r"\s+", " ", str(seg.original_text or "")).strip()
        if not text_tr and not text_orig:
            continue
        start = float(seg.start_sec) + offset_sec
        end = float(seg.end_sec) + offset_sec
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if start >= max_t - 0.15:
            continue  # Drop hallucinated timestamps past audio duration instead of piling at EOF
        start = max(0.0, min(start, max_t - 0.15))
        end = max(start + 0.25, min(end, max_t))
        if end <= start:
            continue
        speaker = re.sub(r"[^A-Za-z0-9_-]", "", str(seg.speaker or "S1"))[:16] or "S1"
        gender = "male" if str(seg.gender or "").strip().lower().startswith("m") else "female"
        tone = re.sub(r"\s+", " ", str(seg.tone or "natural and clear")).strip()[:80] or "natural and clear"
        cleaned.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "speaker": speaker,
                "gender": gender,
                "tone": tone,
                "original_text": text_orig or text_tr,
                "translated_text": text_tr or text_orig,
                "words": [],
            }
        )

    cleaned.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    for i in range(len(cleaned) - 1):
        nxt_start = cleaned[i + 1]["start_sec"]
        if cleaned[i]["end_sec"] > nxt_start:
            cleaned[i]["end_sec"] = round(max(cleaned[i]["start_sec"] + 0.15, nxt_start - 0.02), 3)
    return cleaned[:MAX_SEGMENTS]


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise RuntimeError("Phản hồi JSON từ model AI không hợp lệ.")
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("Phản hồi từ model AI không phải đối tượng JSON.")
    return parsed


def _run_cancellable(
    fn,
    cancel_event: threading.Event | None = None,
    on_cancel=None,
):
    """Run a blocking network/LLM call while aborting promptly (<=100ms) if cancel_event is set."""
    if cancel_event is None:
        return fn()
    if cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng.")

    result_box: dict[str, Any] = {}
    done_ev = threading.Event()

    def _runner() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            result_box["error"] = exc
        finally:
            done_ev.set()

    worker = threading.Thread(target=_runner, name="media-ai-cancellable", daemon=True)
    worker.start()
    while not done_ev.wait(timeout=0.08):
        if cancel_event.is_set():
            if callable(on_cancel):
                try:
                    on_cancel()
                except Exception:
                    pass
            raise AICanceledError("Đã tạm dừng ngay lập tức.")
    if cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng ngay lập tức.")
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


def _openai_chat_json(
    provider: AIProvider,
    api_key: str,
    model_id: str,
    messages: list[dict[str, Any]],
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    chat_url = provider.chat_url
    if not chat_url:
        raise RuntimeError(f"Nhà cung cấp {provider.label} không có endpoint /chat/completions.")
    validate_url(chat_url)
    if cancel_event is not None and cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng.")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **auth_headers(provider, api_key),
    }
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        **provider.chat_completion_extras(model_id),
    }

    def _do_req() -> dict[str, Any]:
        resp = public_request("POST", chat_url, headers=headers, json=body, timeout=(15, 180))
        try:
            if resp.status_code >= 400:
                detail = (resp.text or "")[:240]
                raise requests.HTTPError(f"HTTP {resp.status_code}: {detail}", response=resp)
            data = resp.json()
        finally:
            resp.close()
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return _extract_json_object(str(content))

    return _run_cancellable(_do_req, cancel_event=cancel_event)


def _parse_vtt_ts(h: str | None, m: str, s: str, ms: str) -> float:
    return (int(h) if h else 0) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0


def _clean_caption_text(raw: str) -> str:
    txt = html.unescape(_TAG_STRIP_RE.sub("", str(raw or "")))
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt or _STAGE_NOISE_RE.match(txt):
        return ""
    return txt


def parse_platform_subtitle_file(sub_path: Path) -> list[dict[str, Any]]:
    """Parse yt-dlp downloaded json3 / vtt / srt subtitles into word-aligned cues (0 API quota!)."""
    if not sub_path or not sub_path.is_file():
        return []
    suffix = sub_path.suffix.lower()

    if suffix == ".json3":
        try:
            data = json.loads(sub_path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            return []
        events = data.get("events") if isinstance(data, dict) else None
        if not isinstance(events, list):
            return []

        words: list[dict[str, Any]] = []
        raw_cues: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            segs = ev.get("segs")
            if not isinstance(segs, list) or not segs:
                continue
            start_ms = float(ev.get("tStartMs") or 0.0)
            dur_ms = float(ev.get("dDurationMs") or 0.0)
            ev_start = max(0.0, start_ms / 1000.0)
            ev_end = ev_start + max(0.3, dur_ms / 1000.0)

            has_offsets = any(isinstance(s, dict) and s.get("tOffsetMs") is not None for s in segs)
            if has_offsets:
                ev_words: list[dict[str, Any]] = []
                for idx_s, seg in enumerate(segs):
                    if not isinstance(seg, dict):
                        continue
                    piece = _clean_caption_text(seg.get("utf8") or "")
                    if not piece:
                        continue
                    off_ms = float(seg.get("tOffsetMs") or 0.0)
                    w_start = max(0.0, (start_ms + off_ms) / 1000.0)
                    next_off: float | None = None
                    for nxt in segs[idx_s + 1 :]:
                        if isinstance(nxt, dict) and nxt.get("tOffsetMs") is not None:
                            next_off = float(nxt["tOffsetMs"])
                            break
                    w_end = (
                        (start_ms + next_off) / 1000.0
                        if next_off is not None and next_off > off_ms
                        else min(ev_end, w_start + 0.45)
                    )
                    for tok in piece.split():
                        ev_words.append(
                            {
                                "start_sec": round(w_start, 3),
                                "end_sec": round(max(w_start + 0.05, w_end), 3),
                                "text": tok,
                                "speaker": "S1",
                            }
                        )
                words.extend(ev_words)
            else:
                full_text = _clean_caption_text("".join(str(s.get("utf8") or "") for s in segs if isinstance(s, dict)))
                if full_text:
                    raw_cues.append(
                        {
                            "id": len(raw_cues),
                            "start_sec": round(ev_start, 3),
                            "end_sec": round(ev_end, 3),
                            "speaker": "S1",
                            "gender": "female",
                            "tone": "natural and clear",
                            "original_text": full_text,
                            "translated_text": full_text,
                            "words": [],
                        }
                    )

        if len(words) >= 4:
            return group_words_into_cues(words)
        if len(raw_cues) >= 2:
            raw_cues.sort(key=lambda c: (c["start_sec"], c["end_sec"]))
            for i in range(len(raw_cues) - 1):
                if raw_cues[i]["end_sec"] > raw_cues[i + 1]["start_sec"]:
                    raw_cues[i]["end_sec"] = round(max(raw_cues[i]["start_sec"] + 0.15, raw_cues[i + 1]["start_sec"] - 0.02), 3)
            for idx_c, c in enumerate(raw_cues):
                c["id"] = idx_c
            return raw_cues[:MAX_SEGMENTS]
        return []

    # VTT / SRT fallback parser
    text_body = sub_path.read_text(encoding="utf-8", errors="replace")
    cues: list[dict[str, Any]] = []
    blocks = re.split(r"\r?\n\s*\r?\n", text_body)
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        ts_idx = -1
        m_ts = None
        for idx_l, ln in enumerate(lines):
            m = _VTT_TIME_RE.search(ln)
            if m:
                ts_idx = idx_l
                m_ts = m
                break
        if m_ts is None:
            continue
        st = _parse_vtt_ts(m_ts.group(1), m_ts.group(2), m_ts.group(3), m_ts.group(4))
        en = _parse_vtt_ts(m_ts.group(5), m_ts.group(6), m_ts.group(7), m_ts.group(8))
        body = _clean_caption_text(" ".join(lines[ts_idx + 1 :]))
        if not body or en <= st:
            continue
        if cues and cues[-1]["original_text"].lower() == body.lower() and abs(st - cues[-1]["end_sec"]) < 0.35:
            cues[-1]["end_sec"] = round(max(cues[-1]["end_sec"], en), 3)
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
                "words": [],
            }
        )
    if len(cues) < 2:
        return []
    for i in range(len(cues) - 1):
        if cues[i]["end_sec"] > cues[i + 1]["start_sec"]:
            cues[i]["end_sec"] = round(max(cues[i]["start_sec"] + 0.15, cues[i + 1]["start_sec"] - 0.02), 3)
    return cues[:MAX_SEGMENTS]


def _get_local_whisper_model(model_size: str = LOCAL_WHISPER_DEFAULT_SIZE) -> Any:
    size = model_size if model_size in VALID_WHISPER_MODELS else LOCAL_WHISPER_DEFAULT_SIZE
    with _WHISPER_CACHE_LOCK:
        cached = _WHISPER_CACHE.get(size)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel  # type: ignore

        download_root = BASE_DIR / "models" / "whisper"
        download_root.mkdir(parents=True, exist_ok=True)
        cpu_threads = max(4, min(16, os.cpu_count() or 4))
        model = WhisperModel(
            size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=2,
            download_root=str(download_root),
        )
        _WHISPER_CACHE[size] = model
        return model


def _decode_audio_16k_float32(ffmpeg_bin: str, audio_path: Path, *, vocal_enhance: bool = True) -> Any:
    """Decode audio to 16kHz mono float32 numpy array via FFmpeg with vocal-clarity filtering for ASR."""
    import numpy as np  # type: ignore

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
    ]
    if vocal_enhance:
        # Strip sub-bass rumble (<120Hz) & harsh high hiss (>7.2kHz), lift vocal presence (~2.8kHz),
        # and normalize dynamic level so Whisper hears lyrics & quiet dialogue clearly over BGM.
        cmd += [
            "-af",
            "highpass=f=120,lowpass=f=7200,equalizer=f=2800:t=q:w=1.2:g=3,dynaudnorm=f=150:g=15:p=0.9",
        ]
    cmd += [
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"Không giải mã được âm thanh 16kHz cho Whisper: {(proc.stderr or b'').decode('utf-8', 'replace')[:180]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _decode_audio_16k_stereo_center(ffmpeg_bin: str, audio_path: Path) -> Any:
    """Decode stereo 16kHz float32 and return (mid, side) arrays for center-vocal isolation in DSP alignment."""
    import numpy as np  # type: ignore

    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-af",
            "highpass=f=180,lowpass=f=4500",
            "-ac",
            "2",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    n_pairs = len(raw) // 2
    if n_pairs < 512:
        return None
    lr = raw[: n_pairs * 2].reshape(n_pairs, 2)
    mid = 0.5 * (lr[:, 0] + lr[:, 1])
    side = 0.5 * (lr[:, 0] - lr[:, 1])
    return mid.copy(), side.copy()


def compute_vocal_onset_envelope(
    audio_f32: Any,
    *,
    side_f32: Any | None = None,
    sample_rate: int = 16000,
    hop_ms: float = DSP_ALIGN_HOP_MS,
) -> tuple[Any, Any, float]:
    """Compute a 4ms-hop center-vocal (300–3600 Hz) tonality-weighted P-center onset & energy envelope."""
    import numpy as np  # type: ignore

    hop = max(16, int(round(sample_rate * hop_ms / 1000.0)))
    win = 256 if sample_rate <= 16000 else 512
    n_samples = int(len(audio_f32))
    if n_samples < win + hop:
        zeros = np.zeros(1, dtype=np.float32)
        return zeros, zeros, hop / float(sample_rate)

    frames = np.lib.stride_tricks.sliding_window_view(audio_f32, win)[::hop]
    window = np.hanning(win).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1))

    # Isolate human vocal formant band (300 Hz – 3600 Hz)
    freq_step = sample_rate / float(win)
    k_lo = max(1, int(round(300.0 / freq_step)))
    k_hi = min(spec.shape[1], int(round(3600.0 / freq_step)) + 1)
    vocal_mag = spec[:, k_lo:k_hi].astype(np.float32)

    # Stereo center-vocal subtraction: suppress wide stereo EDM synths/reverb (Side) leaving center vocals (Mid)
    if side_f32 is not None and len(side_f32) >= n_samples:
        s_frames = np.lib.stride_tricks.sliding_window_view(side_f32[:n_samples], win)[::hop]
        s_spec = np.abs(np.fft.rfft(s_frames * window, axis=1))[:, k_lo:k_hi].astype(np.float32)
        if s_spec.shape == vocal_mag.shape:
            vocal_mag = np.maximum(0.0, vocal_mag - 0.85 * s_spec)

    # Harmonic tonality weight = 1 - spectral_flatness (voices are harmonic/peaky; percussion/noise is flat)
    eps = np.float32(1e-6)
    geom_mean = np.exp(np.mean(np.log(vocal_mag + eps), axis=1))
    arith_mean = np.mean(vocal_mag, axis=1) + eps
    tonality = np.clip(1.0 - (geom_mean / arith_mean), 0.0, 1.0)

    # Vocal RMS energy (3-frame = 12ms smoothing)
    raw_energy = np.sqrt(np.mean(vocal_mag * vocal_mag, axis=1)) * (0.45 + 0.55 * tonality)
    if len(raw_energy) >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        energy = np.convolve(raw_energy, kernel, mode="same")
    else:
        energy = raw_energy

    # Positive half-wave log spectral flux weighted by vocal tonality
    log_mag = np.log1p(12.0 * vocal_mag)
    flux = np.zeros(len(energy), dtype=np.float32)
    if len(energy) > 1:
        diff = np.maximum(0.0, log_mag[1:] - log_mag[:-1])
        flux[1:] = np.sum(diff, axis=1) * (0.35 + 0.65 * tonality[1:])
        e_diff = np.maximum(0.0, energy[1:] - energy[:-1])
        e_norm = e_diff / (float(np.percentile(e_diff, 95)) + 1e-6)
        f_norm = flux[1:] / (float(np.percentile(flux[1:], 95)) + 1e-6)
        flux[1:] = 0.65 * f_norm + 0.35 * e_norm

    return energy, flux, hop / float(sample_rate)


def snap_words_to_vocal_onsets(
    words: list[dict[str, Any]],
    audio_f32: Any,
    *,
    side_f32: Any | None = None,
    sample_rate: int = 16000,
    hop_ms: float = DSP_ALIGN_HOP_MS,
) -> list[dict[str, Any]]:
    """Snap ASR word boundaries onto 4ms acoustic center-vocal P-center onsets without jumping early onto BGM beats."""
    if not words:
        return words
    import numpy as np  # type: ignore

    energy, flux, hop_sec = compute_vocal_onset_envelope(
        audio_f32,
        side_f32=side_f32,
        sample_rate=sample_rate,
        hop_ms=hop_ms,
    )
    n_frames = len(energy)
    if n_frames < 8:
        return words

    global_floor = float(np.percentile(energy, 20))
    audio_dur = len(audio_f32) / float(sample_rate)
    aligned = [dict(w) for w in words]

    for idx, w in enumerate(aligned):
        st = float(w["start_sec"])
        en = float(w["end_sec"])
        prev_end = float(aligned[idx - 1]["end_sec"]) if idx > 0 else 0.0
        next_start = float(aligned[idx + 1]["start_sec"]) if idx + 1 < len(aligned) else audio_dur
        is_phrase_head = idx == 0 or (st - prev_end) >= 0.22
        is_phrase_tail = idx + 1 == len(aligned) or (next_start - en) >= 0.22

        # 1. Asymmetric P-center vocal onset search: cap backward shift strictly (<=30ms) so we NEVER
        # jump early onto a pre-vocal drum/synth downbeat, while allowing forward lock (+120ms) onto the true vowel attack.
        back_rad = 0.030 if is_phrase_head else 0.020
        fwd_rad = 0.120 if is_phrase_head else 0.055
        lo_t = max(prev_end + 0.004 if idx > 0 else 0.0, st - back_rad)
        hi_t = min(en - 0.035, st + fwd_rad)
        if hi_t > lo_t + hop_sec * 2:
            i0 = max(0, int(round(lo_t / hop_sec)))
            i1 = min(n_frames - 1, int(round(hi_t / hop_sec)))
            if i1 > i0 + 1:
                win_e = energy[i0 : i1 + 1]
                win_f = flux[i0 : i1 + 1]
                local_floor = float(np.percentile(win_e, 25))
                local_peak = float(np.max(win_e))
                if local_peak > max(global_floor * 1.15, local_floor * 1.35):
                    # Gentle proximity weighting centered slightly forward (+25ms) so real vocal attack beats distant BGM
                    frame_times = (np.arange(i0, i1 + 1, dtype=np.float32)) * hop_sec
                    center_t = st + (0.025 if is_phrase_head else 0.010)
                    sigma = 0.065 if is_phrase_head else 0.035
                    prior = np.exp(-0.5 * ((frame_times - center_t) / sigma) ** 2)
                    weighted_flux = win_f * (0.45 + 0.55 * prior)
                    # Lock directly to the steepest center-vocal P-center onset (no backward walk into silence!)
                    onset_rel = int(np.argmax(weighted_flux))
                    snapped_st = round((i0 + onset_rel) * hop_sec, 3)
                    if idx > 0:
                        snapped_st = max(snapped_st, round(prev_end + 0.004, 3))
                    snapped_st = min(snapped_st, round(en - 0.035, 3))
                    w["start_sec"] = snapped_st
                    st = snapped_st

        # 2. Snap phrase-ending offset cleanly to vocal decay
        if is_phrase_tail:
            e_lo_t = max(st + 0.04, en - 0.06)
            e_hi_t = min(next_start - 0.01 if idx + 1 < len(aligned) else audio_dur, en + 0.10)
            if e_hi_t > e_lo_t + hop_sec * 2:
                j0 = max(0, int(round(e_lo_t / hop_sec)))
                j1 = min(n_frames - 1, int(round(e_hi_t / hop_sec)))
                if j1 > j0 + 1:
                    win_e = energy[j0 : j1 + 1]
                    local_floor = float(np.percentile(win_e, 20))
                    local_peak = float(np.max(win_e))
                    if local_peak > max(global_floor * 1.15, local_floor * 1.35):
                        decay_thresh = local_floor + 0.18 * (local_peak - local_floor)
                        active_idxs = np.where(win_e >= decay_thresh)[0]
                        if len(active_idxs) > 0:
                            snapped_en = round(min(e_hi_t, (j0 + int(active_idxs[-1]) + 1) * hop_sec), 3)
                            w["end_sec"] = max(round(st + 0.04, 3), snapped_en)

    return aligned


def _filter_whisper_hallucinations(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip degenerate Whisper repetition/punctuation hallucination bursts (e.g. '-Đ -Đ -Đ')."""
    cleaned: list[dict[str, Any]] = []
    run_tok = ""
    run_count = 0
    for w in words:
        raw = str(w.get("text") or "").strip()
        tok = _clean_word_token(raw)
        if not tok:
            continue
        if tok == run_tok:
            run_count += 1
            if run_count > 3:
                continue
        else:
            run_tok = tok
            run_count = 1
        cleaned.append(w)
    return cleaned


def _whisper_pass_score(words: list[dict[str, Any]]) -> float:
    """Score ASR pass by unique lexical content + time coverage (penalizes repetition loops)."""
    if not words:
        return 0.0
    toks = [_clean_word_token(w["text"]) for w in words]
    toks = [t for t in toks if len(t) >= 2]
    if not toks:
        return 0.0
    unique_ratio = len(set(toks)) / float(len(toks))
    if unique_ratio < 0.18 and len(toks) > 10:
        return 0.0
    span = max(0.0, float(words[-1]["end_sec"]) - float(words[0]["start_sec"]))
    return len(set(toks)) * 2.0 + len(toks) * 0.35 + span * 0.6


def _find_uncovered_audio_gaps(
    words: list[dict[str, Any]],
    audio_dur: float,
    *,
    min_gap_sec: float = 8.0,
) -> list[tuple[float, float]]:
    """Locate long untranscribed intervals (head, internal, or tail) so only gaps need non-VAD recovery."""
    if not words:
        return [(0.0, audio_dur)]
    gaps: list[tuple[float, float]] = []
    first_st = float(words[0]["start_sec"])
    if first_st >= min_gap_sec:
        gaps.append((0.0, min(audio_dur, first_st + 0.25)))
    for i in range(len(words) - 1):
        g0 = float(words[i]["end_sec"])
        g1 = float(words[i + 1]["start_sec"])
        if g1 - g0 >= min_gap_sec + 2.0:
            gaps.append((max(0.0, g0 - 0.25), min(audio_dur, g1 + 0.25)))
    last_en = float(words[-1]["end_sec"])
    if audio_dur - last_en >= min_gap_sec:
        gaps.append((max(0.0, last_en - 0.30), audio_dur))
    return gaps


def _transcribe_local_whisper_words(
    ffmpeg_bin: str,
    audio_mp3: Path,
    *,
    model_size: str = LOCAL_WHISPER_DEFAULT_SIZE,
    cancel_event: threading.Event | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Turbo 12-core Faster-Whisper + gap-only non-VAD stitching + 4ms DSP sub-10ms vocal-onset alignment."""
    size = model_size if model_size in VALID_WHISPER_MODELS else LOCAL_WHISPER_DEFAULT_SIZE
    if on_status:
        on_status(f"Local Whisper CPU ({size}) đang bóc tách siêu tốc + căn âm học <10ms…")
    model = _get_local_whisper_model(size)
    audio_f32 = _decode_audio_16k_float32(ffmpeg_bin, audio_mp3, vocal_enhance=True)
    audio_dur = len(audio_f32) / 16000.0

    def _collect_segments(segments_iter: Any, offset_sec: float = 0.0) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for seg in segments_iter:
            if cancel_event is not None and cancel_event.is_set():
                raise AICanceledError("Đã hủy tác vụ phiên âm Local Whisper.")
            seg_words = getattr(seg, "words", None) or []
            if seg_words:
                for w in seg_words:
                    txt = str(getattr(w, "word", "") or "").strip()
                    if not txt:
                        continue
                    st = round(offset_sec + max(0.0, float(getattr(w, "start", 0.0))), 3)
                    en = round(offset_sec + max(float(getattr(w, "start", 0.0)) + 0.04, float(getattr(w, "end", 0.1))), 3)
                    collected.append({"start_sec": st, "end_sec": en, "text": txt, "speaker": "S1"})
            else:
                txt = str(getattr(seg, "text", "") or "").strip()
                if txt:
                    st = round(offset_sec + max(0.0, float(getattr(seg, "start", 0.0))), 3)
                    en = round(offset_sec + max(float(getattr(seg, "start", 0.0)) + 0.2, float(getattr(seg, "end", 0.5))), 3)
                    collected.append({"start_sec": st, "end_sec": en, "text": txt, "speaker": "S1"})
        return _filter_whisper_hallucinations(collected)

    # Pass 1: Fast BatchedInferencePipeline (VAD) across all CPU cores
    try:
        from faster_whisper import BatchedInferencePipeline  # type: ignore

        batched = BatchedInferencePipeline(model=model)
        seg_iter, info = batched.transcribe(
            audio_f32,
            batch_size=8,
            beam_size=2,
            best_of=2,
            temperature=(0.0, 0.2),
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.55,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 320, "threshold": 0.30},
            word_timestamps=True,
            condition_on_previous_text=False,
        )
    except Exception:
        seg_iter, info = model.transcribe(
            audio_f32,
            beam_size=2,
            best_of=2,
            temperature=(0.0, 0.2),
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.55,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 320, "threshold": 0.30},
            word_timestamps=True,
            condition_on_previous_text=False,
        )
    src_lang = str(getattr(info, "language", "") or "unknown")
    words = _collect_segments(seg_iter, offset_sec=0.0)

    # Pass 2 (Gap-Only Turbo Recovery!): Only scan uncovered gaps >= 8s (e.g. loud EDM chorus/tail)
    # instead of wasting CPU re-transcribing segments VAD already nailed cleanly.
    gaps = _find_uncovered_audio_gaps(words, audio_dur, min_gap_sec=8.0) if audio_dur >= 8.0 else []
    if gaps:
        if on_status:
            on_status(f"Local Whisper CPU ({size}) đang vá {len(gaps)} đoạn nhạc mạnh + căn mốc <10ms…")
        recovered: list[dict[str, Any]] = []
        for g_start, g_end in gaps:
            s0 = max(0, int(round(g_start * 16000)))
            s1 = min(len(audio_f32), int(round(g_end * 16000)))
            if s1 - s0 < 16000:
                continue
            slice_audio = audio_f32[s0:s1]
            g_iter, g_info = model.transcribe(
                slice_audio,
                language=src_lang if src_lang != "unknown" else None,
                beam_size=3,
                best_of=3,
                temperature=(0.0, 0.2),
                compression_ratio_threshold=2.2,
                no_speech_threshold=0.55,
                vad_filter=False,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            if src_lang == "unknown" and getattr(g_info, "language", None):
                src_lang = str(g_info.language)
            g_words = _collect_segments(g_iter, offset_sec=g_start)
            if _whisper_pass_score(g_words) > 4.0:
                recovered.extend(g_words)
        if recovered:
            # Merge VAD words + gap-recovered words without overlapping existing covered spans
            existing_intervals = [(float(w["start_sec"]), float(w["end_sec"])) for w in words]
            for rw in recovered:
                r_mid = 0.5 * (float(rw["start_sec"]) + float(rw["end_sec"]))
                if not any(s - 0.08 <= r_mid <= e + 0.08 for s, e in existing_intervals):
                    words.append(rw)
            words.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))

    # Pass 3: Sub-10ms 4ms-hop Stereo Center-Vocal P-Center Alignment (suppresses wide EDM synths!)
    if words:
        stereo = _decode_audio_16k_stereo_center(ffmpeg_bin, audio_mp3)
        if stereo is not None:
            mid_f32, side_f32 = stereo
            words = snap_words_to_vocal_onsets(words, mid_f32, side_f32=side_f32, sample_rate=16000, hop_ms=DSP_ALIGN_HOP_MS)
        else:
            words = snap_words_to_vocal_onsets(words, audio_f32, sample_rate=16000, hop_ms=DSP_ALIGN_HOP_MS)

    return src_lang, group_words_into_cues(words)


def _call_structured_llm(
    *,
    provider: AIProvider,
    api_key: str,
    model_id: str,
    prompt: str,
    schema_cls: type[BaseModel],
    cancel_event: threading.Event | None = None,
) -> Any:
    """Invoke Gemini Interactions API (with thinking_level=low + 429/503 backoff & fallback) or OpenAI JSON chat."""
    if cancel_event is not None and cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng.")
    if provider.protocol == "gemini":
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        candidate_models = [model_id]
        for fallback_m in ("gemini-3.5-flash-lite", "gemini-3.8-flash"):
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        def _close_client() -> None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

        interaction = None
        last_exc: Exception | None = None
        for idx_m, cand in enumerate(candidate_models):
            if cancel_event is not None and cancel_event.is_set():
                raise AICanceledError("Đã tạm dừng.")
            create_kwargs: dict[str, Any] = {
                "model": cand,
                "input": prompt,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema_cls.model_json_schema(),
                },
            }
            if cand.lower().startswith(("gemini-4-", "gemini-3.")):
                create_kwargs["generation_config"] = {"thinking_level": "low"}
            try:
                interaction = _run_cancellable(
                    lambda kw=create_kwargs: client.interactions.create(**kw),
                    cancel_event=cancel_event,
                    on_cancel=_close_client,
                )
                break
            except AICanceledError:
                raise
            except Exception as exc:
                msg_low = str(exc).lower()
                if "thinking_level" in create_kwargs.get("generation_config", {}) and "thinking" in msg_low:
                    create_kwargs.pop("generation_config", None)
                    try:
                        interaction = _run_cancellable(
                            lambda kw=create_kwargs: client.interactions.create(**kw),
                            cancel_event=cancel_event,
                            on_cancel=_close_client,
                        )
                        break
                    except AICanceledError:
                        raise
                    except Exception as retry_exc:
                        exc = retry_exc
                        msg_low = str(exc).lower()
                last_exc = exc
                is_transient = any(
                    tok in msg_low
                    for tok in ("503", "service_unavailable", "high demand", "overloaded", "429", "resource_exhausted", "quota", "rate")
                )
                if is_transient and idx_m + 1 < len(candidate_models):
                    m_wait = _RETRY_DELAY_RE.search(str(exc))
                    wait_sec = min(5.0, max(1.0, float(m_wait.group(1)))) if m_wait else 1.5
                    logger.warning(
                        "Gemini model {} rate/load limited ({}), waiting {:.1f}s & rotating to {}",
                        cand,
                        str(exc)[:120],
                        wait_sec,
                        candidate_models[idx_m + 1],
                    )
                    if cancel_event is not None:
                        if cancel_event.wait(wait_sec):
                            raise AICanceledError("Đã tạm dừng.")
                    else:
                        time.sleep(wait_sec)
                    continue
                raise
        if interaction is None:
            raise last_exc or RuntimeError(f"Model {model_id} không phản hồi.")
        raw_text = interaction.output_text or ""
        if not raw_text.strip():
            raise RuntimeError(f"Model {model_id} không trả về dữ liệu JSON.")
        return schema_cls.model_validate_json(raw_text)

    parsed = _openai_chat_json(
        provider,
        api_key,
        model_id,
        [{"role": "user", "content": prompt}],
        cancel_event=cancel_event,
    )
    return schema_cls.model_validate(parsed)


def _build_story_bible(
    cues: list[dict[str, Any]],
    *,
    provider: AIProvider,
    api_key: str,
    model_id: str,
    target_lang: str,
    context_hint: str = "",
    cancel_event: threading.Event | None = None,
) -> StoryBibleSchema:
    """Pass 0 (VideoLingo / llm-subtrans pattern): analyze whole transcript for genre, glossary & pronoun rules."""
    if len(cues) < 3:
        return StoryBibleSchema()
    if cancel_event is not None and cancel_event.is_set():
        raise AICanceledError("Đã tạm dừng.")
    lang_name = TARGET_LANGUAGES.get(target_lang, "Vietnamese")
    sample_lines = [
        f"{c['id']} [{c.get('speaker', 'S1')}]: {c['original_text']}"
        for c in cues[:180]
    ]
    user_note = str(context_hint or "").strip()[:500]
    hint_block = f"\nUser-provided context/pronoun notes (MUST obey strictly):\n{user_note}\n" if user_note else ""
    vi_pronoun_guidance = (
        "- Because target language is Vietnamese (Vietsub), explicitly define natural Vietnamese pronoun rules "
        "(e.g., anh/em, tôi/cậu, ta/ngươi, tao/mày, or lyric 'tôi/em') matching the genre and speaker relationships; "
        "avoid robotic 'tôi - bạn' for intimate, emotional, or cinematic scenes.\n"
        if target_lang == "vi"
        else "- Define consistent tone, register, and form of address across speakers.\n"
    )
    prompt = (
        f"You are preparing a Story Bible before translating video subtitles into {lang_name}.\n"
        "Read the full transcript below and extract a concise, high-signal translation guide.\n"
        f"{vi_pronoun_guidance}"
        "- Extract proper nouns, character names, song titles, and domain terms into glossary as 'Source -> Target' (max 15 items).\n"
        "- Identify content genre (movie, anime, music_video, tutorial, vlog, news) and 1-2 sentence synopsis.\n"
        f"{hint_block}"
        'Return strictly valid JSON: {"source_language": string, "genre": string, "synopsis": string, '
        '"glossary": [string], "pronoun_rules": [string], "style_notes": string}\n'
        "Transcript:\n" + "\n".join(sample_lines)
    )
    try:
        call_kwargs: dict[str, Any] = {
            "provider": provider,
            "api_key": api_key,
            "model_id": model_id,
            "prompt": prompt,
            "schema_cls": StoryBibleSchema,
        }
        if cancel_event is not None:
            call_kwargs["cancel_event"] = cancel_event
        bible = _call_structured_llm(**call_kwargs)
        return bible
    except AICanceledError:
        raise
    except Exception as exc:
        logger.warning("Story bible Pass 0 skipped due to error: {}", str(exc)[:160])
        return StoryBibleSchema()


def _translate_cue_batch(
    batch_cues: list[dict[str, Any]],
    *,
    previous_context: list[dict[str, str]],
    bible: StoryBibleSchema,
    context_hint: str,
    video_title: str = "",
    provider: AIProvider,
    api_key: str,
    model_id: str,
    target_lang: str,
    for_dubbing: bool,
    cancel_event: threading.Event | None = None,
) -> TranslatedCueBatchSchema:
    lang_name = TARGET_LANGUAGES.get(target_lang, "Vietnamese")
    compact_input = [
        {
            "id": c["id"],
            "dur_sec": round(max(0.3, c["end_sec"] - c["start_sec"]), 2),
            "max_chars": max(18, int(round(max(1.0, c["end_sec"] - c["start_sec"]) * (16 if for_dubbing else 20)))),
            "speaker": c["speaker"],
            "text": c["original_text"],
        }
        for c in batch_cues
    ]

    bible_lines: list[str] = []
    if video_title.strip():
        bible_lines.append(f"- Video / Song Title: {video_title.strip()[:220]}")
    if bible.genre:
        bible_lines.append(f"- Genre: {bible.genre}")
    if bible.synopsis:
        bible_lines.append(f"- Synopsis: {bible.synopsis}")
    if bible.glossary:
        bible_lines.append(f"- Glossary: {'; '.join(bible.glossary[:15])}")
    if bible.pronoun_rules:
        bible_lines.append(f"- Pronoun & Register Rules: {'; '.join(bible.pronoun_rules[:10])}")
    if bible.style_notes:
        bible_lines.append(f"- Style Notes: {bible.style_notes}")
    if context_hint.strip():
        bible_lines.append(f"- User Priority Notes: {context_hint.strip()[:500]}")
    bible_block = ("Story Bible (follow consistently):\n" + "\n".join(bible_lines) + "\n\n") if bible_lines else ""

    prev_block = ""
    if previous_context:
        prev_lines = [f"[{p['speaker']}] {p['original']} => {p['translated']}" for p in previous_context]
        prev_block = (
            "Preceding lines (READ-ONLY sliding context for pronoun & flow continuity — do NOT re-output these):\n"
            + "\n".join(prev_lines)
            + "\n\n"
        )

    pacing_rule = (
        f"IMPORTANT FOR DUBBING: Keep translated_text concise enough to speak naturally aloud in {lang_name} within dur_sec (~<= max_chars)."
        if for_dubbing
        else f"SUBTITLE READABILITY (Netflix standard): Keep translated_text punchy and natural in {lang_name}, aiming near <= max_chars (<=20 chars/sec) so viewers read effortlessly."
    )
    prompt = (
        f"You are an expert cinematic subtitle translator & ASR proofreader ({lang_name} Vietsub/localization).\n"
        f"{bible_block}"
        f"{prev_block}"
        "Use a strict 3-stage Proofread -> Translate -> Reflect workflow for EVERY input cue:\n"
        "1. Phonetic ASR Proofreading (`corrected_original`): ASR on loud music or noisy video often mishears words into "
        "nonsensical near-homophones (e.g., 'sisters and crows' -> 'sisters and brothers', 'season harmony' -> 'peace and harmony', "
        "or misheard song/character names from the title). Inspect every awkward/out-of-place word against the title, rhyme, "
        "and surrounding cues, repair the real intended words in `corrected_original`, and NEVER translate obvious ASR nonsense literally.\n"
        f"2. Step 1 Faithful Draft (`draft_translation`): translate the REPAIRED `corrected_original` accurately into {lang_name}.\n"
        f"3. Step 2 Reflect & Polish (`translated_text`): refine into natural, idiomatic {lang_name} film/MV subtitles — "
        "strictly consistent with the Story Bible pronouns & glossary, no stiff literal phrasing.\n"
        f"{pacing_rule}\n"
        "Strict rules:\n"
        "- Return an item for EVERY SINGLE input id (do NOT skip, drop, or merge any id!).\n"
        "- For pure rhythmic vocalizations/scat sounds like 'dam dam didididi' or 'la la la', keep them catchy and rhythmic.\n"
        "- If a line is already in the target language, polish punctuation and flow cleanly.\n"
        'Return strictly valid JSON: {"source_language": string, "items": [{"id": int, "corrected_original": string, '
        '"draft_translation": string, "translated_text": string, "gender": "female"|"male", "tone": string}]}\n'
        f"Cues to translate:\n{json.dumps(compact_input, ensure_ascii=False)}"
    )
    call_kwargs: dict[str, Any] = {
        "provider": provider,
        "api_key": api_key,
        "model_id": model_id,
        "prompt": prompt,
        "schema_cls": TranslatedCueBatchSchema,
    }
    if cancel_event is not None:
        call_kwargs["cancel_event"] = cancel_event
    return _call_structured_llm(**call_kwargs)


def _translate_cues_with_model(
    cues: list[dict[str, Any]],
    *,
    provider: AIProvider,
    api_key: str,
    model_id: str,
    target_lang: str,
    for_dubbing: bool,
    context_hint: str = "",
    video_title: str = "",
    cancel_event: threading.Event | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Translate exact ASR cues 1-to-1 using quota-saving single-call (<=45 cues) or Story Bible + Sliding Window."""
    if not cues:
        return "unknown", []

    # Save API quota: only spend a separate Pass 0 Story Bible call when spanning multiple batches (>45 cues).
    # For single-batch videos (<=45 cues), the full transcript already fits in 1 request with full context!
    if len(cues) > TRANSLATION_BATCH_SIZE:
        if on_status:
            on_status(f"{provider.label} ({model_id}) đang phân tích ngữ cảnh, thuật ngữ & xưng hô…")
        bible = _build_story_bible(
            cues,
            provider=provider,
            api_key=api_key,
            model_id=model_id,
            target_lang=target_lang,
            context_hint=f"{video_title}\n{context_hint}".strip(),
        )
    else:
        vi_note = (
            "Maintain consistent natural Vietnamese pronouns (avoid robotic tôi-bạn) and consistent proper nouns across all cues."
            if target_lang == "vi"
            else "Maintain consistent register, tone, and proper nouns across all cues."
        )
        bible = StoryBibleSchema(style_notes=vi_note)

    detected_lang = bible.source_language if bible.source_language and bible.source_language != "unknown" else "unknown"
    translated_by_id: dict[int, TranslatedCueItemSchema] = {}
    previous_context: list[dict[str, str]] = []
    total_batches = max(1, math.ceil(len(cues) / TRANSLATION_BATCH_SIZE))

    for batch_idx, start_i in enumerate(range(0, len(cues), TRANSLATION_BATCH_SIZE), start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise AICanceledError("Đã hủy dịch phụ đề.")
        batch_cues = cues[start_i : start_i + TRANSLATION_BATCH_SIZE]
        if on_status:
            if total_batches > 1:
                on_status(
                    f"{provider.label} ({model_id}) đang rà soát & dịch lô {batch_idx}/{total_batches} ({len(cues)} câu)…"
                )
            else:
                on_status(
                    f"{provider.label} ({model_id}) đang rà soát ASR & dịch {len(cues)} câu theo ngữ cảnh từng từ…"
                )

        batch_resp = _translate_cue_batch(
            batch_cues,
            previous_context=previous_context[-SLIDING_CONTEXT_WINDOW:],
            bible=bible,
            context_hint=context_hint,
            video_title=video_title,
            provider=provider,
            api_key=api_key,
            model_id=model_id,
            target_lang=target_lang,
            for_dubbing=for_dubbing,
        )
        if detected_lang == "unknown" and batch_resp.source_language and batch_resp.source_language != "unknown":
            detected_lang = str(batch_resp.source_language)
        for item in batch_resp.items:
            translated_by_id[item.id] = item

        # Retry once for any missing IDs inside this batch so 0% of lines go untranslated
        missing_cues = [c for c in batch_cues if c["id"] not in translated_by_id]
        if missing_cues:
            try:
                retry_resp = _translate_cue_batch(
                    missing_cues,
                    previous_context=previous_context[-SLIDING_CONTEXT_WINDOW:],
                    bible=bible,
                    context_hint=context_hint,
                    video_title=video_title,
                    provider=provider,
                    api_key=api_key,
                    model_id=model_id,
                    target_lang=target_lang,
                    for_dubbing=for_dubbing,
                )
                for item in retry_resp.items:
                    translated_by_id[item.id] = item
            except Exception as exc:
                logger.warning("Retry for {} missing subtitle ids failed: {}", len(missing_cues), str(exc)[:140])

        # Advance sliding window with newly translated lines
        for c in batch_cues[-SLIDING_CONTEXT_WINDOW:]:
            tr_item = translated_by_id.get(c["id"])
            previous_context.append(
                {
                    "speaker": str(c.get("speaker") or "S1"),
                    "original": (tr_item.corrected_original.strip() if tr_item and tr_item.corrected_original.strip() else c["original_text"]),
                    "translated": (tr_item.translated_text.strip() if tr_item and tr_item.translated_text.strip() else c["original_text"]),
                }
            )

    result: list[dict[str, Any]] = []
    for cue in cues:
        tr = translated_by_id.get(cue["id"])
        updated = dict(cue)
        if tr is not None:
            if tr.corrected_original.strip():
                updated["original_text"] = tr.corrected_original.strip()
            polished = tr.translated_text.strip() or tr.draft_translation.strip()
            if polished:
                updated["translated_text"] = polished
            updated["gender"] = "male" if str(tr.gender or "").strip().lower().startswith("m") else "female"
            updated["tone"] = re.sub(r"\s+", " ", str(tr.tone or "natural and clear")).strip()[:80] or "natural and clear"
        result.append(updated)
    return detected_lang, result


def _transcribe_openai_words(
    provider: AIProvider,
    audio_mp3: Path,
    *,
    api_key: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Try OpenAI-compatible /audio/transcriptions with word timestamps; return None if unsupported."""
    stt_url = f"{provider.api_base.rstrip('/')}/audio/transcriptions"
    validate_url(stt_url)
    headers = auth_headers(provider, api_key)
    try:
        with audio_mp3.open("rb") as fh:
            resp = public_request(
                "POST",
                stt_url,
                headers=headers,
                files={"file": ("speech.mp3", fh, "audio/mpeg")},
                data=[
                    ("model", "whisper-1"),
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "word"),
                    ("timestamp_granularities[]", "segment"),
                ],
                timeout=(15, 180),
            )
    except requests.RequestException:
        return None
    try:
        if resp.status_code >= 400:
            return None
        stt_data = resp.json()
    except ValueError:
        return None
    finally:
        resp.close()

    src_lang = str(stt_data.get("language") or "unknown")
    raw_words = stt_data.get("words") or []
    if isinstance(raw_words, list) and raw_words:
        words = [
            {
                "start_sec": round(float(w.get("start") or 0.0), 3),
                "end_sec": round(max(float(w.get("start") or 0.0) + 0.05, float(w.get("end") or 0.0)), 3),
                "text": str(w.get("word") or w.get("text") or "").strip(),
                "speaker": "S1",
            }
            for w in raw_words
            if str(w.get("word") or w.get("text") or "").strip()
        ]
        if words:
            return src_lang, group_words_into_cues(words)

    raw_segs = stt_data.get("segments") or []
    if isinstance(raw_segs, list) and raw_segs:
        cues = []
        for idx, s in enumerate(raw_segs[:MAX_SEGMENTS]):
            txt = str(s.get("text") or "").strip()
            if not txt:
                continue
            st = round(float(s.get("start") or 0.0), 3)
            en = round(max(st + 0.25, float(s.get("end") or st + 1.0)), 3)
            cues.append(
                {
                    "id": len(cues),
                    "start_sec": st,
                    "end_sec": en,
                    "speaker": "S1",
                    "gender": "female",
                    "tone": "natural and clear",
                    "original_text": txt,
                    "translated_text": txt,
                    "words": [],
                }
            )
        if cues:
            return src_lang, cues
    return None


def _build_chunk_multimodal_prompt(chunk_dur: float, target_lang: str, for_dubbing: bool) -> str:
    lang_name = TARGET_LANGUAGES.get(target_lang, "Vietnamese")
    dub_guidance = (
        f"Keep each translated_text concise to speak aloud in {lang_name} within (end_sec - start_sec)."
        if for_dubbing
        else f"Translate EVERY line into crisp, natural {lang_name} subtitles."
    )
    return (
        f"Listen carefully to this {chunk_dur:.1f}-second audio clip (timestamps start at 0.0 and must stay <= {chunk_dur:.1f}).\n"
        f"Transcribe EVERY spoken or sung phrase with accurate start_sec and end_sec relative to 0.0, "
        f"and translate into {lang_name}. {dub_guidance}\n"
        "Rules: split into short 1.2s–4.0s phrases; NEVER skip repeated chorus lines; NEVER output timestamps past clip duration.\n"
        'Return strictly valid JSON: {"source_language": string, "segments": [{"start_sec": number, "end_sec": number, "speaker": string, "gender": "female"|"male", "tone": string, "original_text": string, "translated_text": string}]}'
    )


def transcribe_and_translate_audio(
    ffmpeg_bin: str,
    audio_mp3: Path,
    work_dir: Path,
    *,
    provider: AIProvider,
    api_key: str,
    model_id: str,
    duration_sec: float,
    target_lang: str,
    for_dubbing: bool,
    context_hint: str = "",
    video_title: str = "",
    asr_engine: str = "auto",
    whisper_model: str = LOCAL_WHISPER_DEFAULT_SIZE,
    source_subtitle_path: Path | None = None,
    source_media_path: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Decoupled ASR -> Translation pipeline: 0-quota captions / Local Whisper CPU -> any text LLM in Settings."""
    chosen_model = normalize_model_id(model_id, default_transcribe_model_for(provider))
    engine = asr_engine if asr_engine in VALID_ASR_ENGINES else "auto"

    # Zero-quota Path A: Reuse official/auto platform captions when available in 'auto' mode
    if engine == "auto" and source_subtitle_path is not None and source_subtitle_path.is_file():
        caption_cues = parse_platform_subtitle_file(source_subtitle_path)
        if caption_cues:
            if on_status:
                on_status(f"Dùng sẵn {len(caption_cues)} câu phụ đề gốc từ nguồn (0 quota ASR)…")
            return _translate_cues_with_model(
                caption_cues,
                provider=provider,
                api_key=api_key,
                model_id=chosen_model,
                target_lang=target_lang,
                for_dubbing=for_dubbing,
                context_hint=context_hint,
                video_title=video_title,
                cancel_event=cancel_event,
                on_status=on_status,
            )

    # Zero-quota Path B: Local Whisper CPU (faster-whisper INT8 + vocal filter + initial_prompt + word timestamps)
    # Decode directly from source media when provided to avoid ~50-70ms MP3 priming drift.
    if engine == "local_whisper" or (engine == "auto" and is_local_whisper_available()):
        if not is_local_whisper_available():
            raise RuntimeError("Chưa cài thư viện faster-whisper trên máy để chạy Local Whisper CPU.")
        try:
            src_lang, cues = _transcribe_local_whisper_words(
                ffmpeg_bin,
                source_media_path if (source_media_path is not None and source_media_path.is_file()) else audio_mp3,
                model_size=whisper_model,
                cancel_event=cancel_event,
                on_status=on_status,
            )
            if cues:
                tr_lang, translated = _translate_cues_with_model(
                    cues,
                    provider=provider,
                    api_key=api_key,
                    model_id=chosen_model,
                    target_lang=target_lang,
                    for_dubbing=for_dubbing,
                    context_hint=context_hint,
                    video_title=video_title,
                    cancel_event=cancel_event,
                    on_status=on_status,
                )
                return (src_lang if src_lang != "unknown" else tr_lang), translated
            if engine == "local_whisper":
                raise RuntimeError("Local Whisper không phát hiện được lời thoại trong âm thanh.")
        except AICanceledError:
            raise
        except Exception as exc:
            if engine == "local_whisper":
                raise
            logger.warning("Local Whisper CPU failed, falling back to Cloud ASR: {}", str(exc)[:180])

    chunks = _split_audio_chunks(
        ffmpeg_bin,
        audio_mp3,
        duration_sec,
        work_dir,
        cancel_event=cancel_event,
    )

    # Path 1: Gemini — use dedicated gemini-3.5-transcribe word-level ASR across silence-aligned ~40s chunks
    if provider.protocol == "gemini":
        if on_status:
            on_status(f"Đang bóc tách từng từ chuẩn thời gian ({len(chunks)} phân đoạn theo khoảng lặng)…")
        all_words: list[dict[str, Any]] = []
        asr_ok = True
        try:
            with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as pool:
                futs = {
                    pool.submit(
                        _transcribe_words_gemini_chunk,
                        chunk_path,
                        offset_sec=start_off,
                        chunk_dur=(end_off - start_off),
                        api_key=api_key,
                    ): idx
                    for idx, (start_off, end_off, chunk_path) in enumerate(chunks)
                }
                for fut in as_completed(futs):
                    if cancel_event is not None and cancel_event.is_set():
                        raise AICanceledError("Đã hủy tác vụ phiên âm.")
                    all_words.extend(fut.result())
        except AICanceledError:
            raise
        except Exception as exc:
            logger.warning("Dedicated gemini-3.5-transcribe failed, falling back to chunked multimodal: {}", exc)
            asr_ok = False

        if asr_ok and all_words:
            cues = group_words_into_cues(all_words)
            if cues:
                return _translate_cues_with_model(
                    cues,
                    provider=provider,
                    api_key=api_key,
                    model_id=chosen_model,
                    target_lang=target_lang,
                    for_dubbing=for_dubbing,
                    context_hint=context_hint,
                    cancel_event=cancel_event,
                    on_status=on_status,
                )

        # Fallback if dedicated ASR returned empty or unavailable: chunked multimodal with chosen_model
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        all_segs: list[dict[str, Any]] = []
        detected_lang = "unknown"
        for start_off, end_off, chunk_path in chunks:
            if cancel_event is not None and cancel_event.is_set():
                raise AICanceledError("Đã hủy tác vụ phiên âm.")
            chunk_dur = max(1.0, end_off - start_off)
            prompt = _build_chunk_multimodal_prompt(chunk_dur, target_lang, for_dubbing)
            audio_b64 = base64.b64encode(chunk_path.read_bytes()).decode("ascii")
            interaction = client.interactions.create(
                model=chosen_model,
                input=[
                    {"type": "text", "text": prompt},
                    {"type": "audio", "data": audio_b64, "mime_type": "audio/mp3"},
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": TranscriptPlanSchema.model_json_schema(),
                },
            )
            if interaction.output_text:
                plan = TranscriptPlanSchema.model_validate_json(interaction.output_text)
                if plan.source_language and plan.source_language != "unknown":
                    detected_lang = plan.source_language
                all_segs.extend(_normalize_segments(plan.segments, end_off, offset_sec=start_off))
        return detected_lang, all_segs[:MAX_SEGMENTS]

    # Path 2: OpenAI-compatible / Custom Provider
    if on_status:
        on_status(f"{provider.label} đang bóc tách lời thoại theo mốc thời gian…")
    stt_res = _transcribe_openai_words(provider, audio_mp3, api_key=api_key)
    if stt_res is not None:
        src_lang, cues = stt_res
        tr_lang, translated = _translate_cues_with_model(
            cues,
            provider=provider,
            api_key=api_key,
            model_id=chosen_model,
            target_lang=target_lang,
            for_dubbing=for_dubbing,
            context_hint=context_hint,
            cancel_event=cancel_event,
            on_status=on_status,
        )
        return (src_lang if src_lang != "unknown" else tr_lang), translated

    # Fallback: chunked multimodal input_audio chat (40s chunks prevent timestamp drift!)
    all_segs = []
    detected_lang = "unknown"
    for start_off, end_off, chunk_path in chunks:
        if cancel_event is not None and cancel_event.is_set():
            raise AICanceledError("Đã hủy tác vụ phiên âm.")
        chunk_dur = max(1.0, end_off - start_off)
        prompt = _build_chunk_multimodal_prompt(chunk_dur, target_lang, for_dubbing)
        audio_b64 = base64.b64encode(chunk_path.read_bytes()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                ],
            }
        ]
        try:
            parsed = _openai_chat_json(provider, api_key, chosen_model, messages)
            plan = TranscriptPlanSchema.model_validate(parsed)
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"{provider.label} ({chosen_model}) không hỗ trợ đầu vào audio và không có /audio/transcriptions: {exc}"
            ) from exc
        if plan.source_language and plan.source_language != "unknown":
            detected_lang = plan.source_language
        all_segs.extend(_normalize_segments(plan.segments, end_off, offset_sec=start_off))
    return detected_lang, all_segs[:MAX_SEGMENTS]


def _srt_timestamp(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    hours, rem = divmod(ms_total, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def _ass_timestamp(sec: float) -> str:
    cs_total = max(0, int(round(sec * 100)))
    hours, rem = divmod(cs_total, 360000)
    minutes, rem = divmod(rem, 6000)
    seconds, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return (
        str(text or "")
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\n", r"\N")
    )


def _wrap_tokens(tokens: list[str], max_chars: int) -> list[list[str]]:
    if not tokens:
        return []
    lines: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for tok in tokens:
        add = len(tok) + (1 if cur else 0)
        if cur and cur_len + add > max_chars and len(lines) < 2:
            lines.append(cur)
            cur = [tok]
            cur_len = len(tok)
        else:
            cur.append(tok)
            cur_len += add
    if cur:
        lines.append(cur)
    return lines


def _map_translated_word_timings(seg: dict[str, Any], tokens: list[str]) -> list[tuple[float, float]]:
    """Map exact acoustic word timestamps from ASR onto translated words proportionally."""
    n = len(tokens)
    if n == 0:
        return []
    seg_start = float(seg["start_sec"])
    seg_end = max(seg_start + 0.25, float(seg["end_sec"]))
    asr_words = [
        w
        for w in (seg.get("words") or [])
        if isinstance(w, dict) and float(w.get("end_sec", 0)) > float(w.get("start_sec", 0))
    ]

    # Build a piecewise-linear speech timeline from actual spoken word spans (skipping internal pauses!)
    if asr_words:
        spans: list[tuple[float, float]] = []
        for w in asr_words:
            ws = max(seg_start, float(w["start_sec"]))
            we = min(seg_end, max(ws + 0.04, float(w["end_sec"])))
            spans.append((ws, we))
        total_active = sum(we - ws for ws, we in spans) or (seg_end - seg_start)

        def _time_at_fraction(frac: float) -> float:
            target = max(0.0, min(1.0, frac)) * total_active
            acc = 0.0
            for ws, we in spans:
                d = we - ws
                if acc + d >= target:
                    return ws + (target - acc)
                acc += d
            return spans[-1][1]
    else:
        total_span = seg_end - seg_start

        def _time_at_fraction(frac: float) -> float:
            return seg_start + max(0.0, min(1.0, frac)) * total_span

    weights = [max(1.0, len(re.sub(r"[^\w]", "", tok)) + 0.5) for tok in tokens]
    total_w = sum(weights) or float(n)
    timings: list[tuple[float, float]] = []
    acc_w = 0.0
    for w in weights:
        t0 = _time_at_fraction(acc_w / total_w)
        acc_w += w
        t1 = _time_at_fraction(acc_w / total_w)
        timings.append((t0, max(t0 + 0.06, t1)))
    return timings


def write_srt_file(segments: list[dict[str, Any]], out_srt: Path, *, bilingual: bool = False) -> Path:
    blocks: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        ts = f"{_srt_timestamp(seg['start_sec'])} --> {_srt_timestamp(seg['end_sec'])}"
        if bilingual and seg["original_text"].lower() != seg["translated_text"].lower():
            body = f"{seg['original_text']}\n{seg['translated_text']}"
        else:
            body = seg["translated_text"]
        blocks.append(f"{idx}\n{ts}\n{body}")
    out_srt.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return out_srt


def write_ass_file(
    segments: list[dict[str, Any]],
    out_ass: Path,
    *,
    width: int,
    height: int,
    style_preset: str = "classic",
    animation: str = "reveal",
    bilingual: bool = False,
    font_name: str = _BUNDLED_FONT_FAMILY,
) -> Path:
    play_w = max(320, int(width))
    play_h = max(320, int(height))
    is_vertical = play_h > play_w

    base_font = int(round(play_w * 0.056)) if is_vertical else int(round(play_h * 0.062))
    font_size = max(28, min(74, base_font))
    orig_font_size = max(20, int(round(font_size * 0.70)))
    margin_v = int(round(play_h * (0.14 if is_vertical else 0.055)))
    margin_lr = int(round(play_w * 0.06))
    outline = max(3, int(round(font_size * 0.085)))
    shadow = max(1, int(round(font_size * 0.045)))
    max_chars = 24 if is_vertical else 38

    anim = animation if animation in VALID_SUBTITLE_ANIMATIONS else "reveal"

    # ASS colors are &HAABBGGRR
    if style_preset == "box":
        primary = "&H004CE6FF" if anim == "karaoke" else "&H00FFFFFF"
        secondary = "&H00CCCCCC" if anim == "karaoke" else "&H000000FF"
        back = "&H80000000"
        border_style = 3
        outline_val = max(7, int(round(font_size * 0.22)))
        shadow_val = 0
    elif style_preset == "cinema":
        primary = "&H004CE6FF"  # Cinema Yellow (#FFE64C)
        secondary = "&H00B5B5B5" if anim == "karaoke" else "&H000000FF"
        back = "&H80000000"
        border_style = 1
        outline_val = outline
        shadow_val = shadow
    else:
        primary = "&H004CE6FF" if anim == "karaoke" else "&H00FFFFFF"
        secondary = "&H00D8D8D8" if anim == "karaoke" else "&H000000FF"
        back = "&H80000000"
        border_style = 1
        outline_val = outline
        shadow_val = shadow

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\n"
        f"PlayResY: {play_h}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{font_size},{primary},{secondary},&H00000000,{back},"
        f"-1,0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},2,{margin_lr},{margin_lr},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for seg_idx, seg in enumerate(segments):
        seg_start = float(seg["start_sec"])
        seg_end = max(seg_start + 0.25, float(seg["end_sec"]))
        tokens = str(seg["translated_text"] or "").split()
        if not tokens:
            continue
        lines = _wrap_tokens(tokens, max_chars)
        flat_tokens = [tok for line in lines for tok in line]
        timings = _map_translated_word_timings(seg, flat_tokens)

        # Keep word reveals strictly anchored to spoken timestamps, while giving short
        # utterances a comfortable trailing hold so viewers can finish reading before disappear.
        min_readable = min(2.0, max(1.1, 0.26 * len(flat_tokens)))
        display_end = max(seg_end, seg_start + min_readable)
        if seg_idx + 1 < len(segments):
            next_start = float(segments[seg_idx + 1]["start_sec"])
            display_end = min(display_end, max(seg_end, next_start - 0.06))
        display_end = max(seg_end, display_end)

        orig_prefix = ""
        if bilingual and seg["original_text"].lower() != seg["translated_text"].lower():
            orig_lines = _wrap_tokens(str(seg["original_text"]).split(), int(max_chars * 1.25))
            orig_str = r"\N".join(_escape_ass(" ".join(ln)) for ln in orig_lines)
            orig_prefix = rf"{{\fs{orig_font_size}\c&H00D8D8D8&}}{orig_str}\N{{\r}}"

        if anim == "reveal":
            if style_preset == "box":
                # Progressive box reveal: one tight step per word so the box grows naturally with each spoken word
                for step_idx in range(len(flat_tokens)):
                    t0 = timings[step_idx][0]
                    t1 = timings[step_idx + 1][0] if step_idx + 1 < len(flat_tokens) else display_end
                    if t1 <= t0:
                        t1 = min(display_end, t0 + 0.08)
                    shown_lines: list[str] = []
                    count = 0
                    for ln in lines:
                        take = min(len(ln), max(0, (step_idx + 1) - count))
                        if take > 0:
                            shown_lines.append(_escape_ass(" ".join(ln[:take])))
                        count += len(ln)
                    text_body = orig_prefix + r"\N".join(shown_lines)
                    events.append(
                        f"Dialogue: 0,{_ass_timestamp(t0)},{_ass_timestamp(t1)},Default,,0,0,0,,{text_body}"
                    )
            else:
                # Smooth zero-jitter word-by-word reveal inside a single centered dialogue event
                parts_by_line: list[str] = []
                tok_idx = 0
                for ln in lines:
                    word_chunks: list[str] = []
                    for tok in ln:
                        t0, _ = timings[tok_idx]
                        delay_ms = max(0, int(round((t0 - seg_start) * 1000)))
                        end_ms = delay_ms + 70
                        escaped = _escape_ass(tok)
                        if delay_ms <= 20:
                            word_chunks.append(rf"{{\1a&H00&\3a&H00&\4a&H00&}}{escaped}")
                        else:
                            word_chunks.append(
                                rf"{{\1a&HFF&\3a&HFF&\4a&HFF&\t({delay_ms},{end_ms},\1a&H00&\3a&H00&\4a&H00&)}}{escaped}"
                            )
                        tok_idx += 1
                    parts_by_line.append(" ".join(word_chunks))
                text_body = orig_prefix + r"\N".join(parts_by_line)
                events.append(
                    f"Dialogue: 0,{_ass_timestamp(seg_start)},{_ass_timestamp(display_end)},Default,,0,0,0,,{text_body}"
                )
        elif anim == "karaoke":
            parts_by_line = []
            tok_idx = 0
            cursor_sec = seg_start
            for ln in lines:
                word_chunks = []
                for tok in ln:
                    t0, t1 = timings[tok_idx]
                    lead_cs = max(0, int(round((t0 - cursor_sec) * 100)))
                    dur_cs = max(8, int(round((t1 - t0) * 100)))
                    escaped = _escape_ass(tok)
                    prefix_k = rf"{{\k{lead_cs}}}" if lead_cs > 0 else ""
                    word_chunks.append(rf"{prefix_k}{{\kf{dur_cs}}}{escaped}")
                    cursor_sec = max(cursor_sec, t1)
                    tok_idx += 1
                parts_by_line.append(" ".join(word_chunks))
            text_body = orig_prefix + r"\N".join(parts_by_line)
            events.append(
                f"Dialogue: 0,{_ass_timestamp(seg_start)},{_ass_timestamp(display_end)},Default,,0,0,0,,{text_body}"
            )
        else:
            tr_str = r"\N".join(_escape_ass(" ".join(ln)) for ln in lines)
            text_body = orig_prefix + tr_str
            events.append(
                f"Dialogue: 0,{_ass_timestamp(seg_start)},{_ass_timestamp(display_end)},Default,,0,0,0,,{text_body}"
            )

    out_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_ass


def _assign_speaker_voices(segments: list[dict[str, Any]], requested_voice: str) -> dict[str, str]:
    if requested_voice != "auto" and requested_voice in VALID_VOICE_IDS:
        return {seg["speaker"]: requested_voice for seg in segments}
    mapping: dict[str, str] = {}
    f_idx = 0
    m_idx = 0
    for seg in segments:
        spk = seg["speaker"]
        if spk in mapping:
            continue
        if seg.get("gender") == "male":
            mapping[spk] = _MALE_VOICES[m_idx % len(_MALE_VOICES)]
            m_idx += 1
        else:
            mapping[spk] = _FEMALE_VOICES[f_idx % len(_FEMALE_VOICES)]
            f_idx += 1
    return mapping


def _resolve_piper_voice_spec(target_lang: str, voice_id: str) -> tuple[str, str, int | None] | None:
    """Resolve (url_prefix, model_stem, speaker_id) for offline Piper ONNX voices."""
    if target_lang == "vi":
        return _PIPER_VI_VOICE_SPECS.get(voice_id) or _PIPER_VI_VOICE_SPECS["Puck"]
    entry = _PIPER_VOICE_CATALOG.get(target_lang)
    if entry is None:
        return None
    url_prefix, model_stem = entry
    return url_prefix, model_stem, None


def _get_piper_voice(target_lang: str, voice_id: str = "Puck") -> tuple[Any, int | None] | None:
    """Load (and auto-download once if needed) an official/community offline Piper ONNX neural voice."""
    spec = _resolve_piper_voice_spec(target_lang, voice_id)
    if spec is None:
        return None
    url_prefix, voice_name, speaker_id = spec
    with _PIPER_CACHE_LOCK:
        cached = _PIPER_CACHE.get(voice_name)
        if cached is not None:
            return cached, speaker_id
        from piper import PiperVoice  # type: ignore

        models_dir = BASE_DIR / "models" / "piper"
        models_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = models_dir / f"{voice_name}.onnx"
        json_path = models_dir / f"{voice_name}.onnx.json"
        for suffix, dest in ((".onnx", onnx_path), (".onnx.json", json_path)):
            if not dest.is_file() or dest.stat().st_size < 64:
                url = f"{url_prefix}{suffix}"
                validate_url(url)
                resp = public_request("GET", url, timeout=(15, 180))
                try:
                    if resp.status_code >= 400:
                        raise RuntimeError(f"Không tải được giọng Piper {voice_name}{suffix} (HTTP {resp.status_code}).")
                    dest.write_bytes(resp.content)
                finally:
                    resp.close()
        loaded = PiperVoice.load(onnx_path, config_path=json_path)
        _PIPER_CACHE[voice_name] = loaded
        return loaded, speaker_id


def _synthesize_piper_tts_wav(
    text: str,
    *,
    target_lang: str,
    voice: str,
    ffmpeg_bin: str,
    out_wav: Path,
) -> Path:
    """Ultra-fast (~50ms/sentence) 100% offline local neural TTS via Piper ONNX (0 drops, 0 quota)."""
    clean_text = re.sub(r"[\r\n\t]+", " ", str(text or "")).strip()
    if not clean_text or not re.search(r"\w", clean_text):
        raise RuntimeError("Câu thoại rỗng, bỏ qua tổng hợp Piper TTS.")

    resolved = _get_piper_voice(target_lang, voice_id=voice)
    if resolved is None:
        # Languages without an official Piper voice (e.g. ja/ko) fall back cleanly to Edge-TTS
        return _synthesize_edge_tts_wav(
            clean_text,
            target_lang=target_lang,
            voice=voice,
            ffmpeg_bin=ffmpeg_bin,
            out_wav=out_wav,
        )
    piper_voice, speaker_id = resolved

    from piper import SynthesisConfig  # type: ignore

    syn_cfg = SynthesisConfig(speaker_id=speaker_id, length_scale=0.96, noise_scale=0.667, noise_w_scale=0.8)
    raw_wav = out_wav.with_suffix(".piper_raw.wav")
    with wave.open(str(raw_wav), "wb") as wf:
        piper_voice.synthesize_wav(clean_text, wf, syn_config=syn_cfg)

    if not raw_wav.is_file() or raw_wav.stat().st_size <= 64:
        raise RuntimeError("Piper ONNX không tạo được tệp WAV.")

    # Resample Piper native 22050Hz -> 24000Hz mono s16 cleanly in <15ms
    code, stderr = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_wav),
            "-ac",
            "1",
            "-ar",
            str(DUB_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(out_wav),
        ],
        timeout=30.0,
    )
    raw_wav.unlink(missing_ok=True)
    if code != 0 or not out_wav.is_file() or out_wav.stat().st_size <= 64:
        raise RuntimeError(f"Không chuẩn hóa được âm thanh Piper WAV: {stderr.strip()[:160]}")
    return out_wav


def _synthesize_edge_tts_wav(
    text: str,
    *,
    target_lang: str,
    voice: str,
    ffmpeg_bin: str,
    out_wav: Path,
) -> Path:
    """Synthesize natural neural speech via free Microsoft Edge-TTS (0 API key, 0 quota) with fast retry & gTTS safety net."""
    import edge_tts  # type: ignore

    clean_text = re.sub(r"[\r\n\t]+", " ", str(text or "")).strip()
    if not clean_text or not re.search(r"\w", clean_text):
        raise RuntimeError("Câu thoại rỗng, bỏ qua tổng hợp Edge-TTS.")

    edge_voice = _resolve_edge_tts_voice(target_lang, voice)
    tmp_mp3 = out_wav.with_suffix(".edge.mp3")

    async def _gen() -> None:
        comm = edge_tts.Communicate(
            clean_text,
            voice=edge_voice,
            connect_timeout=8,
            receive_timeout=12,
        )
        await comm.save(str(tmp_mp3))

    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            if tmp_mp3.exists():
                tmp_mp3.unlink()
            asyncio.run(_gen())
            if tmp_mp3.is_file() and tmp_mp3.stat().st_size > 128:
                last_exc = None
                time.sleep(0.18)
                break
            last_exc = RuntimeError(f"Edge-TTS ({edge_voice}) trả về tệp âm thanh rỗng.")
        except Exception as exc:
            last_exc = exc
        if attempt < 5:
            time.sleep(0.35 * (attempt + 1))

    # Seamless safety-net fallback to free gTTS so no sentence is EVER left silent
    if last_exc is not None or not tmp_mp3.is_file() or tmp_mp3.stat().st_size <= 128:
        gtts_lang = {"zh": "zh-CN"}.get(target_lang, target_lang or "vi")
        try:
            from gtts import gTTS  # type: ignore

            logger.info("Edge-TTS dropped after 6 tries; seamlessly falling back to gTTS ({}) for segment.", gtts_lang)
            gTTS(text=clean_text, lang=gtts_lang).save(str(tmp_mp3))
            last_exc = None
        except Exception as gtts_exc:
            raise RuntimeError(f"Edge-TTS ({edge_voice}) thất bại sau 6 lần thử ({last_exc}); gTTS fallback lỗi: {gtts_exc}")

    if not tmp_mp3.is_file() or tmp_mp3.stat().st_size == 0:
        raise RuntimeError(f"Không tạo được âm thanh TTS cho câu: {clean_text[:40]}")
    code, stderr = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(tmp_mp3),
            "-ac",
            "1",
            "-ar",
            str(DUB_SAMPLE_RATE),
            "-f",
            "wav",
            str(out_wav),
        ],
        timeout=45.0,
    )
    try:
        tmp_mp3.unlink()
    except OSError:
        pass
    if code != 0 or not out_wav.is_file() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"Không chuyển đổi được âm thanh Edge-TTS sang WAV: {stderr.strip()[:160]}")
    return out_wav


def _synthesize_segment_wav(
    text: str,
    *,
    provider: AIProvider | None,
    tts_backend: str = "",
    api_key: str,
    tts_model: str,
    voice: str,
    tone: str,
    lang_name: str,
    target_lang: str,
    ffmpeg_bin: str,
    out_wav: Path,
) -> Path:
    if provider is None:
        if tts_backend == PIPER_TTS_PROVIDER_ID:
            return _synthesize_piper_tts_wav(
                text,
                target_lang=target_lang,
                voice=voice,
                ffmpeg_bin=ffmpeg_bin,
                out_wav=out_wav,
            )
        return _synthesize_edge_tts_wav(
            text,
            target_lang=target_lang,
            voice=voice,
            ffmpeg_bin=ffmpeg_bin,
            out_wav=out_wav,
        )
    chosen_tts = normalize_model_id(tts_model, default_tts_model_for(provider))
    if provider.protocol == "gemini":
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        style = f"Natural {lang_name} voiceover, {tone}"
        interaction = client.interactions.create(
            model=chosen_tts,
            input=[
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "speech_metadata",
                                    "style": style,
                                }
                            ],
                        }
                    ],
                }
            ],
            response_format={
                "type": "audio",
                "mime_type": "audio/wav",
                "sample_rate": DUB_SAMPLE_RATE,
            },
            generation_config={
                "speech_config": [{"voice": voice}],
            },
        )
        out_audio = interaction.output_audio
        if not out_audio or not out_audio.data:
            raise RuntimeError(f"Model {chosen_tts} không trả về dữ liệu âm thanh.")
        out_wav.write_bytes(base64.b64decode(out_audio.data))
        return out_wav

    speech_url = f"{provider.api_base.rstrip('/')}/audio/speech"
    validate_url(speech_url)
    oai_voice = _OPENAI_VOICE_MAP.get(voice, voice.lower() if voice else "nova")
    headers = {
        "Content-Type": "application/json",
        **auth_headers(provider, api_key),
    }
    payload = {
        "model": chosen_tts,
        "input": text,
        "voice": oai_voice,
        "response_format": "wav",
    }
    resp = public_request("POST", speech_url, headers=headers, json=payload, timeout=(15, 120))
    try:
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{provider.label} /audio/speech ({chosen_tts}) trả lỗi HTTP {resp.status_code}: {(resp.text or '')[:180]}"
            )
        audio_bytes = resp.content
    finally:
        resp.close()

    if not audio_bytes:
        raise RuntimeError(f"{provider.label} không trả về dữ liệu âm thanh.")

    if audio_bytes[:4] == b"RIFF":
        out_wav.write_bytes(audio_bytes)
        return out_wav

    raw_tmp = out_wav.with_suffix(".raw_audio")
    raw_tmp.write_bytes(audio_bytes)
    code, stderr = _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_tmp),
            "-ac",
            "1",
            "-ar",
            str(DUB_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(out_wav),
        ],
        timeout=60.0,
    )
    raw_tmp.unlink(missing_ok=True)
    if code != 0 or not out_wav.is_file():
        raise RuntimeError(f"Không giải mã được âm thanh TTS từ {provider.label}: {stderr.strip()[:180]}")
    return out_wav


def _read_pcm_mono_24k(ffmpeg_bin: str, wav_path: Path, tempo: float = 1.0) -> tuple[array, float]:
    """Read a WAV clip as 24kHz mono 16-bit PCM array, optionally time-stretched via FFmpeg atempo."""
    if abs(tempo - 1.0) > 0.03:
        clamped = max(0.75, min(1.5, float(tempo)))
        stretched = wav_path.with_name(f"{wav_path.stem}_fast.wav")
        code, _ = _run_ffmpeg(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-af",
                f"atempo={clamped:.3f}",
                "-ac",
                "1",
                "-ar",
                str(DUB_SAMPLE_RATE),
                str(stretched),
            ],
            timeout=60.0,
        )
        if code == 0 and stretched.is_file() and stretched.stat().st_size > 44:
            wav_path = stretched

    with wave.open(str(wav_path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if ch != 1 or sw != 2 or sr != DUB_SAMPLE_RATE:
        norm = wav_path.with_name(f"{wav_path.stem}_norm.wav")
        code, _ = _run_ffmpeg(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-ac",
                "1",
                "-ar",
                str(DUB_SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                str(norm),
            ],
            timeout=60.0,
        )
        if code == 0 and norm.is_file():
            with wave.open(str(norm), "rb") as wf:
                raw = wf.readframes(wf.getnframes())

    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    return samples, (len(samples) / float(DUB_SAMPLE_RATE))


def _trim_to_acoustic_onset(samples: array) -> array:
    """Measure exact 2ms (48-sample) TTS vocal attack and trim to 4ms lead so onset locks within <5ms."""
    n = len(samples)
    if n == 0:
        return samples
    import numpy as np  # type: ignore

    arr = np.frombuffer(samples, dtype=np.int16).astype(np.float32)
    step = max(16, int(round(DUB_SAMPLE_RATE * 0.002)))  # 2ms frames @ 24kHz = 48 samples
    n_blocks = n // step
    if n_blocks < 4:
        return samples
    blocks = arr[: n_blocks * step].reshape(n_blocks, step)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    peak = float(np.max(rms))
    if peak < 150.0:
        return samples

    onset_thresh = max(180.0, peak * 0.08)
    tail_thresh = max(140.0, peak * 0.05)
    active_head = np.where(rms >= onset_thresh)[0]
    active_tail = np.where(rms >= tail_thresh)[0]
    if len(active_head) == 0 or len(active_tail) == 0:
        return samples

    # Perceptual P-center sync: leave a natural 18ms breath lead before peak TTS attack so voice never feels early
    lead_pad = int(round(DUB_SAMPLE_RATE * 0.018))
    tail_pad = int(round(DUB_SAMPLE_RATE * 0.020))
    first_idx = max(0, int(active_head[0]) * step - lead_pad)
    last_idx = min(n, (int(active_tail[-1]) + 1) * step + tail_pad)
    if last_idx <= first_idx:
        return samples
    return samples[first_idx:last_idx]


def build_dub_track_wav(
    ffmpeg_bin: str,
    segments: list[dict[str, Any]],
    wav_paths: list[Path | None],
    total_duration_sec: float,
    out_wav: Path,
) -> Path:
    """Assemble P-center synced TTS clips onto a single 24kHz mono WAV track."""
    max_end = max([total_duration_sec] + [seg["end_sec"] for seg in segments] + [1.0])
    total_samples = int(math.ceil((max_end + 1.0) * DUB_SAMPLE_RATE))
    canvas = array("h", [0]) * total_samples

    fade_len = int(DUB_SAMPLE_RATE * 0.020)
    head_fade = int(DUB_SAMPLE_RATE * 0.008)  # 8ms silky fade-in preserving crisp consonant attack
    for idx, seg in enumerate(segments):
        wav_path = wav_paths[idx]
        if wav_path is None or not wav_path.is_file():
            continue

        samples, clip_dur = _read_pcm_mono_24k(ffmpeg_bin, wav_path, tempo=1.0)
        samples = _trim_to_acoustic_onset(samples)
        clip_dur = len(samples) / float(DUB_SAMPLE_RATE)
        if clip_dur <= 0.01:
            continue

        # Psychoacoustic P-center nudge (+24ms): human ears perceive TTS attack slightly earlier than sung vowels over BGM
        next_start = segments[idx + 1]["start_sec"] if idx + 1 < len(segments) else max_end + 0.5
        placed_start = min(
            max(0.0, float(seg["start_sec"]) + 0.024),
            max(0.0, next_start - 0.32),
        )
        avail_window = max(0.40, min(next_start - placed_start - 0.03, (seg["end_sec"] - placed_start) + 0.40))

        if clip_dur > avail_window * 1.03:
            needed_tempo = min(1.45, clip_dur / max(0.32, avail_window))
            samples, _ = _read_pcm_mono_24k(ffmpeg_bin, wav_path, tempo=needed_tempo)
            samples = _trim_to_acoustic_onset(samples)

        max_samples_before_next = max(
            int(0.28 * DUB_SAMPLE_RATE),
            int((next_start - placed_start - 0.015) * DUB_SAMPLE_RATE),
        )
        if len(samples) > max_samples_before_next:
            samples = samples[:max_samples_before_next]
            tail = min(fade_len, len(samples))
            for k in range(tail):
                factor = (tail - k) / float(tail)
                pos = len(samples) - tail + k
                samples[pos] = int(samples[pos] * factor)

        lead = min(head_fade, len(samples))
        for k in range(lead):
            samples[k] = int(samples[k] * (k / float(lead)))

        start_idx = max(0, int(round(placed_start * DUB_SAMPLE_RATE)))
        end_idx = min(total_samples, start_idx + len(samples))
        for j in range(end_idx - start_idx):
            mixed = canvas[start_idx + j] + samples[j]
            if mixed > 32767:
                mixed = 32767
            elif mixed < -32768:
                mixed = -32768
            canvas[start_idx + j] = mixed

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(DUB_SAMPLE_RATE)
        wf.writeframes(canvas.tobytes())
    return out_wav


def _prepare_fonts_dir(job_dir: Path) -> tuple[str, str]:
    fonts_dir = job_dir / "fonts"
    fonts_dir.mkdir(exist_ok=True)
    if _BUNDLED_VI_FONT.is_file():
        dst = fonts_dir / _BUNDLED_VI_FONT.name
        if not dst.exists():
            shutil.copy2(_BUNDLED_VI_FONT, dst)
        return "fonts", _BUNDLED_FONT_FAMILY
    return "fonts", "Arial"


def process_video_with_ai(
    ffmpeg_bin: str,
    input_video: Path,
    job_dir: Path,
    options: AIProcessOptions,
    *,
    source_subtitle_path: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    """Run Hardsub and/or Auto-Dubbing using decoupled 0-quota ASR + Settings LLM + Edge/Cloud TTS."""
    def _report(stage: str, pct: float, detail: str = "") -> None:
        if on_progress:
            on_progress(stage, pct, detail)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise AICanceledError("Đã hủy xử lý AI.")

    provider, api_key = resolve_configured_ai_provider(options.provider_id)
    transcribe_model = normalize_model_id(
        options.transcribe_model, default_transcribe_model_for(provider)
    )

    tts_provider: AIProvider | None = provider
    tts_backend = ""
    tts_api_key = api_key
    tts_model = ""
    tts_label = provider.label
    if options.mode == "dub":
        tts_pid = options.tts_provider_id.strip() or options.provider_id
        if tts_pid == PIPER_TTS_PROVIDER_ID:
            if not is_piper_tts_available():
                raise RuntimeError("Chưa cài thư viện piper-tts trên máy chủ.")
            tts_provider = None
            tts_backend = PIPER_TTS_PROVIDER_ID
            tts_api_key = ""
            tts_model = "piper-onnx"
            tts_label = "Piper ONNX Local (Siêu tốc)"
        elif tts_pid == EDGE_TTS_PROVIDER_ID:
            if not is_edge_tts_available():
                raise RuntimeError("Chưa cài thư viện edge-tts trên máy chủ.")
            tts_provider = None
            tts_backend = EDGE_TTS_PROVIDER_ID
            tts_api_key = ""
            tts_model = "edge-neural"
            tts_label = "Microsoft Edge TTS (Miễn phí)"
        else:
            tts_provider, tts_api_key = resolve_configured_ai_provider(tts_pid)
            tts_backend = tts_provider.id
            tts_model = normalize_model_id(
                options.tts_model, default_tts_model_for(tts_provider)
            )
            tts_label = f"{tts_provider.label} ({tts_model})"

    _check_cancel()
    _report("extracting_audio", 38.0, "Đang phân tích luồng video & tách âm thanh…")
    probe = probe_media_stream(ffmpeg_bin, input_video)
    if not probe.has_audio:
        raise RuntimeError("Video này không có luồng âm thanh để dịch hoặc lồng tiếng.")
    if probe.duration_sec > MAX_AI_DURATION_SECONDS:
        raise RuntimeError(
            f"Video dài {int(probe.duration_sec // 60)} phút — tính năng Hardsub / Lồng tiếng AI hỗ trợ tối đa 15 phút."
        )

    speech_mp3 = job_dir / "source_speech.mp3"
    extract_speech_mp3(ffmpeg_bin, input_video, speech_mp3, cancel_event=cancel_event)

    _check_cancel()
    lang_name = TARGET_LANGUAGES.get(options.target_lang, "Vietnamese")
    _report("transcribing", 44.0, "Đang bóc tách âm thanh theo mốc từng từ…")
    source_lang, segments = transcribe_and_translate_audio(
        ffmpeg_bin,
        speech_mp3,
        job_dir,
        provider=provider,
        api_key=api_key,
        model_id=transcribe_model,
        duration_sec=probe.duration_sec,
        target_lang=options.target_lang,
        for_dubbing=(options.mode == "dub"),
        context_hint=options.context_hint,
        video_title=options.video_title,
        asr_engine=options.asr_engine,
        whisper_model=options.whisper_model,
        source_subtitle_path=source_subtitle_path,
        source_media_path=input_video,
        cancel_event=cancel_event,
        on_status=lambda msg: _report("transcribing", 52.0, msg),
    )
    if not segments:
        raise RuntimeError("Không phát hiện lời thoại nào trong video để tạo phụ đề hoặc lồng tiếng.")

    _check_cancel()
    srt_path = job_dir / "subtitles.srt"
    write_srt_file(segments, srt_path, bilingual=options.bilingual)

    burn_sub = options.mode == "hardsub" or (options.mode == "dub" and options.burn_subtitles)
    ass_path = job_dir / "subs.ass"
    fonts_rel = "fonts"
    if burn_sub:
        fonts_rel, font_family = _prepare_fonts_dir(job_dir)
        write_ass_file(
            segments,
            ass_path,
            width=probe.width,
            height=probe.height,
            style_preset=options.subtitle_style,
            animation=options.subtitle_animation,
            bilingual=options.bilingual,
            font_name=font_family,
        )

    dub_wav_path: Path | None = None
    if options.mode == "dub":
        _report("synthesizing_tts", 60.0, f"{tts_label} đang lồng tiếng 0/{len(segments)} câu…")
        voice_map = _assign_speaker_voices(segments, options.voice)
        tts_dir = job_dir / "tts_clips"
        tts_dir.mkdir(exist_ok=True)
        wav_paths: list[Path | None] = [None] * len(segments)
        completed = 0
        errors: list[str] = []

        def _worker(index: int, seg: dict[str, Any]) -> tuple[int, Path]:
            _check_cancel()
            clip_out = tts_dir / f"seg_{index:04d}.wav"
            voice_name = voice_map.get(seg["speaker"], "Kore")
            _synthesize_segment_wav(
                seg["translated_text"],
                provider=tts_provider,
                tts_backend=tts_backend,
                api_key=tts_api_key,
                tts_model=tts_model,
                voice=voice_name,
                tone=seg["tone"],
                lang_name=lang_name,
                target_lang=options.target_lang,
                ffmpeg_bin=ffmpeg_bin,
                out_wav=clip_out,
            )
            return index, clip_out

        # Edge-TTS enforces strict per-IP WebSocket concurrency (1 worker); Piper ONNX runs ~50ms/sentence locally (2 workers).
        tts_workers = 1 if tts_backend == EDGE_TTS_PROVIDER_ID else (2 if tts_backend == PIPER_TTS_PROVIDER_ID else 4)
        with ThreadPoolExecutor(max_workers=tts_workers) as pool:
            futures = {pool.submit(_worker, i, seg): i for i, seg in enumerate(segments)}
            for fut in as_completed(futures):
                _check_cancel()
                idx = futures[fut]
                try:
                    i_done, path_done = fut.result()
                    wav_paths[i_done] = path_done
                except AICanceledError:
                    raise
                except Exception as exc:
                    logger.warning("TTS segment {} failed: {}", idx, exc)
                    errors.append(str(exc))
                completed += 1
                pct = 60.0 + (22.0 * completed / max(1, len(segments)))
                _report(
                    "synthesizing_tts",
                    round(pct, 1),
                    f"{tts_label} đang lồng tiếng {completed}/{len(segments)} câu…",
                )

        if not any(wav_paths):
            first_err = errors[0] if errors else "Không tạo được giọng lồng tiếng."
            raise RuntimeError(f"Lồng tiếng TTS thất bại: {first_err[:200]}")

        _check_cancel()
        _report("rendering_video", 84.0, "Đang căn chỉnh nhịp thoại & ghép track lồng tiếng…")
        dub_wav_path = job_dir / "dub_track.wav"
        build_dub_track_wav(
            ffmpeg_bin,
            segments,
            wav_paths,
            probe.duration_sec,
            dub_wav_path,
        )

    _check_cancel()
    _report(
        "rendering_video",
        88.0,
        "FFmpeg đang trộn âm thanh & gắn phụ đề cứng…"
        if (options.mode == "dub" and burn_sub)
        else ("FFmpeg đang gắn phụ đề cứng từng từ (Hardsub)…" if burn_sub else "FFmpeg đang trộn âm lồng tiếng…"),
    )

    final_output = job_dir / "media_ai_final.mp4"
    cmd: list[str] = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", input_video.name]

    if options.mode == "hardsub":
        cmd += [
            "-vf",
            f"ass={ass_path.name}:fontsdir={fonts_rel}",
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
            final_output.name,
        ]
    else:
        assert dub_wav_path is not None
        cmd += ["-i", dub_wav_path.name]
        orig_vol = max(0.0, min(0.5, float(options.original_volume)))
        if options.keep_original_audio and orig_vol > 0.01:
            af = (
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={orig_vol:.2f}[bg];"
                "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[dub][dub_sc];"
                "[bg][dub_sc]sidechaincompress=threshold=0.02:ratio=6:attack=25:release=280[bg_duck];"
                "[bg_duck][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                "loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
            )
        else:
            af = (
                "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                "loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
            )

        if burn_sub:
            filter_complex = f"[0:v]ass={ass_path.name}:fontsdir={fonts_rel}[vout];{af}"
            cmd += [
                "-filter_complex",
                filter_complex,
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
                final_output.name,
            ]
        else:
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
                final_output.name,
            ]

    code, stderr = _run_ffmpeg(cmd, cwd=job_dir, cancel_event=cancel_event, timeout=600.0)
    if code != 0 or not final_output.is_file() or final_output.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg xuất video thất bại: {stderr.strip()[:260]}")

    for tmp in (speech_mp3, dub_wav_path, ass_path):
        if tmp and tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
    for scratch_dir in (job_dir / "tts_clips", job_dir / "fonts", job_dir / "asr_chunks"):
        if scratch_dir.is_dir():
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return {
        "output_path": final_output,
        "srt_path": srt_path if srt_path.is_file() else None,
        "source_language": source_lang,
        "target_language": options.target_lang,
        "provider_id": provider.id,
        "tts_provider_id": (
            (tts_provider.id if tts_provider is not None else tts_backend)
            if options.mode == "dub"
            else None
        ),
        "transcribe_model": transcribe_model,
        "tts_model": tts_model if options.mode == "dub" else None,
        "segment_count": len(segments),
        "segments": [
            {k: v for k, v in seg.items() if k != "words"}
            for seg in segments[:150]
        ],
    }
