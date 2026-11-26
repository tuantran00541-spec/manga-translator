from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import mimetypes
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit
import uuid

from fastapi import HTTPException
import requests

from app.config import MEDIA_DIR
from app.downloader.http import public_request
from app.logging_config import logger
from app.media_ai_studio import (
    AICanceledError,
    AIProcessOptions,
    EDGE_TTS_PROVIDER_ID,
    PIPER_TTS_PROVIDER_ID,
    TARGET_LANGUAGES,
    VALID_AI_MODES,
    VALID_ASR_ENGINES,
    VALID_SUBTITLE_ANIMATIONS,
    VALID_SUBTITLE_STYLES,
    VALID_VOICE_IDS,
    VALID_WHISPER_MODELS,
    VOICE_CATALOG,
    any_ai_provider_configured,
    default_transcribe_model_for,
    default_tts_model_for,
    is_edge_tts_available,
    is_piper_tts_available,
    list_configured_ai_providers,
    normalize_model_id,
    process_video_with_ai,
    resolve_configured_ai_provider,
)
from app.security import validate_managed_path, validate_url

JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
VALID_PRESETS = frozenset({"video_1080", "video_720", "audio_only"})

MAX_MEDIA_URL_LENGTH = 2048
MAX_REDIRECT_HOPS = 5
MAX_DURATION_SECONDS = 4 * 3600  # 4 hours
MAX_MEDIA_FILE_BYTES = 1500 * 1024 * 1024  # 1.5 GB
MAX_CONCURRENT_DOWNLOADS = 2
MAX_STORED_JOBS = 30
JOB_TTL_SECONDS = 2 * 3600  # 2 hours

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_BLOCKED_TLDS = frozenset(
    {"localhost", "local", "internal", "lan", "home", "corp", "arpa", "onion", "intranet", "private"}
)
_SPOOFED_BRAND_SUFFIXES = (
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "google.com",
    "paypal.com",
    "apple.com",
)

_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    label: str
    suffixes: tuple[str, ...]


KNOWN_PLATFORMS: tuple[PlatformSpec, ...] = (
    PlatformSpec("youtube", "YouTube", ("youtube.com", "youtu.be", "youtube-nocookie.com")),
    PlatformSpec("tiktok", "TikTok", ("tiktok.com",)),
    PlatformSpec("facebook", "Facebook", ("facebook.com", "fb.watch")),
    PlatformSpec("instagram", "Instagram", ("instagram.com", "instagr.am")),
    PlatformSpec("twitter", "X / Twitter", ("x.com", "twitter.com")),
    PlatformSpec("bilibili", "Bilibili", ("bilibili.com", "b23.tv")),
    PlatformSpec("douyin", "Douyin", ("douyin.com",)),
    PlatformSpec("vimeo", "Vimeo", ("vimeo.com",)),
    PlatformSpec("dailymotion", "Dailymotion", ("dailymotion.com", "dai.ly")),
    PlatformSpec("reddit", "Reddit", ("reddit.com", "redd.it", "v.redd.it")),
    PlatformSpec("twitch", "Twitch", ("twitch.tv",)),
    PlatformSpec("threads", "Threads", ("threads.net", "threads.com")),
    PlatformSpec("pinterest", "Pinterest", ("pinterest.com", "pin.it")),
    PlatformSpec("soundcloud", "SoundCloud", ("soundcloud.com",)),
    PlatformSpec("loom", "Loom", ("loom.com",)),
    PlatformSpec("streamable", "Streamable", ("streamable.com",)),
    PlatformSpec("kick", "Kick", ("kick.com",)),
    PlatformSpec("rumble", "Rumble", ("rumble.com",)),
    PlatformSpec("vk", "VK Video", ("vk.com", "vkvideo.ru")),
    PlatformSpec("ted", "TED", ("ted.com",)),
)


class _DownloadCanceled(Exception):
    """Raised inside yt-dlp hooks when a user cancels an active download."""


def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise HTTPException(400, f"Invalid media job_id: {job_id!r}")
    return job_id


def find_ffmpeg_binary() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return str(exe)
    except Exception:
        pass
    return None


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _is_raw_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _has_homograph_risk(host: str) -> bool:
    if not host.isascii():
        return True
    return any(label.startswith("xn--") for label in host.split("."))


def _match_known_platform(host: str) -> PlatformSpec | None:
    norm = host.lower().rstrip(".")
    for spec in KNOWN_PLATFORMS:
        for suffix in spec.suffixes:
            if norm == suffix or norm.endswith(f".{suffix}"):
                return spec
    return None


def _has_brand_spoof(host: str) -> str | None:
    """Detect lookalike phishing subdomains like youtube.com.evil.example."""
    norm = host.lower().rstrip(".")
    for brand in _SPOOFED_BRAND_SUFFIXES:
        if f"{brand}." in norm and not (norm == brand or norm.endswith(f".{brand}")):
            return brand
    return None


def _validate_public_hostname(host: str) -> tuple[bool, str]:
    norm = host.lower().rstrip(".")
    if not norm or "." not in norm:
        return False, f"Tên miền '{norm or '(trống)'}' không phải tên miền công khai hợp lệ."
    if _is_raw_ip(norm):
        return False, f"Chặn địa chỉ IP trực tiếp ({norm})."
    if _has_homograph_risk(norm):
        return False, f"Phát hiện tên miền quốc tế hóa / Punycode đáng ngờ ({norm})."
    tld = norm.rsplit(".", 1)[-1]
    if tld in _BLOCKED_TLDS:
        return False, f"Chặn tên miền nội bộ / ẩn danh (.{tld})."
    spoof = _has_brand_spoof(norm)
    if spoof:
        return False, f"Phát hiện dấu hiệu giả mạo thương hiệu '{spoof}' trong tên miền '{norm}'."
    return True, ""


def _check_redirect_chain(start_url: str) -> tuple[bool, str, str, PlatformSpec | None]:
    """Follow up to MAX_REDIRECT_HOPS redirects ensuring every hop is HTTPS, non-phishing, and public IP."""
    current = start_url
    hops = 0
    final_platform = _match_known_platform((urlsplit(start_url).hostname or ""))

    for _ in range(MAX_REDIRECT_HOPS + 1):
        resp = None
        try:
            resp = public_request(
                "GET",
                current,
                allow_redirects=False,
                stream=True,
                timeout=(5, 10),
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                },
            )
            status = resp.status_code
            location = resp.headers.get("location")
        except requests.RequestException:
            if hops == 0:
                return (
                    True,
                    current,
                    "Không phát hiện chuyển hướng nguy hiểm (máy chủ đích hạn chế kiểm tra trực tiếp).",
                    final_platform,
                )
            return True, current, f"Đã xác minh {hops} bước chuyển hướng HTTPS công khai an toàn.", final_platform
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

        if status not in _REDIRECT_STATUSES or not location:
            if hops == 0:
                return True, current, "Liên kết trỏ thẳng đến nguồn đích, không có chuyển hướng bất thường.", final_platform
            return True, current, f"Đã xác minh {hops} bước chuyển hướng HTTPS đều an toàn.", final_platform

        hops += 1
        if hops > MAX_REDIRECT_HOPS:
            return False, current, "Chuỗi chuyển hướng vượt quá 5 bước cho phép.", final_platform

        next_url = urljoin(current, location.strip())
        if _has_control_chars(next_url) or len(next_url) > MAX_MEDIA_URL_LENGTH:
            return False, current, "Đích chuyển hướng chứa ký tự không hợp lệ hoặc quá dài.", final_platform

        try:
            next_parsed = urlsplit(next_url)
        except ValueError:
            return False, current, "Đích chuyển hướng không đúng định dạng URL.", final_platform

        if next_parsed.scheme.lower() != "https":
            return False, next_url, f"Chuyển hướng hạ cấp xuống giao thức không an toàn ({next_parsed.scheme or 'trống'}).", final_platform
        if next_parsed.username is not None or next_parsed.password is not None:
            return False, next_url, "Đích chuyển hướng chứa thông tin đăng nhập nhúng.", final_platform
        try:
            next_port = next_parsed.port
        except ValueError:
            return False, next_url, "Đích chuyển hướng có cổng mạng không hợp lệ.", final_platform
        if next_port not in (None, 443):
            return False, next_url, f"Đích chuyển hướng dùng cổng :{next_port} bất thường.", final_platform

        next_host = (next_parsed.hostname or "").lower().rstrip(".")
        host_ok, host_err = _validate_public_hostname(next_host)
        if not host_ok:
            return False, next_url, f"Chuyển hướng tới tên miền bị chặn: {host_err}", final_platform

        try:
            validate_url(next_url)
        except HTTPException as exc:
            return False, next_url, f"Đích chuyển hướng vi phạm kiểm tra mạng an toàn: {exc.detail}", final_platform

        matched = _match_known_platform(next_host)
        if matched is not None:
            final_platform = matched
        current = next_url

    return False, current, "Chuỗi chuyển hướng vượt quá giới hạn an toàn.", final_platform


def inspect_link_safety(
    raw_url: str,
    *,
    resolve_redirects: bool = True,
) -> dict[str, Any]:
    """Run multi-layer link safety verification for any public HTTPS video URL."""
    url = str(raw_url or "").strip()
    checks: list[dict[str, Any]] = []

    def _result(
        *,
        safe: bool,
        summary: str,
        platform: PlatformSpec | None = None,
        hostname: str = "",
        resolved_url: str = "",
    ) -> dict[str, Any]:
        return {
            "safe": safe,
            "status": "safe" if safe else "unsafe",
            "risk_level": "low" if safe else "high",
            "platform": platform.key if platform else ("web" if safe else None),
            "platform_label": platform.label if platform else (hostname or "Web Video" if safe else None),
            "normalized_url": url,
            "resolved_url": resolved_url or url,
            "hostname": hostname,
            "summary": summary,
            "checks": checks,
        }

    # 1. URL format & HTTPS protocol check
    if not url:
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": "Vui lòng dán liên kết video trước khi kiểm tra.",
            }
        )
        return _result(safe=False, summary="Chưa nhập liên kết.")

    if len(url) > MAX_MEDIA_URL_LENGTH or _has_control_chars(url):
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": "Liên kết quá dài hoặc chứa ký tự điều khiển không hợp lệ.",
            }
        )
        return _result(safe=False, summary="Liên kết không hợp lệ.")

    try:
        parsed = urlsplit(url)
    except ValueError:
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": "Không thể phân tích cú pháp đường dẫn.",
            }
        )
        return _result(safe=False, summary="Định dạng URL không hợp lệ.")

    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        detail = (
            "Liên kết đang dùng HTTP không mã hóa — chỉ chấp nhận HTTPS."
            if scheme == "http"
            else f"Giao thức '{scheme or 'thiếu https://'}' không được phép."
        )
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": detail,
            }
        )
        return _result(safe=False, summary=detail)

    if parsed.username is not None or parsed.password is not None:
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": "Không cho phép liên kết chứa thông tin đăng nhập (user:pass@).",
            }
        )
        return _result(safe=False, summary="Liên kết chứa thông tin đăng nhập nhúng — bị chặn vì rủi ro lừa đảo.")

    try:
        port = parsed.port
    except ValueError:
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": "Cổng mạng trong liên kết không hợp lệ.",
            }
        )
        return _result(safe=False, summary="Cổng mạng không hợp lệ.")

    if port not in (None, 443):
        checks.append(
            {
                "id": "https_protocol",
                "label": "Giao thức HTTPS & cấu trúc URL",
                "passed": False,
                "detail": f"Cổng :{port} bất thường — chỉ chấp nhận cổng HTTPS tiêu chuẩn (443).",
            }
        )
        return _result(safe=False, summary="Liên kết dùng cổng mạng bất thường.")

    checks.append(
        {
            "id": "https_protocol",
            "label": "Giao thức HTTPS & cấu trúc URL",
            "passed": True,
            "detail": "Sử dụng HTTPS chuẩn, không chứa ký tự lạ hoặc thông tin đăng nhập nhúng.",
        }
    )

    # 2. Domain reputation & anti-phishing check (supports 1,000+ public video websites)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    host_ok, host_err = _validate_public_hostname(hostname)
    if not host_ok:
        checks.append(
            {
                "id": "official_domain",
                "label": "Kiểm tra tên miền & chống giả mạo",
                "passed": False,
                "detail": host_err,
            }
        )
        return _result(safe=False, hostname=hostname, summary=host_err)

    platform = _match_known_platform(hostname)
    has_path_or_query = bool((parsed.path or "").strip("/") or (parsed.query or "").strip())
    if not has_path_or_query:
        detail = f"Đây là trang chủ ({hostname}), chưa phải đường dẫn tới một video cụ thể."
        checks.append(
            {
                "id": "official_domain",
                "label": "Kiểm tra tên miền & chống giả mạo",
                "passed": False,
                "detail": detail,
            }
        )
        return _result(safe=False, platform=platform, hostname=hostname, summary=detail)

    domain_detail = (
        f"Xác nhận nền tảng phổ biến {platform.label} ({hostname})."
        if platform
        else f"Tên miền công khai hợp lệ ({hostname}) — hỗ trợ trích xuất video mở rộng."
    )
    checks.append(
        {
            "id": "official_domain",
            "label": "Kiểm tra tên miền & chống giả mạo",
            "passed": True,
            "detail": domain_detail,
        }
    )

    # 3. DNS & SSRF public-IP check
    try:
        validate_url(url)
    except HTTPException as exc:
        checks.append(
            {
                "id": "network_ssrf",
                "label": "Kiểm tra DNS & chống SSRF",
                "passed": False,
                "detail": f"Bị chặn bởi bộ lọc an toàn mạng: {exc.detail}",
            }
        )
        return _result(
            safe=False,
            platform=platform,
            hostname=hostname,
            summary="Tên miền trỏ tới địa chỉ mạng nội bộ hoặc không phân giải được.",
        )

    checks.append(
        {
            "id": "network_ssrf",
            "label": "Kiểm tra DNS & chống SSRF",
            "passed": True,
            "detail": "DNS phân giải ra IP công khai hợp lệ, không trỏ vào mạng nội bộ.",
        }
    )

    # 4. Redirect chain safety verification
    resolved_url = url
    if resolve_redirects:
        ok_redirect, resolved_url, redirect_detail, redir_platform = _check_redirect_chain(url)
        if redir_platform is not None:
            platform = redir_platform
        checks.append(
            {
                "id": "redirect_chain",
                "label": "Kiểm tra chuyển hướng (Redirect)",
                "passed": ok_redirect,
                "detail": redirect_detail,
            }
        )
        if not ok_redirect:
            return _result(
                safe=False,
                platform=platform,
                hostname=hostname,
                resolved_url=resolved_url,
                summary=redirect_detail,
            )
    else:
        checks.append(
            {
                "id": "redirect_chain",
                "label": "Kiểm tra chuyển hướng (Redirect)",
                "passed": True,
                "detail": "Đã bỏ qua kiểm tra chuyển hướng mạng theo cấu hình kiểm thử.",
            }
        )

    label_name = platform.label if platform else hostname
    return _result(
        safe=True,
        platform=platform,
        hostname=hostname,
        resolved_url=resolved_url,
        summary=f"Liên kết {label_name} an toàn và hợp lệ.",
    )


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "Không rõ"
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "Không rõ"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_size(num_bytes: int | float | None) -> str | None:
    if num_bytes is None:
        return None
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"~{size:.1f} {unit}" if unit != "B" else f"~{int(size)} B"
        size /= 1024
    return None


def _short_side(fmt: dict[str, Any]) -> int:
    w = fmt.get("width")
    h = fmt.get("height")
    w_i = int(w) if isinstance(w, (int, float)) and w > 0 else 0
    h_i = int(h) if isinstance(h, (int, float)) and h > 0 else 0
    if w_i > 0 and h_i > 0:
        return min(w_i, h_i)
    return max(w_i, h_i)


def _estimate_bytes(fmt: dict[str, Any] | None, duration: float | None) -> int | None:
    if not fmt:
        return None
    for key in ("filesize", "filesize_approx"):
        val = fmt.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    tbr = fmt.get("tbr") or fmt.get("vbr") or fmt.get("abr")
    if isinstance(tbr, (int, float)) and tbr > 0 and duration and duration > 0:
        return int(tbr * 1000 / 8 * duration)
    return None


def _pick_best_video(formats: list[dict[str, Any]], max_short: int) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for fmt in formats:
        if fmt.get("vcodec") in (None, "none") or fmt.get("has_drm"):
            continue
        side = _short_side(fmt)
        if side <= 0 or side > max_short:
            continue
        candidates.append(fmt)
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: (
            _short_side(f),
            1 if (f.get("ext") == "mp4") else 0,
            float(f.get("fps") or 0),
            float(f.get("tbr") or f.get("vbr") or 0),
        ),
        reverse=True,
    )
    return candidates[0]


def _pick_best_audio(formats: list[dict[str, Any]]) -> dict[str, Any] | None:
    audio_only = [
        f
        for f in formats
        if f.get("acodec") not in (None, "none")
        and f.get("vcodec") in (None, "none")
        and not f.get("has_drm")
    ]
    if audio_only:
        audio_only.sort(
            key=lambda f: (
                1 if f.get("ext") in ("m4a", "mp3", "aac") else 0,
                float(f.get("abr") or f.get("tbr") or 0),
            ),
            reverse=True,
        )
        return audio_only[0]
    with_audio = [
        f
        for f in formats
        if f.get("acodec") not in (None, "none") and not f.get("has_drm")
    ]
    if with_audio:
        with_audio.sort(
            key=lambda f: float(f.get("abr") or f.get("tbr") or 0),
            reverse=True,
        )
        return with_audio[0]
    return None


def build_quality_options(
    info: dict[str, Any],
    *,
    has_ffmpeg: bool,
) -> tuple[list[dict[str, Any]], str, bool]:
    """Build selectable 1080p, 720p, and audio-only options from yt-dlp metadata."""
    raw_formats = info.get("formats")
    formats = [f for f in (raw_formats or []) if isinstance(f, dict)]
    if not formats and info.get("url"):
        formats = [info]
    duration = info.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = None

    all_videos = [
        f
        for f in formats
        if f.get("vcodec") not in ("none",) and not f.get("has_drm") and (_short_side(f) > 0 or f.get("ext") in ("mp4", "webm", "mkv", "mov"))
    ]
    is_vertical = any(
        (f.get("height") or 0) > (f.get("width") or 0)
        for f in all_videos
        if f.get("width") and f.get("height")
    )
    max_source_side = max((_short_side(f) for f in all_videos), default=0)

    best_audio = _pick_best_audio(formats)
    audio_bytes = _estimate_bytes(best_audio, duration)

    best_1080 = _pick_best_video(formats, 1120)
    side_1080 = _short_side(best_1080) if best_1080 else 0
    has_1080 = side_1080 >= 900
    v1080_bytes = _estimate_bytes(best_1080, duration)
    if best_1080 and best_1080.get("acodec") in (None, "none") and audio_bytes:
        v1080_bytes = (v1080_bytes or 0) + audio_bytes
    res_1080 = (
        f"{int(best_1080['width'])}×{int(best_1080['height'])}"
        if best_1080 and best_1080.get("width") and best_1080.get("height")
        else "1080p"
    )
    fps_1080 = int(round(float(best_1080["fps"]))) if best_1080 and best_1080.get("fps") else None

    best_720 = _pick_best_video(formats, 760)
    if best_720 is None and all_videos:
        best_720 = min(all_videos, key=lambda f: _short_side(f) or 99999)
    side_720 = _short_side(best_720) if best_720 else 0
    has_video_fallback = bool(best_720 or all_videos or formats)
    v720_bytes = _estimate_bytes(best_720, duration)
    if best_720 and best_720.get("acodec") in (None, "none") and audio_bytes:
        v720_bytes = (v720_bytes or 0) + audio_bytes
    res_720 = (
        f"{int(best_720['width'])}×{int(best_720['height'])}"
        if best_720 and best_720.get("width") and best_720.get("height")
        else (f"{side_720}p" if side_720 else "Chất lượng gốc")
    )
    fps_720 = int(round(float(best_720["fps"]))) if best_720 and best_720.get("fps") else None

    label_720 = (
        "720p HD"
        if side_720 == 0 or side_720 >= 680
        else f"720p / Tối đa nguồn ({side_720}p)"
    )

    audio_ext = "MP3" if has_ffmpeg else "M4A"
    audio_abr = int(round(float(best_audio["abr"]))) if best_audio and best_audio.get("abr") else (192 if has_ffmpeg else 128)
    audio_est = int(192 * 1000 / 8 * duration) if has_ffmpeg and duration else audio_bytes

    options: list[dict[str, Any]] = [
        {
            "id": "video_1080",
            "kind": "video",
            "label": "1080p Full HD",
            "container": "MP4",
            "resolution": res_1080 if has_1080 else (f"Nguồn tối đa {max_source_side}p" if max_source_side else "Không rõ luồng 1080p"),
            "fps": fps_1080 if has_1080 else None,
            "estimated_size": _format_size(v1080_bytes) if has_1080 else None,
            "available": has_1080,
            "description": (
                "Sắc nét nhất, tự động ghép hình ảnh + âm thanh MP4."
                if has_1080
                else "Video nguồn không có luồng 1080p — hãy chọn 720p/Gốc hoặc Chỉ Audio."
            ),
        },
        {
            "id": "video_720",
            "kind": "video",
            "label": label_720,
            "container": "MP4",
            "resolution": res_720,
            "fps": fps_720,
            "estimated_size": _format_size(v720_bytes),
            "available": has_video_fallback,
            "description": "Cân bằng giữa độ nét, tốc độ tải và dung lượng gọn nhẹ.",
        },
        {
            "id": "audio_only",
            "kind": "audio",
            "label": f"Chỉ xuất Audio ({audio_ext})",
            "container": audio_ext,
            "resolution": f"{audio_abr} kbps",
            "fps": None,
            "estimated_size": _format_size(audio_est),
            "available": bool(best_audio or formats),
            "description": (
                "Tách riêng âm thanh chất lượng cao sang tệp MP3."
                if has_ffmpeg
                else "Tải riêng luồng âm thanh M4A (không cần chuyển mã)."
            ),
        },
    ]

    default_preset = "video_1080" if has_1080 else ("video_720" if has_video_fallback else "audio_only")
    return options, default_preset, is_vertical


def _safe_thumbnail_url(raw_thumb: Any) -> str | None:
    if not isinstance(raw_thumb, str):
        return None
    thumb = raw_thumb.strip()
    if not thumb or len(thumb) > MAX_MEDIA_URL_LENGTH or _has_control_chars(thumb):
        return None
    try:
        parsed = urlsplit(thumb)
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return thumb


def _has_unsafe_stream_protocol(info: dict[str, Any]) -> bool:
    for item in [info, *(info.get("formats") or [])]:
        if not isinstance(item, dict):
            continue
        proto = str(item.get("protocol") or "").lower()
        if proto and proto in ("file", "ftp", "ftps", "sftp", "smb"):
            return True
        stream_url = str(item.get("url") or "").strip().lower()
        if stream_url and stream_url.startswith(("file:", "ftp:", "javascript:", "data:")):
            return True
    return False


def _extract_info(target_url: str, platform_label: str) -> dict[str, Any]:
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            503,
            "Thiếu thư viện yt-dlp trên máy chủ. Hãy chạy: pip install yt-dlp imageio-ffmpeg",
        ) from exc

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "playlist_items": "1",
        "socket_timeout": 15,
        "retries": 2,
        "extractor_retries": 2,
        "skip_download": True,
        "enable_file_urls": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target_url, download=False)
    except Exception as exc:
        msg = str(exc).strip()
        lowered = msg.lower()
        if "drm" in lowered:
            raise HTTPException(400, "Nội dung có bảo vệ DRM nên không hỗ trợ tải xuống.") from exc
        if "unsupported url" in lowered or "no video formats" in lowered:
            raise HTTPException(
                404,
                f"Liên kết an toàn nhưng không tìm thấy luồng video/âm thanh nào có thể tải trên {platform_label}.",
            ) from exc
        if "private" in lowered or "login" in lowered or "sign in" in lowered or "cookies" in lowered:
            raise HTTPException(
                400,
                f"Video trên {platform_label} đang ở chế độ riêng tư hoặc yêu cầu đăng nhập.",
            ) from exc
        if "unavailable" in lowered or "not exist" in lowered or "removed" in lowered:
            raise HTTPException(404, f"Không tìm thấy video hoặc video trên {platform_label} đã bị gỡ.") from exc
        logger.warning("Media inspect failed for {}: {}", platform_label, msg)
        raise HTTPException(
            502,
            f"Không lấy được thông tin video từ {platform_label}: {msg[:240] or 'Lỗi kết nối'}",
        ) from exc

    if not isinstance(info, dict):
        raise HTTPException(502, f"Không đọc được dữ liệu video từ {platform_label}.")
    if "entries" in info and isinstance(info["entries"], list):
        first = next((item for item in info["entries"] if isinstance(item, dict)), None)
        if first is None:
            raise HTTPException(404, "Liên kết không chứa video nào khả dụng.")
        info = first
    return info


def inspect_media_url(raw_url: str) -> dict[str, Any]:
    """Verify link safety first; if safe, fetch video metadata, quality presets, and Settings providers."""
    safety = inspect_link_safety(raw_url, resolve_redirects=True)
    providers_summary = list_configured_ai_providers()
    ai_ready = any(p["configured"] for p in providers_summary)

    if not safety["safe"]:
        return {
            "safe": False,
            "safety": safety,
            "media": None,
            "options": [],
            "default_preset": None,
            "ffmpeg_available": bool(find_ffmpeg_binary()),
            "ai_available": ai_ready,
            "ai_providers": providers_summary,
        }

    platform_label = str(safety.get("platform_label") or safety.get("hostname") or "Web Video")
    target_url = safety["resolved_url"] or safety["normalized_url"]
    info = _extract_info(target_url, platform_label)

    # 5. Content & policy check (reject local protocols, DRM, livestreams, >4h streams)
    if _has_unsafe_stream_protocol(info):
        safety["checks"].append(
            {
                "id": "content_policy",
                "label": "Kiểm tra nội dung & chính sách tải",
                "passed": False,
                "detail": "Luồng media sử dụng giao thức không được phép (không phải HTTP/HTTPS).",
            }
        )
        safety["safe"] = False
        safety["status"] = "unsafe"
        safety["risk_level"] = "high"
        safety["summary"] = "Luồng media không an toàn — đã chặn."
        return {
            "safe": False,
            "safety": safety,
            "media": None,
            "options": [],
            "default_preset": None,
            "ffmpeg_available": bool(find_ffmpeg_binary()),
            "ai_available": ai_ready,
            "ai_providers": providers_summary,
        }

    formats = [f for f in (info.get("formats") or []) if isinstance(f, dict)]
    has_drm = bool(info.get("_has_drm")) or (
        bool(formats) and all(bool(f.get("has_drm")) for f in formats)
    )
    if has_drm:
        safety["checks"].append(
            {
                "id": "content_policy",
                "label": "Kiểm tra nội dung & chính sách tải",
                "passed": False,
                "detail": "Nội dung có bảo vệ bản quyền kỹ thuật số (DRM) — không hỗ trợ tải.",
            }
        )
        safety["safe"] = False
        safety["status"] = "unsafe"
        safety["risk_level"] = "high"
        safety["summary"] = "Nội dung được bảo vệ DRM không được phép tải xuống."
        return {
            "safe": False,
            "safety": safety,
            "media": None,
            "options": [],
            "default_preset": None,
            "ffmpeg_available": bool(find_ffmpeg_binary()),
            "ai_available": ai_ready,
            "ai_providers": providers_summary,
        }

    if info.get("is_live"):
        safety["checks"].append(
            {
                "id": "content_policy",
                "label": "Kiểm tra nội dung & chính sách tải",
                "passed": False,
                "detail": "Không hỗ trợ tải luồng phát trực tiếp (Livestream) đang diễn ra.",
            }
        )
        safety["safe"] = False
        safety["status"] = "unsafe"
        safety["summary"] = "Video đang phát trực tiếp — hãy chờ phát xong rồi thử lại."
        return {
            "safe": False,
            "safety": safety,
            "media": None,
            "options": [],
            "default_preset": None,
            "ffmpeg_available": bool(find_ffmpeg_binary()),
            "ai_available": ai_ready,
            "ai_providers": providers_summary,
        }

    duration = info.get("duration")
    if isinstance(duration, (int, float)) and duration > MAX_DURATION_SECONDS:
        safety["checks"].append(
            {
                "id": "content_policy",
                "label": "Kiểm tra nội dung & chính sách tải",
                "passed": False,
                "detail": f"Thời lượng video ({_format_duration(duration)}) vượt giới hạn 4 giờ.",
            }
        )
        safety["safe"] = False
        safety["status"] = "unsafe"
        safety["summary"] = "Video quá dài (tối đa 4 giờ)."
        return {
            "safe": False,
            "safety": safety,
            "media": None,
            "options": [],
            "default_preset": None,
            "ffmpeg_available": bool(find_ffmpeg_binary()),
            "ai_available": ai_ready,
            "ai_providers": providers_summary,
        }

    extractor_label = str(info.get("extractor_key") or info.get("extractor") or "").strip()
    if safety.get("platform") == "web" and extractor_label and extractor_label.lower() != "generic":
        platform_label = extractor_label

    safety["checks"].append(
        {
            "id": "content_policy",
            "label": "Kiểm tra nội dung & chính sách tải",
            "passed": True,
            "detail": "Video công khai hợp lệ, không có DRM, thời lượng trong giới hạn an toàn.",
        }
    )

    has_ffmpeg = bool(find_ffmpeg_binary())
    options, default_preset, is_vertical = build_quality_options(info, has_ffmpeg=has_ffmpeg)

    title = str(info.get("title") or info.get("fulltitle") or f"Video {platform_label}").strip()
    uploader = str(info.get("uploader") or info.get("channel") or info.get("creator") or "").strip()

    return {
        "safe": True,
        "safety": safety,
        "media": {
            "title": title,
            "uploader": uploader,
            "duration": int(duration) if isinstance(duration, (int, float)) else None,
            "duration_text": _format_duration(duration),
            "thumbnail": _safe_thumbnail_url(info.get("thumbnail")),
            "webpage_url": target_url,
            "platform": safety.get("platform") or "web",
            "platform_label": platform_label,
            "is_vertical": is_vertical,
        },
        "options": options,
        "default_preset": default_preset,
        "ffmpeg_available": has_ffmpeg,
        "ai_available": ai_ready,
        "ai_providers": providers_summary,
        "ai_voices": list(VOICE_CATALOG),
        "ai_languages": [
            {"code": code, "label": label} for code, label in TARGET_LANGUAGES.items()
        ],
    }


def sanitize_download_filename(title: str, suffix: str, ext: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", str(title or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(". ")
    if not cleaned or cleaned.upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = "video"
    if len(cleaned) > 96:
        cleaned = cleaned[:96].rstrip(". ")
    ext_clean = re.sub(r"[^a-zA-Z0-9]", "", str(ext or "mp4")).lower() or "mp4"
    return f"{cleaned} [{suffix}].{ext_clean}"


def _format_selector_for_preset(preset: str, *, is_vertical: bool, has_ffmpeg: bool) -> str:
    if preset == "audio_only":
        return "bestaudio[ext=m4a]/bestaudio/best"
    if preset == "video_1080":
        if is_vertical:
            if has_ffmpeg:
                return (
                    "bestvideo[width<=1080][ext=mp4]+bestaudio[ext=m4a]/"
                    "bestvideo[width<=1080]+bestaudio/"
                    "best[width<=1080][ext=mp4]/best[width<=1080]/"
                    "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
                )
            return "best[width<=1080][ext=mp4]/best[width<=1080]/best[height<=1080]/best"
        if has_ffmpeg:
            return (
                "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height<=1080]+bestaudio/"
                "best[height<=1080][ext=mp4]/best[height<=1080]/best"
            )
        return "best[height<=1080][ext=mp4]/best[height<=1080]/best"
    if is_vertical:
        if has_ffmpeg:
            return (
                "bestvideo[width<=720][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[width<=720]+bestaudio/"
                "best[width<=720][ext=mp4]/best[width<=720]/"
                "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
            )
        return "best[width<=720][ext=mp4]/best[width<=720]/best[height<=720]/best"
    if has_ffmpeg:
        return (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=720]+bestaudio/"
            "best[height<=720][ext=mp4]/best[height<=720]/best"
        )
    return "best[height<=720][ext=mp4]/best[height<=720]/best"


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _snapshot_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "stage_detail": job.get("stage_detail") or "",
        "preset": job["preset"],
        "ai_mode": job.get("ai_mode") or "none",
        "target_lang": job.get("target_lang") or "vi",
        "transcribe_model": job.get("transcribe_model"),
        "tts_model": job.get("tts_model"),
        "platform": job["platform"],
        "platform_label": job["platform_label"],
        "title": job["title"],
        "percent": job["percent"],
        "downloaded_bytes": job["downloaded_bytes"],
        "total_bytes": job["total_bytes"],
        "speed": job["speed"],
        "eta": job["eta"],
        "filename": job["filename"],
        "file_size": job["file_size"],
        "file_size_text": _format_size(job["file_size"]),
        "download_url": f"/api/media/files/{job['job_id']}" if job["status"] == "completed" and job["file_path"] else None,
        "subtitle_url": f"/api/media/subtitles/{job['job_id']}" if job["status"] == "completed" and job.get("subtitle_path") else None,
        "subtitle_filename": job.get("subtitle_filename"),
        "source_language": job.get("source_language"),
        "segment_count": job.get("segment_count") or 0,
        "segments": job.get("segments") or [],
        "error": job["error"],
        "created_at": job["created_at"],
    }


def _cleanup_job_dir(job_dir: Path) -> None:
    try:
        resolved = validate_managed_path(job_dir, MEDIA_DIR)
    except HTTPException:
        return
    if resolved == MEDIA_DIR.resolve(strict=False):
        return
    if resolved.exists():
        shutil.rmtree(resolved, ignore_errors=True)


def cleanup_expired_media_jobs(max_age_seconds: float = JOB_TTL_SECONDS) -> int:
    """Remove expired media download directories and prune in-memory history."""
    now = time.time()
    removed = 0
    to_remove_dirs: list[Path] = []
    with _JOBS_LOCK:
        stale_ids = [
            jid
            for jid, job in _JOBS.items()
            if job["status"] not in ("pending", "running")
            and (now - float(job["created_at"])) > max_age_seconds
        ]
        for jid in stale_ids:
            job = _JOBS.pop(jid, None)
            if job and job.get("job_dir"):
                to_remove_dirs.append(Path(job["job_dir"]))

        if len(_JOBS) > MAX_STORED_JOBS:
            finished = sorted(
                (j for j in _JOBS.values() if j["status"] not in ("pending", "running")),
                key=lambda item: float(item["created_at"]),
            )
            excess = len(_JOBS) - MAX_STORED_JOBS
            for job in finished[:excess]:
                _JOBS.pop(job["job_id"], None)
                if job.get("job_dir"):
                    to_remove_dirs.append(Path(job["job_dir"]))

    for d in to_remove_dirs:
        _cleanup_job_dir(d)
        removed += 1

    if MEDIA_DIR.exists():
        with _JOBS_LOCK:
            live_ids = set(_JOBS.keys())
        for child in MEDIA_DIR.iterdir():
            if not child.is_dir() or child.is_symlink() or child.name in live_ids:
                continue
            try:
                if (now - child.stat().st_mtime) > max_age_seconds:
                    _cleanup_job_dir(child)
                    removed += 1
            except OSError:
                continue
    return removed


def _fetch_platform_subtitles(target_url: str, job_dir: Path, info: Any) -> Path | None:
    """Best-effort 0-quota fetch of official/auto platform captions (preferring word-timed json3)."""
    if not isinstance(info, dict):
        return None
    subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}
    if not subs and not auto_subs:
        return None

    # Prefer original video language, then English / common languages
    orig_lang = str(info.get("language") or "").strip()
    preferred = [c for c in (orig_lang, "en", "en-US", "en-orig", "ja", "ko", "zh", "vi") if c]
    chosen_lang = ""
    use_auto = False
    for lang in preferred:
        if lang in subs:
            chosen_lang = lang
            use_auto = False
            break
    if not chosen_lang and subs:
        chosen_lang = next((k for k in subs.keys() if k != "live_chat"), "")
    if not chosen_lang:
        for lang in preferred:
            if lang in auto_subs:
                chosen_lang = lang
                use_auto = True
                break
    if not chosen_lang:
        return None

    sub_dir = job_dir / "platform_subs"
    sub_dir.mkdir(exist_ok=True)
    try:
        import yt_dlp  # type: ignore

        sub_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": 12,
            "writesubtitles": not use_auto,
            "writeautomaticsub": use_auto,
            "subtitleslangs": [chosen_lang],
            "subtitlesformat": "json3/vtt/srt",
            "outtmpl": str(sub_dir / "cap.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(sub_opts) as ydl:
            ydl.download([target_url])
    except Exception as exc:
        logger.debug("Optional platform subtitle fetch skipped: {}", str(exc)[:140])
        return None

    for ext in (".json3", ".vtt", ".srt"):
        matches = sorted(sub_dir.glob(f"*{ext}"))
        if matches and matches[0].is_file() and matches[0].stat().st_size > 32:
            return matches[0]
    return None


def _find_downloaded_file(job_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for entry in job_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            continue
        if entry.name.endswith((".part", ".ytdl", ".temp")):
            continue
        candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _run_download_worker(
    job_id: str,
    target_url: str,
    platform_label: str,
    preset: str,
    is_vertical: bool,
    ai_options: AIProcessOptions | None = None,
) -> None:
    job_dir = validate_managed_path(MEDIA_DIR / job_id, MEDIA_DIR)
    job_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = find_ffmpeg_binary()
    has_ffmpeg = bool(ffmpeg_bin)
    ai_enabled = bool(ai_options and ai_options.mode != "none" and preset != "audio_only")

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            _cleanup_job_dir(job_dir)
            return
        cancel_event: threading.Event = job["cancel_event"]
        if cancel_event.is_set():
            job["status"] = "canceled"
            job["stage"] = "canceled"
            _cleanup_job_dir(job_dir)
            return
        job["status"] = "running"
        job["stage"] = "downloading"
        job["stage_detail"] = "Đang tải video gốc trước khi xử lý AI…" if ai_enabled else ""

    def progress_hook(data: dict[str, Any]) -> None:
        if cancel_event.is_set():
            raise _DownloadCanceled()
        status = data.get("status")
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if not entry:
                return
            if status == "downloading":
                downloaded = int(data.get("downloaded_bytes") or 0)
                total = int(data.get("total_bytes") or data.get("total_bytes_estimate") or 0)
                raw_pct = min(99.0, (downloaded / total) * 100.0) if total > 0 else 0.0
                pct = round(raw_pct * 0.35, 1) if ai_enabled else (round(raw_pct, 1) if total > 0 else entry["percent"])
                entry["stage"] = "downloading"
                entry["downloaded_bytes"] = downloaded
                entry["total_bytes"] = total or None
                entry["percent"] = pct
                speed = data.get("speed")
                entry["speed"] = float(speed) if isinstance(speed, (int, float)) and speed > 0 else None
                eta = data.get("eta")
                entry["eta"] = int(eta) if isinstance(eta, (int, float)) and eta >= 0 else None
            elif status == "finished":
                entry["stage"] = "merging"
                entry["percent"] = 35.0 if ai_enabled else 99.0
                entry["speed"] = None
                entry["eta"] = 0

    def postprocessor_hook(data: dict[str, Any]) -> None:
        if cancel_event.is_set():
            raise _DownloadCanceled()
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["stage"] = "merging"
                entry["percent"] = 36.0 if ai_enabled else 99.5

    def ai_progress(stage: str, pct: float, detail: str) -> None:
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["stage"] = stage
                entry["percent"] = round(min(99.5, max(0.0, float(pct))), 1)
                entry["stage_detail"] = detail
                entry["speed"] = None
                entry["eta"] = None

    try:
        import yt_dlp  # type: ignore
    except ImportError:
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["status"] = "failed"
                entry["stage"] = "failed"
                entry["error"] = "Thiếu thư viện yt-dlp trên máy chủ."
        _cleanup_job_dir(job_dir)
        return

    fmt_selector = _format_selector_for_preset(
        preset,
        is_vertical=is_vertical,
        has_ffmpeg=has_ffmpeg,
    )
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "playlist_items": "1",
        "socket_timeout": 20,
        "retries": 3,
        "fragment_retries": 3,
        "restrictfilenames": True,
        "max_filesize": MAX_MEDIA_FILE_BYTES,
        "enable_file_urls": False,
        "format": fmt_selector,
        "outtmpl": str(job_dir / "media.%(ext)s"),
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [postprocessor_hook],
    }
    if ffmpeg_bin:
        ydl_opts["ffmpeg_location"] = ffmpeg_bin

    if preset == "audio_only":
        if has_ffmpeg:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ]
    elif has_ffmpeg:
        ydl_opts["merge_output_format"] = "mp4"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
        if cancel_event.is_set():
            raise _DownloadCanceled()

        output_path = _find_downloaded_file(job_dir)
        if output_path is None:
            raise RuntimeError("Không tìm thấy tệp sau khi tải hoàn tất.")
        safe_path = validate_managed_path(output_path, job_dir)

        ai_result: dict[str, Any] | None = None
        if ai_enabled and ai_options is not None:
            if not ffmpeg_bin:
                raise RuntimeError("Thiếu FFmpeg nên không thể gắn phụ đề cứng hoặc lồng tiếng.")
            platform_sub_path: Path | None = None
            if ai_options.asr_engine == "auto":
                platform_sub_path = _fetch_platform_subtitles(target_url, job_dir, info)
            ai_result = process_video_with_ai(
                ffmpeg_bin,
                safe_path,
                job_dir,
                ai_options,
                source_subtitle_path=platform_sub_path,
                cancel_event=cancel_event,
                on_progress=ai_progress,
            )
            sub_tmp_dir = job_dir / "platform_subs"
            if sub_tmp_dir.is_dir():
                shutil.rmtree(sub_tmp_dir, ignore_errors=True)
            processed_path = validate_managed_path(ai_result["output_path"], job_dir)
            if safe_path != processed_path and safe_path.is_file():
                try:
                    safe_path.unlink()
                except OSError:
                    pass
            safe_path = processed_path

        file_size = safe_path.stat().st_size

        title = ""
        if isinstance(info, dict):
            title = str(info.get("title") or info.get("fulltitle") or "").strip()
        suffix_map = {
            "video_1080": "1080p",
            "video_720": "720p",
            "audio_only": "audio",
        }
        base_suffix = suffix_map.get(preset, "media")
        if ai_enabled and ai_options is not None:
            lang_tag = ai_options.target_lang.upper()
            if ai_options.mode == "hardsub":
                base_suffix = f"{base_suffix} Sub-{lang_tag}"
            elif ai_options.burn_subtitles:
                base_suffix = f"{base_suffix} Dub+Sub-{lang_tag}"
            else:
                base_suffix = f"{base_suffix} Dub-{lang_tag}"

        srt_safe_path: Path | None = None
        if ai_result and ai_result.get("srt_path"):
            srt_candidate = validate_managed_path(ai_result["srt_path"], job_dir)
            if srt_candidate.is_file() and not srt_candidate.is_symlink():
                srt_safe_path = srt_candidate

        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                final_title = title or entry["title"] or f"Video {platform_label}"
                entry["title"] = final_title
                entry["file_path"] = str(safe_path)
                entry["filename"] = sanitize_download_filename(
                    final_title,
                    base_suffix,
                    safe_path.suffix.lstrip("."),
                )
                entry["file_size"] = int(file_size)
                if srt_safe_path is not None:
                    entry["subtitle_path"] = str(srt_safe_path)
                    entry["subtitle_filename"] = sanitize_download_filename(
                        final_title,
                        f"Sub-{ai_options.target_lang.upper() if ai_options else 'VI'}",
                        "srt",
                    )
                if ai_result:
                    entry["source_language"] = ai_result.get("source_language")
                    entry["segment_count"] = int(ai_result.get("segment_count") or 0)
                    entry["segments"] = ai_result.get("segments") or []
                entry["percent"] = 100.0
                entry["speed"] = None
                entry["eta"] = 0
                entry["stage"] = "completed"
                entry["stage_detail"] = "Hoàn tất!"
                entry["status"] = "completed"
    except (_DownloadCanceled, AICanceledError):
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["status"] = "canceled"
                entry["stage"] = "canceled"
                entry["stage_detail"] = "Đã hủy tác vụ."
                entry["error"] = "Đã hủy tải xuống."
        _cleanup_job_dir(job_dir)
    except Exception as exc:
        logger.warning("Media download {} failed: {}", job_id, exc)
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["status"] = "failed"
                entry["stage"] = "failed"
                entry["stage_detail"] = ""
                entry["error"] = str(exc).strip()[:280] or "Tải xuống thất bại."
        _cleanup_job_dir(job_dir)


def start_media_download(
    raw_url: str,
    preset: str,
    *,
    title: str = "",
    is_vertical: bool = False,
    ai_options: AIProcessOptions | None = None,
) -> dict[str, Any]:
    """Re-verify URL safety and launch a background media download (and optional AI studio) job."""
    if preset not in VALID_PRESETS:
        raise HTTPException(
            400,
            f"Chất lượng '{preset}' không hợp lệ. Chỉ hỗ trợ: video_1080, video_720, audio_only.",
        )

    opts = ai_options or AIProcessOptions()
    if opts.mode not in VALID_AI_MODES:
        raise HTTPException(400, f"Chế độ AI '{opts.mode}' không hợp lệ.")
    if opts.mode != "none":
        if preset == "audio_only":
            raise HTTPException(400, "Chế độ Hardsub / Lồng tiếng AI chỉ áp dụng cho video (1080p hoặc 720p).")
        if opts.target_lang not in TARGET_LANGUAGES:
            raise HTTPException(400, f"Ngôn ngữ đích '{opts.target_lang}' chưa được hỗ trợ.")
        if opts.voice not in VALID_VOICE_IDS:
            raise HTTPException(400, f"Giọng đọc '{opts.voice}' không hợp lệ.")
        if opts.subtitle_style not in VALID_SUBTITLE_STYLES:
            raise HTTPException(400, f"Kiểu phụ đề '{opts.subtitle_style}' không hợp lệ.")
        if opts.subtitle_animation not in VALID_SUBTITLE_ANIMATIONS:
            raise HTTPException(400, f"Hiệu ứng phụ đề '{opts.subtitle_animation}' không hợp lệ.")
        if opts.asr_engine not in VALID_ASR_ENGINES:
            raise HTTPException(400, f"Nguồn ASR '{opts.asr_engine}' không hợp lệ.")
        if opts.whisper_model not in VALID_WHISPER_MODELS:
            raise HTTPException(400, f"Model Whisper '{opts.whisper_model}' không hợp lệ.")
        try:
            provider, _ = resolve_configured_ai_provider(opts.provider_id)
            tr_model = normalize_model_id(
                opts.transcribe_model, default_transcribe_model_for(provider)
            )
            tts_provider_id = opts.tts_provider_id.strip() or provider.id
            tts_model = ""
            if opts.mode == "dub":
                if tts_provider_id == PIPER_TTS_PROVIDER_ID:
                    if not is_piper_tts_available():
                        raise RuntimeError("Thiếu thư viện piper-tts trên máy chủ.")
                    tts_model = "piper-onnx"
                elif tts_provider_id == EDGE_TTS_PROVIDER_ID:
                    if not is_edge_tts_available():
                        raise RuntimeError("Thiếu thư viện edge-tts trên máy chủ.")
                    tts_model = "edge-neural"
                else:
                    tts_provider, _ = resolve_configured_ai_provider(tts_provider_id)
                    tts_provider_id = tts_provider.id
                    tts_model = normalize_model_id(
                        opts.tts_model, default_tts_model_for(tts_provider)
                    )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        opts = AIProcessOptions(
            mode=opts.mode,
            provider_id=provider.id,
            tts_provider_id=tts_provider_id,
            target_lang=opts.target_lang,
            voice=opts.voice,
            burn_subtitles=opts.burn_subtitles,
            bilingual=opts.bilingual,
            subtitle_style=opts.subtitle_style,
            subtitle_animation=opts.subtitle_animation,
            keep_original_audio=opts.keep_original_audio,
            original_volume=opts.original_volume,
            transcribe_model=tr_model,
            tts_model=tts_model,
            context_hint=str(opts.context_hint or "").strip()[:500],
            asr_engine=opts.asr_engine,
            whisper_model=opts.whisper_model,
            video_title=str(opts.video_title or title or "").strip()[:240],
        )
        if not find_ffmpeg_binary():
            raise HTTPException(503, "Thiếu FFmpeg trên máy chủ để xử lý phụ đề cứng và lồng tiếng.")

    safety = inspect_link_safety(raw_url, resolve_redirects=True)
    if not safety["safe"]:
        raise HTTPException(400, safety["summary"])

    cleanup_expired_media_jobs()
    platform_key = str(safety.get("platform") or "web")
    platform_label = str(safety.get("platform_label") or safety.get("hostname") or "Web Video")
    target_url = safety["resolved_url"] or safety["normalized_url"]
    job_id = uuid.uuid4().hex[:12]
    job_dir = validate_managed_path(MEDIA_DIR / job_id, MEDIA_DIR)

    with _JOBS_LOCK:
        active_count = sum(
            1 for j in _JOBS.values() if j["status"] in ("pending", "running")
        )
        if active_count >= MAX_CONCURRENT_DOWNLOADS:
            raise HTTPException(
                429,
                f"Đang có {active_count} lượt tải chạy cùng lúc. Hãy chờ hoàn tất hoặc hủy bớt.",
            )
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "pending",
            "stage": "preparing",
            "stage_detail": "",
            "preset": preset,
            "ai_mode": opts.mode,
            "target_lang": opts.target_lang,
            "provider_id": opts.provider_id if opts.mode != "none" else None,
            "tts_provider_id": opts.tts_provider_id if opts.mode == "dub" else None,
            "transcribe_model": opts.transcribe_model if opts.mode != "none" else None,
            "tts_model": opts.tts_model if opts.mode == "dub" else None,
            "platform": platform_key,
            "platform_label": platform_label,
            "title": str(title or "").strip() or f"Video {platform_label}",
            "percent": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": None,
            "speed": None,
            "eta": None,
            "filename": None,
            "file_path": None,
            "file_size": None,
            "subtitle_path": None,
            "subtitle_filename": None,
            "source_language": None,
            "segment_count": 0,
            "segments": [],
            "job_dir": str(job_dir),
            "error": None,
            "created_at": time.time(),
            "cancel_event": threading.Event(),
        }
        _JOBS[job_id] = job
        snapshot = _snapshot_job(job)

    worker = threading.Thread(
        target=_run_download_worker,
        args=(job_id, target_url, platform_label, preset, bool(is_vertical), opts),
        name=f"media-dl-{job_id}",
        daemon=True,
    )
    worker.start()
    return snapshot


def get_media_job(job_id: str) -> dict[str, Any]:
    validate_job_id(job_id)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Không tìm thấy tác vụ tải video.")
        return _snapshot_job(job)


def list_media_jobs() -> list[dict[str, Any]]:
    cleanup_expired_media_jobs()
    with _JOBS_LOCK:
        items = sorted(
            (_snapshot_job(j) for j in _JOBS.values()),
            key=lambda item: float(item["created_at"]),
            reverse=True,
        )
    return items


def cancel_media_job(job_id: str) -> dict[str, Any]:
    validate_job_id(job_id)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Không tìm thấy tác vụ tải video.")
        if job["status"] in ("pending", "running"):
            job["cancel_event"].set()
            job["stage"] = "canceling"
        return _snapshot_job(job)


def delete_media_job(job_id: str) -> dict[str, Any]:
    validate_job_id(job_id)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Không tìm thấy tác vụ tải video.")
        if job["status"] in ("pending", "running"):
            job["cancel_event"].set()
        _JOBS.pop(job_id, None)
        job_dir = Path(job["job_dir"])
    _cleanup_job_dir(job_dir)
    return {"deleted": True, "job_id": job_id}


def resolve_media_job_file(job_id: str) -> tuple[Path, str, str]:
    validate_job_id(job_id)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Không tìm thấy tệp tải xuống.")
        if job["status"] != "completed" or not job["file_path"]:
            raise HTTPException(409, "Tệp chưa sẵn sàng để tải xuống.")
        raw_path = Path(job["file_path"])
        filename = str(job["filename"] or raw_path.name)

    safe_path = validate_managed_path(raw_path, MEDIA_DIR)
    if safe_path.is_symlink() or not safe_path.is_file():
        raise HTTPException(404, "Tệp đã bị xóa hoặc không còn tồn tại.")
    media_type, _ = mimetypes.guess_type(filename)
    return safe_path, filename, media_type or "application/octet-stream"


def resolve_media_job_subtitle(job_id: str) -> tuple[Path, str]:
    validate_job_id(job_id)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Không tìm thấy tác vụ tải video.")
        if job["status"] != "completed" or not job.get("subtitle_path"):
            raise HTTPException(404, "Tác vụ này không có tệp phụ đề SRT.")
        raw_path = Path(job["subtitle_path"])
        filename = str(job.get("subtitle_filename") or "subtitles.srt")

    safe_path = validate_managed_path(raw_path, MEDIA_DIR)
    if safe_path.is_symlink() or not safe_path.is_file():
        raise HTTPException(404, "Tệp phụ đề đã bị xóa hoặc không còn tồn tại.")
    return safe_path, filename
