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
    domains: frozenset[str]
    short_domains: frozenset[str]
    allowed_extractors: tuple[str, ...]


PLATFORMS: tuple[PlatformSpec, ...] = (
    PlatformSpec(
        key="youtube",
        label="YouTube",
        domains=frozenset(
            {
                "youtube.com",
                "www.youtube.com",
                "m.youtube.com",
                "music.youtube.com",
                "youtu.be",
                "www.youtu.be",
                "youtube-nocookie.com",
                "www.youtube-nocookie.com",
            }
        ),
        short_domains=frozenset({"youtu.be", "www.youtu.be"}),
        allowed_extractors=(r"(?i)^youtube(?::.*)?$", r"(?i)^YoutubeYtBe$"),
    ),
    PlatformSpec(
        key="tiktok",
        label="TikTok",
        domains=frozenset(
            {
                "tiktok.com",
                "www.tiktok.com",
                "m.tiktok.com",
                "vm.tiktok.com",
                "vt.tiktok.com",
            }
        ),
        short_domains=frozenset({"vm.tiktok.com", "vt.tiktok.com"}),
        allowed_extractors=(r"(?i)^tiktok(?::.*)?$", r"(?i)^vm\.tiktok$"),
    ),
    PlatformSpec(
        key="facebook",
        label="Facebook",
        domains=frozenset(
            {
                "facebook.com",
                "www.facebook.com",
                "m.facebook.com",
                "web.facebook.com",
                "mbasic.facebook.com",
                "fb.watch",
                "www.fb.watch",
            }
        ),
        short_domains=frozenset({"fb.watch", "www.fb.watch"}),
        allowed_extractors=(
            r"(?i)^facebook(?::.*)?$",
            r"(?i)^FacebookPluginsVideo$",
            r"(?i)^FacebookRedirectURL$",
        ),
    ),
)

_PLATFORM_BY_KEY: dict[str, PlatformSpec] = {p.key: p for p in PLATFORMS}
_HOST_TO_PLATFORM: dict[str, PlatformSpec] = {
    host: spec for spec in PLATFORMS for host in spec.domains
}


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


def _match_platform(host: str) -> PlatformSpec | None:
    return _HOST_TO_PLATFORM.get(host.lower().rstrip("."))


def _looks_like_video_path(parsed, platform: PlatformSpec) -> bool:
    path = (parsed.path or "").strip("/")
    query = (parsed.query or "").strip()
    if platform.key == "youtube":
        if parsed.hostname and parsed.hostname.lower().rstrip(".") in platform.short_domains:
            return bool(path)
        return bool(path) or bool(query)
    if platform.key == "tiktok":
        return bool(path)
    if platform.key == "facebook":
        return bool(path) or bool(query)
    return bool(path)


def _check_redirect_chain(
    start_url: str,
    platform: PlatformSpec,
) -> tuple[bool, str, str]:
    """Follow up to MAX_REDIRECT_HOPS redirects without leaving the platform allowlist."""
    current = start_url
    hops = 0

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
            # Some social CDNs throttle or reset direct requests even when yt-dlp succeeds.
            if hops == 0:
                return (
                    True,
                    current,
                    "Không phát hiện chuyển hướng nguy hiểm (máy chủ đích hạn chế kiểm tra trực tiếp).",
                )
            return True, current, f"Đã xác minh {hops} bước chuyển hướng an toàn trong {platform.label}."
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

        if status not in _REDIRECT_STATUSES or not location:
            if hops == 0:
                return True, current, f"Liên kết trỏ thẳng đến {platform.label}, không có chuyển hướng bất thường."
            return True, current, f"Đã xác minh {hops} bước chuyển hướng đều nằm trên tên miền {platform.label}."

        hops += 1
        if hops > MAX_REDIRECT_HOPS:
            return False, current, "Chuỗi chuyển hướng vượt quá 5 bước cho phép."

        next_url = urljoin(current, location.strip())
        if _has_control_chars(next_url) or len(next_url) > MAX_MEDIA_URL_LENGTH:
            return False, current, "Đích chuyển hướng chứa ký tự không hợp lệ hoặc quá dài."

        try:
            next_parsed = urlsplit(next_url)
        except ValueError:
            return False, current, "Đích chuyển hướng không đúng định dạng URL."

        if next_parsed.scheme.lower() != "https":
            return False, next_url, f"Chuyển hướng hạ cấp xuống giao thức không an toàn ({next_parsed.scheme or 'trống'})."
        if next_parsed.username is not None or next_parsed.password is not None:
            return False, next_url, "Đích chuyển hướng chứa thông tin đăng nhập nhúng."

        next_host = (next_parsed.hostname or "").lower().rstrip(".")
        if not next_host or _has_homograph_risk(next_host) or _is_raw_ip(next_host):
            return False, next_url, f"Tên miền chuyển hướng đáng ngờ: {next_host or '(trống)'}."

        next_platform = _match_platform(next_host)
        if next_platform is None or next_platform.key != platform.key:
            return (
                False,
                next_url,
                f"Liên kết chuyển hướng ra ngoài {platform.label} tới tên miền không tin cậy: {next_host}.",
            )

        try:
            validate_url(next_url)
        except HTTPException as exc:
            return False, next_url, f"Đích chuyển hướng vi phạm kiểm tra mạng an toàn: {exc.detail}"

        current = next_url

    return False, current, "Chuỗi chuyển hướng vượt quá giới hạn an toàn."


def inspect_link_safety(
    raw_url: str,
    *,
    resolve_redirects: bool = True,
) -> dict[str, Any]:
    """Run multi-layer link safety verification for TikTok, YouTube, and Facebook URLs."""
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
            "platform": platform.key if platform else None,
            "platform_label": platform.label if platform else None,
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

    # 2. Official platform domain & anti-phishing check
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        checks.append(
            {
                "id": "official_domain",
                "label": "Tên miền chính chủ (TikTok / YouTube / Facebook)",
                "passed": False,
                "detail": "Liên kết không có tên miền.",
            }
        )
        return _result(safe=False, summary="Liên kết thiếu tên miền.")

    if _is_raw_ip(hostname):
        checks.append(
            {
                "id": "official_domain",
                "label": "Tên miền chính chủ (TikTok / YouTube / Facebook)",
                "passed": False,
                "detail": f"Chặn địa chỉ IP trực tiếp ({hostname}).",
            }
        )
        return _result(safe=False, hostname=hostname, summary="Không cho phép tải từ địa chỉ IP trực tiếp.")

    if _has_homograph_risk(hostname):
        checks.append(
            {
                "id": "official_domain",
                "label": "Tên miền chính chủ (TikTok / YouTube / Facebook)",
                "passed": False,
                "detail": f"Phát hiện tên miền quốc tế hóa / Punycode đáng ngờ ({hostname}).",
            }
        )
        return _result(safe=False, hostname=hostname, summary="Tên miền có dấu hiệu giả mạo (IDN/Punycode).")

    platform = _match_platform(hostname)
    if platform is None:
        checks.append(
            {
                "id": "official_domain",
                "label": "Tên miền chính chủ (TikTok / YouTube / Facebook)",
                "passed": False,
                "detail": f"Tên miền '{hostname}' không nằm trong danh sách chính chủ của TikTok, YouTube hoặc Facebook.",
            }
        )
        return _result(
            safe=False,
            hostname=hostname,
            summary=f"Tên miền '{hostname}' không được hỗ trợ hoặc không phải trang chính chủ.",
        )

    if not _looks_like_video_path(parsed, platform):
        checks.append(
            {
                "id": "official_domain",
                "label": "Tên miền chính chủ (TikTok / YouTube / Facebook)",
                "passed": False,
                "detail": f"Đây là trang chủ {platform.label}, chưa phải đường dẫn tới một video cụ thể.",
            }
        )
        return _result(
            safe=False,
            platform=platform,
            hostname=hostname,
            summary=f"Hãy dán đường dẫn trực tiếp của video {platform.label} thay vì trang chủ.",
        )

    checks.append(
        {
            "id": "official_domain",
            "label": "Tên miền chính chủ (TikTok / YouTube / Facebook)",
            "passed": True,
            "detail": f"Xác nhận tên miền chính chủ {platform.label} ({hostname}).",
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

    # 4. Redirect chain safety verification (especially important for short links)
    resolved_url = url
    if resolve_redirects:
        ok_redirect, resolved_url, redirect_detail = _check_redirect_chain(url, platform)
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

    return _result(
        safe=True,
        platform=platform,
        hostname=hostname,
        resolved_url=resolved_url,
        summary=f"Liên kết {platform.label} an toàn và hợp lệ.",
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
    formats = [f for f in (info.get("formats") or []) if isinstance(f, dict)]
    duration = info.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = None

    all_videos = [
        f
        for f in formats
        if f.get("vcodec") not in (None, "none") and not f.get("has_drm") and _short_side(f) > 0
    ]
    is_vertical = any(
        (f.get("height") or 0) > (f.get("width") or 0)
        for f in all_videos
        if f.get("width") and f.get("height")
    )
    max_source_side = max((_short_side(f) for f in all_videos), default=0)

    best_audio = _pick_best_audio(formats)
    audio_bytes = _estimate_bytes(best_audio, duration)

    # 1080p option
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

    # 720p option (also acts as fallback SD video option if source max < 720p)
    best_720 = _pick_best_video(formats, 760)
    if best_720 is None and all_videos:
        # Some sources only report a single progressive stream > 760p; still allow 720p selector fallback
        best_720 = min(all_videos, key=_short_side)
    side_720 = _short_side(best_720) if best_720 else 0
    has_video_fallback = bool(best_720 or all_videos or formats)
    v720_bytes = _estimate_bytes(best_720, duration)
    if best_720 and best_720.get("acodec") in (None, "none") and audio_bytes:
        v720_bytes = (v720_bytes or 0) + audio_bytes
    res_720 = (
        f"{int(best_720['width'])}×{int(best_720['height'])}"
        if best_720 and best_720.get("width") and best_720.get("height")
        else (f"{side_720}p" if side_720 else "720p")
    )
    fps_720 = int(round(float(best_720["fps"]))) if best_720 and best_720.get("fps") else None

    label_720 = (
        "720p HD"
        if side_720 == 0 or side_720 >= 680
        else f"720p / Tối đa nguồn ({side_720}p)"
    )

    audio_ext = "MP3" if has_ffmpeg else "M4A"
    audio_abr = int(round(float(best_audio["abr"]))) if best_audio and best_audio.get("abr") else (192 if has_ffmpeg else 128)
    audio_est = (
        int(192 * 1000 / 8 * duration)
        if has_ffmpeg and duration
        else audio_bytes
    )

    options: list[dict[str, Any]] = [
        {
            "id": "video_1080",
            "kind": "video",
            "label": "1080p Full HD",
            "container": "MP4",
            "resolution": res_1080 if has_1080 else (f"Nguồn tối đa {max_source_side}p" if max_source_side else "Không có bản 1080p"),
            "fps": fps_1080 if has_1080 else None,
            "estimated_size": _format_size(v1080_bytes) if has_1080 else None,
            "available": has_1080,
            "description": (
                "Sắc nét nhất, tự động ghép hình ảnh + âm thanh MP4."
                if has_1080
                else "Video gốc không có luồng 1080p — hãy chọn 720p hoặc Chỉ Audio."
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


def _extract_info(target_url: str, platform: PlatformSpec) -> dict[str, Any]:
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
        "allowed_extractors": list(platform.allowed_extractors),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target_url, download=False)
    except Exception as exc:
        msg = str(exc).strip()
        lowered = msg.lower()
        if "drm" in lowered:
            raise HTTPException(400, "Nội dung có bảo vệ DRM nên không hỗ trợ tải xuống.") from exc
        if "private" in lowered or "login" in lowered or "sign in" in lowered or "cookies" in lowered:
            raise HTTPException(
                400,
                f"Video {platform.label} này ở chế độ riêng tư hoặc yêu cầu đăng nhập.",
            ) from exc
        if "unavailable" in lowered or "not exist" in lowered or "removed" in lowered:
            raise HTTPException(404, f"Không tìm thấy video hoặc video trên {platform.label} đã bị gỡ.") from exc
        logger.warning("Media inspect failed for {}: {}", platform.key, msg)
        raise HTTPException(
            502,
            f"Không lấy được thông tin video từ {platform.label}: {msg[:240] or 'Lỗi kết nối'}",
        ) from exc

    if not isinstance(info, dict):
        raise HTTPException(502, f"Không đọc được dữ liệu video từ {platform.label}.")
    if "entries" in info and isinstance(info["entries"], list):
        first = next((item for item in info["entries"] if isinstance(item, dict)), None)
        if first is None:
            raise HTTPException(404, "Liên kết không chứa video nào khả dụng.")
        info = first
    return info


def inspect_media_url(raw_url: str) -> dict[str, Any]:
    """Verify link safety first; if safe, fetch video metadata and quality presets."""
    safety = inspect_link_safety(raw_url, resolve_redirects=True)
    if not safety["safe"]:
        return {
            "safe": False,
            "safety": safety,
            "media": None,
            "options": [],
            "default_preset": None,
            "ffmpeg_available": bool(find_ffmpeg_binary()),
        }

    platform = _PLATFORM_BY_KEY[safety["platform"]]
    target_url = safety["resolved_url"] or safety["normalized_url"]
    info = _extract_info(target_url, platform)

    # 5. Content & policy check (reject DRM, active livestreams, overly long streams)
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
        }

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

    title = str(info.get("title") or info.get("fulltitle") or f"Video {platform.label}").strip()
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
            "platform": platform.key,
            "platform_label": platform.label,
            "is_vertical": is_vertical,
        },
        "options": options,
        "default_preset": default_preset,
        "ffmpeg_available": has_ffmpeg,
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
    # video_720
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
        "preset": job["preset"],
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

    # Sweep orphaned directories on disk left over from previous app runs
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
    platform: PlatformSpec,
    preset: str,
    is_vertical: bool,
) -> None:
    job_dir = validate_managed_path(MEDIA_DIR / job_id, MEDIA_DIR)
    job_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = find_ffmpeg_binary()
    has_ffmpeg = bool(ffmpeg_bin)

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
                pct = round(min(99.0, (downloaded / total) * 100.0), 1) if total > 0 else entry["percent"]
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
                entry["percent"] = 99.0
                entry["speed"] = None
                entry["eta"] = 0

    def postprocessor_hook(data: dict[str, Any]) -> None:
        if cancel_event.is_set():
            raise _DownloadCanceled()
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["stage"] = "merging"
                entry["percent"] = 99.5

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
        "format": fmt_selector,
        "outtmpl": str(job_dir / "media.%(ext)s"),
        "allowed_extractors": list(platform.allowed_extractors),
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
        file_size = safe_path.stat().st_size

        title = ""
        if isinstance(info, dict):
            title = str(info.get("title") or info.get("fulltitle") or "").strip()
        suffix_map = {
            "video_1080": "1080p",
            "video_720": "720p",
            "audio_only": "audio",
        }
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                final_title = title or entry["title"] or f"Video {platform.label}"
                entry["title"] = final_title
                entry["file_path"] = str(safe_path)
                entry["filename"] = sanitize_download_filename(
                    final_title,
                    suffix_map.get(preset, "media"),
                    safe_path.suffix.lstrip("."),
                )
                entry["file_size"] = int(file_size)
                entry["percent"] = 100.0
                entry["speed"] = None
                entry["eta"] = 0
                entry["stage"] = "completed"
                entry["status"] = "completed"
    except _DownloadCanceled:
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["status"] = "canceled"
                entry["stage"] = "canceled"
                entry["error"] = "Đã hủy tải xuống."
        _cleanup_job_dir(job_dir)
    except Exception as exc:
        logger.warning("Media download {} failed: {}", job_id, exc)
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry:
                entry["status"] = "failed"
                entry["stage"] = "failed"
                entry["error"] = str(exc).strip()[:280] or "Tải xuống thất bại."
        _cleanup_job_dir(job_dir)


def start_media_download(
    raw_url: str,
    preset: str,
    *,
    title: str = "",
    is_vertical: bool = False,
) -> dict[str, Any]:
    """Re-verify URL safety and launch a background media download job."""
    if preset not in VALID_PRESETS:
        raise HTTPException(
            400,
            f"Chất lượng '{preset}' không hợp lệ. Chỉ hỗ trợ: video_1080, video_720, audio_only.",
        )

    safety = inspect_link_safety(raw_url, resolve_redirects=True)
    if not safety["safe"]:
        raise HTTPException(400, safety["summary"])

    cleanup_expired_media_jobs()
    platform = _PLATFORM_BY_KEY[safety["platform"]]
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
            "preset": preset,
            "platform": platform.key,
            "platform_label": platform.label,
            "title": str(title or "").strip() or f"Video {platform.label}",
            "percent": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": None,
            "speed": None,
            "eta": None,
            "filename": None,
            "file_path": None,
            "file_size": None,
            "job_dir": str(job_dir),
            "error": None,
            "created_at": time.time(),
            "cancel_event": threading.Event(),
        }
        _JOBS[job_id] = job
        snapshot = _snapshot_job(job)

    worker = threading.Thread(
        target=_run_download_worker,
        args=(job_id, target_url, platform, preset, bool(is_vertical)),
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
