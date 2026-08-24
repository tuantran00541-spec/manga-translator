from __future__ import annotations

import io
import ipaddress
from pathlib import Path
import re
import socket
from urllib.parse import urlparse

from fastapi import HTTPException
from PIL import Image

CHAPTER_ID_RE = re.compile(r"^[a-f0-9]{8}$")

MAX_REQUEST_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_RENDER_TRANSLATIONS = 100
MAX_RENDER_TEXT_LEN = 5000

MAX_UPLOAD_FILE_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024
MAX_UPLOAD_FILES = 300
ALLOWED_UPLOAD_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}

MAX_REMOTE_IMAGE_BYTES = MAX_UPLOAD_FILE_BYTES
MAX_REMOTE_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_REMOTE_REDIRECTS = 5
MAX_REMOTE_URL_LENGTH = 8192

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_BLOCKED_NETWORKS = [
    ipaddress.ip_network(n) for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
        "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
        "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24",
        "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4",
        "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8",
    )
]


def validate_chapter_id(chapter_id: str) -> str:
    if not chapter_id or not CHAPTER_ID_RE.match(chapter_id):
        raise HTTPException(400, f"Invalid chapter_id: {chapter_id!r}")
    return chapter_id


def validate_url(url: str) -> str:
    if not url or not isinstance(url, str):
        raise HTTPException(400, "URL is required")
    if len(url) > MAX_REMOTE_URL_LENGTH:
        raise HTTPException(400, "URL is too long")

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise HTTPException(400, "Malformed URL") from exc
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"URL scheme '{parsed.scheme}' not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(400, "Credentials in URLs are not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, "URL has no hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(400, "URL has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise HTTPException(400, "URL has an invalid port")

    try:
        ip_obj = ipaddress.ip_address(hostname)
        _check_ip(ip_obj, hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise HTTPException(400, f"Cannot resolve hostname: {hostname}") from exc
        if not infos:
            raise HTTPException(400, f"Cannot resolve hostname: {hostname}")

        checked = 0
        for _, _, _, _, sockaddr in infos:
            if not sockaddr:
                continue
            try:
                ip_obj = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            _check_ip(ip_obj, hostname)
            checked += 1
        if checked == 0:
            raise HTTPException(400, f"Cannot resolve hostname: {hostname}")

    return url


def _check_ip(ip_obj, hostname: str) -> None:
    mapped = getattr(ip_obj, "ipv4_mapped", None)
    if mapped is not None:
        _check_ip(mapped, hostname)
        return

    if not ip_obj.is_global:
        raise HTTPException(
            400,
            f"Hostname '{hostname}' resolves to non-public IP {ip_obj}",
        )
    for net in _BLOCKED_NETWORKS:
        if ip_obj in net:
            raise HTTPException(
                400,
                f"Hostname '{hostname}' resolves to blocked IP {ip_obj}",
            )


def browser_request_allowed(url: str) -> bool:
    if not isinstance(url, str) or not url:
        return False
    try:
        scheme = urlparse(url).scheme.lower()
    except ValueError:
        return False
    if scheme in {"about", "blob", "data"}:
        return True
    if scheme not in {"http", "https"}:
        return False
    try:
        validate_url(url)
        return True
    except HTTPException:
        return False


def validate_managed_path(path: str | Path, root: str | Path) -> Path:
    try:
        root_path = Path(root).resolve(strict=False)
        resolved = Path(path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(403, "Managed file path is invalid") from exc
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise HTTPException(403, "Managed file path escapes the allowed project directory") from exc
    return resolved


def validate_upload_image(data: bytes, filename: str) -> str:
    if not data:
        raise HTTPException(400, f"Empty file: {filename}")
    if len(data) > MAX_UPLOAD_FILE_BYTES:
        raise HTTPException(
            413,
            f"File too large: {filename} exceeds {MAX_UPLOAD_FILE_BYTES // (1024*1024)}MB",
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            w, h = img.size
    except Exception as exc:
        raise HTTPException(400, f"Not a valid image file: {filename}") from exc

    if fmt not in ALLOWED_UPLOAD_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported image format '{fmt}' for {filename}. Only PNG, JPEG, WEBP, BMP are allowed.",
        )
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(
            413,
            f"Image too large: {filename} is {w}x{h}, exceeds {MAX_IMAGE_PIXELS} pixels",
        )
    return fmt


def validate_image_file(path: str | Path) -> str:
    try:
        with Image.open(path) as img:
            fmt = img.format
            w, h = img.size
            if fmt not in ALLOWED_UPLOAD_FORMATS:
                raise HTTPException(400, "Downloaded file is not a supported image format")
            if w * h > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    413,
                    f"Image too large: {w}x{h} exceeds {MAX_IMAGE_PIXELS} pixels",
                )
            img.verify()
        return str(fmt)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Invalid or unreadable image") from exc


def validate_image_size(path) -> None:
    try:
        with Image.open(path) as img:
            w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            raise HTTPException(
                413,
                f"Image too large: {w}x{h} exceeds {MAX_IMAGE_PIXELS} pixels",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "Invalid or unreadable image") from exc
