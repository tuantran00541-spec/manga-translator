from __future__ import annotations

import io
import ipaddress
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

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_BLOCKED_NETWORKS = [
    ipaddress.ip_network(n) for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
        "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
        "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24",
        "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4",
        "::1/128", "fc00::/7", "fe80::/10",
    )
]


def validate_chapter_id(chapter_id: str) -> str:
    if not chapter_id or not CHAPTER_ID_RE.match(chapter_id):
        raise HTTPException(400, f"Invalid chapter_id: {chapter_id!r}")
    return chapter_id


def validate_url(url: str) -> str:
    if not url or not isinstance(url, str):
        raise HTTPException(400, "URL is required")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"URL scheme '{parsed.scheme}' not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, "URL has no hostname")

    try:
        ip_obj = ipaddress.ip_address(hostname)
        _check_ip(ip_obj, hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            raise HTTPException(400, f"Cannot resolve hostname: {hostname}")

        for _, _, _, _, sockaddr in infos:
            try:
                _check_ip(ipaddress.ip_address(sockaddr[0]), hostname)
            except ValueError:
                continue

    return url


def _check_ip(ip_obj, hostname: str) -> None:
    for net in _BLOCKED_NETWORKS:
        if ip_obj in net:
            raise HTTPException(
                400,
                f"Hostname '{hostname}' resolves to blocked IP {ip_obj}",
            )


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
    except Exception:
        raise HTTPException(400, f"Not a valid image file: {filename}")

    if fmt not in ALLOWED_UPLOAD_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported image format '{fmt}' for {filename}. Only PNG, JPEG, WEBP are allowed.",
        )
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(
            413,
            f"Image too large: {filename} is {w}x{h}, exceeds {MAX_IMAGE_PIXELS} pixels",
        )
    return fmt


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
    except Exception:
        raise HTTPException(400, "Invalid or unreadable image")
