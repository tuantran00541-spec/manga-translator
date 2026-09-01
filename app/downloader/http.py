from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin
import uuid

import requests

from app.parameters import REMOTE_CHUNK_BYTES, REMOTE_CONNECT_TIMEOUT_SECONDS
from app.security import (
    MAX_REMOTE_IMAGE_BYTES,
    MAX_REMOTE_REDIRECTS,
    validate_image_file,
    validate_url,
)

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def safe_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int | float = REMOTE_CONNECT_TIMEOUT_SECONDS,
    stream: bool = True,
    max_redirects: int = MAX_REMOTE_REDIRECTS,
) -> requests.Response:
    current = validate_url(url)
    redirects = 0

    while True:
        response = requests.get(  # NOSONAR(S5144): validate_url rejects non-public addresses before the initial request and every redirect hop.
            current,
            headers=headers,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
        )
        if response.status_code not in _REDIRECT_STATUS_CODES:
            response.raise_for_status()
            return response

        location = response.headers.get("Location")
        response.close()
        if not location:
            raise ValueError("Redirect response is missing Location header")
        redirects += 1
        if redirects > max_redirects:
            raise ValueError(f"Too many redirects (>{max_redirects})")
        current = validate_url(urljoin(current, location))


def read_response_limited(response: requests.Response, *, limit_bytes: int) -> bytes:
    if limit_bytes < 1:
        raise ValueError("limit_bytes must be >= 1")
    _reject_oversized_content_length(response, limit_bytes)

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=REMOTE_CHUNK_BYTES):
        if not chunk:
            continue
        total += len(chunk)
        if total > limit_bytes:
            raise ValueError(f"Remote response exceeds {limit_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def safe_download_file(
    url: str,
    out_path: Path,
    *,
    headers: dict[str, str] | None = None,
    timeout: int | float = REMOTE_CONNECT_TIMEOUT_SECONDS,
    limit_bytes: int = MAX_REMOTE_IMAGE_BYTES,
) -> None:
    if limit_bytes < 1:
        raise ValueError("limit_bytes must be >= 1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.{uuid.uuid4().hex}.part")
    response = None
    try:
        response = safe_get(url, headers=headers, timeout=timeout, stream=True)
        _reject_oversized_content_length(response, limit_bytes)
        total = 0
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=REMOTE_CHUNK_BYTES):
                if not chunk:
                    continue
                total += len(chunk)
                if total > limit_bytes:
                    raise ValueError(f"Remote image exceeds {limit_bytes} bytes")
                handle.write(chunk)
        validate_image_file(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if response is not None:
            response.close()
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _reject_oversized_content_length(response: requests.Response, limit_bytes: int) -> None:
    raw = response.headers.get("Content-Length")
    if raw is None:
        return
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return
    if value < 0:
        raise ValueError("Remote response has an invalid Content-Length")
    if value > limit_bytes:
        raise ValueError(f"Remote response exceeds {limit_bytes} bytes")
