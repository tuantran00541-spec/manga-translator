from __future__ import annotations

import pytest
import requests
from fastapi import HTTPException

from app.downloader.base import _retry_delay
from app.routers import chapters as chapters_router
from app.schemas import ChapterRequest


def _http_error(status: int, headers: dict | None = None) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return requests.HTTPError(f"{status} error", response=response)


def test_too_many_requests_waits_for_the_site():
    assert _retry_delay(_http_error(429, {"Retry-After": "7"}), 1) == 7.0
    assert _retry_delay(_http_error(429, {"Retry-After": "600"}), 1) == 60.0
    assert _retry_delay(_http_error(429), 2) == 20.0
    assert _retry_delay(_http_error(503), 2) == 2.0


def test_a_rate_limited_chapter_says_why_in_vietnamese(monkeypatch):
    def refuse(*args, **kwargs):
        raise _http_error(429)

    monkeypatch.setattr(chapters_router.pipeline, "download_chapter", refuse)
    monkeypatch.setattr(chapters_router, "validate_url", lambda url: url)
    with pytest.raises(HTTPException) as refused:
        chapters_router.create_chapter(ChapterRequest(url="https://example.com/chapter/1"))
    assert refused.value.status_code == 429
    assert "tải quá nhiều" in refused.value.detail
