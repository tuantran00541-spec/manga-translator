import socket

import pytest
from fastapi import HTTPException

from app.downloader import http


def test_a_name_that_turns_private_after_the_check_is_refused(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(http, "validate_url", lambda url: url)  # the first check saw a public address
    real = socket.getaddrinfo

    def rebound(host, *args, **kwargs):
        if host == "rebind.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebound)
    with pytest.raises(HTTPException) as caught:
        http.safe_get("http://rebind.example/page")
    assert caught.value.status_code == 400 and "127.0.0.1" in str(caught.value.detail)


def test_a_chapter_that_cannot_be_read_says_why(monkeypatch):
    from app.downloader import registry

    monkeypatch.setattr(registry, "STATIC_ADAPTER", registry.GenericStaticAdapter())
    monkeypatch.setattr("app.security.validate_url", lambda url: url)

    def refused(url):
        raise ConnectionError("503 from the site")

    monkeypatch.setattr(registry.STATIC_ADAPTER, "extract_image_urls", refused)
    monkeypatch.setattr(registry.JS_ADAPTER, "extract_image_urls", refused)
    with pytest.raises(ValueError, match="503 from the site"):
        registry.download_chapter("https://example.com/chapter/1", None)
