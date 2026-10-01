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


def test_a_plain_page_keeps_its_comic_pages_and_drops_icons_and_thumbnails(monkeypatch, tmp_path):
    from PIL import Image

    from app.downloader import registry

    html = (
        b'<img src="/menu.svg"><img src="/thumb.jpg"><img src="/p1.jpg"><img src="/p2.jpg">'
    )

    class Page:
        def close(self):
            pass

    monkeypatch.setattr(registry, "safe_get", lambda *a, **k: Page())
    monkeypatch.setattr(registry, "read_response_limited", lambda *a, **k: html)
    urls = registry.GenericStaticAdapter().extract_image_urls("https://comic.example/ep1.html")
    assert urls[0] == "https://comic.example/menu.svg"

    sizes = {"thumb.jpg": (120, 120), "p1.jpg": (1200, 1660), "p2.jpg": (1200, 1660)}

    def fake_download(self, url, out_path, referer):
        Image.new("RGB", sizes[url.rsplit("/", 1)[1]]).save(out_path)

    monkeypatch.setattr("app.security.validate_url", lambda url: url)
    monkeypatch.setattr(registry.STATIC_ADAPTER, "extract_image_urls", lambda url: urls)
    monkeypatch.setattr(registry.JS_ADAPTER, "extract_image_urls", lambda url: [])
    monkeypatch.setattr(registry.GenericStaticAdapter, "_download_file", fake_download)
    pages = registry.download_chapter("https://comic.example/ep1.html", tmp_path)
    assert [path.name for path in pages] == ["001.jpg", "002.jpg"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["001.jpg", "002.jpg"]


def test_the_browser_path_drops_a_wide_svg_site_logo(monkeypatch, tmp_path):
    from PIL import Image

    from app.downloader import registry

    def fake_download(self, url, out_path, referer):
        Image.new("RGB", (1200, 1660)).save(out_path)

    monkeypatch.setattr("app.security.validate_url", lambda url: url)
    monkeypatch.setattr(registry.STATIC_ADAPTER, "extract_image_urls", lambda url: [])
    monkeypatch.setattr(
        registry.JS_ADAPTER,
        "extract_image_urls",
        lambda url: ["https://comic.example/logo.svg", "https://comic.example/p1.jpg", "https://comic.example/p2.jpg"],
    )
    monkeypatch.setattr(registry.GenericStaticAdapter, "_download_file", fake_download)
    pages = registry.download_chapter("https://comic.example/ep1.html", tmp_path)
    assert [path.name for path in pages] == ["000.jpg", "001.jpg"]
