import requests

import app.downloader.base as base_module
from app.downloader.base import BaseAdapter


class _Adapter(BaseAdapter):
    def can_handle(self, url: str) -> bool:
        return True

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        return []


def test_image_download_retries_transient_connection_errors(monkeypatch, tmp_path):
    calls = []
    sleeps = []

    def fake_download(url, out_path, *, headers=None):
        calls.append((url, out_path, headers))
        if len(calls) < 3:
            raise requests.ConnectionError("synthetic timeout")
        out_path.write_bytes(b"ok")

    monkeypatch.setattr(base_module, "safe_download_file", fake_download)
    monkeypatch.setattr(base_module, "DOWNLOAD_RETRY_ATTEMPTS", 4)
    monkeypatch.setattr(base_module, "DOWNLOAD_RETRY_BACKOFF_SECONDS", 0.25)
    monkeypatch.setattr(base_module.time, "sleep", sleeps.append)

    target = tmp_path / "000.jpg"
    _Adapter()._download_file(
        "https://cdn.example.test/000.jpg",
        target,
        "https://example.test/chapter/1",
    )

    assert len(calls) == 3
    assert sleeps == [0.25, 0.5]
    assert target.read_bytes() == b"ok"
    assert calls[0][2]["Referer"] == "https://example.test/chapter/1"


def test_image_download_does_not_retry_validation_errors(monkeypatch, tmp_path):
    calls = []

    def fake_download(url, out_path, *, headers=None):
        calls.append(url)
        raise ValueError("invalid image")

    monkeypatch.setattr(base_module, "safe_download_file", fake_download)
    monkeypatch.setattr(base_module, "DOWNLOAD_RETRY_ATTEMPTS", 4)

    try:
        _Adapter()._download_file(
            "https://cdn.example.test/000.jpg",
            tmp_path / "000.jpg",
            "https://example.test/chapter/1",
        )
    except ValueError as exc:
        assert str(exc) == "invalid image"
    else:
        raise AssertionError("validation error must propagate")

    assert calls == ["https://cdn.example.test/000.jpg"]
