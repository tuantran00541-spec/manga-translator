"""Audit §8.16 (S2): the Asura static adapter must not silently truncate a
chapter at the first gap in "Page N" numbering. A gap fails loudly so the
registry falls through to the JS adapter (DOM order) instead of shipping a
short chapter with no error.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import app.downloader.registry as registry
from app.downloader.asura import _ordered_reader_urls


def _asset(n: int) -> str:
    return f"https://example.com/asura-images/chapters/c/{n}.jpg"


def test_ordered_reader_urls_keeps_contiguous_pages_in_order():
    assert _ordered_reader_urls([(2, _asset(2)), (1, _asset(1)), (3, _asset(3))]) == [
        _asset(1), _asset(2), _asset(3),
    ]


def test_ordered_reader_urls_raises_on_gap_in_the_middle():
    with pytest.raises(ValueError, match="gap at page 3"):
        _ordered_reader_urls([(1, _asset(1)), (2, _asset(2)), (4, _asset(4)), (5, _asset(5))])


def test_ordered_reader_urls_raises_when_first_page_is_missing():
    with pytest.raises(ValueError, match="gap at page 1"):
        _ordered_reader_urls([(2, _asset(2)), (3, _asset(3))])


def test_ordered_reader_urls_keeps_first_duplicate_and_ignores_non_assets():
    assert _ordered_reader_urls([
        (1, _asset(1)),
        (1, "https://example.com/asura-images/chapters/c/1-dup.jpg"),
        (2, "https://example.com/other/2.jpg"),
        (2, _asset(2)),
    ]) == [_asset(1), _asset(2)]


def test_registry_falls_through_to_js_adapter_on_static_gap(monkeypatch, tmp_path):
    """A gapped static list must not win the registry's len>=2 preference:
    the static attempt fails, so the JS adapter's DOM-order list is used."""
    monkeypatch.setattr("app.security.validate_url", lambda url: url)
    monkeypatch.setattr(registry, "is_asura_chapter_page", lambda url: True)

    class StaticAdapter(registry.ASURA_STATIC_ADAPTER.__class__):
        def extract_image_urls(self, chapter_url):
            raise ValueError("Asura chapter page list has a gap at page 3")

    class JsAdapter(registry.ASURA_JS_ADAPTER.__class__):
        def extract_image_urls(self, chapter_url):
            return ["http://example.com/js1.png", "http://example.com/js2.png", "http://example.com/js3.png"]

    class StaticDownloader(registry.GenericStaticAdapter):
        def download_urls(self, urls, output_dir, referer):
            paths = []
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for index in range(len(urls)):
                path = output_dir / f"{index:03d}.png"
                Image.new("RGB", (10, 10), "white").save(path)
                paths.append(path)
            return paths

    monkeypatch.setattr(registry, "ASURA_STATIC_ADAPTER", StaticAdapter())
    monkeypatch.setattr(registry, "ASURA_JS_ADAPTER", JsAdapter())
    monkeypatch.setattr(registry, "STATIC_ADAPTER", StaticDownloader())

    result = registry.download_chapter("https://asurascans.com/chapter/1", tmp_path / "out")
    assert len(result) == 3
    assert all(Path(path).exists() for path in result)
