from pathlib import Path

from PIL import Image

import app.downloader.registry as registry


def _write_small_images(output_dir: Path, count: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = output_dir / f"{index:03d}.png"
        Image.new("RGB", (10, 10), "white").save(path)
        paths.append(path)
    return paths


def test_download_chapter_empty_filtered_pages_returns_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.security.validate_url", lambda url: url)
    monkeypatch.setattr(registry, "is_asura_chapter_page", lambda url: False)

    class StaticAdapter(registry.GenericStaticAdapter):
        def extract_image_urls(self, chapter_url):
            return ["http://example.com/a.png", "http://example.com/b.png"]

        def download_urls(self, urls, output_dir, referer):
            return _write_small_images(Path(output_dir), len(urls))

    class JsAdapter(registry.GenericStaticAdapter):
        def extract_image_urls(self, chapter_url):
            return []

    monkeypatch.setattr(registry, "STATIC_ADAPTER", StaticAdapter())
    monkeypatch.setattr(registry, "JS_ADAPTER", JsAdapter())

    result = registry.download_chapter("http://example.com/chapter", tmp_path / "out")

    assert all(Path(path).exists() for path in result)
