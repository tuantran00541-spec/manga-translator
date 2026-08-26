from app.downloader.image_urls import best_srcset_candidate, resolve_image_candidate
from app.downloader.registry import GenericStaticAdapter, _choose_image_urls


class FakeResponse:
    def close(self):
        return None


def test_relative_and_srcset_urls_are_resolved(monkeypatch):
    html = b"""
    <html><body>
      <img width='80' src='/logo.png'>
      <img src='/chapter/001.jpg'>
      <img data-srcset='/chapter/002-small.jpg 400w, /chapter/002.jpg 1200w'>
    </body></html>
    """
    monkeypatch.setattr("app.downloader.registry.safe_get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr("app.downloader.registry.read_response_limited", lambda *a, **k: html)

    urls = GenericStaticAdapter().extract_image_urls("https://reader.example/manga/1")
    assert urls == [
        "https://reader.example/chapter/001.jpg",
        "https://reader.example/chapter/002.jpg",
    ]


def test_js_candidates_win_when_browser_found_multiple_content_images():
    static = ["https://x/logo.jpg", "https://x/banner.jpg", "https://x/one.jpg"]
    js = ["https://cdn/001.jpg", "https://cdn/002.jpg"]
    assert _choose_image_urls(static, js) == js
    assert _choose_image_urls(static, []) == static


def test_srcset_prefers_highest_descriptor_and_rejects_data_urls():
    assert best_srcset_candidate("a.jpg 400w, b.jpg 1200w") == "b.jpg"
    assert resolve_image_candidate("https://x/a/", "data:image/png;base64,abc", "../b.jpg") == "https://x/b.jpg"
