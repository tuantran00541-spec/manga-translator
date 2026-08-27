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


def test_asura_reader_precision_uses_dom_and_asset_provenance():
    from app.downloader.asura import AsuraScansJsAdapter, filter_asura_chapter_assets

    chapter = "https://asurascans.com/comics/title-b57aa235/chapter/92"
    mixed = [
        "https://cdn.asurascans.com/asura-images/covers/title.webp",
        "https://cdn.asurascans.com/asura-images/chapters/title/92/page-a.webp?v=1",
        "https://cdn.asurascans.com/asura-images/profiles/123.webp",
        "https://cdn.asurascans.com/asura-images/comments/123-abc.webp",
        "https://cdn.asurascans.com/asura-images/gifs/abc.webp",
        "https://cdn.asurascans.com/asura-images/chapters/title/92/page-b.webp?v=1",
    ]

    adapter = AsuraScansJsAdapter()
    assert adapter.can_handle(chapter)
    assert adapter.img_selector == "img[alt^='Page ']"
    assert filter_asura_chapter_assets(mixed) == [mixed[1], mixed[5]]


def test_asura_asset_filter_does_not_use_image_dimensions():
    from app.downloader.asura import filter_asura_chapter_assets

    # A tiny legitimate reader page is retained while a huge comment image is
    # rejected because provenance, not dimensions, decides membership.
    reader = "https://cdn.asurascans.com/asura-images/chapters/title/7/tiny.webp"
    huge_comment = "https://cdn.asurascans.com/asura-images/comments/1-huge.webp"
    assert filter_asura_chapter_assets([huge_comment, reader]) == [reader]


def test_asura_static_fallback_ignores_non_reader_img_even_on_chapter_cdn(monkeypatch):
    from app.downloader.asura import AsuraScansStaticAdapter

    html = b"""
    <html><body>
      <img alt='Page 1 - Chapter 92 - Title' src='https://cdn.asurascans.com/asura-images/chapters/title/92/a.webp'>
      <img alt='Recommendation' src='https://cdn.asurascans.com/asura-images/chapters/other/12/rec.webp'>
      <img alt='Comment media' src='https://cdn.asurascans.com/asura-images/comments/x.webp'>
      <img alt='Page 2 - Chapter 92 - Title' data-src='https://cdn.asurascans.com/asura-images/chapters/title/92/b.webp'>
    </body></html>
    """

    class Response:
        def close(self):
            pass

    monkeypatch.setattr("app.downloader.asura.safe_get", lambda *a, **k: Response())
    monkeypatch.setattr("app.downloader.asura.read_response_limited", lambda *a, **k: html)
    urls = AsuraScansStaticAdapter().extract_image_urls(
        "https://asurascans.com/comics/title-b57aa235/chapter/92"
    )
    assert urls == [
        "https://cdn.asurascans.com/asura-images/chapters/title/92/a.webp",
        "https://cdn.asurascans.com/asura-images/chapters/title/92/b.webp",
    ]
