from pathlib import Path

from bs4 import BeautifulSoup

from app.downloader.base import BaseAdapter
from app.downloader.asura import (
    AsuraScansJsAdapter,
    AsuraScansStaticAdapter,
    is_asura_chapter_page,
)
from app.downloader.generic_js import GenericJsAdapter
from app.downloader.http import read_response_limited, safe_get
from app.downloader.image_urls import best_srcset_candidate, resolve_image_candidate
from app.security import MAX_REMOTE_DOCUMENT_BYTES


class GenericStaticAdapter(BaseAdapter):
    img_selector = "img"
    min_declared_width = 240

    def can_handle(self, url: str) -> bool:
        return True

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        response = safe_get(chapter_url, headers=self.headers, timeout=30, stream=True)
        try:
            body = read_response_limited(response, limit_bytes=MAX_REMOTE_DOCUMENT_BYTES)
        finally:
            response.close()
        soup = BeautifulSoup(body, "lxml")
        urls = []
        for img in soup.select(self.img_selector):
            declared_width = img.get("width")
            if declared_width:
                try:
                    if int(str(declared_width).replace("px", "").strip()) < self.min_declared_width:
                        continue
                except ValueError:
                    pass
            srcset = best_srcset_candidate(img.get("data-srcset") or img.get("srcset"))
            url = resolve_image_candidate(
                chapter_url,
                srcset,
                img.get("data-src"),
                img.get("data-original"),
                img.get("data-lazy"),
                img.get("src"),
            )
            if url:
                urls.append(url)
        return self._dedupe(urls)


STATIC_ADAPTER = GenericStaticAdapter()
JS_ADAPTER = GenericJsAdapter()
ASURA_STATIC_ADAPTER = AsuraScansStaticAdapter()
ASURA_JS_ADAPTER = AsuraScansJsAdapter()


def _choose_image_urls(static_urls: list[str], js_urls: list[str]) -> list[str]:
    static_urls = STATIC_ADAPTER._dedupe(static_urls)
    js_urls = STATIC_ADAPTER._dedupe(js_urls)
    if len(js_urls) >= 2:
        return js_urls
    if static_urls:
        return static_urls
    return js_urls


def download_chapter(chapter_url: str, output_dir: Path) -> list[Path]:
    from app.security import validate_url

    validate_url(chapter_url)
    static_urls: list[str] = []
    js_urls: list[str] = []

    if is_asura_chapter_page(chapter_url):
        # Asura comments/profile UI contains many large intrinsic images.  Both
        # discovery paths are therefore scoped to reader page elements; never
        # fall back to the generic all-<img> adapter for a recognized Asura
        # chapter because that would silently reintroduce website assets.
        try:
            static_urls = ASURA_STATIC_ADAPTER.extract_image_urls(chapter_url)
        except Exception:
            static_urls = []
        try:
            js_urls = ASURA_JS_ADAPTER.extract_image_urls(chapter_url)
        except Exception:
            js_urls = []
        selected = js_urls or static_urls
    else:
        try:
            static_urls = STATIC_ADAPTER.extract_image_urls(chapter_url)
        except Exception:
            static_urls = []
        try:
            js_urls = JS_ADAPTER.extract_image_urls(chapter_url)
        except Exception:
            js_urls = []
        selected = _choose_image_urls(static_urls, js_urls)
    if not selected:
        raise ValueError("Không tìm thấy ảnh chương hợp lệ từ URL này")

    return STATIC_ADAPTER.download_urls(selected, output_dir, referer=chapter_url)
