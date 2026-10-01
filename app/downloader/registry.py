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
from app.parameters import (
    DOWNLOAD_JS_MIN_IMAGE_WIDTH,
    DOWNLOAD_STATIC_MIN_DECLARED_WIDTH,
    REMOTE_CONNECT_TIMEOUT_SECONDS,
)
from app.security import MAX_REMOTE_DOCUMENT_BYTES


class GenericStaticAdapter(BaseAdapter):
    img_selector = "img"
    min_declared_width = DOWNLOAD_STATIC_MIN_DECLARED_WIDTH

    def can_handle(self, url: str) -> bool:
        return True

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        response = safe_get(
            chapter_url,
            headers=self.headers,
            timeout=REMOTE_CONNECT_TIMEOUT_SECONDS,
            stream=True,
        )
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
            # Vector icons (menus, flags, stars) are never comic pages.
            if url and not url.lower().split("?")[0].endswith(".svg"):
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
    failures: list[str] = []

    def attempt(label: str, adapter) -> list[str]:
        try:
            return adapter.extract_image_urls(chapter_url)
        except Exception as exc:  # one way failing still leaves the other
            failures.append(f"{label}: {getattr(exc, 'detail', None) or exc or type(exc).__name__}")
            return []

    if is_asura_chapter_page(chapter_url):
        static_urls = attempt("trang tĩnh", ASURA_STATIC_ADAPTER)
        if len(static_urls) >= 2:
            selected = static_urls
        else:
            js_urls = attempt("trình duyệt", ASURA_JS_ADAPTER)
            selected = js_urls or static_urls
    else:
        static_urls = attempt("trang tĩnh", STATIC_ADAPTER)
        js_urls = attempt("trình duyệt", JS_ADAPTER)
        selected = _choose_image_urls(static_urls, js_urls)
    if not selected:
        reason = "; ".join(str(item)[:200] for item in failures)
        raise ValueError("Không tìm thấy ảnh chương hợp lệ từ URL này" + (f" ({reason})" if reason else ""))

    paths = STATIC_ADAPTER.download_urls(selected, output_dir, referer=chapter_url)
    # Static HTML gives no rendered sizes, so thumbnails and logos are dropped by the browser path's width bar.
    pages = [path for path in paths if _image_width(path) >= DOWNLOAD_JS_MIN_IMAGE_WIDTH]
    for path in set(paths) - set(pages):
        path.unlink(missing_ok=True)
    return pages or paths


def _image_width(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        return image.width
