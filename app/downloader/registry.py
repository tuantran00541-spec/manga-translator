from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from app.downloader.base import BaseAdapter
from app.downloader.generic_js import GenericJsAdapter


def _is_safe_image_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


class GenericStaticAdapter(BaseAdapter):
    img_selector = "img"

    def can_handle(self, url: str) -> bool:
        return True

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        resp = requests.get(chapter_url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        urls = []
        for img in soup.select(self.img_selector):
            src = img.get("data-src") or img.get("src")
            if src and _is_safe_image_url(src):
                urls.append(src)
        return urls


STATIC_ADAPTER = GenericStaticAdapter()
JS_ADAPTER = GenericJsAdapter()


def download_chapter(chapter_url: str, output_dir: Path) -> list[Path]:
    from app.security import validate_url
    validate_url(chapter_url)

    paths = STATIC_ADAPTER.download(chapter_url, output_dir)
    if paths:
        return paths
    return JS_ADAPTER.download(chapter_url, output_dir)
