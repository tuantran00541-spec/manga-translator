from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from app.downloader.base import BaseAdapter
from app.downloader.generic_js import GenericJsAdapter
from app.downloader.http import read_response_limited, safe_get
from app.downloader.image_urls import best_srcset_candidate, resolve_image_candidate
from app.security import MAX_REMOTE_DOCUMENT_BYTES


_ASURA_HOSTS = {"asurascans.com", "www.asurascans.com"}
_ASURA_CHAPTER_ASSET_PREFIX = "/asura-images/chapters/"
_PAGE_ALT_RE = re.compile(r"^\s*Page\s+(\d+)\b", re.IGNORECASE)


def is_asura_chapter_page(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return host in _ASURA_HOSTS and "/chapter/" in parsed.path.lower()


def is_asura_chapter_asset(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.path.lower().startswith(_ASURA_CHAPTER_ASSET_PREFIX)


def filter_asura_chapter_assets(urls: list[str]) -> list[str]:
    """Keep only Asura chapter-reader asset URLs, preserving input order."""
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not is_asura_chapter_asset(url) or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _ordered_reader_urls(candidates: list[tuple[int, str]]) -> list[str]:
    """Return the contiguous Page 1..N reader sequence.

    Page labels are DOM provenance, not a size heuristic.  Requiring the sequence
    to start at Page 1 also prevents a stray page-labelled image elsewhere on the
    document from being appended to the current chapter.
    """
    by_page: dict[int, str] = {}
    for page_number, url in candidates:
        if page_number < 1 or not is_asura_chapter_asset(url):
            continue
        by_page.setdefault(page_number, url)

    out: list[str] = []
    page_number = 1
    while page_number in by_page:
        out.append(by_page[page_number])
        page_number += 1
    return out


class AsuraScansStaticAdapter(BaseAdapter):
    """Static fallback scoped to Asura's explicit reader-page DOM semantics."""

    def can_handle(self, url: str) -> bool:
        return is_asura_chapter_page(url)

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        if not self.can_handle(chapter_url):
            return []
        response = safe_get(chapter_url, headers=self.headers, timeout=30, stream=True)
        try:
            body = read_response_limited(response, limit_bytes=MAX_REMOTE_DOCUMENT_BYTES)
        finally:
            response.close()

        soup = BeautifulSoup(body, "lxml")
        candidates: list[tuple[int, str]] = []
        for img in soup.select("img[alt]"):
            match = _PAGE_ALT_RE.match(str(img.get("alt") or ""))
            if not match:
                continue
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
                candidates.append((int(match.group(1)), url))
        return _ordered_reader_urls(candidates)


class AsuraScansJsAdapter(GenericJsAdapter):
    """Asura reader discovery scoped to page elements instead of all site images."""

    img_selector = "img[alt^='Page ']"
    min_width = 0

    def can_handle(self, url: str) -> bool:
        return is_asura_chapter_page(url)

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        if not self.can_handle(chapter_url):
            return []
        # GenericJsAdapter preserves DOM order.  The selector removes site UI,
        # avatars and comment media; asset provenance is a second independent
        # guard against unrelated page-labelled images.
        return filter_asura_chapter_assets(super().extract_image_urls(chapter_url))
