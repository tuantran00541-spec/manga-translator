"""Discover ordered chapter links from a series table-of-contents page."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup

from app.downloader.http import read_response_limited, safe_get
from app.logging_config import logger
from app.parameters import DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS, REMOTE_CONNECT_TIMEOUT_SECONDS
from app.security import MAX_REMOTE_DOCUMENT_BYTES, browser_request_allowed, validate_url

_CHAPTER_URL_RE = re.compile(
    r"(?:^|/)(?:chapter|chap|ch|chuong|episode|ep|c)[-_./]*(\d+(?:[.-]\d+)?)(?:/|$)|/(\d+(?:[.-]\d+)?)/?$",
    re.I,
)
_CHAPTER_TEXT_RE = re.compile(
    r"\b(?:chapter|chap|chuong|chương|episode|ep|ch)\s*[-.:#]?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)
_MEDIA_EXT_RE = re.compile(r"\.(?:jpg|jpeg|png|webp|gif|bmp|svg|css|js|pdf|zip|cbz)$", re.I)


def _normalize_host(host: str | None) -> str:
    return (host or "").strip().lower().removeprefix("www.")


def _chapter_sort_key(url: str, label: str, order: int) -> tuple[int, tuple[float, ...], int]:
    path = urlparse(url).path
    match = _CHAPTER_URL_RE.search(path) or _CHAPTER_TEXT_RE.search(label)
    if match:
        raw_num = next((g for g in match.groups() if g), None)
        if raw_num:
            parts = tuple(float(p) for p in re.split(r"[.-]", raw_num) if p.isdigit())
            if parts:
                return (0, parts, order)
    return (1, (float(order),), order)


def _series_parts(url: str) -> tuple[str, str]:
    parts = [p for p in urlparse(url).path.split("/") if p]
    while parts and _CHAPTER_URL_RE.search("/" + parts[-1]):
        parts.pop()
    if not parts:
        return ("", "")
    prefix = "/" + "/".join(parts).lower()
    slug = re.sub(r"-[0-9a-f]{6,}$", "", parts[-1].lower())
    return (prefix, slug)


def extract_chapter_links(html: str, base_url: str, *, limit: int = 2000) -> list[dict[str, str]]:
    """Extract deduplicated, reading-order chapter links from HTML."""
    soup = BeautifulSoup(html or "", "lxml")
    base_clean, _ = urldefrag(base_url)
    base_parsed = urlparse(base_clean)
    base_host = _normalize_host(base_parsed.hostname)
    series_pref, series_slug = _series_parts(base_clean)

    seen: set[str] = set()
    scoped: list[tuple[tuple[int, tuple[float, ...], int], str, str]] = []
    relaxed: list[tuple[tuple[int, tuple[float, ...], int], str, str]] = []

    for idx, anchor in enumerate(soup.find_all("a", href=True)):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        resolved, _ = urldefrag(urljoin(base_clean, href))
        parsed = urlparse(resolved)
        if parsed.scheme not in {"http", "https"}:
            continue
        if _normalize_host(parsed.hostname) != base_host:
            continue
        if resolved.rstrip("/") == base_clean.rstrip("/"):
            continue
        if _MEDIA_EXT_RE.search(parsed.path):
            continue

        label = " ".join(anchor.get_text(" ", strip=True).split())[:120]
        path_lower = parsed.path.lower()
        url_match = _CHAPTER_URL_RE.search(parsed.path)
        text_match = _CHAPTER_TEXT_RE.search(label)
        if not url_match and not text_match:
            continue

        norm_key = resolved.rstrip("/")
        if norm_key in seen:
            continue
        seen.add(norm_key)
        entry = (_chapter_sort_key(resolved, label, idx), resolved, label or resolved)
        matches_series = bool(
            (series_pref and (path_lower.startswith(series_pref + "/") or path_lower == series_pref))
            or (series_slug and f"/{series_slug}/" in path_lower)
        )
        if matches_series:
            scoped.append(entry)
        else:
            relaxed.append(entry)

    candidates = scoped if scoped else relaxed
    candidates.sort(key=lambda item: item[0])
    return [{"url": url, "label": label} for _, url, label in candidates[: max(1, int(limit))]]


def _fetch_html_playwright(toc_url: str) -> str:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
    from playwright_stealth import Stealth

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-zygote",
                "--disable-extensions",
                "--disable-background-networking",
            ]
        )
        try:
            context = browser.new_context(service_workers="block", viewport={"width": 1440, "height": 900})
            context.route(
                "**/*",
                lambda route: route.continue_() if browser_request_allowed(route.request.url) else route.abort(),
            )
            page = context.new_page()
            Stealth(chrome_runtime=True).apply_stealth_sync(page)
            try:
                page.goto(toc_url, wait_until="domcontentloaded", timeout=DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS)
                page.wait_for_timeout(1200)
            except PlaywrightTimeoutError:
                logger.warning("Timed out waiting for series TOC page: {}", toc_url)
            return page.content()
        finally:
            browser.close()


def discover_chapter_urls(toc_url: str, *, allow_js: bool = True, limit: int = 2000) -> list[dict[str, str]]:
    """Validate toc_url and return ordered chapter links from static HTTP or JS fallback."""
    validate_url(toc_url)
    html = ""
    try:
        response = safe_get(toc_url, timeout=REMOTE_CONNECT_TIMEOUT_SECONDS, stream=True)
        try:
            html = read_response_limited(response, limit_bytes=MAX_REMOTE_DOCUMENT_BYTES).decode("utf-8", errors="replace")
        finally:
            response.close()
    except Exception as exc:
        logger.warning("Static TOC fetch failed for {}: {}", toc_url, type(exc).__name__)
        if not allow_js:
            raise

    chapters = extract_chapter_links(html, toc_url, limit=limit) if html else []
    if not chapters and allow_js:
        try:
            js_html = _fetch_html_playwright(toc_url)
            chapters = extract_chapter_links(js_html, toc_url, limit=limit)
        except Exception as exc:
            logger.warning("Playwright TOC fetch failed for {}: {}", toc_url, type(exc).__name__)
    return chapters
