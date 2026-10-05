"""Web search for the agent: Tavily or Brave when a key is set in the environment, else DuckDuckGo's plain HTML page."""
from __future__ import annotations

import os
from urllib.parse import parse_qs, quote_plus, urlparse

from bs4 import BeautifulSoup
import requests

from app.downloader.http import read_response_limited, safe_get

TIMEOUT = (10, 30)
UA = "Mozilla/5.0 manga-translator-agent"


class SearchError(Exception):
    pass


def _rows(results: list[tuple[str, str, str]]) -> str:
    if not results:
        return "No results"
    return "\n\n".join(f"{i}. {title}\n{url}\n{snippet}".rstrip() for i, (title, url, snippet) in enumerate(results, 1))


def _tavily(query: str, count: int, key: str) -> list[tuple[str, str, str]]:
    reply = requests.post("https://api.tavily.com/search", json={"query": query, "max_results": count}, timeout=TIMEOUT,
                          headers={"Authorization": f"Bearer {key}"})
    reply.raise_for_status()
    return [(r.get("title", ""), r.get("url", ""), (r.get("content") or "")[:400]) for r in reply.json().get("results", [])]


def _brave(query: str, count: int, key: str) -> list[tuple[str, str, str]]:
    reply = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": count}, timeout=TIMEOUT,
                         headers={"X-Subscription-Token": key, "Accept": "application/json"})
    reply.raise_for_status()
    return [(r.get("title", ""), r.get("url", ""), BeautifulSoup(r.get("description", ""), "lxml").get_text()[:400])
            for r in reply.json().get("web", {}).get("results", [])]


def parse_duckduckgo(html: str, count: int) -> list[tuple[str, str, str]]:
    out = []
    for block in BeautifulSoup(html, "lxml").select("div.result"):
        link = block.select_one("a.result__a")
        if link is None:
            continue
        href = link.get("href", "")
        target = parse_qs(urlparse(href).query).get("uddg", [href])[0]
        snippet = block.select_one(".result__snippet")
        out.append((link.get_text(" ", strip=True), target, snippet.get_text(" ", strip=True) if snippet else ""))
        if len(out) >= count:
            break
    return out


def _duckduckgo(query: str, count: int) -> list[tuple[str, str, str]]:
    response = safe_get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", timeout=TIMEOUT, headers={"User-Agent": UA})
    try:
        html = read_response_limited(response, limit_bytes=2_000_000).decode(response.encoding or "utf-8", errors="replace")
    finally:
        response.close()
    found = parse_duckduckgo(html, count)
    if not found and ("anomaly" in html.lower() or "captcha" in html.lower()):
        raise SearchError("DuckDuckGo refused this machine; set TAVILY_API_KEY or BRAVE_API_KEY for a search service")
    return found


def search(query: str, count: int = 8) -> str:
    query = " ".join(str(query).split())[:300]
    if not query:
        raise SearchError("query is empty")
    count = max(1, min(15, int(count)))
    try:
        if os.environ.get("TAVILY_API_KEY"):
            return _rows(_tavily(query, count, os.environ["TAVILY_API_KEY"]))
        brave = os.environ.get("BRAVE_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY")
        if brave:
            return _rows(_brave(query, count, brave))
        return _rows(_duckduckgo(query, count))
    except requests.RequestException as exc:
        raise SearchError(f"search failed: {type(exc).__name__}") from exc
