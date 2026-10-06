"""Web search for the agent: Tavily when a key is set in the environment, else DuckDuckGo's plain HTML page; DuckDuckGo answers when Tavily fails or finds nothing."""
from __future__ import annotations

import os
import re
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


RECENCY = {"day": ("day", "d"), "week": ("week", "w"), "month": ("month", "m"), "year": ("year", "y")}


def _tavily(query: str, count: int, key: str, recency: str = "") -> list[tuple[str, str, str]]:
    body = {"query": query, "max_results": count, **({"time_range": RECENCY[recency][0]} if recency else {})}
    reply = requests.post("https://api.tavily.com/search", json=body, timeout=TIMEOUT,
                          headers={"Authorization": f"Bearer {key}"})
    reply.raise_for_status()
    return [(r.get("title", ""), r.get("url", ""), (r.get("content") or "")[:400]) for r in reply.json().get("results", [])]


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


def _duckduckgo(query: str, count: int, recency: str = "") -> list[tuple[str, str, str]]:
    when = f"&df={RECENCY[recency][1]}" if recency else ""
    response = safe_get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}{when}", timeout=TIMEOUT, headers={"User-Agent": UA})
    try:
        html = read_response_limited(response, limit_bytes=2_000_000).decode(response.encoding or "utf-8", errors="replace")
    finally:
        response.close()
    found = parse_duckduckgo(html, count)
    if not found and ("anomaly" in html.lower() or "captcha" in html.lower()):
        raise SearchError("DuckDuckGo refused this machine; set TAVILY_API_KEY for a search service")
    return found


def constraints(query: str) -> dict:
    """The site:, -site: and filetype: words of a query; a search engine may ignore them, so the results are checked against them too (from omp)."""
    found = {"sites": [], "not_sites": [], "types": []}
    for sign, word, value in re.findall(r"(?<!\S)(-?)(site|filetype|ext):([^\s]+)", query, re.I):
        value = value.lower().strip("\"'")
        key = "types" if word.lower() != "site" else ("not_sites" if sign else "sites")
        found[key].append(value.lstrip(".") if key == "types" else re.sub(r"^https?://", "", value).rstrip("/"))
    return found


def _on_site(url: str, site: str) -> bool:
    """The host (or a subdomain of it) matches, and the path starts with the site's own path when it has one."""
    host, _, path = site.partition("/")
    parts = urlparse(url)
    name = parts.netloc.lower().removeprefix("www.")
    return (name == host or name.endswith("." + host)) and parts.path.lstrip("/").lower().startswith(path)


def filter_results(results: list[tuple[str, str, str]], wanted: dict) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Keep what matches each constraint; a constraint that would leave nothing is dropped and named, not applied."""
    dropped = []
    checks = (("sites", "site:", lambda url, values: any(_on_site(url, v) for v in values)),
              ("not_sites", "-site:", lambda url, values: not any(_on_site(url, v) for v in values)),
              ("types", "filetype:", lambda url, values: any(urlparse(url).path.lower().endswith("." + v) for v in values)))
    for key, label, match in checks:
        values = wanted.get(key) or []
        if not values:
            continue
        kept = [row for row in results if match(row[1], values)]
        if kept:
            results = kept
        else:
            dropped.append(f"{label}{','.join(values)}")
    return results, dropped


def search(query: str, count: int = 8, recency: str = "") -> str:
    query = " ".join(str(query).split())[:300]
    if not query:
        raise SearchError("query is empty")
    count = max(1, min(15, int(count)))
    extra = {"recency": recency} if recency in RECENCY else {}
    chain = []
    if os.environ.get("TAVILY_API_KEY"):
        chain.append(("Tavily", lambda: _tavily(query, count, os.environ["TAVILY_API_KEY"], **extra)))
    chain.append(("DuckDuckGo", lambda: _duckduckgo(query, count, **extra)))
    failures, answered = [], False
    for name, run in chain:
        try:
            found = run()
        except (requests.RequestException, SearchError) as exc:
            failures.append(f"{name}: {exc if isinstance(exc, SearchError) else type(exc).__name__}")
            continue
        wanted = constraints(query)
        found, dropped = filter_results(found, wanted) if found else (found, [])
        if not found:
            answered = True
            continue
        notes = "".join(f"Note: no result matched `{label}`; that constraint was not applied\n" for label in dropped)
        return notes + _rows(found)
    if answered:
        return "No results"
    raise SearchError("search failed: " + "; ".join(failures))
