from __future__ import annotations

import json
import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, FeatureNotFound, NavigableString, Tag

SKIP = ("script", "style", "noscript", "svg", "form", "iframe", "template", "button")
CHROME = ("nav", "footer", "aside")
JS_GATED = (
    "enable javascript",
    "javascript required",
    "turn on javascript",
    "please enable javascript",
    "browser not supported",
)
RAW_GITHUB = "https://raw.githubusercontent.com"
GITHUB_PAGES = {
    "trending", "topics", "collections", "explore", "search", "marketplace", "orgs", "users",
    "settings", "sponsors", "features", "about", "login", "notifications", "issues", "pulls",
    "apps", "enterprise", "pricing", "readme", "events", "codespaces", "new",
}
_MD_LINK_URL_RE = re.compile(r"\]\([^)]*\)")
_HEADING_RE = re.compile(r"h[1-6]")
_MAX_WALK_DEPTH = 200
_MIN_MAIN_TEXT_CHARS = 200


def rewrite(url: str) -> str:
    parts = urlparse(url)
    if parts.scheme.lower() not in ("http", "https"):
        return url
    if parts.netloc.lower() not in ("github.com", "www.github.com"):
        return url
    path = [p for p in parts.path.split("/") if p]
    if not path or path[0].lower() in GITHUB_PAGES or parts.query:
        return url
    if len(path) >= 5 and path[2] == "blob":
        return f"{RAW_GITHUB}/{path[0]}/{path[1]}/{'/'.join(path[3:])}"
    if len(path) == 2:
        return f"{RAW_GITHUB}/{path[0]}/{path[1]}/HEAD/README.md"
    return url


def _safe_href(raw_href: object, base: str) -> str:
    if isinstance(raw_href, list):
        raw_href = raw_href[0] if raw_href else ""
    href = str(raw_href or "").strip()
    if not href or href.startswith("#"):
        return ""
    lowered = href.lower()
    if lowered.startswith(("javascript:", "data:", "vbscript:")):
        return ""
    try:
        resolved = urljoin(base, href) if base else href
    except ValueError:
        return ""
    scheme = urlparse(resolved).scheme.lower()
    if scheme and scheme not in ("http", "https"):
        return ""
    return resolved


def to_markdown(html: str, base: str = "") -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(SKIP):
        tag.decompose()
    root = None
    for candidate in (soup.find("main"), soup.find("article")):
        if candidate is not None and len(candidate.get_text(" ", strip=True)) >= _MIN_MAIN_TEXT_CHARS:
            root = candidate
            break
    if root is None:
        root = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in root.find_all(CHROME):
        tag.decompose()
    lines: list[str] = []
    _walk(root, lines, base, depth=0)
    text = "\n".join(line.rstrip() for line in "".join(lines).split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _walk(node, out: list[str], base: str, depth: int = 0) -> None:
    if depth >= _MAX_WALK_DEPTH:
        out.append(node.get_text(" ", strip=True) if isinstance(node, Tag) else str(node))
        return
    for child in node.children:
        if isinstance(child, NavigableString):
            out.append(re.sub(r"\s+", " ", str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in SKIP:
            continue
        if _HEADING_RE.fullmatch(name):
            out.append(f"\n\n{'#' * int(name[1])} {child.get_text(' ', strip=True)}\n\n")
        elif name == "a" and child.get("href"):
            label = child.get_text(" ", strip=True)
            href = _safe_href(child.get("href"), base)
            out.append(f"[{label}]({href})" if label and href else label)
        elif name == "pre":
            out.append(f"\n\n```\n{child.get_text()}\n```\n\n")
        elif name == "code":
            out.append(f"`{child.get_text()}`")
        elif name == "li":
            out.append("\n- ")
            _walk(child, out, base, depth + 1)
        elif name == "tr":
            out.append("\n" + " | ".join(cell.get_text(" ", strip=True) for cell in child.find_all(["th", "td"])))
        elif name in ("table", "ul", "ol", "p", "div", "section", "blockquote", "figure", "dl"):
            out.append("\n\n" if name in ("p", "blockquote", "table") else "\n")
            _walk(child, out, base, depth + 1)
            out.append("\n")
        elif name == "br":
            out.append("\n")
        else:
            _walk(child, out, base, depth + 1)


def low_quality(text: str) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    if len(stripped) < 1024 and any(word in lower for word in JS_GATED):
        return True
    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    if len(lines) <= 10:
        return False
    nav_lines = sum(
        "](" in line and len(_MD_LINK_URL_RE.sub("]", line)) < 60
        for line in lines
    )
    return nav_lines / len(lines) > 0.6


def readable(body: bytes, kind: str, encoding: str | None, base: str) -> tuple[str, list[str]]:
    try:
        text = body.decode(encoding or "utf-8", errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    notes: list[str] = []
    kind_lower = str(kind or "").lower()
    if "json" in kind_lower:
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False), notes
        except ValueError:
            return text, notes
    if "html" in kind_lower or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        text = to_markdown(text, base)
        if low_quality(text):
            notes.append(
                "The page looks like it needs JavaScript or is mostly menus; what is here may not be its content. "
                "Try another source, a raw or API URL, or search for the text elsewhere."
            )
    return text, notes


def find(text: str, pattern: str, context: int = 2, limit: int = 40) -> str:
    context = max(0, int(context))
    limit = max(1, int(limit))
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I)
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if rx.search(line)]
    if not hits:
        return f"No line matches {pattern!r}."
    shown, last = [], -1
    for i in hits[:limit]:
        start, end = max(i - context, 0, last + 1), min(i + context + 1, len(lines))
        if start >= end:
            continue
        if shown and start > last + 1:
            shown.append("…")
        shown += [f"{n + 1}: {lines[n]}" for n in range(start, end)]
        last = end - 1
    more = f"\n[{len(hits) - limit} more matching lines]" if len(hits) > limit else ""
    return "\n".join(shown) + more


def browsed(url: str, key: str) -> str:
    endpoint = os.environ.get("TINYFISH_FETCH_URL") or "https://api.fetch.tinyfish.ai"
    with requests.post(
        endpoint,
        json={"urls": [url], "format": "markdown"},
        timeout=(10, 60),
        allow_redirects=False,
        headers={"X-API-Key": key, "Content-Type": "application/json", "Accept": "application/json"},
    ) as reply:
        reply.raise_for_status()
        try:
            payload = reply.json()
        except ValueError:
            return ""
    if not isinstance(payload, dict):
        return ""
    rows = payload.get("results") or []
    return str(rows[0].get("text") or "").strip() if isinstance(rows, list) and rows and isinstance(rows[0], dict) else ""
