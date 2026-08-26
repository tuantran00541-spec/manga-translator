from __future__ import annotations

from urllib.parse import urljoin, urlparse


def best_srcset_candidate(srcset: str | None) -> str | None:
    if not srcset or not isinstance(srcset, str):
        return None
    best_url = None
    best_score = -1.0
    for index, part in enumerate(srcset.split(",")):
        item = part.strip()
        if not item:
            continue
        pieces = item.split()
        url = pieces[0].strip()
        if not url:
            continue
        score = float(index)
        if len(pieces) > 1:
            descriptor = pieces[-1].lower()
            try:
                if descriptor.endswith("w"):
                    score = float(descriptor[:-1])
                elif descriptor.endswith("x"):
                    score = float(descriptor[:-1]) * 10_000.0
            except ValueError:
                pass
        if score >= best_score:
            best_score = score
            best_url = url
    return best_url


def resolve_image_candidate(base_url: str, *candidates: str | None) -> str | None:
    for raw in candidates:
        if not raw or not isinstance(raw, str):
            continue
        value = raw.strip()
        if not value or value.startswith("data:") or value.startswith("blob:"):
            continue
        absolute = urljoin(base_url, value)
        try:
            parsed = urlparse(absolute)
        except ValueError:
            continue
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return absolute
    return None
