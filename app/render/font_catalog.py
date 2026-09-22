"""Catalog and safe path resolution for bundled comic fonts.

The catalog is the source of truth for both the API and the renderer.  Font
identifiers are stable slugs; callers never need to send filesystem paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DEFAULT_FONT


class FontNotFoundError(ValueError):
    """Raised when a user or AI requests an unknown font identifier."""


@dataclass(frozen=True)
class FontRecord:
    id: str
    name: str
    path: Path
    category: str
    tags: tuple[str, ...]
    vietnamese: bool
    license: str
    license_file: str
    source_url: str
    default_rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path.as_posix(),
            "category": self.category,
            "tags": list(self.tags),
            "vietnamese": self.vietnamese,
            "license": self.license,
            "license_file": self.license_file,
            "source_url": self.source_url,
            "default_rank": self.default_rank,
        }


@dataclass(frozen=True)
class FontCatalog:
    root: Path
    records: tuple[FontRecord, ...]
    aliases: dict[str, Path]

    def resolve(self, font_id: str | None) -> Path:
        requested = str(font_id or "default").strip()
        if not requested or requested.lower() == "default":
            return self.root / "default.ttf"

        record = next((item for item in self.records if item.id == requested), None)
        if record is not None:
            return record.path

        alias_key = requested.lower()
        if alias_key in self.aliases:
            return self.aliases[alias_key]
        raise FontNotFoundError(f"Unknown font id: {requested}")

    def records_as_dicts(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records]


def _catalog_path() -> Path:
    return DEFAULT_FONT.parent / "font_catalog.json"


def _safe_font_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"Font catalog path escapes font root: {relative}")
    return candidate


@lru_cache(maxsize=1)
def load_font_catalog() -> FontCatalog:
    root = DEFAULT_FONT.parent
    catalog_path = _catalog_path()
    records: list[FontRecord] = []
    if catalog_path.is_file():
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            path = _safe_font_path(root, str(item["path"]))
            records.append(
                FontRecord(
                    id=str(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    path=path,
                    category=str(item.get("category") or "other"),
                    tags=tuple(str(tag) for tag in item.get("tags", [])),
                    vietnamese=bool(item.get("vietnamese", False)),
                    license=str(item.get("license") or "unknown"),
                    license_file=str(item.get("license_file") or ""),
                    source_url=str(item.get("source_url") or ""),
                    default_rank=int(item.get("default_rank", len(records))),
                )
            )

    aliases: dict[str, Path] = {}
    for path in root.glob("*.[tT][tT][fF]"):
        aliases.setdefault(path.stem.lower(), path.resolve())
        aliases.setdefault(path.name.lower(), path.resolve())
    aliases["default"] = DEFAULT_FONT.resolve()

    return FontCatalog(
        root=root.resolve(),
        records=tuple(sorted(records, key=lambda record: record.default_rank)),
        aliases=aliases,
    )


def clear_font_catalog_cache() -> None:
    load_font_catalog.cache_clear()


def resolve_font_id(font_id: str | None) -> Path:
    return load_font_catalog().resolve(font_id)


def list_font_records() -> list[dict[str, Any]]:
    return load_font_catalog().records_as_dicts()
