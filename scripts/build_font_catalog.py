""" The download step is intentionally separate: assets are sourced from the upstream Google Fonts repository, while this script only validates paths, licenses, and stable identifiers in a checkout. """

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTFont


def validate(root: Path, catalog_path: Path) -> list[str]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    records = payload.get("records", [])
    if not 60 <= len(records) <= 80:
        errors.append(f"expected 60-80 records, got {len(records)}")
    seen: set[str] = set()
    for record in records:
        font_id = record.get("id")
        if not font_id or font_id in seen:
            errors.append(f"duplicate or empty id: {font_id!r}")
        seen.add(font_id)
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"missing font file: {path}")
            continue
        try:
            TTFont(path, lazy=True).close()
        except Exception as exc:  # pragma: no cover - diagnostic CLI path
            errors.append(f"invalid font {path}: {exc}")
        license_path = root / record["license_file"]
        if not license_path.is_file():
            errors.append(f"missing license file: {license_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-root", type=Path, default=Path("app/static/fonts"))
    args = parser.parse_args()
    catalog_path = args.font_root / "font_catalog.json"
    errors = validate(args.font_root, catalog_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated {len(json.loads(catalog_path.read_text())['records'])} fonts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
