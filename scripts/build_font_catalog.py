from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTFont

# A font counts as Vietnamese only if it has all of these letters.
VIETNAMESE_LETTERS = (
    "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
)
VIETNAMESE_LETTERS += VIETNAMESE_LETTERS.upper()


def covers_vietnamese(path: Path) -> bool:
    font = TTFont(path, lazy=True)
    try:
        cmap = font.getBestCmap() or {}
    finally:
        font.close()
    return all(ord(char) in cmap for char in VIETNAMESE_LETTERS)


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
            covers = covers_vietnamese(path)
        except Exception as exc:  # pragma: no cover
            errors.append(f"invalid font {path}: {exc}")
        else:
            if bool(record.get("vietnamese")) != covers:
                errors.append(f"{font_id}: vietnamese is {record.get('vietnamese')} but the font "
                              f"{'maps' if covers else 'lacks'} the Vietnamese letters")
        license_path = root / record["license_file"]
        if not license_path.is_file():
            errors.append(f"missing license file: {license_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-root", type=Path, default=Path("app/static/fonts"))
    parser.add_argument("--fix-vietnamese", action="store_true",
                        help="rewrite each record's vietnamese flag from the font's own character map")
    args = parser.parse_args()
    catalog_path = args.font_root / "font_catalog.json"
    if args.fix_vietnamese:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            record["vietnamese"] = covers_vietnamese(args.font_root / record["path"])
        catalog_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = validate(args.font_root, catalog_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated {len(json.loads(catalog_path.read_text())['records'])} fonts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
