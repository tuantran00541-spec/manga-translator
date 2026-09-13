from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("checkpoint/04-translation/translated-manifest.json")
TARGET = "box_4f2f668087d845f2"
# The detector/OCR bbox only enclosed the original short glyph line (34 px high),
# not the actual black circular speech balloon. Human visual review of source page
# 010 slice 00 establishes this safe inner balloon region.
REVIEWED_REGION = {"x1": 430, "y1": 1390, "x2": 720, "y2": 1660}

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
found = 0
for page in manifest.get("pages") or []:
    for obj in page.get("text_objects") or []:
        refs = [v for v in (obj.get("source_boxes") or []) if isinstance(v, str)]
        if TARGET not in refs:
            continue
        if int(page.get("source_page") or -1) != 10 or int(page.get("slice_index") or -1) != 0:
            raise SystemExit(f"unexpected location for {TARGET}")
        before = dict(obj.get("region") or {})
        if before != {"x1": 442, "y1": 1510, "x2": 708, "y2": 1544}:
            raise SystemExit(f"geometry drift for {TARGET}: {before}")
        obj["typography_region_before_review"] = before
        obj["region"] = dict(REVIEWED_REGION)
        obj["typography_geometry_reviewed"] = True
        found += 1

if found != 1:
    raise SystemExit(f"expected one target object, found {found}")
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"box_id": TARGET, "region": REVIEWED_REGION, "status": "PASS"}, ensure_ascii=False))
