from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("checkpoint/04-translation/translated-manifest.json")
TARGET = "box_4f2f668087d845f2"
# The detector/OCR bbox only enclosed the original short glyph line (34 px high),
# not the actual black circular speech balloon. Human visual review of source page
# 010 slice 00 establishes this safe inner balloon region.
REVIEWED_REGION = {"x1": 430, "y1": 1390, "x2": 720, "y2": 1660}

# Visual review showed these objects were falling all the way to the neutral
# BeVietnamPro fallback only because the manga fonts do not contain U+2014/U+2026.
# Normalize punctuation without changing wording/meaning so skill/shout/dialogue
# stay in their intended semantic font family. Guard both before/after strings so
# any upstream translation drift fails closed instead of being silently patched.
TEXT_OVERRIDES = {
    "box_ebbb5475e55d4a32": ("……?!", "......?!"),
    "box_b6f5d5d07b01453a": ("VÕ Ý—\nNHẬT NGUYỆT\nBẠO VŨ", "VÕ Ý\nNHẬT NGUYỆT\nBẠO VŨ"),
    "box_a17f4b50bce94125": ("NHẬT NGUYỆT\nBẠO VŨ—\nLIÊN HOÀN\nTHẦN KỸ.", "NHẬT NGUYỆT\nBẠO VŨ\nLIÊN HOÀN\nTHẦN KỸ."),
    "box_82028db6e4d14c94": ("VÕ Ý—\nTHIÊN ĐỊA\nTRẢM TIÊN", "VÕ Ý\nTHIÊN ĐỊA\nTRẢM TIÊN"),
    "box_670cd228d5b649d2": ("VÕ Ý—\nNGŨ HÀNH\nTỊCH DIỆT", "VÕ Ý\nNGŨ HÀNH\nTỊCH DIỆT"),
    "box_c5f930806b2a48ce": ("RA NGOÀI VÕ ĐÀI—\nXỬ THUA...!", "RA NGOÀI VÕ ĐÀI,\nXỬ THUA...!"),
    "box_1bf8f8b8b4fd4bdf": ("ĐẠN CHỈ\nTHẦN THÔNG—\nBIẾN THỨC\nJIN CHEONHEE.", "ĐẠN CHỈ\nTHẦN THÔNG\nBIẾN THỨC\nJIN CHEONHEE."),
    "box_f0e75aca70d0444b": ("A—", "A-"),
}

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
geometry_found = 0
text_found: set[str] = set()
text_repairs = []
for page in manifest.get("pages") or []:
    for obj in page.get("text_objects") or []:
        refs = [v for v in (obj.get("source_boxes") or []) if isinstance(v, str)]
        if len(refs) != 1:
            continue
        box_id = refs[0]

        if box_id == TARGET:
            source_page = page.get("source_page")
            slice_index = page.get("slice_index")
            if source_page is None or slice_index is None or int(source_page) != 10 or int(slice_index) != 0:
                raise SystemExit(
                    f"unexpected location for {TARGET}: source_page={source_page!r} slice_index={slice_index!r}"
                )
            before = dict(obj.get("region") or {})
            if before != {"x1": 442, "y1": 1510, "x2": 708, "y2": 1544}:
                raise SystemExit(f"geometry drift for {TARGET}: {before}")
            obj["typography_region_before_review"] = before
            obj["region"] = dict(REVIEWED_REGION)
            obj["typography_geometry_reviewed"] = True
            geometry_found += 1

        if box_id in TEXT_OVERRIDES:
            if box_id in text_found:
                raise SystemExit(f"duplicate typography text target {box_id}")
            before_text, after_text = TEXT_OVERRIDES[box_id]
            actual = str(obj.get("translation") or "")
            if actual != before_text:
                raise SystemExit(
                    f"translation drift for {box_id}: expected {before_text!r}, got {actual!r}"
                )
            obj["typography_translation_before_review"] = actual
            obj["translation"] = after_text
            obj["typography_punctuation_normalized"] = True
            text_found.add(box_id)
            text_repairs.append({
                "box_id": box_id,
                "source_page": int(page.get("source_page") or 0),
                "slice_index": int(page.get("slice_index") or 0),
                "before": before_text,
                "after": after_text,
            })

if geometry_found != 1:
    raise SystemExit(f"expected one geometry target, found {geometry_found}")
missing = sorted(set(TEXT_OVERRIDES) - text_found)
if missing:
    raise SystemExit(f"missing typography text targets: {missing}")

MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
report = {
    "status": "PASS",
    "geometry_override": {"box_id": TARGET, "region": REVIEWED_REGION},
    "punctuation_normalization_count": len(text_repairs),
    "punctuation_normalizations": text_repairs,
}
Path("checkpoint/04-translation/typography-overrides-applied.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
