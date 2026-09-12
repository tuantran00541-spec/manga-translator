from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

KEYS = {
    0: ["box_4d9b409ec3de4429", "box_ede6b33a00f94745", "box_b21c08e1d96d498a", "box_5bb018a7360241a7"],
    18: ["box_1e1457279d9b4ad7"],
    20: ["box_cc01706c6ea64712", "box_e9e84335b5d540c2"],
    21: ["box_f513c6697c844fd2", "box_968485f59cb14b5d"],
    30: ["box_15aed6043c954767"],
    31: ["box_355a8dc43c8b40ee", "box_21b9d9a748934a8b"],
    36: ["box_a5ec1161458544ab"],
    37: ["box_f7382b1a76f24c0c"],
    38: ["box_a3c90509c54642f3"],
    43: ["box_b6bd91a26f6f4533"],
    44: ["box_cdb593dcfed34114", "box_c14efb2c87fa442e"],
    49: ["box_d809d9a845424212", "box_39e8ec1471384216"],
    52: ["box_36300e8061d742e0", "box_771a3d7577834c28"],
    53: ["box_2a55ae1b428242ca"],
    55: ["box_42b33c7c024d4c6e"],
    56: ["box_d900bc50b4374bd9", "box_5043cba0cd1a4a5c"],
    57: ["box_cfb3d1372e1f4966", "box_8571f5814b764f0c"],
    60: ["box_a684dac5f29045eb"],
    61: ["box_22748b6c37534964"],
    63: ["box_9faef21adfdf4b20", "box_39c2a2f793a24eea", "box_43ec017503714dd9", "box_25b772a92acf4584", "box_4ecb09a9dbb24208", "box_26ff7288259847c2"],
    64: ["box_1a1f8c4c2daf4c77"],
}


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(s or "").upper()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "processed" / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((root / "ocr-review" / "ocr-results.json").read_text(encoding="utf-8"))
    skipped = json.loads((root / "ocr-review" / "ocr-skipped.json").read_text(encoding="utf-8"))
    objects = json.loads((root / "ocr-review" / "ocr-objects.json").read_text(encoding="utf-8"))

    result_map = {}
    for r in results:
        pi = r.get("page_index")
        bid = r.get("box_id") or r.get("id")
        if pi is not None and bid:
            result_map[(int(pi), str(bid))] = r

    selected = []
    for pi, bids in KEYS.items():
        page = manifest["pages"][pi]
        box_map = {str(b.get("id")): b for b in page.get("boxes", [])}
        for bid in bids:
            b = box_map.get(bid)
            selected.append({
                "page_index": pi,
                "source_page": page.get("source_page"),
                "slice_index": page.get("slice_index"),
                "box_id": bid,
                "box": b,
                "ocr_result": result_map.get((pi, bid)),
            })

    dup_groups = []
    by_page_text = defaultdict(list)
    for r in objects:
        txt = norm(r.get("ocr_text"))
        if txt:
            by_page_text[(int(r["page_index"]), txt)].append(r)
    for (pi, txt), rows in sorted(by_page_text.items()):
        if len(rows) > 1:
            dup_groups.append({
                "page_index": pi,
                "normalized_text": txt,
                "rows": [{
                    "box_id": r.get("box_id"),
                    "region": r.get("region"),
                    "semantic_type": r.get("semantic_type"),
                    "ocr_text": r.get("ocr_text"),
                } for r in rows],
            })

    payload = {
        "chapter_id": manifest.get("chapter_id"),
        "pages": len(manifest.get("pages", [])),
        "manifest_text_objects": sum(len(p.get("text_objects") or []) for p in manifest.get("pages", [])),
        "ocr_results": len(results),
        "ocr_objects": len(objects),
        "skipped": len(skipped),
        "skip_reason_counts": dict(Counter(str(r.get("skip_reason")) for r in skipped)),
        "non_deferred_skipped": sum(1 for r in skipped if r.get("skip_reason") != "deferred-review-region"),
        "selected": selected,
        "duplicate_groups": dup_groups,
    }
    out = root / "ocr-curation-inventory.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
