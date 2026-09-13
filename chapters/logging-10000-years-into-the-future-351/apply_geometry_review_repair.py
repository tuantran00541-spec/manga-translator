from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("checkpoint/04-translation/translated-manifest.json")
OUT = Path("checkpoint/05-render")

# Human visual-review repair: these text blocks were technically centered but
# occupied almost the full detector-owned source-text box. Symmetric horizontal
# insets preserve the original center while restoring optical breathing room.
REPAIRS = {
    "box_a4b9b09505e94389": {"expected": [110, 171, 857, 396], "inset_x": 24},
    "box_4e3af9963bf444db": {"expected": [185, 455, 820, 660], "inset_x": 18},
    "box_f7b1da74ae064805": {"expected": [137, 1675, 733, 1831], "inset_x": 24},
    "box_9585976688394dad": {"expected": [140, 2880, 759, 3241], "inset_x": 22},
    "box_b7a9cef01ad041b6": {"expected": [136, 1526, 841, 1801], "inset_x": 22},
    "box_d421009df70342bc": {"expected": [36, 2007, 776, 2279], "inset_x": 22},
    "box_ffc320d1ace04ad4": {"expected": [190, 1768, 844, 1954], "inset_x": 18},
    "box_3ddab9ac78fe4aec": {"expected": [424, 767, 879, 983], "inset_x": 18},
    "box_7133ff406e6d44a8": {"expected": [214, 74, 820, 397], "inset_x": 20},
    "box_d6a49729a0bc45e8": {"expected": [44, 496, 561, 715], "inset_x": 24},
    "box_adcf708670964b29": {"expected": [411, 144, 835, 378], "inset_x": 18},
    "box_a99699fc3278422f": {"expected": [376, 908, 900, 1064], "inset_x": 18},
    "box_f4cbd538a9694c6f": {"expected": [90, 208, 731, 455], "inset_x": 18},
}


def _center(region: dict[str, int]) -> tuple[float, float]:
    return ((region["x1"] + region["x2"]) / 2.0, (region["y1"] + region["y2"]) / 2.0)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_box: dict[str, dict] = {}
    for page in manifest.get("pages") or []:
        for obj in page.get("text_objects") or []:
            if not isinstance(obj, dict) or obj.get("source_missing"):
                continue
            refs = [str(v) for v in (obj.get("source_boxes") or []) if isinstance(v, str)]
            if len(refs) == 1:
                by_box[refs[0]] = obj

    applied = []
    for box_id, spec in REPAIRS.items():
        obj = by_box.get(box_id)
        if obj is None:
            raise SystemExit(f"geometry repair target missing: {box_id}")
        region = dict(obj.get("region") or {})
        current = [int(region[k]) for k in ("x1", "y1", "x2", "y2")]
        expected = [int(v) for v in spec["expected"]]
        if current != expected:
            raise SystemExit(f"geometry source-region drift for {box_id}: {current} != {expected}")

        inset = int(spec["inset_x"])
        after = {
            "x1": current[0] + inset,
            "y1": current[1],
            "x2": current[2] - inset,
            "y2": current[3],
        }
        if after["x2"] <= after["x1"]:
            raise SystemExit(f"geometry repair collapsed region for {box_id}: {after}")

        before = {"x1": current[0], "y1": current[1], "x2": current[2], "y2": current[3]}
        if _center(before) != _center(after):
            raise SystemExit(f"geometry repair shifted center for {box_id}: {before} -> {after}")

        obj["region_before_optical_breathing_review"] = before
        obj["region"] = after
        obj["optical_breathing_reviewed"] = True
        obj["optical_breathing_inset_x"] = inset
        applied.append({
            "box_id": box_id,
            "before": before,
            "after": after,
            "inset_x": inset,
            "center_preserved": True,
            "reason": "Human visual review: localized text was too close to the horizontal edges while technically fitting.",
        })

    if len(applied) != len(REPAIRS):
        raise SystemExit(f"geometry repair count mismatch: {len(applied)}/{len(REPAIRS)}")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": "05c-optical-bubble-geometry-repair",
        "chapter_id": "c3513510",
        "status": "PASS",
        "repair_count": len(applied),
        "policy": "symmetric_horizontal_inset_preserve_center_then_refit_fixed_typography",
        "repairs": applied,
        "next_action": "RERENDER_AND_HUMAN_VISUAL_REVIEW",
    }
    (OUT / "geometry-review-repair-applied.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
