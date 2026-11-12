from app.region_policy import seam_preserve_plan


def _page(objects, preserve=(), skipped=False):
    return {
        "skipped": skipped,
        "process_required": False,
        "clean": "clean.png",
        "preserve_regions": list(preserve),
        "text_objects": objects,
    }


def _obj(oid, region, seam_owner=None):
    obj = {"id": oid, "region": dict(region)}
    if seam_owner is not None:
        obj["seam_owner"] = seam_owner
    return obj


OWNER_REGION = {"x1": 88, "y1": 792, "x2": 523, "y2": 960}
MIRROR_REGION = {"x1": 88, "y1": 2275, "x2": 521, "y2": 2443}


def _manifest():
    owner = _obj("text_owner", OWNER_REGION)
    mirror = _obj("text_mirror", MIRROR_REGION, {"page": 1, "id": "text_owner"})
    return {"pages": [_page([mirror]), _page([owner])]}


def test_preserving_owner_also_preserves_the_mirror_slice():
    plan = seam_preserve_plan(_manifest(), 1, [dict(OWNER_REGION)])
    assert plan[1] == [OWNER_REGION]
    assert plan[0] == [MIRROR_REGION]


def test_preserving_mirror_also_preserves_the_owner_slice():
    plan = seam_preserve_plan(_manifest(), 0, [dict(MIRROR_REGION)])
    assert plan[0] == [MIRROR_REGION]
    assert plan[1] == [OWNER_REGION]


def test_unlinked_region_stays_on_its_own_page():
    manifest = _manifest()
    lone = {"x1": 10, "y1": 10, "x2": 60, "y2": 60}
    manifest["pages"][1]["text_objects"].append(_obj("text_lone", lone))
    plan = seam_preserve_plan(manifest, 1, [dict(lone)])
    assert plan == {1: [lone]}


def test_region_already_preserved_on_mirror_is_not_repainted_again():
    manifest = _manifest()
    manifest["pages"][0]["preserve_regions"] = [dict(MIRROR_REGION)]
    plan = seam_preserve_plan(manifest, 1, [dict(OWNER_REGION)])
    assert plan == {1: [OWNER_REGION]}


def test_skipped_mirror_page_is_left_out():
    manifest = _manifest()
    manifest["pages"][0]["skipped"] = True
    plan = seam_preserve_plan(manifest, 1, [dict(OWNER_REGION)])
    assert plan == {1: [OWNER_REGION]}


def test_invalid_page_index_yields_empty_plan():
    assert seam_preserve_plan(_manifest(), 7, [dict(OWNER_REGION)]) == {}