
from app.ai_mode.seams import _shared, join_stacked_lines, sync_seam_mirrors


def test_shared_overlap_is_share_of_smaller_box():
    assert _shared((0, 0, 10, 10), (5, 5, 15, 15)) == 0.25


def test_join_stacked_lines_merges_a_vertical_block():
    manifest = {
        "pages": [
            {
                "text_objects": [
                    {"id": "a", "region": {"x1": 10, "y1": 10, "x2": 50, "y2": 30}},
                    {"id": "b", "region": {"x1": 12, "y1": 32, "x2": 48, "y2": 50}},
                ]
            }
        ]
    }

    assert join_stacked_lines(manifest, [0]) == 1
    objects = manifest["pages"][0]["text_objects"]
    assert objects[0]["joined_lines"] == ["b"]
    assert objects[1]["joined_into"] == "a"


def test_sync_seam_mirrors_does_not_revive_dropped_text():
    manifest = {
        "pages": [
            {
                "skipped": False,
                "source_page": 0,
                "stitch_core": {
                    "source_y1": 0,
                    "core_source_y1": 0,
                    "core_source_y2": 50,
                },
                "text_objects": [
                    {
                        "id": "mirror",
                        "region": {"x1": 0, "y1": 40, "x2": 10, "y2": 60},
                        "translation": "",
                        "overlap_dropped": True,
                    }
                ],
            },
            {
                "skipped": False,
                "source_page": 0,
                "stitch_core": {
                    "source_y1": 50,
                    "core_source_y1": 50,
                    "core_source_y2": 100,
                },
                "text_objects": [
                    {
                        "id": "owner",
                        "region": {"x1": 0, "y1": -10, "x2": 10, "y2": 10},
                        "translation": "owner text",
                    }
                ],
            },
        ]
    }

    sync_seam_mirrors(manifest)
    assert manifest["pages"][0]["text_objects"][0]["translation"] == ""
    assert manifest["pages"][0]["text_objects"][0].get("overlap_dropped") is True
