import asyncio

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app import config, manifest_utils
from app.ai_mode.seams import drop_duplicated_leading_text, drop_overlapping_letters
from app.detector.bubble_mask import clip_bubble_mask, derive_bubble_mask
from app.ocr.crop_geometry import active_overlap_signatures
from app.render.balloon_layout import plan_balloon_regions
from app.render.composition import compose_text
from app.render.font_catalog import resolve_font_id
from app.render.identity import render_input_signature
from app.render.text_fitting import DEFAULT_BODY_FONT_SIZE
from app.routers import export, image as image_routes, render_commit, translation
from app.text_objects import ensure_page_text_objects, sync_existing_auto_text_object
from app.translation.vision import VisionTranslationResult


def _region(bounds):
    return dict(zip(("x1", "y1", "x2", "y2"), bounds))


def _page():
    clean = Image.new("RGB", (520, 400), "white")
    ImageDraw.Draw(clean).ellipse((16, 46, 503, 356), outline="black", width=4)
    original = clean.copy()
    draw = ImageDraw.Draw(original)
    draw.text((105, 178), "WELL", fill="black", font_size=52)
    draw.text((277, 177), "DONE.", fill="black", font_size=52)
    raw, cleared = (cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in (original, clean))
    whole = derive_bubble_mask(cleared, (85, 148, 432, 260))
    boxes = [
        {"id": "box_a", **_region((85, 155, 260, 238)), "ocr_text": "WELL", "ocr_source": "manual",
         "bubble_mask": clip_bubble_mask(whole, (16, 46, 260, 356))},
        {"id": "box_b", **_region((260, 151, 440, 239)), "ocr_text": "DONE.", "ocr_source": "manual",
         "bubble_mask": clip_bubble_mask(whole, (260, 67, 503, 356))},
    ]
    return original, clean, raw, cleared, boxes


def _save_page(monkeypatch, tmp_path, original, clean, boxes):
    chapter = "abcd0622"
    raw_dir, processed, output = (tmp_path / p for p in ("raw", "processed", "output"))
    for module, name, value in (
        (manifest_utils, "PROCESSED_DIR", processed), (translation, "PROCESSED_DIR", processed),
        (config, "PROCESSED_DIR", processed), (translation, "RAW_DIR", raw_dir),
        (config, "RAW_DIR", raw_dir), (config, "OUTPUT_DIR", output),
        (render_commit, "OUTPUT_DIR", output), (export, "OUTPUT_DIR", output), (image_routes, "OUTPUT_DIR", output),
    ):
        monkeypatch.setattr(module, name, value)
    (raw_dir / chapter).mkdir(parents=True)
    (processed / chapter).mkdir(parents=True)
    raw_path, clean_path = raw_dir / chapter / "0.png", processed / chapter / "0.png"
    original.save(raw_path)
    clean.save(clean_path)
    manifest_utils.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [{
        "original": str(raw_path), "clean": str(clean_path), "boxes": boxes,
    }]})
    return chapter


def test_vision_translates_one_verified_balloon_and_render_export_share_it(monkeypatch, tmp_path):
    original, clean, _, _, boxes = _page()
    chapter = _save_page(monkeypatch, tmp_path, original, clean, boxes)
    calls = []

    def fake_translate(self, raw, clean, items, **kwargs):
        calls.append(items)
        assert len(items) == 1
        assert items[0]["text"] == "WELL\nDONE."
        assert items[0]["balloon_sources"] == ["box_a", "box_b"]
        w = lambda rect: rect[2] - rect[0]
        assert w(items[0]["fit_region"]) >= w(items[0]["region"]) - 15
        return VisionTranslationResult({"text_a": "Vậy. Xong."}, "test", {}, None,
                                       roles={"text_a": "dialogue"}, containers={"text_a": "bubble"})

    monkeypatch.setattr(translation, "get_provider_api_key", lambda *args, **kwargs: "test")
    monkeypatch.setattr(translation.VisionPageTranslator, "translate_page", fake_translate)
    result = asyncio.run(translation.translate_page_in_context(translation.TranslateVisionPageRequest(
        chapter_id=chapter, page_index=0, provider="openai", source_lang="en",
    )))
    assert len(calls) == 1 and result["translation_run"]["translated"] == 1
    assert result["translation_run"]["rendered_pages"] == [0]
    page = manifest_utils.load_manifest_raw(chapter)["pages"][0]
    obj, = page["text_objects"]
    assert obj["translation"] == "Vậy. Xong." and obj["verified_balloon"]
    assert obj["region"] == obj["auto_geometry"]
    assert ensure_page_text_objects(page) == (0, False)
    summary = export.render_chapter(chapter, force=True)["chapter_render"]
    assert summary["reused"] == 1 and summary["layout_conflicts"] == 0


def test_two_connected_lobes_create_two_ownership_regions():
    clean = Image.new("RGB", (710, 530), "black")
    draw = ImageDraw.Draw(clean)
    draw.ellipse((24, 40, 396, 325), fill="white")
    draw.ellipse((300, 181, 676, 465), fill="white")
    image = cv2.cvtColor(np.asarray(clean), cv2.COLOR_RGB2BGR)
    mask = derive_bubble_mask(image, (85, 112, 605, 410))
    boxes = [
        {"id": "box_a", **_region((85, 112, 318, 226)), "bubble_mask": mask},
        {"id": "box_b", **_region((384, 244, 605, 410)), "bubble_mask": mask},
    ]
    plans = plan_balloon_regions(image, image, boxes)
    assert [p["source_order"] for p in plans] == [["box_a"], ["box_b"]]
    assert plans[0]["bubble_mask"]["bbox"][2] < 500
    assert plans[1]["bubble_mask"]["bbox"][0] > 220
    page = {"boxes": boxes, "balloons": plans}
    assert ensure_page_text_objects(page) == (2, True)
    for obj, oid in zip(page["text_objects"], ("box_a", "box_b")):
        assert [ref for ref, _ in active_overlap_signatures(page, obj["region"], tuple(obj["source_boxes"]))] == [oid]
    first, second = page["text_objects"]
    first.update(translation="Tôi hiểu rồi.", ocr_text="I get it.")
    candidates = translation._vision_candidates(page, force=True)
    feedback = VisionTranslationResult(
        {"text_a": "Tôi hiểu rồi.", "text_b": "Tôi hiểu rồi."}, "test", {}, None,
        sources={"text_a": "I get it.", "text_b": "I get it."},
    )
    assert translation._duplicate_translation_ids(candidates, feedback) == {}
    assert translation._committed_duplicate(page, candidates[1], "Tôi hiểu rồi.", "I get it.") is None
    second.update(translation="Tôi hiểu rồi. Chuẩn bị đi.", region=first["region"].copy())
    first["letter_bounds"] = second["letter_bounds"]
    assert drop_duplicated_leading_text({"pages": [page]}) == []
    second["translation"] = first["translation"]
    assert drop_overlapping_letters({"pages": [page]}) == []


def test_verified_group_keeps_full_geometry_and_ocr_when_one_fragment_updates():
    _, _, original, clean, boxes = _page()
    page = {"boxes": boxes, "balloons": plan_balloon_regions(original, clean, boxes)}
    ensure_page_text_objects(page)
    obj, = page["text_objects"]
    initial = obj["region"]
    boxes[1]["ocr_text"] = "PLAYED."
    assert sync_existing_auto_text_object(page, boxes[1])
    assert obj["ocr_text"] == "WELL\nPLAYED."
    assert obj["region"] == obj["auto_geometry"] == initial
    assert obj["source_boxes"] == ["box_a", "box_b"]
    boxes[1]["removed"] = True
    ensure_page_text_objects(page)
    assert obj["source_missing"]


def test_forced_page_translation_can_upgrade_unchanged_machine_fragments_only():
    _, _, original, clean, boxes = _page()
    page = {"boxes": boxes}
    ensure_page_text_objects(page)
    first, second = page["text_objects"]
    for obj, line in zip((first, second), ("Vậy.", "Xong.")):
        obj.update(translation=line, auto_translation=line, translation_input_text=obj["ocr_text"],
                   translation_source="test")
    page["balloons"] = plan_balloon_regions(original, clean, boxes)
    assert ensure_page_text_objects(page) == (0, False)
    second["translation"] = "Xong rồi."
    assert ensure_page_text_objects(page, regroup_machine=True) == (0, False)
    second["translation"] = second["auto_translation"]
    ensure_page_text_objects(page, regroup_machine=True)
    assert page["text_objects"] == [first] and first["translation"] == "Vậy. Xong."
    assert first["translation_input_text"] == "WELL\nDONE."


def test_overlong_balloon_remains_editable_without_restoring_original(monkeypatch, tmp_path):
    original, clean, raw, cleared, boxes = _page()
    chapter = _save_page(monkeypatch, tmp_path, original, clean, boxes)
    translation.prepare_page_balloons(chapter, 0)
    manifest = manifest_utils.load_manifest_raw(chapter)
    page = manifest["pages"][0]
    obj, = page["text_objects"]
    obj["translation"] = "W" * 300
    manifest_utils.save_manifest_raw(chapter, manifest)
    request = export._render_request_from_page(chapter, 0, page)
    result = render_commit.render_page(request)
    assert result["skipped_objects"] == [] and result["layout_conflicts"] == [obj["id"]]
    summary = export.render_chapter(chapter, force=True)["chapter_render"]
    assert summary["restored_regions"] == 0 and summary["layout_conflicts"] == 1
    stored = manifest_utils.load_manifest_raw(chapter)["pages"][0]
    assert stored["text_objects"][0]["translation"] == "W" * 300
    assert stored["render_layout_issue_types"] == {obj["id"]: "balloon_overset"}


def test_render_identity_includes_the_chapters_body_size_for_verified_dialogue(tmp_path):
    original, clean, raw, cleared, boxes = _page()
    path = tmp_path / "clean.png"
    clean.save(path)
    page = {"clean": str(path), "boxes": boxes, "balloons": plan_balloon_regions(raw, cleared, boxes)}
    ensure_page_text_objects(page)
    obj, = page["text_objects"]
    obj["translation"] = "Vậy. Xong."
    samples = [{"source_cap_px": cap, "plain_outline": True} for cap in (13, 19, 21, 28, 30)]
    manifest = {"pages": [page, {"balloons": samples}]}
    baseline = render_input_signature(manifest, 0)
    for sample in samples:
        sample["source_cap_px"] += 12
    assert render_input_signature(manifest, 0) != baseline


def test_a_short_phrase_stays_on_one_line_when_it_fits_at_a_reading_size():
    _, clean, original, cleared, boxes = _page()
    mask = plan_balloon_regions(original, cleared, boxes)[0]["bubble_mask"]
    fit = compose_text(clean, "Một tiếng nữa!", tuple(mask["bbox"]), str(resolve_font_id("dialogue.mac-dinh-3")),
                       preferred=92, floor=12, stroke=2, bold=False, mask=mask, verified_balloon=True)
    assert fit is not None and fit.size >= DEFAULT_BODY_FONT_SIZE
    assert [line for _, _, line in fit.lines] == ["Một tiếng nữa!"]
