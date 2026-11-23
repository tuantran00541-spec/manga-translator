from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from app.detector.boxes import BubbleBox
from app.image_io import read_image, write_image
from app.inpaint import api_inpaint
from app.inpaint.lama_inpainter import Inpainter
from app.pipeline import ChapterPipeline


def _encode_png_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_textured_panel(h: int = 240, w: int = 260) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), (185, 200, 215), dtype=np.uint8)
    for x in range(0, w, 12):
        cv2.line(img, (x, 0), (x + 35, h - 1), (35, 65, 160), 2)
    for y in range(0, h, 18):
        cv2.line(img, (0, y), (w - 1, y + 10), (140, 40, 50), 2)
    img = np.clip(img.astype(np.int16) + rng.integers(-18, 19, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[70:170, 75:185] = 255
    return img, mask


def test_inpaint_settings_defaults_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(api_inpaint, "INPAINT_SETTINGS_PATH", tmp_path / "inpaint_settings.json")
    defaults = api_inpaint.load_inpaint_settings()
    assert defaults["enabled"] is False
    assert defaults["hybrid_auto"] is False
    assert defaults["model"] == "openai/gpt-image-2.6-flare"
    assert defaults["quality"] == "low"
    assert defaults["pack_atlas"] is True
    assert defaults["max_api_calls_per_chapter"] == 3

    saved = api_inpaint.save_inpaint_settings({
        "enabled": True,
        "hybrid_auto": True,
        "provider_id": "custom-gw",
        "model": "openai/gpt-image-2.6-flare",
        "quality": "auto",  # must be coerced to "low" to prevent expensive auto billing
        "max_api_calls_per_chapter": 99,
        "pack_atlas": False,
    })
    assert saved["enabled"] is True
    assert saved["provider_id"] == "custom-gw"
    assert saved["quality"] == "low"
    assert saved["max_api_calls_per_chapter"] == 20  # clamped
    assert saved["pack_atlas"] is False
    assert api_inpaint.load_inpaint_settings() == saved


def test_complexity_score_separates_flat_bubble_from_sfx_art():
    flat = np.full((220, 220, 3), 252, dtype=np.uint8)
    flat_mask = np.zeros((220, 220), dtype=np.uint8)
    flat_mask[60:160, 60:160] = 255
    assert api_inpaint.region_complexity_score(flat, flat_mask) == 0.0
    assert api_inpaint.is_complex_for_ai(flat, flat_mask) is False

    art, art_mask = _make_textured_panel()
    score = api_inpaint.region_complexity_score(art, art_mask)
    assert score >= 0.52
    assert api_inpaint.is_complex_for_ai(art, art_mask) is True


def test_pack_and_unpack_single_2x2_and_3x3_atlas():
    c1 = np.full((180, 240, 3), (10, 40, 220), dtype=np.uint8)
    c2 = np.full((260, 140, 3), (30, 210, 50), dtype=np.uint8)
    c3 = np.full((120, 300, 3), (220, 60, 20), dtype=np.uint8)
    c4 = np.full((200, 200, 3), (180, 180, 40), dtype=np.uint8)

    single_atlas, single_slots, grid_single = api_inpaint.pack_regions_to_atlas([c1])
    assert single_atlas.size == (1024, 1024)
    assert grid_single == 1
    unpacked_single = api_inpaint.unpack_atlas(single_atlas, single_slots)
    assert unpacked_single[0].shape == c1.shape

    grid_atlas, grid_slots, grid_2x2 = api_inpaint.pack_regions_to_atlas([c1, c2, c3, c4])
    assert grid_atlas.size == (1024, 1024)
    assert grid_2x2 == 2
    unpacked = api_inpaint.unpack_atlas(grid_atlas, grid_slots)
    assert [u.shape for u in unpacked] == [c1.shape, c2.shape, c3.shape, c4.shape]
    for orig, rec in zip((c1, c2, c3, c4), unpacked):
        assert np.max(np.abs(orig.astype(int) - rec.astype(int))) <= 2

    nine_crops = [
        np.full((140 + (i % 3) * 20, 160 + (i % 2) * 30, 3), (20 * i + 10, 255 - 20 * i, 120), dtype=np.uint8)
        for i in range(9)
    ]
    atlas_3x3, slots_3x3, grid_3x3 = api_inpaint.pack_regions_to_atlas(nine_crops)
    assert atlas_3x3.size == (1024, 1024)
    assert grid_3x3 == 3
    unpacked_9 = api_inpaint.unpack_atlas(atlas_3x3, slots_3x3)
    assert [u.shape for u in unpacked_9] == [c.shape for c in nine_crops]
    for orig, rec in zip(nine_crops, unpacked_9):
        assert np.max(np.abs(orig.astype(int) - rec.astype(int))) <= 2


def test_plan_atlas_batches_dynamic_3x3_and_2x2():
    compact_regions = [{"crop": np.zeros((220, 260, 3), dtype=np.uint8)} for _ in range(9)]
    batches = api_inpaint.plan_atlas_batches(compact_regions, pack_atlas=True, max_calls=3)
    assert batches == [list(range(9))]

    large_regions = [{"crop": np.zeros((520, 480, 3), dtype=np.uint8)} for _ in range(6)]
    batches_large = api_inpaint.plan_atlas_batches(large_regions, pack_atlas=True, max_calls=1)
    assert batches_large == [[0, 1, 2, 3]]

    mixed = [
        {"crop": np.zeros((500, 450, 3), dtype=np.uint8)},  # 0 large
        {"crop": np.zeros((240, 240, 3), dtype=np.uint8)},  # 1 compact
        {"crop": np.zeros((500, 450, 3), dtype=np.uint8)},  # 2 large
        {"crop": np.zeros((240, 240, 3), dtype=np.uint8)},  # 3 compact
        {"crop": np.zeros((240, 240, 3), dtype=np.uint8)},  # 4 compact
        {"crop": np.zeros((240, 240, 3), dtype=np.uint8)},  # 5 compact
        {"crop": np.zeros((240, 240, 3), dtype=np.uint8)},  # 6 compact
    ]
    batches_mixed = api_inpaint.plan_atlas_batches(mixed, pack_atlas=True, max_calls=2)
    assert batches_mixed == [[0, 2], [1, 3, 4, 5, 6]]


def test_align_ai_crop_corrects_vertical_shift_and_scale_drift():
    orig, mask = _make_textured_panel(240, 260)
    h, w = orig.shape[:2]
    # Simulate gpt-image-2 subtle 3.5% vertical scale + 4px shift drift
    drift_mat = np.array([[1.0, 0.0, 2.0], [0.0, 1.035, -4.0]], dtype=np.float32)
    drifted = cv2.warpAffine(orig, drift_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

    valid = cv2.dilate((mask > 127).astype(np.uint8) * 255, np.ones((11, 11), np.uint8)) == 0
    mse_before = float(np.mean((orig[valid].astype(np.float32) - drifted[valid].astype(np.float32)) ** 2))

    aligned = api_inpaint.align_ai_crop(orig, drifted, mask)
    mse_after = float(np.mean((orig[valid].astype(np.float32) - aligned[valid].astype(np.float32)) ** 2))
    assert mse_after < mse_before * 0.35


def test_inpaint_regions_via_api_packs_4_into_1_call_and_respects_budget(monkeypatch):
    fake_provider = SimpleNamespace(
        id="generic-gw",
        label="Generic Gateway",
        api_base="https://gateway.example.test/v1",
        protocol="openai",
        auth_scheme="bearer",
    )
    monkeypatch.setattr(
        api_inpaint,
        "resolve_api_inpaint_target",
        lambda **kwargs: (fake_provider, "sk-test", "openai/gpt-image-2.6-flare", "low"),
    )
    monkeypatch.setattr(api_inpaint, "validate_url", lambda url: url)

    calls = []

    class FakeResp:
        ok = True
        status_code = 200
        text = ""

        def __init__(self, img: Image.Image):
            self._b64 = _encode_png_b64(img)

        def json(self):
            return {"data": [{"b64_json": self._b64}]}

    def fake_post(url, headers=None, data=None, files=None, timeout=None, allow_redirects=False):
        calls.append({"url": url, "data": dict(data or {})})
        sent_bytes = files["image"][1]
        sent_img = Image.open(io.BytesIO(sent_bytes)).convert("RGB")
        return FakeResp(sent_img)

    monkeypatch.setattr(api_inpaint, "public_request", lambda _method, url, **kw: fake_post(url, **kw))

    regions = [
        {"crop": np.full((120, 140, 3), 100 + i * 10, dtype=np.uint8), "mask": np.full((120, 140), 255, dtype=np.uint8)}
        for i in range(11)
    ]
    outputs = api_inpaint.inpaint_regions_via_api(regions, pack_atlas=True, max_calls=1)
    assert len(calls) == 1
    assert calls[0]["url"] == "https://gateway.example.test/v1/images/edits"
    assert calls[0]["data"]["model"] == "openai/gpt-image-2.6-flare"
    assert calls[0]["data"]["size"] == "1024x1024"
    assert "3x3 grid" in calls[0]["data"]["prompt"]
    assert all(out is not None for out in outputs[:9])
    assert outputs[9] is None and outputs[10] is None


def test_atlas_failure_keeps_results_in_slots_from_other_batches(monkeypatch):
    monkeypatch.setattr(api_inpaint, "resolve_api_inpaint_target", lambda **kw: (object(), "key", "model", "low"))

    def edit(provider, key, model, quality, atlas, *, grid_dim):
        if grid_dim == 2:
            raise RuntimeError("simulated image API failure")
        return atlas

    monkeypatch.setattr(api_inpaint, "_call_images_edit_endpoint", edit)
    crops = [np.full((440, 460, 3) if i in (0, 3) else (140, 150, 3), 30 + i, np.uint8)
             for i in range(7)]
    outputs = api_inpaint.inpaint_regions_via_api([{"crop": crop} for crop in crops], max_calls=2)

    for i, (crop, output) in enumerate(zip(crops, outputs, strict=True)):
        if i in (0, 3):
            assert output is None
        else:
            assert np.array_equal(output, crop)

    with pytest.raises(RuntimeError, match="simulated image API failure"):
        api_inpaint.inpaint_regions_via_api([{"crop": crops[0]}, {"crop": crops[3]}], max_calls=1)


def test_hybrid_chapter_preserves_brush_and_restore_layers(tmp_path, monkeypatch):
    pipe = ChapterPipeline()
    image = np.full((48, 48, 3), 220, np.uint8)
    raw_path = tmp_path / "original.png"
    write_image(raw_path, image)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[18:30, 18:34] = 255
    brush = np.zeros_like(mask)
    brush[22:35, 23:40] = 255
    restore = np.zeros_like(mask)
    restore[19:21, 19:21] = 255
    write_image(pipe._manual_mask_path(tmp_path, raw_path), brush)
    write_image(pipe._restore_mask_path(tmp_path, raw_path), restore)

    def defer(image, boxes, *, protected_regions, deferred_ai_items):
        deferred_ai_items.append({
            "crop": image, "local_mask": mask, "crop_box": (0, 0, 48, 48), "score": 0.9,
        })
        return image.copy()

    def brush_paint(image, local_mask, *, force_lama):
        image[local_mask > 0] = 44
        return image

    def composite(image, painted, local_mask, crop_box, *, feather):
        image[local_mask > 0] = painted[local_mask > 0]
        return image

    pipe._detector = SimpleNamespace(detect=lambda image: [], last_metrics=lambda: {})
    pipe._inpainter = SimpleNamespace(inpaint=defer, inpaint_mask=brush_paint, last_metrics=lambda: {})
    monkeypatch.setattr(api_inpaint, "composite_api_region", composite)
    monkeypatch.setattr(api_inpaint, "inpaint_regions_via_api", lambda *a, **kw: [np.full_like(image, 90)])
    page = pipe._process_page(raw_path, tmp_path, defer_chapter_ai=True)
    assert np.array_equal(read_image(Path(page["tmp_clean"])), image)

    observed = {}

    def commit(chapter_id, directory, page_index, page_data, snapshot):
        observed["auto"] = read_image(Path(page_data["tmp_auto_clean"]))
        observed["clean"] = read_image(Path(page_data["tmp_clean"]))
        return True

    monkeypatch.setattr(pipe, "_commit_processed_page", commit)
    committed, stale, errors = [], [], []
    pipe._finalize_chapter_ai_batch("chapter", tmp_path, [(0, page, None)], {}, committed, stale, errors, None)

    assert committed == [0] and not stale and not errors
    assert np.all(observed["auto"][mask > 0] == 90)
    assert np.all(observed["auto"][(brush > 0) & (mask == 0)] == 220)
    assert np.all(observed["clean"][brush > 0] == 44)
    assert np.all(observed["clean"][restore > 0] == 220)
    assert page["processing_metrics"]["auto_inpaint"]["api_inpaint_regions"] == 1
    assert not list(tmp_path.glob("*.tmp.png"))


def test_hybrid_finalize_discards_temporary_files_on_failure(tmp_path):
    pipe = ChapterPipeline()
    intermediate = tmp_path / "interrupted-clean.tmp.png"
    auto = tmp_path / "interrupted-auto.tmp.png"
    intermediate.write_bytes(b"invalid image")
    auto.touch()
    page = {"tmp_clean": str(intermediate), "tmp_auto_clean": str(auto)}
    errors = []

    pipe._finalize_chapter_ai_batch("chapter", tmp_path, [(0, page, None)], {}, [], [], errors, None)

    assert len(errors) == 1 and errors[0][0] == 0
    assert not intermediate.exists() and not auto.exists()


def test_hybrid_inpainter_routes_complex_to_api_and_falls_back_on_error(monkeypatch):
    art, _ = _make_textured_panel(300, 320)
    box = BubbleBox(70, 70, 190, 180, 0.95, np.full((110, 120), 255, np.uint8), safe_to_inpaint=True)

    monkeypatch.setattr(
        "app.inpaint.lama_inpainter.load_inpaint_settings",
        lambda: {
            "enabled": True,
            "hybrid_auto": True,
            "provider_id": "generic-gw",
            "model": "openai/gpt-image-2",
            "quality": "low",
            "max_api_calls_per_chapter": 2,
            "pack_atlas": True,
        },
    )

    api_called = []

    def fake_api(regions, **kwargs):
        api_called.append(len(regions))
        return [np.full_like(r["crop"], (190, 200, 210)) for r in regions]

    monkeypatch.setattr("app.inpaint.lama_inpainter.inpaint_regions_via_api", fake_api)

    inpainter = Inpainter()
    lama_calls = []
    inpainter._lama_fill = lambda *args, **kwargs: (lama_calls.append(1), args[0])[1]

    out = inpainter.inpaint(art, [box])
    assert api_called == [1]
    assert lama_calls == []
    assert inpainter.last_metrics()["api_inpaint_regions"] == 1
    assert out.shape == art.shape

    # Simulate API failure -> must fall back cleanly to local LaMa without raising
    monkeypatch.setattr(
        "app.inpaint.lama_inpainter.inpaint_regions_via_api",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("upstream 503")),
    )
    out2 = inpainter.inpaint(art, [box])
    assert lama_calls == [1]
    assert out2.shape == art.shape


def test_telea_fast_bypass_skips_lama_for_tiny_low_edge_speck():
    # Split two-tone background (Smart Fill declines because surround has 2 colors,
    # while Canny edge density stays < 0.08) with a tiny 10x10 speck.
    h, w = 120, 140
    img = np.full((h, w, 3), (110, 145, 210), dtype=np.uint8)
    img[:, w // 2:] = (210, 170, 120)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[55:65, 65:75] = 255  # area = 100 <= 180

    inpainter = Inpainter()
    inpainter._begin_metrics()
    lama_calls = []
    inpainter._lama_fill = lambda *args, **kwargs: (lama_calls.append(1), args[0])[1]

    res = inpainter._smart_paint_region(img.copy(), mask, (0, 0, w, h), force_lama=False)
    assert lama_calls == []
    assert inpainter.last_metrics()["telea_fast_regions"] == 1
    assert np.array_equal(res[mask == 0], img[mask == 0])

    # When force_lama=True, Telea fast-path MUST NOT bypass LaMa
    inpainter._begin_metrics()
    inpainter._smart_paint_region(img.copy(), mask, (0, 0, w, h), force_lama=True)
    assert lama_calls == [1]


def test_inpaint_settings_endpoints_and_ai_repaint_route(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import app.routers.editor as editor_router

    monkeypatch.setattr(api_inpaint, "INPAINT_SETTINGS_PATH", tmp_path / "inpaint_settings.json")
    monkeypatch.setattr(
        editor_router,
        "list_configured_image_providers",
        lambda: [{"id": "generic-gw", "label": "Generic Gateway", "api_base": "https://gw.test/v1", "protocol": "openai"}],
    )

    test_app = FastAPI()
    test_app.include_router(editor_router.router)
    client = TestClient(test_app)

    r_get = client.get("/api/inpaint/settings")
    assert r_get.status_code == 200
    body = r_get.json()
    assert body["model"] == "openai/gpt-image-2.6-flare"
    assert body["providers"][0]["id"] == "generic-gw"

    r_put = client.put("/api/inpaint/settings", json={
        "enabled": True,
        "hybrid_auto": True,
        "provider_id": "generic-gw",
        "model": "openai/gpt-image-2.6-flare",
        "quality": "low",
        "max_api_calls_per_chapter": 4,
        "pack_atlas": True,
    })
    assert r_put.status_code == 200
    assert r_put.json()["enabled"] is True
    assert r_put.json()["max_api_calls_per_chapter"] == 4

    # Test POST /api/repaint_mask with mode="ai" dispatches to pipeline.repaint_mask_ai
    cid = "deadbeef"
    ch_dir = tmp_path / "raw" / cid
    ch_dir.mkdir(parents=True)
    orig_path = ch_dir / "page_001.png"
    cv2.imwrite(str(orig_path), np.full((64, 64, 3), 240, dtype=np.uint8))

    monkeypatch.setattr(editor_router._app_config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        editor_router,
        "load_manifest_raw",
        lambda c: {"chapter_id": c, "pages": [{"original": orig_path.as_posix(), "clean": orig_path.as_posix()}]},
    )

    dispatched = {}

    def fake_repaint_ai(chapter_id, page_index, mask_array, *, provider_id=None, model=None, quality=None):
        dispatched.update({
            "chapter_id": chapter_id,
            "page_index": page_index,
            "shape": mask_array.shape,
            "provider_id": provider_id,
            "model": model,
            "quality": quality,
        })
        return {"chapter_id": chapter_id, "pages": [{"original": orig_path.as_posix(), "clean": orig_path.as_posix(), "manual_ai_applied": True}]}

    monkeypatch.setattr(editor_router.pipeline, "repaint_mask_ai", fake_repaint_ai)

    mask_png = io.BytesIO()
    Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(mask_png, format="PNG")
    r_ai = client.post(
        "/api/repaint_mask",
        data={
            "chapter_id": cid,
            "page_index": "0",
            "mode": "ai",
            "provider_id": "generic-gw",
            "model": "openai/gpt-image-2",
            "quality": "low",
        },
        files={"mask": ("mask.png", mask_png.getvalue(), "image/png")},
    )
    assert r_ai.status_code == 200, r_ai.text
    assert dispatched == {
        "chapter_id": cid,
        "page_index": 0,
        "shape": (64, 64),
        "provider_id": "generic-gw",
        "model": "openai/gpt-image-2",
        "quality": "low",
    }


def test_cross_page_chapter_batching_pools_6_pages_into_1_3x3_api_call(tmp_path, monkeypatch):
    import app.pipeline as pipeline_mod
    import app.manifest_utils as manifest_mod
    from app.pipeline import ChapterPipeline

    cid = "cafe1234"
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    out_dir = tmp_path / "output"
    (raw_dir / cid).mkdir(parents=True)
    (proc_dir / cid).mkdir(parents=True)
    (out_dir / cid).mkdir(parents=True)

    monkeypatch.setattr(pipeline_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(pipeline_mod, "PROCESSED_DIR", proc_dir)
    monkeypatch.setattr(manifest_mod, "PROCESSED_DIR", proc_dir)
    import app.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(cfg_mod, "PROCESSED_DIR", proc_dir)
    monkeypatch.setattr(cfg_mod, "OUTPUT_DIR", out_dir)

    art, _ = _make_textured_panel(260, 280)
    pages = []
    for i in range(6):
        p = raw_dir / cid / f"page_{i:03d}.png"
        cv2.imwrite(str(p), art)
        pages.append({
            "original": p.as_posix(),
            "clean": None,
            "boxes": [],
            "skipped": False,
            "process_required": True,
            "preserve_regions": [],
        })
    manifest_mod.save_manifest_raw(cid, {"chapter_id": cid, "pages": pages})

    hybrid_cfg = {
        "enabled": True,
        "hybrid_auto": True,
        "provider_id": "generic-gw",
        "model": "openai/gpt-image-2.6-flare",
        "quality": "low",
        "max_api_calls_per_chapter": 1,
        "pack_atlas": True,
    }
    monkeypatch.setattr(api_inpaint, "load_inpaint_settings", lambda: dict(hybrid_cfg))
    monkeypatch.setattr("app.inpaint.lama_inpainter.load_inpaint_settings", lambda: dict(hybrid_cfg))

    fake_provider = SimpleNamespace(
        id="generic-gw",
        label="Generic Gateway",
        api_base="https://gateway.example.test/v1",
        protocol="openai",
        auth_scheme="bearer",
    )
    monkeypatch.setattr(
        api_inpaint,
        "resolve_api_inpaint_target",
        lambda **kwargs: (fake_provider, "sk-test", "openai/gpt-image-2.6-flare", "low"),
    )
    monkeypatch.setattr(api_inpaint, "validate_url", lambda url: url)

    http_calls = []

    class FakeResp:
        ok = True
        status_code = 200
        text = ""

        def __init__(self, img: Image.Image):
            self._b64 = _encode_png_b64(img)

        def json(self):
            return {"data": [{"b64_json": self._b64}]}

    def fake_post(url, headers=None, data=None, files=None, timeout=None, allow_redirects=False):
        http_calls.append(dict(data or {}))
        sent_img = Image.open(io.BytesIO(files["image"][1])).convert("RGB")
        return FakeResp(sent_img)

    monkeypatch.setattr(api_inpaint, "public_request", lambda _method, url, **kw: fake_post(url, **kw))

    class StubDetector:
        def detect(self, _img):
            return [BubbleBox(75, 70, 185, 170, 0.95, np.full((100, 110), 255, np.uint8), safe_to_inpaint=True)]

        def leftover_boxes(self, _img, _boxes):
            return []

        def last_metrics(self):
            return {}

    pipe = ChapterPipeline.__new__(ChapterPipeline)
    pipe._detector = StubDetector()
    inp = Inpainter()
    lama_calls = []
    inp._lama_fill = lambda *args, **kwargs: (lama_calls.append(1), args[0])[1]
    pipe._inpainter = inp

    progress_done = []
    manifest = pipe.process_pages(cid, list(range(6)), workers=2, progress_callback=progress_done.append)

    # All 6 pages' complex SFX regions MUST be pooled into ONE single 3x3 HTTP call!
    assert len(http_calls) == 1
    assert "3x3 grid" in http_calls[0]["prompt"]
    assert lama_calls == []
    assert sorted(progress_done) == [0, 1, 2, 3, 4, 5]
    assert manifest["last_processing_run"]["inpaint"]["api_inpaint_regions"] == 6
    for i in range(6):
        assert manifest["pages"][i]["clean"] is not None
        assert manifest["pages"][i]["processing_metrics"]["auto_inpaint"]["api_inpaint_regions"] == 1
