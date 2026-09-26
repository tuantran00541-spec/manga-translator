from __future__ import annotations

from pathlib import Path

import numpy as np

from app.image_io import read_image, write_image
from app.optimized_pipeline import OptimizedChapterPipeline
from app.region_policy import subtract_regions_from_mask


class _FakeInpainter:
    def __init__(self):
        self.manual_inputs: list[tuple[bool, np.ndarray, np.ndarray]] = []

    def inpaint(self, image, boxes, *, protected_regions=None):
        return np.full_like(image, 100)

    def inpaint_mask(self, image, mask, *, force_lama=False):
        raise AssertionError("optimized manual repaint must bypass dilating inpaint_mask")

    def _smart_paint_region(
        self,
        image,
        local_mask,
        crop_box,
        feather=False,
        force_lama=False,
    ):
        assert feather is False
        self.manual_inputs.append(
            (bool(force_lama), image.copy(), local_mask.copy())
        )
        out = image.copy()
        cx1, cy1, cx2, cy2 = crop_box
        authority = local_mask > 0
        value = 220 if force_lama else 180
        target = out[cy1:cy2, cx1:cx2]
        target[authority] = value
        return out


def test_lama_repaint_uses_exact_user_mask_without_growth(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    img_path = tmp_path / "page.png"

    original = np.full((40, 48, 3), 10, dtype=np.uint8)
    original[5:35, 5:43] = 20
    write_image(img_path, original)

    user_mask = np.zeros((40, 48), dtype=np.uint8)
    user_mask[8:32, 8:40] = 255
    user_mask_path = processed_dir / "manual_lama_mask_page.png"
    write_image(user_mask_path, user_mask)

    boxes = [
        {
            "id": "segmenter_box",
            "x1": 4,
            "y1": 7,
            "x2": 42,
            "y2": 31,
            "confidence": 0.95,
            "source_model": "text_segmenter.onnx",
            "mask_source": "text_segmenter",
            "safe_to_inpaint": True,
        }
    ]

    preserve = [{"x1": 12, "y1": 12, "x2": 18, "y2": 20}]
    effective_user_mask = subtract_regions_from_mask(user_mask, preserve)

    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    fake_inpainter = _FakeInpainter()
    pipeline._inpainter = fake_inpainter

    clean_path = pipeline._do_reinpaint(
        processed_dir,
        img_path,
        original,
        boxes,
        manual_lama_mask_posix=user_mask_path.as_posix(),
        preserve_regions=preserve,
    )
    clean = read_image(Path(clean_path))

    assert len(fake_inpainter.manual_inputs) == 1
    force_lama, model_input, model_mask = fake_inpainter.manual_inputs[0]
    assert force_lama is True
    assert np.array_equal(model_input, original)
    assert np.array_equal(model_mask, effective_user_mask)

    stats = pipeline._last_repaint_detector_stats
    expected_pixels = int(np.count_nonzero(effective_user_mask > 0))
    assert stats["mask_mode"] == "exact_user_mask"
    assert stats["detector_source"] == "not_used"
    assert stats["user_mask_pixels"] == expected_pixels
    assert stats["inference_mask_pixels"] == expected_pixels
    assert stats["detector_added_pixels"] == 0
    assert stats["dilation_pixels"] == 0
    assert stats["feather_outside_mask"] is False

    authority = effective_user_mask > 0
    assert np.all(clean[authority] == 220)

    p = preserve[0]
    assert np.array_equal(
        clean[p["y1"]:p["y2"], p["x1"]:p["x2"]],
        original[p["y1"]:p["y2"], p["x1"]:p["x2"]],
    )


def test_standard_repaint_uses_exact_user_mask_and_preserve(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    img_path = tmp_path / "page.png"

    original = np.full((32, 36, 3), 25, dtype=np.uint8)
    write_image(img_path, original)

    standard_mask = np.zeros((32, 36), dtype=np.uint8)
    standard_mask[6:26, 6:30] = 255
    standard_mask_path = processed_dir / "manual_mask_page.png"
    write_image(standard_mask_path, standard_mask)

    preserve = [{"x1": 10, "y1": 10, "x2": 16, "y2": 18}]
    effective_mask = subtract_regions_from_mask(standard_mask, preserve)

    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    fake = _FakeInpainter()
    pipeline._inpainter = fake

    clean_path = pipeline._do_reinpaint(
        processed_dir,
        img_path,
        original,
        [],
        manual_mask_posix=standard_mask_path.as_posix(),
        preserve_regions=preserve,
    )
    clean = read_image(Path(clean_path))

    assert len(fake.manual_inputs) == 1
    force_lama, model_input, model_mask = fake.manual_inputs[0]
    assert force_lama is False
    assert np.all(model_input == 100)
    assert np.array_equal(model_mask, effective_mask)

    authority = effective_mask > 0
    assert np.all(clean[authority] == 180)

    p = preserve[0]
    assert np.array_equal(
        clean[p["y1"]:p["y2"], p["x1"]:p["x2"]],
        original[p["y1"]:p["y2"], p["x1"]:p["x2"]],
    )


def test_preserve_and_reinpaint_restores_original_pixels_and_keeps_translations(tmp_path: Path, monkeypatch):
    import app.config as config
    import app.manifest_utils as manifests
    import app.pipeline_editing as editing
    from app.region_policy import text_object_in_preserve_region

    chapter = "d4c3b2a1"
    processed, output = tmp_path / "processed", tmp_path / "output"
    (processed / chapter).mkdir(parents=True)
    for module, name, value in ((editing, "PROCESSED_DIR", processed), (manifests, "PROCESSED_DIR", processed),
                                (config, "PROCESSED_DIR", processed), (config, "OUTPUT_DIR", output)):
        monkeypatch.setattr(module, name, value)
    original = np.full((60, 80, 3), 30, dtype=np.uint8)
    img_path = tmp_path / "page.png"
    write_image(img_path, original)
    boxes = [
        {"id": "b1", "x1": 5, "y1": 5, "x2": 35, "y2": 25, "confidence": 0.9, "safe_to_inpaint": True},
        {"id": "b2", "x1": 45, "y1": 30, "x2": 75, "y2": 55, "confidence": 0.9, "safe_to_inpaint": True},
    ]
    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    pipeline._inpainter = _FakeInpainter()
    clean = pipeline._do_reinpaint(processed / chapter, img_path, original, boxes,
                                   manual_mask_posix=None, manual_lama_mask_posix=None, preserve_regions=[])
    objects = [
        {"id": "o1", "region": {"x1": 5, "y1": 5, "x2": 35, "y2": 25}, "translation": "Chào", "source_boxes": ["b1"]},
        {"id": "o2", "region": {"x1": 45, "y1": 30, "x2": 75, "y2": 55}, "translation": "", "source_boxes": ["b2"]},
    ]
    manifests.save_manifest_raw(chapter, {"chapter_id": chapter, "pages": [{
        "original": img_path.as_posix(), "clean": clean, "boxes": boxes, "text_objects": objects,
        "preserve_regions": [], "skipped": False, "process_required": False,
        "source_revision": 1, "clean_revision": 1, "render_revision": 0, "rendered": False,
    }]})

    region = {"x1": 43, "y1": 28, "x2": 77, "y2": 57}
    pipeline.preserve_and_reinpaint(chapter, 0, [region])

    page = manifests.load_manifest_raw(chapter)["pages"][0]
    assert page["preserve_regions"] == [region]
    assert page["process_required"] is False and page["clean_revision"] == 2, "no second detection pass"
    assert [obj["translation"] for obj in page["text_objects"]] == ["Chào", ""], "translations survive"
    assert text_object_in_preserve_region(page, page["text_objects"][1])
    cleaned = read_image(Path(page["clean"]))
    assert (cleaned[30:55, 45:75] == 30).all(), "the preserved object shows the original pixels again"
    assert (cleaned[5:25, 5:35] == 100).all(), "the other box stays erased"
