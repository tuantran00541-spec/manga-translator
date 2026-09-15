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
        # Deliberately ignore preserve regions here. The optimized repaint policy
        # must hard-restore exact preserve pixels after every repaint pass.
        return np.full_like(image, 100)

    def inpaint_mask(self, image, mask, *, force_lama=False):
        self.manual_inputs.append((bool(force_lama), image.copy(), mask.copy()))
        value = 220 if force_lama else 180
        return np.full_like(image, value)


def test_lama_repaint_uses_original_context_and_exact_authority(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    img_path = tmp_path / "page.png"

    original = np.full((40, 48, 3), 10, dtype=np.uint8)
    original[5:35, 5:43] = 20
    write_image(img_path, original)

    lama_mask = np.zeros((40, 48), dtype=np.uint8)
    lama_mask[8:32, 8:40] = 255
    lama_mask_path = processed_dir / "manual_lama_mask_page.png"
    write_image(lama_mask_path, lama_mask)

    preserve = [{"x1": 12, "y1": 12, "x2": 18, "y2": 20}]
    effective_mask = subtract_regions_from_mask(lama_mask, preserve)

    pipeline = OptimizedChapterPipeline.__new__(OptimizedChapterPipeline)
    fake = _FakeInpainter()
    pipeline._inpainter = fake

    clean_path = pipeline._do_reinpaint(
        processed_dir,
        img_path,
        original,
        [],
        manual_lama_mask_posix=lama_mask_path.as_posix(),
        preserve_regions=preserve,
    )
    clean = read_image(Path(clean_path))

    assert len(fake.manual_inputs) == 1
    force_lama, model_input, model_mask = fake.manual_inputs[0]
    assert force_lama is True
    assert np.array_equal(model_input, original)
    assert np.array_equal(model_mask, effective_mask)

    authority = effective_mask > 0
    assert np.all(clean[authority] == 220)
    assert np.all(clean[0:5, 0:5] == 100)

    p = preserve[0]
    assert np.array_equal(
        clean[p["y1"]:p["y2"], p["x1"]:p["x2"]],
        original[p["y1"]:p["y2"], p["x1"]:p["x2"]],
    )


def test_standard_repaint_cannot_modify_preserve_pixels(tmp_path: Path):
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

    p = preserve[0]
    assert np.array_equal(
        clean[p["y1"]:p["y2"], p["x1"]:p["x2"]],
        original[p["y1"]:p["y2"], p["x1"]:p["x2"]],
    )
