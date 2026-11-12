import numpy as np
import pytest

import app.inpaint.lama_inpainter as lama_module
from app.inpaint.lama_inpainter import Inpainter


class _FakeEntry:
    def __init__(self, name):
        self.name = name


class _FakeMiganSession:
    def get_inputs(self):
        return [_FakeEntry("image"), _FakeEntry("mask")]

    def get_outputs(self):
        return [_FakeEntry("result")]

    def run(self, outputs, feed):
        image = feed["image"]
        assert image.shape[0] == 1 and image.shape[1] == 3
        assert feed["mask"].shape == (1, 1, image.shape[2], image.shape[3])
        return [np.full_like(image, 255)]


def _migan_inpainter(monkeypatch, tmp_path):
    monkeypatch.setattr(lama_module, "INPAINTER", "migan")
    model = tmp_path / "migan_pipeline_v2.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(lama_module, "MIGAN_MODEL", model)
    monkeypatch.setattr(
        lama_module, "make_session", lambda *_a, **_k: _FakeMiganSession()
    )
    return Inpainter()


def test_default_inpainter_is_lama():
    assert Inpainter()._use_migan is False


def test_migan_fill_paints_the_masked_region(monkeypatch, tmp_path):
    inpainter = _migan_inpainter(monkeypatch, tmp_path)
    crop = np.full((40, 60, 3), 30, np.uint8)
    crop[10:20, 20:40] = 200
    mask = np.zeros((40, 60), np.uint8)
    mask[10:20, 20:40] = 255
    image = crop.copy()
    result = inpainter._migan_fill(image, crop, mask, (0, 0, 60, 40), feather=False)
    assert result.shape == crop.shape
    assert np.all(result[10:20, 20:40] == 255)
    assert np.all(result[0:10, 0:20] == 30)
    metrics = inpainter.last_metrics()
    assert metrics["migan_regions"] == 1
    assert metrics["migan_model_runs"] == 1


def test_migan_fill_without_a_model_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(lama_module, "INPAINTER", "migan")
    monkeypatch.setattr(lama_module, "MIGAN_MODEL", tmp_path / "missing.onnx")
    inpainter = Inpainter()
    with pytest.raises(RuntimeError, match="MI-GAN model missing"):
        inpainter._ensure_migan_session()
