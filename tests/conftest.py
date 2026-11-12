import pytest

from app.detector import ctd_mask
from ctd_fake import InkModel


@pytest.fixture(autouse=True)
def fake_letter_model(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=0))


@pytest.fixture(autouse=True)
def series_memory_in_tmp(monkeypatch, tmp_path_factory):
    from app import series_memory
    monkeypatch.setattr(series_memory, "SERIES_DIR", tmp_path_factory.mktemp("series"))
