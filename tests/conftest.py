import pytest

from app.detector import ctd_mask
from ctd_fake import InkModel


@pytest.fixture(autouse=True)
def fake_letter_model(monkeypatch):
    """Tests never load the real comic-text-detector model; by default it reads no letters."""
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=0))
