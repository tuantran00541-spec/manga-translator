import pytest

from app.detector import ctd_mask
from ctd_fake import InkModel


@pytest.fixture(autouse=True)
def fake_letter_model(monkeypatch):
    """Tests never load the real comic-text-detector model; by default it reads no letters."""
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=0))


@pytest.fixture(autouse=True)
def legacy_tools(monkeypatch):
    """Most agent tests drive the shell one call at a time, so they run without the chain-only defaults."""
    from app.agent import registry
    monkeypatch.setattr(registry, "CHAIN_DEFAULTS", {"shell_tool": True, "min_calls": 1})


@pytest.fixture(autouse=True)
def series_memory_in_tmp(monkeypatch, tmp_path_factory):
    """Series memory written by a test never lands in the real data folder."""
    from app import series_memory
    monkeypatch.setattr(series_memory, "SERIES_DIR", tmp_path_factory.mktemp("series"))
