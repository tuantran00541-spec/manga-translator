import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import app.routers.editor as editor
from app.schemas import StylePresetsRequest


@pytest.fixture
def preset_file(tmp_path, monkeypatch):
    path = tmp_path / "style_presets.json"
    monkeypatch.setattr(editor, "STYLE_PRESETS_PATH", path)
    return path


def _request(*names):
    return StylePresetsRequest.model_validate(
        {"presets": [{"name": n, "style": {"font": "default", "bold": True, "strokeWidth": "3"}} for n in names]}
    )


def test_presets_start_empty_and_round_trip(preset_file):
    assert editor.get_style_presets() == {"presets": []}
    editor.put_style_presets(_request("Lời thoại", "Hiệu ứng"))
    presets = editor.get_style_presets()["presets"]
    assert [p["name"] for p in presets] == ["Lời thoại", "Hiệu ứng"]
    assert presets[1]["style"]["bold"] is True and presets[1]["style"]["strokeWidth"] == "3"


def test_duplicate_or_blank_names_are_refused(preset_file):
    with pytest.raises(HTTPException):
        editor.put_style_presets(_request("A", "A"))
    with pytest.raises(HTTPException):
        editor.put_style_presets(_request("A", "   "))
    assert not preset_file.exists()


def test_a_corrupt_file_reads_as_no_presets(preset_file):
    preset_file.write_text("{not json", encoding="utf-8")
    assert editor.get_style_presets() == {"presets": []}


def test_style_fields_are_validated():
    with pytest.raises(ValidationError):
        StylePresetsRequest.model_validate({"presets": [{"name": "x", "style": {"horizontalAlign": "sideways"}}]})
