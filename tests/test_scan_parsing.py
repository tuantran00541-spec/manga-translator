from app.ai_mode.page_scan import SCAN_PROMPT, parse_scan
from app.translation.vision import parse_vision_translation


def test_scan_prompt_treats_a_title_page_as_story():
    assert "title page carrying the series title" in SCAN_PROMPT
    assert "not uploader credits" in SCAN_PROMPT


def test_parse_scan_rejects_a_logo_covering_most_of_the_slice():
    data = {
        "slices": [
            {
                "slice": 0,
                "is_credit": False,
                "credit_confidence": 0.0,
                "logos": [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9}],
            }
        ]
    }
    scans = parse_scan(data, {0: (100, 100)})
    assert scans[0].logos == ()


def test_parse_vision_translation_fills_omitted_ids_when_allowed():
    content = '{"translations":[{"id":"1","translated_text":"xin chao"}]}'
    result = parse_vision_translation(content, {"1", "2"}, allow_missing=True)
    assert result == {"1": "xin chao", "2": ""}


def test_parse_scan_reports_malformed_entries_instead_of_silently_dropping_them():
    errors = []
    data = {"slices": ["not an object",
                       {"slice": "x", "is_credit": False, "credit_confidence": 0.0},
                       {"slice": 0, "is_credit": False, "credit_confidence": 0.0},
                       {"slice": 0, "is_credit": True, "credit_confidence": 0.9}]}
    scans = parse_scan(data, {0: (100, 100)}, errors=errors)
    assert len(scans) == 1 and scans[0].is_credit is False
    assert [entry["reason"] for entry in errors] == [
        "entry is not an object",
        "slice is not an integer",
        "duplicate slice entry; first entry kept",
    ]
    assert [entry["slice"] for entry in errors] == [None, "x", 0]
