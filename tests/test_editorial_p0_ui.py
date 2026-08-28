from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage_rail_orders_script_and_final_qc_around_typeset():
    html = read("app/templates/index.html")
    order = [
        'data-stage="review"',
        'data-stage="script"',
        'data-stage="editor"',
        'data-stage="final_qc"',
    ]
    positions = [html.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "/static/js/script-workspace.js" in html
    assert "/static/js/final-qc.js" in html


def test_script_workspace_is_keyboard_first_and_persists_on_text_objects():
    js = read("app/static/js/script-workspace.js")
    assert 'event.ctrlKey && event.key === "Enter"' in js
    assert 'event.altKey && event.key === "ArrowDown"' in js
    assert '"/api/text_object/update"' in js
    assert '"/api/script/review"' in js
    assert '"/api/text_objects/ensure"' in js
    assert 'window.renderScript = renderScript' in js


def test_final_qc_is_mandatory_export_surface():
    final_qc = read("app/static/js/final-qc.js")
    export = read("app/static/js/chapter-export.js")
    assert '"/api/final_qc/page"' in final_qc
    assert 'ready_for_export' in final_qc
    assert 'Xuất chapter (.zip)' in final_qc
    assert '/api/render/chapter?chapter_id=' in export
    assert 'window.renderFinalQC' in export
    assert '/api/export/' not in export


def test_editorial_p0_styles_are_loaded_by_app_bundle():
    css = read("app/static/css/app.css")
    assert '@import url("./editorial-p0.css") layer(stage);' in css
