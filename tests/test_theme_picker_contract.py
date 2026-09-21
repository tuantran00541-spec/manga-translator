from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_theme_control_is_native_and_has_exactly_three_modes():
    theme_js = (ROOT / "app/static/js/theme.js").read_text(encoding="utf-8")
    app_css = (ROOT / "app/static/css/app.css").read_text(encoding="utf-8")
    index_html = (ROOT / "app/templates/index.html").read_text(encoding="utf-8")

    assert 'const MODES = new Set(["light", "dark", "system"])' in theme_js
    assert 'const resolved = normalized === "system"' in theme_js
    assert 'select.addEventListener("change", () => applyMode(select.value))' in theme_js
    assert "function buildThemePicker()" not in theme_js
    assert "theme-trigger" not in theme_js
    assert "theme-menu" not in theme_js

    assert '<option value="system">Hệ thống</option>' in index_html
    assert '<option value="light">Sáng</option>' in index_html
    assert '<option value="dark">Tối</option>' in index_html
    assert index_html.count('<option value="system">') == 1
    assert index_html.count('<option value="light">') == 1
    assert index_html.count('<option value="dark">') == 1

    assert ".theme-control > select" in app_css
    assert ".theme-control.theme-picker" not in app_css
    assert ".theme-select-native" not in app_css
    assert ".theme-menu" not in app_css
