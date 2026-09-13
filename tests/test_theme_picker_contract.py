from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_theme_picker_replaces_native_select_with_accessible_custom_menu():
    theme_js = (ROOT / "app/static/js/theme.js").read_text(encoding="utf-8")
    picker_css = (ROOT / "app/static/css/theme-picker.css").read_text(encoding="utf-8")
    app_css = (ROOT / "app/static/css/app.css").read_text(encoding="utf-8")

    assert "function buildThemePicker()" in theme_js
    assert 'trigger.setAttribute("aria-haspopup", "listbox")' in theme_js
    assert 'menu.setAttribute("role", "listbox")' in theme_js
    assert 'option.setAttribute("role", "option")' in theme_js
    assert 'document.addEventListener("pointerdown"' in theme_js
    assert 'document.addEventListener("focusin"' in theme_js
    assert 'event.key === "Escape"' in theme_js
    assert 'event.key === "ArrowDown"' in theme_js
    assert 'event.key === "ArrowUp"' in theme_js
    assert 'system: "Hệ thống"' in theme_js
    assert 'light: "Sáng"' in theme_js
    assert 'dark: "Tối"' in theme_js

    assert ".theme-select-native" in picker_css
    assert ".theme-menu" in picker_css
    assert '.theme-option[aria-selected="true"]' in picker_css
    assert '@import url("./theme-picker.css") layer(studio);' in app_css
