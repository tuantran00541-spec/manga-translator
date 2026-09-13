from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_theme_picker_uses_existing_stylesheet_and_hides_native_select():
    theme_js = (ROOT / "app/static/js/theme.js").read_text(encoding="utf-8")
    app_css = (ROOT / "app/static/css/app.css").read_text(encoding="utf-8")
    picker_css = ROOT / "app/static/css/theme-picker.css"

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
    assert "Object.assign(select.style" in theme_js
    assert 'pointerEvents: "none"' in theme_js

    assert not picker_css.exists()
    assert '@import url("./theme-picker.css")' not in app_css
    assert ".theme-control.theme-picker" in app_css
    assert ".theme-select-native" in app_css
    assert ".theme-menu" in app_css
    assert '.theme-option[aria-selected="true"]' in app_css
