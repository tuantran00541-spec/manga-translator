from __future__ import annotations

from pathlib import Path
import re


def test_i18n_switcher_wiring_and_contracts():
    root = Path(__file__).resolve().parents[1]
    index_html = (root / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    i18n_js = (root / "app" / "static" / "js" / "i18n.js").read_text(encoding="utf-8")
    workspace_js = (root / "app" / "static" / "js" / "review-workspace.js").read_text(encoding="utf-8")
    app_css = (root / "app" / "static" / "css" / "app.css").read_text(encoding="utf-8")

    assert re.search(r'<script\s+src="/static/js/i18n\.js(?:\?[^"]*)?"></script>', index_html)
    assert 'id="lang-toggle"' in index_html
    assert ".lang-toggle" in app_css
    assert 'data-ui-lang-select' in workspace_js
    assert 'DEFAULT_LANG = "vi"' in i18n_js
    assert "window.setUiLang = setUiLang" in i18n_js
    assert "window.getUiLang = () => currentLang" in i18n_js
    assert "MutationObserver" in i18n_js
    assert "Object.create(null)" in i18n_js
    assert "NodeFilter.FILTER_REJECT" in i18n_js
    assert ".review-proof-source" in i18n_js
    assert ".translation-textarea" in i18n_js
    assert '"Xong": "Done"' in i18n_js
    assert '"Cứu:": "Rescued:"' in i18n_js
    assert i18n_js.count('"Chọn":') == 1
