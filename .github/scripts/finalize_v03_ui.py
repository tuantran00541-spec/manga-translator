from __future__ import annotations

from pathlib import Path
import re
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
css_root = root / "app/static/css"
js_root = root / "app/static/js"

FORBIDDEN_CSS = (
    "workspace-nav-",
    "preview-navigation",
    "preview-nav-btn",
    "preview-thumbnail",
    "review-page-nav",
    "review-nav-btn",
    "review-position",
    "translation-page-nav",
    "translation-nav-btn",
)


def match_close(text: str, opening: int) -> int:
    depth = 1
    i = opening + 1
    quote: str | None = None
    comment = False
    while i < len(text):
        if comment:
            if text.startswith("*/", i):
                comment = False
                i += 2
                continue
            i += 1
            continue
        if quote:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == quote:
                quote = None
            i += 1
            continue
        if text.startswith("/*", i):
            comment = True
            i += 2
            continue
        if text[i] in ('"', "'"):
            quote = text[i]
            i += 1
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced CSS block")


def strip_legacy_rules(text: str) -> str:
    out: list[str] = []
    cursor = 0
    while True:
        opening = text.find("{", cursor)
        if opening < 0:
            out.append(text[cursor:])
            break
        closing = match_close(text, opening)
        header = text[cursor:opening]
        body = text[opening + 1 : closing]
        normalized = re.sub(r"^(?:\s|/\*.*?\*/)*", "", header, flags=re.S)
        if any(marker in header for marker in FORBIDDEN_CSS):
            cursor = closing + 1
            continue
        if normalized.startswith(("@media", "@supports", "@container", "@layer")):
            body = strip_legacy_rules(body)
        out.extend((header, "{", body, "}"))
        cursor = closing + 1
    return "".join(out)


# 1. Remove replaced navigation CSS and specificity escalation.
for path in sorted(css_root.glob("*.css")):
    if path.name == "app.css":
        continue
    text = path.read_text(encoding="utf-8")
    text = strip_legacy_rules(text)
    text = re.sub(r"\s*!important\b", "", text)
    path.write_text(text, encoding="utf-8")

# 2. Small destructive overlay control must be keyboard-revealable and >=24 px.
preview = css_root / "preview.css"
text = preview.read_text(encoding="utf-8")
old = """.excluded-region-del {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 18px;
  height: 18px;
  padding: 0;
  line-height: 16px;
  text-align: center;
  background: var(--danger, #ff4d4d);
  color: #fff;
  border: 0;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  box-shadow: 0 1px 3px rgba(0,0,0,0.4);
}

.excluded-region-box:hover .excluded-region-del {
  display: block;
}"""
new = """.excluded-region-del {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 24px;
  height: 24px;
  padding: 0;
  line-height: 22px;
  text-align: center;
  background: var(--danger, #ff4d4d);
  color: #fff;
  border: 0;
  border-radius: 50%;
  cursor: pointer;
  display: block;
  opacity: 0;
  pointer-events: none;
  transition: opacity var(--transition-fast);
  box-shadow: 0 1px 3px rgba(0,0,0,0.4);
}

.excluded-region-box:hover .excluded-region-del,
.excluded-region-box:focus-within .excluded-region-del,
.excluded-region-del:focus-visible {
  opacity: 1;
  pointer-events: auto;
}"""
if old not in text:
    raise SystemExit("excluded-region accessibility target not found")
preview.write_text(text.replace(old, new, 1), encoding="utf-8")

# 3. Shared navigator is the only programmatic page-selection API.
navigator_path = js_root / "page-navigator.js"
navigator = navigator_path.read_text(encoding="utf-8")
old = """      setItems(nextItems, nextActiveIndex = currentIndex) {
        currentItems = Array.isArray(nextItems) ? nextItems.slice() : [];
        currentIndex = clampIndex(nextActiveIndex, currentItems.length);
        render();
      },
"""
new = """      setItems(nextItems, nextActiveIndex = currentIndex) {
        currentItems = Array.isArray(nextItems) ? nextItems.slice() : [];
        currentIndex = clampIndex(nextActiveIndex, currentItems.length);
        render();
      },
      select(index) {
        select(index);
      },
      selectByKey(key) {
        const index = currentItems.findIndex((item) => String(item?.key) === String(key));
        if (index >= 0) select(index);
      },
"""
if old not in navigator:
    raise SystemExit("page navigator public API target not found")
navigator_path.write_text(navigator.replace(old, new, 1), encoding="utf-8")

review_path = js_root / "review-workspace.js"
review = review_path.read_text(encoding="utf-8")
needle = "    layout.append(navigator.element, canvasHost, inspector);\n"
if needle not in review:
    raise SystemExit("review navigator mount target not found")
review_path.write_text(review.replace(needle, "    workspace._pageNavigator = navigator;\n" + needle, 1), encoding="utf-8")

qc_path = js_root / "chapter-qc.js"
qc = qc_path.read_text(encoding="utf-8")
old = """    const visibleIndex = visiblePageIndices().indexOf(pageIndex);
    if (visibleIndex < 0) return;
    const jump = workspace.querySelector(".workspace-nav-jump-input");
    if (!jump) return;
    jump.value = String(visibleIndex + 1);
    jump.dispatchEvent(new Event("change", { bubbles: true }));
    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);
"""
new = """    if (!visiblePageIndices().includes(pageIndex)) return;
    const navigator = workspace._pageNavigator;
    if (!navigator || typeof navigator.selectByKey !== "function") return;
    navigator.selectByKey(pageIndex);
    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);
"""
if old not in qc:
    raise SystemExit("chapter QC legacy jump target not found")
qc_path.write_text(qc.replace(old, new, 1), encoding="utf-8")

# 4. One layered CSS entrypoint. Legacy stage files remain as migration inputs,
# but system/components/workbench own the cascade without !important wars.
app_css = '''/* Manga Translator v0.3 CSS entrypoint. */
@layer base, stage, system, components, workbench, utilities;

@import url("base.css") layer(base);

@import url("upload.css") layer(stage);
@import url("preview.css") layer(stage);
@import url("editor.css") layer(stage);
@import url("editor-workspace.css") layer(stage);
@import url("editor-box-transform.css") layer(stage);
@import url("box-panel.css") layer(stage);
@import url("text-object-editor.css") layer(stage);
@import url("toolbar.css") layer(stage);
@import url("recent-toast.css") layer(stage);
@import url("review-workspace.css") layer(stage);
@import url("deepseek-qc.css") layer(stage);
@import url("chapter-ocr.css") layer(stage);

@import url("ui-system.css") layer(system);
@import url("page-navigator.css") layer(components);
@import url("workbench.css") layer(workbench);

@layer utilities {
  .app-workspace {
    container-name: workspace;
    container-type: inline-size;
  }

  .page-navigator-item {
    content-visibility: auto;
    contain-intrinsic-size: auto 56px;
  }

  :where(button, [role="button"]) {
    min-width: 24px;
    min-height: 24px;
  }

  :where(button, input, select, textarea, [role="button"], [tabindex]):focus-visible {
    outline: 2px solid var(--focus, var(--accent-1));
    outline-offset: 2px;
  }

  @container workspace (max-width: 1100px) {
    .workbench-stage-grid {
      grid-template-columns: 190px minmax(0, 1fr) 220px;
    }
    .translation-workspace-body.editor-workbench-grid {
      grid-template-columns: 190px minmax(0, 1fr) 260px;
    }
  }

  @container workspace (max-width: 900px) {
    .workbench-stage-grid,
    .translation-workspace-body.editor-workbench-grid {
      grid-template-columns: 1fr;
    }
    .context-inspector,
    .translation-panel-host.context-inspector {
      position: static;
      max-height: none;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      scroll-behavior: auto !important;
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
      transition-delay: 0ms !important;
    }
  }
}
'''
(css_root / "app.css").write_text(app_css, encoding="utf-8")

# 5. HTML loads one CSS entrypoint.
html_path = root / "app/templates/index.html"
html = html_path.read_text(encoding="utf-8")
html, count = re.subn(
    r'(?:<link rel="stylesheet" href="/static/css/[^"]+">\n)+',
    '<link rel="stylesheet" href="/static/css/app.css">\n',
    html,
    count=1,
)
if count != 1:
    raise SystemExit("stylesheet link block not found exactly once")
html_path.write_text(html, encoding="utf-8")

# 6. Update gate contracts.
test_path = root / "tests/test_ui_system_contract.py"
tests = test_path.read_text(encoding="utf-8")
old = "    assert '/static/css/ui-system.css' in html\n    assert '/static/css/workbench.css' in html\n"
new = "    assert html.count('/static/css/') == 1\n    assert '/static/css/app.css' in html\n"
if old not in tests:
    raise SystemExit("UI CSS-link contract target not found")
tests = tests.replace(old, new, 1)
extra = '''

def test_v03_css_uses_layered_single_entrypoint():
    html = read("app/templates/index.html")
    css = read("app/static/css/app.css")
    assert html.count('/static/css/') == 1
    assert '/static/css/app.css' in html
    assert '@layer base, stage, system, components, workbench, utilities;' in css
    assert 'layer(workbench)' in css


def test_v03_css_removes_legacy_navigation_and_specificity_debt():
    roots = [ROOT / "app/static/css", ROOT / "app/static/js"]
    text = "\\n".join(
        path.read_text(encoding="utf-8")
        for folder in roots
        for path in folder.glob("*.*")
        if path.suffix in {".css", ".js"}
    )
    for legacy in (
        "workspace-nav-", "preview-navigation", "preview-thumbnail",
        "review-page-nav", "translation-page-nav",
    ):
        assert legacy not in text
    assert text.count("!important") == 5


def test_v03_css_keeps_accessibility_and_long_list_performance_contracts():
    css = read("app/static/css/app.css")
    preview = read("app/static/css/preview.css")
    assert "prefers-reduced-motion: reduce" in css
    assert "min-height: 24px" in css
    assert ":focus-visible" in css
    assert "content-visibility: auto" in css
    assert "contain-intrinsic-size" in css
    assert "container-type: inline-size" in css
    assert "@container workspace" in css
    assert "width: 24px" in preview
    assert ".excluded-region-box:focus-within .excluded-region-del" in preview


def test_v03_shared_page_navigator_replaces_duplicate_jump_logic():
    html = read("app/templates/index.html")
    navigator = read("app/static/js/page-navigator.js")
    stage_js = "\\n".join(read(path) for path in (
        "app/static/js/preview.js",
        "app/static/js/review-workspace.js",
        "app/static/js/editor.js",
    ))
    assert '/static/js/page-navigator.js' in html
    assert 'window.createPageNavigator' in navigator
    assert 'selectByKey(key)' in navigator
    assert 'const doJump' not in stage_js
'''
if "test_v03_css_uses_layered_single_entrypoint" not in tests:
    tests += extra
test_path.write_text(tests, encoding="utf-8")

qc_test = root / "tests/test_chapter_qc_ui.py"
text = qc_test.read_text(encoding="utf-8")
old = '    assert \'jump.dispatchEvent(new Event("change", { bubbles: true }))\' in js\n'
new = '    assert \'navigator.selectByKey(pageIndex)\' in js\n    assert \'workspace._pageNavigator\' in js\n'
if old not in text:
    raise SystemExit("chapter QC test target not found")
qc_test.write_text(text.replace(old, new, 1), encoding="utf-8")

# 7. Fail closed on any migration residue.
css_text = "\n".join(path.read_text(encoding="utf-8") for path in css_root.glob("*.css"))
js_text = "\n".join(path.read_text(encoding="utf-8") for path in js_root.glob("*.js"))
combined = css_text + "\n" + js_text
stale = [marker for marker in ("workspace-nav-", "preview-navigation", "preview-thumbnail", "review-page-nav", "translation-page-nav") if marker in combined]
if stale:
    raise SystemExit(f"legacy navigation residue remains: {stale}")
if css_text.count("!important") != 5:
    raise SystemExit(f"expected 5 reduced-motion !important rules, got {css_text.count('!important')}")
if html.count("/static/css/") != 1:
    raise SystemExit("expected one CSS entrypoint")
