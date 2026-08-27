from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()

def replace_once(path, old, new):
    p = root / path
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"target not found in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")

replace_once(
    "app/static/css/preview.css",
    """.excluded-region-del {
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
}""",
    """.excluded-region-del {
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
}""",
)

replace_once(
    "app/static/js/page-navigator.js",
    """      select(index) {
        select(index);
      },
      setBusy(value) {
""",
    """      select(index) {
        select(index);
      },
      selectByKey(key) {
        const index = currentItems.findIndex((item) => String(item?.key) === String(key));
        if (index >= 0) select(index);
      },
      setBusy(value) {
""",
)

replace_once(
    "app/static/js/chapter-qc.js",
    """    const visibleIndex = visiblePageIndices().indexOf(pageIndex);
    if (visibleIndex < 0) return;
    const navigator = workspace._pageNavigator;
    if (!navigator || typeof navigator.select !== "function") return;
    navigator.select(visibleIndex);
    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);
""",
    """    if (!visiblePageIndices().includes(pageIndex)) return;
    const navigator = workspace._pageNavigator;
    if (!navigator || typeof navigator.selectByKey !== "function") return;
    navigator.selectByKey(pageIndex);
    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);
""",
)

replace_once(
    "tests/test_chapter_qc_ui.py",
    """    assert 'jump.dispatchEvent(new Event(\"change\", { bubbles: true }))' in js
""",
    """    assert 'navigator.selectByKey(pageIndex)' in js
    assert 'workspace._pageNavigator' in js
""",
)

ui_test = root / "tests/test_ui_system_contract.py"
text = ui_test.read_text(encoding="utf-8")
needle = '    assert "@container workbench" in css\n'
addition = '    assert "@container workbench" in css\n    preview = read("app/static/css/preview.css")\n    assert "width: 24px" in preview\n    assert ".excluded-region-box:focus-within .excluded-region-del" in preview\n'
if needle not in text:
    raise SystemExit("UI accessibility contract target not found")
ui_test.write_text(text.replace(needle, addition, 1), encoding="utf-8")

print("V03_FINAL_POLISH_OK")
