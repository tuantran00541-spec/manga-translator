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
    "app/static/js/page-navigator.js",
    """    return {\n      element: root,\n      setBusy(value) {\n""",
    """    return {\n      element: root,\n      select(index) {\n        select(index);\n      },\n      setBusy(value) {\n""",
)

replace_once(
    "app/static/js/review-workspace.js",
    """    layout.append(navigator.element, canvasHost, inspector);\n""",
    """    workspace._pageNavigator = navigator;\n    layout.append(navigator.element, canvasHost, inspector);\n""",
)

replace_once(
    "app/static/js/chapter-qc.js",
    """    const jump = workspace.querySelector(\".workspace-nav-jump-input\");\n    if (!jump) return;\n    jump.value = String(visibleIndex + 1);\n    jump.dispatchEvent(new Event(\"change\", { bubbles: true }));\n    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);\n""",
    """    const navigator = workspace._pageNavigator;\n    if (!navigator || typeof navigator.select !== \"function\") return;\n    navigator.select(visibleIndex);\n    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);\n""",
)

print("V03_QC_NAV_BRIDGE_OK")
