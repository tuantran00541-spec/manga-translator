from pathlib import Path


def replace_exact(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor, found {count}")
    path.write_text(source.replace(old, new), encoding="utf-8")


workbench = Path("app/static/css/workbench.css")
tokens = Path("app/static/css/tokens.css")
browser = Path("scripts/browser_sanity.py")

replace_exact(
    workbench,
    "@media (max-width: 720px) {\n  :root {\n    --sticky-header-offset: 92px;\n  }\n\n",
    "@media (max-width: 720px) {\n",
)

source = tokens.read_text(encoding="utf-8")
responsive = "\n@media (max-width: 720px) {\n  :root {\n    --sticky-header-offset: 92px;\n  }\n}\n"
if responsive.strip() not in source:
    tokens.write_text(source.rstrip() + "\n" + responsive, encoding="utf-8")

replace_exact(
    browser,
    '        "--nav-width: 168px;",\n        "--inspector-width: 288px;",',
    '        "--nav-width: var(--studio-rail-width, 250px);",\n        "--inspector-width: var(--studio-inspector-width, 320px);",',
)
replace_exact(
    browser,
    '            "card._reviewBusy = true",',
    '            "activeCard._reviewBusy = true",',
)

print("phase9 release gate patch applied")
