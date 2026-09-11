from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

JS_PATHS = sorted(Path("app/static/js").rglob("*.js"))
HTML_PATHS = sorted(Path("app/templates").rglob("*.html"))
CSS_PATHS = sorted(Path("app/static/css").rglob("*.css"))
STATIC_ROOT = Path("app/static")
TOKENS_PATH = STATIC_ROOT / "css" / "tokens.css"
WORKBENCH_PATH = STATIC_ROOT / "css" / "studio.css"
CSS_IMPORT_PATTERN = re.compile(
    r"@import\s+url\(\s*(?:['\"])?([^'\")\s]+)(?:['\"])?\s*\)",
    re.IGNORECASE,
)
HTML_STATIC_REF_PATTERN = re.compile(
    r"(?:src|href)\s*=\s*['\"](/?static/[^'\"?#]+)(?:[?#][^'\"]*)?['\"]",
    re.IGNORECASE,
)
TOKEN_REFERENCE_PATTERN = re.compile(r"var\((--[A-Za-z0-9_-]+)")
TOKEN_DEFINITION_PATTERN = re.compile(r"(?:^|[;{\s])(--[A-Za-z0-9_-]+)\s*:")
LEGACY_TOKEN_PATTERN = re.compile(
    r"--(?:"
    r"color-[A-Za-z0-9_-]+|"
    r"ink(?:-raised)?|panel|line|paper(?:-dim)?|"
    r"text(?:-dim|-faint)?|blue(?:-dim)?|red(?:-dim)?|ok|"
    r"transition-(?:fast|normal)|surface-[0-9]|border-[0-9]|text-[123]|"
    r"accent-[12]|font-(?:body|display|heading)|muted-text|border-color|"
    r"ui-border|ui-panel"
    r")(?![A-Za-z0-9_-])"
)
LOCAL_LAYOUT_TOKENS = {"--nav-width", "--inspector-width"}
STRUCTURAL_GLYPH_PATTERN = re.compile(
    r"[\u2190-\u21ff\u25a0-\u25ff\u2600-\u27ff\U0001f000-\U0001faff]"
)
REQUIRED_SEMANTIC_TOKENS = {
    "--surface-app",
    "--surface-canvas",
    "--surface-panel",
    "--surface-raised",
    "--surface-overlay",
    "--border-subtle",
    "--border-strong",
    "--text-primary",
    "--text-secondary",
    "--text-muted",
    "--text-disabled",
    "--accent",
    "--accent-hover",
    "--accent-soft",
    "--success",
    "--warning",
    "--danger",
    "--radius-xs",
    "--radius-sm",
    "--radius-md",
    "--motion-fast",
    "--motion-normal",
    "--z-canvas",
    "--z-panel",
    "--z-toolbar",
    "--z-popover",
    "--z-modal",
    "--z-toast",
}


class _MarkupParser(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.ids: dict[str, int] = {}
        self.failures: list[str] = []

    def _check_attrs(self, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value for name, value in attrs}
        element_id = attr_map.get("id")
        if element_id:
            line = self.getpos()[0]
            previous = self.ids.get(element_id)
            if previous is not None:
                self.failures.append(
                    f"{self.path}:{line}: duplicate id={element_id!r}; first declared on line {previous}"
                )
            else:
                self.ids[element_id] = line

        for name, value in attrs:
            if name.lower().startswith("on") and value:
                self.failures.append(
                    f"{self.path}:{self.getpos()[0]}: inline event handler {name!r} is not allowed"
                )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._check_attrs(attrs)
        attr_map = {name.lower(): value for name, value in attrs}
        local_ref = None
        if tag.lower() == "script":
            local_ref = attr_map.get("src")
        elif tag.lower() == "link":
            local_ref = attr_map.get("href")
        if local_ref:
            _check_static_ref(self.path, self.getpos()[0], local_ref, self.failures)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _fail(title: str, failures: list[str]) -> None:
    if not failures:
        return
    print(title)
    for failure in failures:
        print(f"  {failure}")
    raise SystemExit(1)


def _static_path(ref: str) -> Path | None:
    clean = ref.split("?", 1)[0].split("#", 1)[0]
    if clean.startswith("/static/"):
        return STATIC_ROOT / clean.removeprefix("/static/")
    if clean.startswith("static/"):
        return STATIC_ROOT / clean.removeprefix("static/")
    return None


def _check_static_ref(source_path: Path, line: int, ref: str, failures: list[str]) -> None:
    target = _static_path(ref)
    if target is not None and not target.is_file():
        failures.append(f"{source_path}:{line}: missing static asset {ref!r} -> {target}")


def _css_local_imports(path: Path) -> list[Path]:
    source = path.read_text(encoding="utf-8")
    imports: list[Path] = []
    for match in CSS_IMPORT_PATTERN.finditer(source):
        ref = match.group(1)
        if ref.startswith(("http://", "https://", "data:")):
            continue
        imports.append(
            (path.parent / ref.split("?", 1)[0].split("#", 1)[0]).resolve()
        )
    return imports


def check_markup_integrity() -> None:
    failures: list[str] = []
    for path in HTML_PATHS:
        parser = _MarkupParser(path)
        try:
            parser.feed(path.read_text(encoding="utf-8"))
            parser.close()
        except Exception as exc:
            failures.append(f"{path}: HTML parse failed: {exc}")
        failures.extend(parser.failures)

    for path in CSS_PATHS:
        source = path.read_text(encoding="utf-8")
        for match in CSS_IMPORT_PATTERN.finditer(source):
            ref = match.group(1)
            if ref.startswith(("http://", "https://", "data:")):
                continue
            target = (path.parent / ref.split("?", 1)[0].split("#", 1)[0]).resolve()
            if not target.is_file():
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{path}:{line}: missing CSS import {ref!r}")

    _fail("Browser markup integrity failures:", failures)
    print(f"Browser markup integrity OK: {len(HTML_PATHS)} HTML, {len(CSS_PATHS)} CSS files")


def check_browser_asset_reachability() -> None:
    entrypoints: set[Path] = set()
    failures: list[str] = []
    for path in HTML_PATHS:
        source = path.read_text(encoding="utf-8")
        for match in HTML_STATIC_REF_PATTERN.finditer(source):
            target = _static_path(match.group(1))
            if target is not None:
                entrypoints.add(target.resolve())

    reachable_css = {
        path for path in entrypoints if path.suffix.lower() == ".css"
    }
    queue = list(reachable_css)
    while queue:
        current = queue.pop()
        if not current.is_file():
            continue
        for imported in _css_local_imports(current):
            if imported not in reachable_css:
                reachable_css.add(imported)
                queue.append(imported)

    reachable_js = {
        path for path in entrypoints if path.suffix.lower() == ".js"
    }
    for path in JS_PATHS:
        resolved = path.resolve()
        if resolved not in reachable_js:
            failures.append(f"{path}: JavaScript file is not loaded by any HTML entrypoint")
    for path in CSS_PATHS:
        resolved = path.resolve()
        if resolved not in reachable_css:
            failures.append(f"{path}: CSS file is unreachable from HTML/CSS entrypoints")

    _fail("Orphan browser assets found:", failures)
    print(
        f"Browser asset reachability OK: {len(reachable_js)} JS, "
        f"{len(reachable_css)} CSS assets reachable"
    )


def check_design_token_convergence() -> None:
    failures: list[str] = []
    if not TOKENS_PATH.is_file():
        _fail("Design token convergence failures:", [f"missing {TOKENS_PATH}"])

    token_source = TOKENS_PATH.read_text(encoding="utf-8")
    defined_tokens = set(TOKEN_DEFINITION_PATTERN.findall(token_source))
    missing_required = sorted(REQUIRED_SEMANTIC_TOKENS - defined_tokens)
    if missing_required:
        failures.append(f"{TOKENS_PATH}: missing semantic tokens {missing_required}")

    for path in [*CSS_PATHS, *JS_PATHS]:
        source = path.read_text(encoding="utf-8")
        if path != TOKENS_PATH and ":root" in source:
            failures.append(f"{path}: token definitions must live in {TOKENS_PATH}")
        for match in LEGACY_TOKEN_PATTERN.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            failures.append(f"{path}:{line}: legacy token {match.group()!r} is not allowed")
        for name in TOKEN_REFERENCE_PATTERN.findall(source):
            if name not in defined_tokens and name not in LOCAL_LAYOUT_TOKENS:
                failures.append(f"{path}: undefined design token {name!r}")

    _fail("Design token convergence failures:", failures)
    print(f"Design token convergence OK: {len(defined_tokens)} canonical tokens")


def _consume_js_string_literal(source: str, start: int) -> str | None:
    if start >= len(source) or source[start] not in {'"', "'", "`"}:
        return None
    quote = source[start]
    index = start + 1
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return source[start:index + 1]
        index += 1
    return None


def _trusted_inner_html_template(path: Path, literal: str) -> bool:
    # This editor helper receives only hard-coded section labels (Văn bản / Kiểu dáng / Nền).
    # Keep the exception narrow so any other interpolated innerHTML still fails the gate.
    if path.as_posix() != "app/static/js/editor.js":
        return False
    expressions = re.findall(r"\$\{([^{}]+)\}", literal)
    return (
        expressions == ["title"]
        and "section-caret" in literal
        and literal.lstrip().startswith("`<span>${title}</span>")
    )


def check_unsafe_html_sinks() -> None:
    failures: list[str] = []
    banned_patterns = (
        ("outerHTML assignment", re.compile(r"\.outerHTML\s*=")),
        ("innerHTML append", re.compile(r"\.innerHTML\s*\+=")),
        ("insertAdjacentHTML", re.compile(r"\.insertAdjacentHTML\s*\(")),
        ("srcdoc assignment", re.compile(r"\.srcdoc\s*=")),
        ("document.write", re.compile(r"\bdocument\.writeln?\s*\(")),
    )
    inner_assignment = re.compile(r"\.innerHTML\s*=(?!=)")

    for path in [*JS_PATHS, *HTML_PATHS]:
        source = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(source.splitlines(), 1):
            for label, pattern in banned_patterns:
                if pattern.search(line):
                    failures.append(f"{path}:{line_no}: unsafe browser sink: {label}")

        for match in inner_assignment.finditer(source):
            rhs_start = match.end()
            while rhs_start < len(source) and source[rhs_start].isspace():
                rhs_start += 1
            line_no = source.count("\n", 0, match.start()) + 1
            literal = _consume_js_string_literal(source, rhs_start)
            if literal is None:
                failures.append(
                    f"{path}:{line_no}: dynamic innerHTML expression is not allowed"
                )
                continue
            if literal.startswith("`") and "${" in literal:
                if not _trusted_inner_html_template(path, literal):
                    failures.append(
                        f"{path}:{line_no}: interpolated innerHTML template is not allowed"
                    )

    _fail("Unsafe browser HTML sinks found:", failures)
    print("Browser HTML sink check OK")


def check_browser_state_contracts() -> None:
    contracts = {
        Path("app/static/js/api.js"): (
            "async function flushExcludedRegionSaves(",
            "await window.flushExcludedRegionSaves(chapterId);",
            "state.persistedVersion < state.version",
        ),
        Path("app/static/js/editor-box-transform.js"): (
            "browser request alone cannot undo a server commit",
            "if (currentGen === geomGeneration)",
            "window.editorImageMetrics(img)",
            "metrics.offsetX + r.x1 * metrics.sx",
        ),
        Path("app/static/js/editor.js"): (
            "function editorImageMetrics(img)",
            "const rect = img.getBoundingClientRect();",
            "if (!point || !point.inside) return;",
        ),
        Path("app/static/js/review-workspace.js"): (
            "container.querySelectorAll(\".review-card\").forEach(captureMaskSnapshot);",
            "if (maskSnapshots.size > 0)",
        ),
        Path("app/static/js/review.js"): (
            "activeCard._reviewBusy = true",
            "chapterId !== currentChapterId",
        ),
    }
    failures: list[str] = []
    for path, markers in contracts.items():
        source = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in source:
                failures.append(f"{path}: missing browser state contract marker {marker!r}")
    _fail("Browser state persistence contract failures:", failures)
    print("Browser state persistence contracts OK")


def check_structural_glyphs() -> None:
    failures: list[str] = []
    for path in [*HTML_PATHS, *JS_PATHS]:
        source = path.read_text(encoding="utf-8")
        for match in STRUCTURAL_GLYPH_PATTERN.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            failures.append(
                f"{path}:{line}: structural glyph {match.group()!r} is not allowed; use the SVG icon system"
            )
    _fail("Structural icon glyphs found:", failures)
    print("Structural icon glyph check OK")


def check_workbench_shell_contract() -> None:
    failures: list[str] = []
    source = WORKBENCH_PATH.read_text(encoding="utf-8")
    for marker in (
        ".workbench-stage-grid, .translation-workspace-body",
        "grid-template-columns: var(--studio-rail-width) minmax(0, 1fr) var(--studio-inspector-width)",
        "@media (max-width: 1000px)",
    ):
        if marker not in source:
            failures.append(f"{WORKBENCH_PATH}: missing shell geometry marker {marker!r}")
    _fail("Workbench shell contract failures:", failures)
    print("Workbench shell geometry contract OK")


def main() -> None:
    check_markup_integrity()
    check_browser_asset_reachability()
    check_design_token_convergence()
    check_unsafe_html_sinks()
    check_browser_state_contracts()
    check_structural_glyphs()
    check_workbench_shell_contract()


if __name__ == "__main__":
    main()
