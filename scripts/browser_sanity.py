from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

JS_PATHS = sorted(Path("app/static/js").rglob("*.js"))
HTML_PATHS = sorted(Path("app/templates").rglob("*.html"))
CSS_PATHS = sorted(Path("app/static/css").rglob("*.css"))
STATIC_ROOT = Path("app/static")


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


def _check_static_ref(source_path: Path, line: int, ref: str, failures: list[str]) -> None:
    clean = ref.split("?", 1)[0].split("#", 1)[0]
    if clean.startswith("/static/"):
        target = STATIC_ROOT / clean.removeprefix("/static/")
    elif clean.startswith("static/"):
        target = STATIC_ROOT / clean.removeprefix("static/")
    else:
        return
    if not target.is_file():
        failures.append(f"{source_path}:{line}: missing static asset {ref!r} -> {target}")


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

    import_pattern = re.compile(
        r"@import\s+url\(\s*(?:['\"])?([^'\")\s]+)(?:['\"])?\s*\)",
        re.IGNORECASE,
    )
    for path in CSS_PATHS:
        source = path.read_text(encoding="utf-8")
        for match in import_pattern.finditer(source):
            ref = match.group(1)
            if ref.startswith(("http://", "https://", "data:")):
                continue
            target = (path.parent / ref.split("?", 1)[0].split("#", 1)[0]).resolve()
            if not target.is_file():
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{path}:{line}: missing CSS import {ref!r}")

    _fail("Browser markup integrity failures:", failures)
    print(f"Browser markup integrity OK: {len(HTML_PATHS)} HTML, {len(CSS_PATHS)} CSS files")


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


def main() -> None:
    check_markup_integrity()
    check_unsafe_html_sinks()


if __name__ == "__main__":
    main()
