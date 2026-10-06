"""Tools the agent works with: files inside one workspace, sandboxed commands and web pages."""
from __future__ import annotations

import ast
import base64
import difflib
import fnmatch
import json
import os
from pathlib import Path
import re
import time
import unicodedata

import requests

from app.agent import gitguard, hashline, patch as patches, sandbox, services, webread, websearch
from app.downloader.http import read_response_limited, safe_get

BROAD_FOLDERS = {"/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/var", "/boot", "/dev", "/proc", "/sys", "/opt", "/root", "/home", "/Users",
                 "/System", "/Library", "/Applications", "/private", "/mnt", "/media"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
# Chapter images, model weights and logs at the app's root would drown every listing and search.
ROOT_SKIP_DIRS = {"data", "models", "logs"}
MAX_READ_LINES = 2000
MAX_OUTPUT_CHARS = 20_000
MAX_SEARCH_HITS = 200
MAX_LIST_ENTRIES = 500
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_FETCH_BYTES = 5_000_000
MAX_DOWNLOAD_BYTES = 50_000_000
MAX_IMAGE_BYTES = 4_000_000
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
SYMBOL_LINE = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:(?:public|private|protected|static|final|abstract|pub|override)\s+)*"
                         r"(def|class|function|func|fn|struct|enum|interface|trait|type|impl|module|object)\s+\*?([A-Za-z_$][\w$]*)")
COMMAND_TIMEOUT = 120
MAX_COMMAND_TIMEOUT = 1800

# What each tool can do decides whether it waits for the user: read, edit, exec or net.
KIND = {"list_dir": "read", "read_file": "read", "symbols": "read", "view_image": "read", "search": "read", "glob": "read", "write_file": "edit",
        "edit_file": "edit", "edit_lines": "edit", "apply_patch": "edit", "run_command": "exec", "run_script": "exec", "web_fetch": "net", "web_search": "net", "web_download": "edit"}

SPECS = [
    {"name": "list_dir", "description": "List files and folders under a path in the workspace.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Folder relative to the workspace root; '.' is the root."},
         "depth": {"type": "integer", "description": "How many folder levels to show, 1 to 4."}}}},
    {"name": "read_file", "description": "Read a text file with line numbers. Skill files may be read by their absolute path.",
     "parameters": {"type": "object", "required": ["path"], "properties": {
         "path": {"type": "string"},
         "offset": {"type": "integer", "description": "First line to read, from 1."},
         "limit": {"type": "integer", "description": "How many lines to read, at most 2000."},
         "anchors": {"type": "boolean", "description": "Show each line as N#hh|text so edit_lines can point at it."}}}},
    {"name": "symbols", "description": "Outline a file's classes and functions with line numbers, or find where a name is defined across the workspace.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "A file to outline, or a folder to search; '.' by default."},
         "name": {"type": "string", "description": "Find definitions whose name contains this text."}}}},
    {"name": "view_image", "description": "Look at an image file (png, jpg, gif, webp, up to 4 MB); it appears in your next turn.",
     "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}},
    {"name": "search", "description": "Search file contents with a regular expression; returns file:line: text.",
     "parameters": {"type": "object", "required": ["pattern"], "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"},
         "glob": {"type": "string", "description": "Only files whose name matches, such as *.py."},
         "ignore_case": {"type": "boolean"}}}},
    {"name": "glob", "description": "Find files whose path matches a pattern such as **/*.py or app/**/test_*.py, newest first.",
     "parameters": {"type": "object", "required": ["pattern"], "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"}}}},
    {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object", "required": ["path", "content"], "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "edit_file", "description": "Replace old_text with new_text in a file; old_text must match once unless replace_all. "
                                         "For several changes to one file pass edits, a list of {old_text, new_text}, applied in order and saved together. "
                                         "Whole lines that differ only in trailing spaces, quote style or indentation still match.",
     "parameters": {"type": "object", "required": ["path"], "properties": {
         "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
         "replace_all": {"type": "boolean"},
         "edits": {"type": "array", "items": {"type": "object", "required": ["old_text", "new_text"], "properties": {
             "old_text": {"type": "string"}, "new_text": {"type": "string"}}}}}}},
    {"name": "edit_lines", "description": hashline.GUIDE,
     "parameters": {"type": "object", "required": ["path", "edits"], "properties": {
         "path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "required": ["op", "anchor"], "properties": {
             "op": {"type": "string", "enum": list(hashline.OPS)}, "anchor": {"type": "string"},
             "end": {"type": "string"}, "text": {"type": "string"}}}}}}},
    {"name": "apply_patch", "description": "Add, update, move or delete several files in one patch.\n" + patches.GUIDE,
     "parameters": {"type": "object", "required": ["patch"], "properties": {"patch": {"type": "string"}}}},
    {"name": "run_command", "description": "Run a shell command in the workspace root and return its output and exit code. "
                                           "It runs in the session's sandbox; set outside_sandbox only when the sandbox blocks "
                                           "something the task truly needs, and say why.",
     "parameters": {"type": "object", "required": ["command"], "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": "Seconds before it is stopped, at most 1800."},
         "background": {"type": "boolean", "description": "Start it and return a job id at once (a dev server, a watcher); read it with job_output, stop it with job_stop. "
                                                           "A server needs the network switch on to listen on a port."},
         "tty": {"type": "boolean", "description": "With background: run it on a terminal, for programs that want one (python -i, interactive prompts); type into it with job_input."},
         "outside_sandbox": {"type": "boolean"}}}},
    {"name": "run_script", "description": "Run a Python script in the sandbox that can call read-only tools as functions and print what you need: "
                                          "tools.read_file(path=...), tools.search(pattern=...), tools.glob(pattern=...), tools.list_dir(path=...), "
                                          "tools.symbols(path=..., name=...), tools.web_fetch(url=...), tools.web_search(query=...). Each returns the tool's text "
                                          "and raises on error. Use it to look through many files or results in one step instead of many calls; only what the "
                                          "script prints comes back. tools.fan_out(jobs=[{\"prompt\": ...}], schema={...}) runs helper agents and returns a JSON list of {job, report} "
                                          "(report is null when a helper failed), so a script can fan out, check the reports, and run another round on what is left (at most 40 helpers per script; timeout up to 3600).",
     "parameters": {"type": "object", "required": ["code"], "properties": {
         "code": {"type": "string"}, "timeout": {"type": "integer", "description": "Seconds, at most 3600."}}}},
    {"name": "web_fetch", "description": "Fetch a public web page and return it as Markdown (headings, links, code kept); a GitHub file or repository page is read as raw text. "
                                         "A long page is cut with a note giving the offset to continue from; find= returns only the matching lines.",
     "parameters": {"type": "object", "required": ["url"], "properties": {
         "url": {"type": "string"}, "max_chars": {"type": "integer"}, "offset": {"type": "integer", "description": "Character to start from, to read on in a long page."},
         "find": {"type": "string", "description": "A word or regular expression; returns the matching lines with a little around each instead of the page."}}}},
    {"name": "web_search", "description": "Search the web; returns titles, links and snippets. Follow a link with web_fetch. The query may use site:, -site:, filetype:, \"phrases\" and -words. "
                                          "Write short plain queries of 3 to 6 words; if two searches miss, fetch an index page, the docs or an API URL instead. "
                                          "Prefer primary sources (official docs, the project's own repository) and cite the links you used.",
     "parameters": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string"}, "count": {"type": "integer", "description": "How many results, 1 to 15."},
         "recency": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "Only results from this long ago."}}}},
    {"name": "web_download", "description": "Download a public file (an archive, a PDF, an image, a dataset) into the workspace, up to 50 MB.",
     "parameters": {"type": "object", "required": ["url", "path"], "properties": {
         "url": {"type": "string"}, "path": {"type": "string", "description": "Where to save it, relative to the workspace root."}}}},
]


class ToolError(Exception):
    """A tool call the agent made wrongly; its message goes back to the model."""


LOOSE = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def _loose(line: str, level: int) -> str:
    line = unicodedata.normalize("NFKC", line).translate(LOOSE).rstrip()
    return line.strip() if level == 2 else line


def _replace(text: str, old: str, new: str, replace_all: bool, where: str = "") -> tuple[str, int]:
    """One replacement: exact first, then whole lines compared loosely (trailing spaces, quotes, then indentation), as Codex and pi do."""
    count = text.count(old) if old else 0
    if count == 1 or (count > 1 and replace_all):
        return text.replace(old, new), count
    if count > 1:
        raise ToolError(f"{where}old_text matches {count} places; add more surrounding lines or set replace_all")
    lines = text.splitlines(keepends=True)
    want = old.strip("\r\n").splitlines()
    for level in (1, 2):
        wanted = [_loose(line, level) for line in want]
        have = [_loose(line, level) for line in lines]
        hits = [i for i in range(len(lines) - len(wanted) + 1) if wanted and have[i:i + len(wanted)] == wanted]
        if len(hits) == 1:
            start, end = hits[0], hits[0] + len(wanted)
            ending = "\r\n" if lines[end - 1].endswith("\r\n") else "\n" if lines[end - 1].endswith("\n") else ""
            body = new.strip("\r\n")
            replacement = (body.replace("\n", "\r\n") if ending == "\r\n" else body) + (ending if body or not ending else "")
            return "".join(lines[:start]) + replacement + "".join(lines[end:]), 1
        if len(hits) > 1:
            raise ToolError(f"{where}old_text matches {len(hits)} places once spacing is ignored; add more surrounding lines")
    raise ToolError(f"{where}old_text was not found; read the file again and copy it exactly" + _closest(lines, want))


def _closest(lines: list[str], want: list[str]) -> str:
    """Where the file has text most like what the model was looking for, so it can copy the real lines."""
    if not want or len(lines) > 20_000:
        return ""
    size, target, best = len(want), "\n".join(line.strip() for line in want), (0.0, 0)
    for i in range(0, max(1, len(lines) - size + 1)):
        ratio = difflib.SequenceMatcher(None, target, "\n".join(line.strip() for line in lines[i:i + size])).quick_ratio()
        if ratio > best[0]:
            best = (ratio, i)
    if best[0] < 0.6:
        return ""
    start = best[1]
    snippet = "".join(lines[start:start + size]).rstrip("\n")
    return f". The closest text is at lines {start + 1}-{start + size}:\n{snippet[:1500]}"


def _checkers() -> dict:
    found = {".py": ast.parse, ".json": json.loads}
    try:
        import tomllib
        found[".toml"] = tomllib.loads
    except ImportError:
        pass
    try:
        import yaml
        found[".yaml"] = found[".yml"] = lambda text: list(yaml.safe_load_all(text))
    except ImportError:
        pass
    return found


CHECKERS = _checkers()


# Shell commands a project tool does better, each with the tip that follows its output (omp's bash interceptor, as advice instead of a refusal).
SHELL_HINTS = ((re.compile(r"^(cat|head|tail|less|more)\s+[^|<>;&]*$"), "read_file shows a file with line numbers and records that you read it, which edit_file and write_file rely on"),
               (re.compile(r"^(grep\s+(-\w*r|--recursive)|rg\s|ag\s|ack\s)"), "search looks through the project for a pattern, skips ignored and data folders and lists file:line"),
               (re.compile(r"^find\s.*-i?name\b"), "glob finds files by name pattern and skips ignored folders"),
               (re.compile(r"^(curl|wget)\s.*https?://"), "web_fetch reads a URL (HTML as Markdown, JSON pretty-printed), keeps the page for offset and find, and marks it as web content"),
               (re.compile(r"^(sed|perl)\s+(-\w*i|--in-place)"), "edit_file changes text and shows what changed; a file changed with sed -i must be read again before edit_file can change it"))
SHELL_HINT_LIMIT = 3
SEARCH_STREAK = 3  # Searches in a row before the result suggests reading instead.
PAGE_CACHE_S, PAGE_CACHE_SIZE = 600, 20  # A fetched page is kept this long for offset and find.


def clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} characters cut] …\n{text[-half:]}"


class Workspace:
    """A folder the agent may read and, as its sandbox policy allows, change and run commands in."""

    def __init__(self, root: str | Path, policy: sandbox.Policy | None = None, read_roots: list[Path] | None = None):
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace folder does not exist: {root}")
        if root == Path(root.anchor) or root == Path.home().resolve() or str(root) in BROAD_FOLDERS:
            raise ValueError(f"Pick a project folder, not {root}: the agent could read and change everything in it")
        self.root = root
        self.policy = policy or sandbox.Policy()
        self.read_roots = [Path(p).resolve() for p in read_roots or []]
        # Files the agent has read or written, with their modification time then; an edit over a newer one is refused.
        self.seen: dict[Path, int] = {}
        self.hints = 0
        self.searches_in_row = 0
        self.pages: dict[str, tuple[str, list[str], float]] = {}
        self.new_images: list[tuple[str, str]] = []
        # The providers behind web search, page downloads and commands; a session swaps in the user's choice.
        self.services = services.Services()

    def resolve(self, path: str | None, *, write: bool = False) -> Path:
        target = (self.root / (path or ".")).resolve()
        roots = [self.root] + ([] if write else self.read_roots)
        if not any(target == r or r in target.parents for r in roots):
            raise ToolError(f"{path!r} is outside the workspace")
        return target

    def skipped(self, folder: Path) -> bool:
        return folder.name in SKIP_DIRS or (folder.parent == self.root and folder.name in ROOT_SKIP_DIRS)

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix() or "."
        except ValueError:
            return str(path)

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool {name!r}")
        if not isinstance(args, dict):
            raise ToolError("Tool arguments must be a JSON object")
        args = self._coerce(name, args)
        if KIND.get(name) == "edit" and self.policy.mode == "read-only":
            raise ToolError("The session is read-only; ask the user to allow edits")
        try:
            output = handler(**args)
        except TypeError as exc:
            raise ToolError(f"Bad arguments for {name}: {exc}") from exc
        if KIND.get(name) == "edit":
            changed = self.targets(name, args)
            for path in changed:
                self._stamp(path)
            problems = self.diagnose(changed)
            if problems:
                output += f"\nSyntax check failed:\n{problems}"
        return output

    @staticmethod
    def _coerce(name: str, args: dict) -> dict:
        """Models often send an array, number or flag as a JSON string; read it as the type the tool declares."""
        spec = next((s for s in SPECS if s["name"] == name), None)
        props = (spec or {}).get("parameters", {}).get("properties", {})
        fixed = dict(args)
        for key, value in args.items():
            kind = (props.get(key) or {}).get("type")
            if isinstance(value, str) and kind in ("array", "object", "integer", "boolean"):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    continue
                if (kind == "array" and isinstance(parsed, list)) or (kind == "object" and isinstance(parsed, dict)) or (
                        kind == "integer" and isinstance(parsed, int) and not isinstance(parsed, bool)) or (kind == "boolean" and isinstance(parsed, bool)):
                    fixed[key] = parsed
        return fixed

    def targets(self, name: str, args: dict) -> list[Path]:
        """The files an edit tool call is about to change."""
        try:
            if name == "apply_patch":
                return [self.resolve(p, write=True) for h in patches.parse(str(args.get("patch") or "")) for p in (h.path, h.move_to) if p]
            if name in ("write_file", "edit_file", "edit_lines", "web_download"):
                return [self.resolve(args.get("path"), write=True)]
        except (ToolError, patches.PatchError):
            pass
        return []

    def _stamp(self, path: Path) -> None:
        try:
            self.seen[path] = path.stat().st_mtime_ns
        except OSError:
            self.seen.pop(path, None)

    def _fresh(self, target: Path, shown: str, overwrite: bool = False) -> None:
        """Refuse to change a file that was never read (when overwriting) or that changed after the agent last saw it."""
        if not target.is_file():
            return
        if target not in self.seen:
            if overwrite:
                raise ToolError(f"{shown} already exists; read_file it first, or use edit_file to change part of it")
            return
        if target.stat().st_mtime_ns != self.seen[target]:
            raise ToolError(f"{shown} changed since you last read it; read it again before changing it")

    def diagnose(self, paths: list[Path]) -> str:
        """Syntax errors in changed Python, JSON, TOML and YAML files, so the model sees them at once."""
        rows = []
        for path in paths:
            check = CHECKERS.get(path.suffix)
            if check is None or not path.is_file():
                continue
            try:
                check(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                rows.append(f"{self.rel(path)}:{exc.lineno}: {exc.msg}")
            except Exception as exc:
                rows.append(f"{self.rel(path)}: {' '.join(str(exc).split())[:240] or type(exc).__name__}")
        return "\n".join(rows)

    def _files(self, top: Path, name_glob: str | None = None):
        if top.is_file():
            yield top
            return
        for folder, dirs, names in os.walk(top):
            dirs[:] = sorted(d for d in dirs if not self.skipped(Path(folder) / d))
            for n in sorted(names):
                if not name_glob or fnmatch.fnmatch(n, name_glob):
                    yield Path(folder) / n

    def _tool_list_dir(self, path: str = ".", depth: int = 1) -> str:
        top = self.resolve(path)
        if not top.is_dir():
            raise ToolError(f"{path!r} is not a folder")
        depth = max(1, min(4, int(depth)))
        rows: list[str] = []

        def walk(folder: Path, level: int) -> None:
            for entry in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if len(rows) >= MAX_LIST_ENTRIES:
                    return
                if entry.is_dir() and self.skipped(entry):
                    continue
                rows.append(f"{'  ' * (level - 1)}{entry.name}{'/' if entry.is_dir() else ''}")
                if entry.is_dir() and not entry.is_symlink() and level < depth:
                    walk(entry, level + 1)

        walk(top, 1)
        if len(rows) >= MAX_LIST_ENTRIES:
            rows.append(f"… stopped at {MAX_LIST_ENTRIES} entries")
        return "\n".join(rows) or "(empty folder)"

    def _tool_read_file(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES, anchors: bool = False) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file")
        data = target.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError(f"{path!r} is a binary file")
        self._stamp(target)
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(offset))
        count = max(1, min(MAX_READ_LINES, int(limit)))
        chunk = lines[start - 1:start - 1 + count]
        body = hashline.render(chunk, start) if anchors else "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(chunk))
        more = len(lines) - (start - 1 + len(chunk))
        tail = f"\n… {more} more lines" if more > 0 else ""
        return clip(body + tail) if chunk else f"(file has {len(lines)} lines)"

    def _tool_symbols(self, path: str = ".", name: str | None = None) -> str:
        top = self.resolve(path)
        if top.is_file() and not name:
            rows = self._outline(top)
            return "\n".join(rows) or "No definitions found"
        needle = (name or "").lower()
        hits: list[str] = []
        for file in self._files(top):
            if file.suffix in IMAGE_TYPES or file.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            for row in self._outline(file, flat=True):
                if not needle or needle in row.split(" ", 2)[-1].lower():
                    hits.append(f"{self.rel(file)}:{row}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join(hits) + f"\n… stopped at {MAX_SEARCH_HITS} matches"
        return "\n".join(hits) or "No definitions found"

    def _outline(self, file: Path, flat: bool = False) -> list[str]:
        """Definitions in a file as 'line kind name'; indented by nesting unless flat."""
        try:
            data = file.read_bytes()
        except OSError:
            return []
        if b"\0" in data[:8192]:
            return []
        source = data.decode("utf-8", errors="replace")
        rows: list[str] = []
        if file.suffix == ".py":
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
            if tree is not None:
                def walk(node: ast.AST, prefix: str, level: int) -> None:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            kind = "class" if isinstance(child, ast.ClassDef) else "def"
                            rows.append(f"{child.lineno}: {kind} {prefix}{child.name}" if flat else f"{child.lineno}: {'  ' * level}{kind} {child.name}")
                            walk(child, f"{prefix}{child.name}.", level + 1)
                        else:
                            walk(child, prefix, level)
                walk(tree, "", 0)
                return rows
        for number, line in enumerate(source.splitlines(), 1):
            found = SYMBOL_LINE.match(line)
            if found:
                indent = 0 if flat else (len(line) - len(line.lstrip())) // 2
                rows.append(f"{number}: {'  ' * indent}{found.group(1)} {found.group(2)}")
        return rows

    def _tool_view_image(self, path: str) -> str:
        target = self.resolve(path)
        kind = IMAGE_TYPES.get(target.suffix.lower())
        if not target.is_file() or kind is None:
            raise ToolError(f"{path!r} is not a png, jpg, gif or webp image")
        if target.stat().st_size > MAX_IMAGE_BYTES:
            raise ToolError(f"{path!r} is larger than {MAX_IMAGE_BYTES // 1_000_000} MB; make a smaller copy with a command first")
        self.new_images.append((self.rel(target), f"data:{kind};base64,{base64.b64encode(target.read_bytes()).decode()}"))
        return f"Image {self.rel(target)} loaded; it is shown to you in the next message."

    def _tool_search(self, pattern: str, path: str = ".", glob: str | None = None, ignore_case: bool = False) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"Bad regular expression: {exc}") from exc
        hits: list[str] = []
        for file in self._files(self.resolve(path), glob):
            try:
                if file.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{self.rel(file)}:{number}: {line.strip()[:300]}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join(hits) + f"\n… stopped at {MAX_SEARCH_HITS} matches"
        return "\n".join(hits) or "No matches"

    def _tool_glob(self, pattern: str, path: str = ".") -> str:
        top = self.resolve(path)
        pattern = pattern.lstrip("./")
        matched = [f for f in self._files(top) if fnmatch.fnmatch(f.relative_to(top).as_posix(), pattern)
                   or ("/" not in pattern and fnmatch.fnmatch(f.name, pattern))
                   or (pattern.startswith("**/") and fnmatch.fnmatch(f.relative_to(top).as_posix(), pattern[3:]))]
        matched.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)
        rows = [self.rel(f) for f in matched[:MAX_LIST_ENTRIES]]
        return "\n".join(rows) + (f"\n… {len(matched) - len(rows)} more" if len(matched) > len(rows) else "") if rows else "No files"

    def _tool_write_file(self, path: str, content: str) -> str:
        target = self.resolve(path, write=True)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder")
        self._fresh(target, path, overwrite=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Wrote {self.rel(target)} ({len(str(content).splitlines())} lines)"

    def _tool_edit_file(self, path: str, old_text: str | None = None, new_text: str | None = None, replace_all: bool = False,
                        edits: list | None = None) -> str:
        target = self.resolve(path, write=True)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file")
        self._fresh(target, path)
        if not edits and old_text is not None:
            edits = None
        if edits is None:
            if old_text is None or new_text is None:
                raise ToolError("give old_text and new_text, or edits")
            edits = [{"old_text": old_text, "new_text": new_text}]
        if not isinstance(edits, list) or not edits:
            raise ToolError("edits must be a non-empty list of {old_text, new_text}")
        text, changes = target.read_text(encoding="utf-8"), 0
        for number, edit in enumerate(edits, 1):
            where = f"edit {number}: " if len(edits) > 1 else ""
            if not isinstance(edit, dict) or "old_text" not in edit or "new_text" not in edit:
                raise ToolError(f"{where}each edit needs old_text and new_text")
            text, count = _replace(text, str(edit["old_text"]), str(edit["new_text"]), bool(replace_all), where)
            changes += count
        target.write_text(text, encoding="utf-8")
        return f"Edited {self.rel(target)} ({changes} change{'s' if changes != 1 else ''})"

    def _tool_edit_lines(self, path: str, edits: list) -> str:
        target = self.resolve(path, write=True)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file")
        self._fresh(target, path)
        try:
            target.write_text(hashline.apply(target.read_bytes().decode("utf-8"), edits), encoding="utf-8", newline="")
        except hashline.HashlineError as exc:
            raise ToolError(str(exc)) from exc
        return f"Edited {self.rel(target)} ({len(edits)} edits)"

    def _tool_apply_patch(self, patch: str) -> str:
        for path in self.targets("apply_patch", {"patch": patch}):
            self._fresh(path, self.rel(path))
        try:
            return patches.apply(patch, lambda p: self.resolve(p, write=True), self.rel)
        except patches.PatchError as exc:
            raise ToolError(str(exc)) from exc

    def _tool_run_command(self, command: str, timeout: int = COMMAND_TIMEOUT, outside_sandbox: bool = False, background: bool = False, tty: bool = False) -> str:
        if not str(command).strip():
            raise ToolError("command is empty")
        limit = max(1, min(MAX_COMMAND_TIMEOUT, int(timeout)))
        policy = sandbox.Policy("full-access", True) if outside_sandbox else self.policy
        guard = gitguard.snapshot(self.root)
        if self.services.is_local_shell():
            code, output = sandbox.run(str(command), policy, self.root, limit)
        else:
            job, end = self.services.shell(str(command), policy, self.root), time.time() + limit
            while job.code is None and time.time() < end:
                time.sleep(0.05)
            code, output = job.code, job.read()
            job.stop()
        status = f"[stopped after {limit} s]" if code is None else f"[exit code {code}]"
        undone = gitguard.restore(self.root, guard)
        warning = f"\n[blocked: the command changed {', '.join(undone)}; git hooks and config run outside the sandbox, so they were put back]" if undone else ""
        return clip(f"{output.strip()}\n{status}{warning}{self._shell_hint(str(command))}", 400_000)

    def _shell_hint(self, command: str) -> str:
        bare = re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", command).strip()
        tip = next((tip for pattern, tip in SHELL_HINTS if pattern.search(bare)), "")
        if not tip or self.hints >= SHELL_HINT_LIMIT:
            return ""
        self.hints += 1
        return f"\n[Tip: {tip}.]"

    def _tool_web_download(self, url: str, path: str) -> str:
        target = self.resolve(path, write=True)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder")
        try:
            response = safe_get(str(url), timeout=(10, 60), headers={"User-Agent": "Mozilla/5.0 manga-translator-agent"})
            body = read_response_limited(response, limit_bytes=MAX_DOWNLOAD_BYTES)
            response.close()
        except Exception as exc:
            raise ToolError(f"Could not download {url}: {getattr(exc, 'detail', exc)}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return f"Saved {self.rel(target)} ({len(body)} bytes)"

    def _tool_web_fetch(self, url: str, max_chars: int = MAX_OUTPUT_CHARS, offset: int = 0, find: str = "") -> str:
        self.searches_in_row = 0
        url = str(url)
        page = self.pages.get(url)
        if page is None or time.time() - page[2] > PAGE_CACHE_S:
            text, notes = self._read_page(url)
            self.pages[url] = (text, notes, time.time())
            while len(self.pages) > PAGE_CACHE_SIZE:
                self.pages.pop(next(iter(self.pages)))
        else:
            # Reading on in a page (offset, find) does not download it again.
            text, notes = page[0], page[1]
        head = "".join(f"[{note}]\n" for note in notes)
        if find:
            return head + webread.find(text, str(find))
        limit, start = max(1000, min(100_000, int(max_chars))), max(0, int(offset))
        more = len(text) - start - limit
        footer = f"\n[{more} more characters; call again with offset={start + limit}, or find=... to search the page]" if more > 0 else ""
        return head + text[start:start + limit] + footer

    def _read_page(self, url: str) -> tuple[str, list[str]]:
        raw = webread.rewrite(url)
        try:
            body, kind, encoding = self._download(raw)
        except ToolError:
            if raw == url:
                raise
            # A GitHub page that has no raw file (a missing README) is read as the page itself.
            raw = url
            body, kind, encoding = self._download(url)
        text, notes = webread.readable(body, kind, encoding, raw)
        text = text.strip()
        key = os.environ.get("TINYFISH_API_KEY")
        if notes and key:
            # Only a page that needs JavaScript or is all menus goes to the browser service; a page that refused us is not sent there.
            try:
                seen = webread.browsed(url, key)
            except requests.RequestException:
                seen = ""
            if seen and not webread.low_quality(seen) and len(seen) > len(text):
                text, notes = seen, ["This page needs JavaScript; it was read through TinyFish's browser."]
        return text, notes

    def _download(self, url: str) -> tuple[bytes, str, str | None]:
        fetch = self.services.fetcher()
        if fetch is not None:
            try:
                return fetch(url)
            except Exception as exc:
                raise ToolError(f"Could not fetch {url}: {exc}") from exc
        try:
            response = safe_get(url, timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0 manga-translator-agent", "Accept": "text/html,application/xhtml+xml;q=0.9,text/markdown;q=0.5,*/*;q=0.3"})
            body = read_response_limited(response, limit_bytes=MAX_FETCH_BYTES)
            kind = response.headers.get("Content-Type", "")
            response.close()
        except Exception as exc:
            raise ToolError(f"Could not fetch {url}: {getattr(exc, 'detail', exc)}") from exc
        return body, kind, response.encoding

    def _tool_web_search(self, query: str, count: int = 8, recency: str = "") -> str:
        self.searches_in_row += 1
        try:
            found = self.services.search(query, count, recency)
        except websearch.SearchError as exc:
            raise ToolError(str(exc)) from exc
        if self.searches_in_row >= SEARCH_STREAK:
            # Searching again with longer queries rarely helps; reading the best lead does (seen on Qwen: eight searches before one fetch).
            found += (f"\n\n[Tip: this is search {self.searches_in_row} in a row. Open the most promising result with web_fetch, or fetch a page that "
                      "lists what you need (a project's docs, an index page such as https://github.com/trending/python, or an API URL), instead of "
                      "searching again. Short plain queries of 3 to 6 words work best; quotes and stacked terms often return nothing.]")
        return found
