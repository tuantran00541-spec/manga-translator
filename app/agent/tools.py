"""Tools the agent works with: files inside one workspace, sandboxed commands and web pages."""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re

from bs4 import BeautifulSoup

from app.agent import patch as patches, sandbox
from app.downloader.http import read_response_limited, safe_get

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
# Chapter images, model weights and logs at the app's root would drown every listing and search.
ROOT_SKIP_DIRS = {"data", "models", "logs"}
MAX_READ_LINES = 2000
MAX_OUTPUT_CHARS = 20_000
MAX_SEARCH_HITS = 200
MAX_LIST_ENTRIES = 500
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_FETCH_BYTES = 5_000_000
COMMAND_TIMEOUT = 120
MAX_COMMAND_TIMEOUT = 1800

# What each tool can do decides whether it waits for the user: read, edit, exec or net.
KIND = {"list_dir": "read", "read_file": "read", "search": "read", "glob": "read", "write_file": "edit",
        "edit_file": "edit", "apply_patch": "edit", "run_command": "exec", "web_fetch": "net"}

SPECS = [
    {"name": "list_dir", "description": "List files and folders under a path in the workspace.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Folder relative to the workspace root; '.' is the root."},
         "depth": {"type": "integer", "description": "How many folder levels to show, 1 to 4."}}}},
    {"name": "read_file", "description": "Read a text file with line numbers. Skill files may be read by their absolute path.",
     "parameters": {"type": "object", "required": ["path"], "properties": {
         "path": {"type": "string"},
         "offset": {"type": "integer", "description": "First line to read, from 1."},
         "limit": {"type": "integer", "description": "How many lines to read, at most 2000."}}}},
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
    {"name": "edit_file", "description": "Replace old_text with new_text in a file; old_text must match exactly once unless replace_all.",
     "parameters": {"type": "object", "required": ["path", "old_text", "new_text"], "properties": {
         "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
         "replace_all": {"type": "boolean"}}}},
    {"name": "apply_patch", "description": "Add, update, move or delete several files in one patch.\n" + patches.GUIDE,
     "parameters": {"type": "object", "required": ["patch"], "properties": {"patch": {"type": "string"}}}},
    {"name": "run_command", "description": "Run a shell command in the workspace root and return its output and exit code. "
                                           "It runs in the session's sandbox; set outside_sandbox only when the sandbox blocks "
                                           "something the task truly needs, and say why.",
     "parameters": {"type": "object", "required": ["command"], "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": "Seconds before it is stopped, at most 1800."},
         "outside_sandbox": {"type": "boolean"}}}},
    {"name": "web_fetch", "description": "Fetch a public web page and return its readable text.",
     "parameters": {"type": "object", "required": ["url"], "properties": {
         "url": {"type": "string"}, "max_chars": {"type": "integer"}}}},
]


class ToolError(Exception):
    """A tool call the agent made wrongly; its message goes back to the model."""


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
        self.root = root
        self.policy = policy or sandbox.Policy()
        self.read_roots = [Path(p).resolve() for p in read_roots or []]

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
        if KIND.get(name) == "edit" and self.policy.mode == "read-only":
            raise ToolError("The session is read-only; ask the user to allow edits")
        try:
            return handler(**args)
        except TypeError as exc:
            raise ToolError(f"Bad arguments for {name}: {exc}") from exc

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

    def _tool_read_file(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file")
        data = target.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError(f"{path!r} is a binary file")
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(offset))
        count = max(1, min(MAX_READ_LINES, int(limit)))
        chunk = lines[start - 1:start - 1 + count]
        body = "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(chunk))
        more = len(lines) - (start - 1 + len(chunk))
        tail = f"\n… {more} more lines" if more > 0 else ""
        return clip(body + tail) if chunk else f"(file has {len(lines)} lines)"

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
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Wrote {self.rel(target)} ({len(str(content).splitlines())} lines)"

    def _tool_edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        target = self.resolve(path, write=True)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file")
        text = target.read_text(encoding="utf-8")
        count = text.count(old_text) if old_text else 0
        if count == 0:
            raise ToolError("old_text was not found; read the file again and copy it exactly")
        if count > 1 and not replace_all:
            raise ToolError(f"old_text matches {count} places; add more surrounding lines or set replace_all")
        target.write_text(text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1),
                          encoding="utf-8")
        return f"Edited {self.rel(target)} ({count if replace_all else 1} change)"

    def _tool_apply_patch(self, patch: str) -> str:
        try:
            return patches.apply(patch, lambda p: self.resolve(p, write=True), self.rel)
        except patches.PatchError as exc:
            raise ToolError(str(exc)) from exc

    def _tool_run_command(self, command: str, timeout: int = COMMAND_TIMEOUT, outside_sandbox: bool = False) -> str:
        if not str(command).strip():
            raise ToolError("command is empty")
        limit = max(1, min(MAX_COMMAND_TIMEOUT, int(timeout)))
        policy = sandbox.Policy("full-access", True) if outside_sandbox else self.policy
        code, output = sandbox.run(str(command), policy, self.root, limit)
        status = f"[stopped after {limit} s]" if code is None else f"[exit code {code}]"
        return clip(f"{output.strip()}\n{status}")

    def _tool_web_fetch(self, url: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
        try:
            response = safe_get(str(url), timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0 manga-translator-agent"})
            body = read_response_limited(response, limit_bytes=MAX_FETCH_BYTES)
            kind = response.headers.get("Content-Type", "")
            response.close()
        except Exception as exc:
            raise ToolError(f"Could not fetch {url}: {getattr(exc, 'detail', exc)}") from exc
        text = body.decode(response.encoding or "utf-8", errors="replace")
        if "html" in kind or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            soup = BeautifulSoup(text, "lxml")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
        return clip(text.strip(), max(1000, min(100_000, int(max_chars))))
