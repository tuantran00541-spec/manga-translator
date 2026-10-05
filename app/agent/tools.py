"""Tools the agent works with, all kept inside one workspace folder."""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re
import subprocess

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
# Chapter images, model weights and logs at the app's root would drown every listing and search.
ROOT_SKIP_DIRS = {"data", "models", "logs"}
MAX_READ_LINES = 2000
MAX_OUTPUT_CHARS = 20_000
MAX_SEARCH_HITS = 200
MAX_LIST_ENTRIES = 500
MAX_SEARCH_FILE_BYTES = 1_000_000
COMMAND_TIMEOUT = 120
MAX_COMMAND_TIMEOUT = 600

# Tools that change files or run programs wait for the user unless the session allows them.
EDIT_TOOLS = {"write_file", "edit_file"}
COMMAND_TOOLS = {"run_command"}

SPECS = [
    {"name": "list_dir", "description": "List files and folders under a path in the workspace.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Folder relative to the workspace root; '.' is the root."},
         "depth": {"type": "integer", "description": "How many folder levels to show, 1 to 4."}}}},
    {"name": "read_file", "description": "Read a text file with line numbers.",
     "parameters": {"type": "object", "required": ["path"], "properties": {
         "path": {"type": "string"},
         "offset": {"type": "integer", "description": "First line to read, from 1."},
         "limit": {"type": "integer", "description": "How many lines to read, at most 2000."}}}},
    {"name": "search", "description": "Search file contents with a regular expression; returns file:line: text.",
     "parameters": {"type": "object", "required": ["pattern"], "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"},
         "glob": {"type": "string", "description": "Only files whose name matches, such as *.py."},
         "ignore_case": {"type": "boolean"}}}},
    {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object", "required": ["path", "content"], "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "edit_file", "description": "Replace old_text with new_text in a file; old_text must match exactly once unless replace_all.",
     "parameters": {"type": "object", "required": ["path", "old_text", "new_text"], "properties": {
         "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
         "replace_all": {"type": "boolean"}}}},
    {"name": "run_command", "description": "Run a shell command in the workspace root and return its output and exit code.",
     "parameters": {"type": "object", "required": ["command"], "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": "Seconds before it is stopped, at most 600."}}}},
]


class ToolError(Exception):
    """A tool call the agent made wrongly; its message goes back to the model."""


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} characters cut] …\n{text[-half:]}"


class Workspace:
    """A folder the agent may read, change and run commands in."""

    def __init__(self, root: str | Path):
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Workspace folder does not exist: {root}")
        self.root = root

    def resolve(self, path: str | None) -> Path:
        target = (self.root / (path or ".")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"{path!r} is outside the workspace")
        return target

    def skipped(self, folder: Path) -> bool:
        return folder.name in SKIP_DIRS or (folder.parent == self.root and folder.name in ROOT_SKIP_DIRS)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool {name!r}")
        if not isinstance(args, dict):
            raise ToolError("Tool arguments must be a JSON object")
        try:
            return handler(**args)
        except TypeError as exc:
            raise ToolError(f"Bad arguments for {name}: {exc}") from exc

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
        return _clip(body + tail) if chunk else f"(file has {len(lines)} lines)"

    def _tool_search(self, pattern: str, path: str = ".", glob: str | None = None, ignore_case: bool = False) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"Bad regular expression: {exc}") from exc
        top = self.resolve(path)
        files = [top] if top.is_file() else []
        if top.is_dir():
            for folder, dirs, names in os.walk(top):
                dirs[:] = sorted(d for d in dirs if not self.skipped(Path(folder) / d))
                files.extend(Path(folder) / n for n in sorted(names) if not glob or fnmatch.fnmatch(n, glob))
        hits: list[str] = []
        for file in files:
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

    def _tool_write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Wrote {self.rel(target)} ({len(str(content).splitlines())} lines)"

    def _tool_edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        target = self.resolve(path)
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

    def _tool_run_command(self, command: str, timeout: int = COMMAND_TIMEOUT) -> str:
        if not str(command).strip():
            raise ToolError("command is empty")
        limit = max(1, min(MAX_COMMAND_TIMEOUT, int(timeout)))
        try:
            done = subprocess.run(str(command), shell=True, cwd=self.root, capture_output=True, text=True,
                                  errors="replace", timeout=limit, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") + (exc.stderr or "")
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
            return _clip(f"{out}\n[stopped after {limit} s]")
        output = (done.stdout or "") + (f"\n[stderr]\n{done.stderr}" if done.stderr else "")
        return _clip(f"{output.strip()}\n[exit code {done.returncode}]")
