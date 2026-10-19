"""The agent's four tools: read, write and edit files in its folder, and run a command there."""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

MAX_OUTPUT = 30_000  # characters of a result the model sees; the middle of a longer one is cut
MAX_READ_BYTES = 5_000_000
READ_LINES = 2000
COMMAND_TIMEOUT_S = 120
MAX_COMMAND_TIMEOUT_S = 1800
# Commands never see the app's keys or the user's tokens.
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
CHANGES = {"write_file", "edit_file", "run_command"}

SPECS = [
    {"name": "read_file", "description": "Read a text file in the folder, with line numbers.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Path relative to the folder."},
         "offset": {"type": "integer", "description": "First line to read, from 1."},
         "limit": {"type": "integer", "description": f"Lines to read, {READ_LINES} by default."}},
         "required": ["path"]}},
    {"name": "write_file", "description": "Create a file, or replace all of it, with the given content.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace one exact piece of a file; old_text must appear exactly once.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
         "required": ["path", "old_text", "new_text"]}},
    {"name": "run_command", "description": "Run a shell command in the folder and return its output and exit code. "
     "Use it to list and search files, run tests and use any installed tool.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": f"Seconds, {COMMAND_TIMEOUT_S} by default."}},
         "required": ["command"]}},
]


class ToolError(Exception):
    """A tool call that cannot run as asked; its message goes back to the model."""


def clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return f"{text[:half]}\n… [{len(text) - MAX_OUTPUT} characters cut] …\n{text[-half:]}"


class Workspace:
    """The folder the agent works in; no tool reaches outside it."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a folder: {root}")

    def path(self, value: str) -> Path:
        target = (self.root / str(value or "")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"{value} is outside the folder {self.root}")
        return target

    def read_file(self, path: str, offset: int = 1, limit: int = READ_LINES) -> str:
        target = self.path(path)
        if not target.is_file():
            raise ToolError(f"No such file: {path}")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"{path} is over {MAX_READ_BYTES // 1_000_000} MB; read parts of it with run_command")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(offset or 1))
        shown = lines[start - 1:start - 1 + max(1, int(limit or READ_LINES))]
        body = "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(shown))
        rest = len(lines) - (start - 1 + len(shown))
        return clip(body + (f"\n… {rest} more lines; read on with offset={start + len(shown)}" if rest > 0 else ""))

    def write_file(self, path: str, content: str) -> str:
        target = self.path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8", newline="")
        return f"Wrote {path} ({len(str(content))} characters)"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        target = self.path(path)
        if not target.is_file():
            raise ToolError(f"No such file: {path}")
        with target.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
        count = text.count(old_text) if old_text else 0
        if count != 1:
            raise ToolError(f"old_text appears {count} times in {path}; it must appear exactly once")
        target.write_text(text.replace(old_text, str(new_text), 1), encoding="utf-8", newline="")
        return f"Edited {path}"

    def run_command(self, command: str, timeout: int = COMMAND_TIMEOUT_S) -> str:
        seconds = min(MAX_COMMAND_TIMEOUT_S, max(1, int(timeout or COMMAND_TIMEOUT_S)))
        env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
        if sys.platform == "win32":
            argv, extra = ["powershell", "-NoProfile", "-Command", command], {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            argv, extra = [shutil.which("bash") or "/bin/sh", "-c", command], {"start_new_session": True}
        process = subprocess.Popen(argv, cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **extra)
        try:
            output, _ = process.communicate(timeout=seconds)
        except subprocess.TimeoutExpired:
            _kill(process)
            output, _ = process.communicate()
            return clip(f"{output}\n[stopped after {seconds} s]")
        return clip(f"{output}\n[exit code {process.returncode}]")

    def run(self, name: str, args: dict) -> str:
        """One tool call's result, or the error that stopped it, as text for the model."""
        tool = getattr(self, name, None) if name in {s["name"] for s in SPECS} else None
        if tool is None:
            return f"Error: there is no tool named {name!r}"
        try:
            return tool(**args)
        except TypeError as exc:
            return f"Error: wrong arguments for {name}: {exc}"
        except (ToolError, OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"


def _kill(process: subprocess.Popen) -> None:
    """Stop a command and everything it started."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()
