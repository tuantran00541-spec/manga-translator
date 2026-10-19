"""The fixed API a register(api) plugin gets, and the ctx its handlers get, instead of the raw session."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading
from typing import Callable
import uuid

from app.logging_config import logger

VERSION = 1


class State:
    """A plugin's own JSON store, kept across sessions."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def all(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str, default=None):
        return self.all().get(key, default)

    def set(self, key: str, value) -> None:
        with self._lock:
            data = self.all()
            data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            tmp.replace(self.path)


class Context:
    """What a plugin handler sees of a session: the workspace, gated actions and its events; no key, mode or registry."""

    def __init__(self, session, plugin: str):
        self._session = session
        self.plugin = plugin

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def root(self) -> Path:
        return self._session.workspace.root

    @property
    def model(self) -> str:
        return self._session.model

    @property
    def history(self) -> list[dict]:
        return json.loads(json.dumps(self._session.history, default=str))

    def read(self, path: str) -> str:
        """Read a file the sandbox lets the agent read."""
        return self._session.workspace.resolve(str(path)).read_text(encoding="utf-8", errors="replace")

    def call(self, tool: str, args: dict) -> tuple[str, bool]:
        """Call any tool as the model would: the same approval, hooks and sandbox apply."""
        return self._session._run_call({"id": f"plugin-{self.plugin}-{uuid.uuid4().hex[:8]}", "name": tool, "args": dict(args)})

    def run(self, command: str) -> tuple[str, bool]:
        return self.call("run_command", {"command": str(command)})

    def ask_user(self, question: str, options: list[str] | None = None) -> str:
        output, _ = self.call("ask_user", {"question": str(question), **({"options": list(options)} if options else {})})
        return output

    def emit(self, kind: str, **data) -> None:
        self._session.emit("plugin", plugin=self.plugin, name=str(kind)[:40], data=data)

    # For a plugin loop: one model turn, running its calls, and the default loop.

    def turn(self) -> dict:
        return self._session._turn()

    def run_calls(self, calls: list[dict]) -> None:
        self._session._run_calls(calls)

    def append(self, message: dict) -> None:
        self._session.history.append(dict(message))

    @property
    def stopped(self) -> bool:
        return bool(self._session._stop)

    def default_loop(self, max_steps: int) -> None:
        type(self._session)._steps(self._session, max_steps)


class API:
    """What register(api) receives: the slots one plugin fills, each handler called with a ctx."""

    version = VERSION

    def __init__(self, reg, name: str, config: dict | None = None, state_dir: Path | None = None):
        self._reg = reg
        self.name = name
        self.config = dict(config or {})
        self.state = State((state_dir or Path.home() / ".manga-agent" / "plugin-state") / f"{re.sub(r'[^A-Za-z0-9_-]', '_', name)}.json")

    def _wrap(self, fn: Callable) -> Callable:
        if not callable(fn):
            raise ValueError("a plugin handler must be a function")
        name = self.name
        return lambda session, *args: fn(Context(session, name), *args)

    def tool(self, spec: dict, handler: Callable, kind: str = "exec", always_ask: bool = False) -> None:
        """handler(ctx, args) returns the text the model reads; kind is read, edit, exec or net."""
        self._reg.tool(spec, self._wrap(handler), kind, always_ask=always_ask)

    def command(self, name: str, description: str, handler: Callable) -> None:
        """handler(ctx, args) for /name returns a message."""
        self._reg.command(name, description, self._wrap(handler))

    def on(self, event: str, fn: Callable) -> None:
        """pre_tool(ctx, call) returns a reason to block; post_tool(ctx, call, output) returns text to append; the rest only observe."""
        self._reg.hook(event, self._wrap(fn))

    def prompt(self, fn: Callable) -> None:
        """fn(ctx) returns text added to the system prompt."""
        self._reg.prompt(self._wrap(fn))

    def loop(self, name: str, fn: Callable) -> None:
        """fn(ctx, max_steps) replaces the step loop when profile.json sets "loop" to this name."""
        self._reg.loop(name, self._wrap(fn))

    def log(self, message: str) -> None:
        logger.info("Agent plugin {}: {}", self.name, str(message)[:2000])
