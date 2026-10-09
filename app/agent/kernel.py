"""A small plugin kernel after Cordis (DeepSeek Harness): plugins inject services, register reversible effects and chain waterfall events."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Callable

from loguru import logger

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass
class Row:
    """One mounted plugin: its id, the object with apply(ctx, config), and what it left in the context."""
    id: str
    plugin: Any
    config: dict = field(default_factory=dict)
    disabled: bool = False
    source: str = "builtin"
    state: str = "waiting"
    error: str = ""
    effects: list[Callable] = field(default_factory=list)


class Context:
    """Services by name, event listeners and the plugin tree; whatever a plugin registers is undone when it stops."""

    def __init__(self):
        self.services: dict[str, tuple[str, Any]] = {}
        self.listeners: dict[str, list[tuple[str, Callable]]] = {}
        self.rows: dict[str, Row] = {}
        self._owner = ""

    # Effects: each is recorded against the plugin being applied, and undone in reverse when it stops.

    def effect(self, undo: Callable) -> None:
        if self._owner:
            self.rows[self._owner].effects.append(undo)

    def provide(self, name: str, value: Any) -> None:
        holder = self.services.get(name)
        if holder and holder[0] != self._owner:
            raise ValueError(f"service {name} is already provided by {holder[0] or 'the core'}")
        self.services[name] = (self._owner, value)
        self.effect(lambda: self.services.pop(name, None))

    def get(self, name: str, default: Any = None) -> Any:
        return self.services[name][1] if name in self.services else default

    def __contains__(self, name: str) -> bool:
        return name in self.services

    def on(self, event: str, fn: Callable) -> None:
        """A waterfall listener is fn(payload, next) and returns next(payload) or its own answer; a broadcast one is fn(payload)."""
        entry = (self._owner, fn)
        self.listeners.setdefault(event, []).append(entry)
        self.effect(lambda: self.listeners.get(event, []).remove(entry) if entry in self.listeners.get(event, []) else None)

    def waterfall(self, event: str, payload: Any, last: Callable) -> Any:
        """Pass payload through every listener in order; the last step is the core behavior."""
        chain = [fn for _, fn in self.listeners.get(event, [])]

        def step(index: int) -> Callable:
            if index == len(chain):
                return last
            def run(value):
                try:
                    return chain[index](value, step(index + 1))
                except Exception as exc:
                    # A failing plugin must not break the chain (fail-closed: the payload passes
                    # through unchanged, and the failure is logged).
                    logger.warning("Plugin waterfall for {} failed: {}", event, exc)
                    return step(index + 1)(value)
            return run
        return step(0)(payload)

    def emit(self, event: str, payload: Any) -> None:
        for _, fn in list(self.listeners.get(event, [])):
            try:
                fn(payload)
            except Exception as exc:
                logger.warning("Plugin listener for {} failed: {}", event, exc)

    def subscribers(self, event: str) -> int:
        return len(self.listeners.get(event, []))

    # The plugin tree.

    def mount(self, row: Row) -> None:
        if row.id in self.rows:
            self.unmount(row.id)
        self.rows[row.id] = row
        self.settle()

    def unmount(self, row_id: str) -> None:
        row = self.rows.pop(row_id, None)
        if row is not None:
            self._stop(row)
            self.settle()

    def set_disabled(self, row_id: str, disabled: bool) -> None:
        row = self.rows[row_id]
        row.disabled = disabled
        if not disabled and row.state in ("disabled", "failed"):
            row.state = "waiting"
        self.settle()

    def settle(self) -> None:
        """Start every plugin whose injected services exist and stop every one that lost them, until nothing changes."""
        changed = True
        while changed:
            changed = False
            for row in list(self.rows.values()):
                if row.disabled:
                    if row.state == "active":
                        self._stop(row)
                        changed = True
                    row.state = "disabled"
                    continue
                need = [n for n in getattr(row.plugin, "inject", ()) or () if n not in self.services]
                if row.state == "active" and need:
                    self._stop(row)
                    row.state, row.error, changed = "waiting", f"waiting for {', '.join(need)}", True
                elif row.state == "waiting":
                    if need:
                        row.error = f"waiting for {', '.join(need)}"
                    else:
                        self._apply(row)
                        changed = True

    def _apply(self, row: Row) -> None:
        self._owner = row.id
        try:
            config = {**(getattr(row.plugin, "defaults", None) or {}), **row.config}
            dispose = row.plugin.apply(self, config)
            if callable(dispose):
                row.effects.append(dispose)
            row.state, row.error = "active", ""
        except Exception as exc:
            self._owner = ""
            self._stop(row)
            row.state, row.error = "failed", f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self._owner = ""

    def _stop(self, row: Row) -> None:
        for undo in reversed(row.effects):
            try:
                undo()
            except Exception as exc:
                logger.warning("Plugin {} could not undo an effect: {}", row.id, exc)
        row.effects.clear()
        if row.state == "active":
            row.state = "waiting"

    def close(self) -> None:
        for row in list(self.rows.values()):
            self._stop(row)

    def tree(self) -> list[str]:
        rows = []
        for row in self.rows.values():
            needs = ", ".join(getattr(row.plugin, "inject", ()) or ())
            rows.append(f"{row.id} [{row.source}] {row.state}" + (f" (cần {needs})" if needs else "") + (f" — {row.error}" if row.error and row.state != "active" else ""))
        return rows


class Plugin:
    """A built-in plugin made from a function, with the services it needs."""

    def __init__(self, apply: Callable, inject: tuple[str, ...] = (), defaults: dict | None = None):
        self.apply, self.inject, self.defaults = apply, inject, defaults or {}


def provider(name: str, value: Any) -> Plugin:
    return Plugin(lambda ctx, config: ctx.provide(name, value))


def load_module(path: Path, label: str, source: bytes | None = None):
    """Run a plugin file from its source every time, so a reload never picks up a stale cached .pyc.

    source pins the exact bytes to exec (M25): when the caller already read the file for a trust
    digest, passing those bytes closes the read-then-exec TOCTOU window."""
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec)
    data = source if source is not None else path.read_bytes()
    exec(compile(data.decode("utf-8"), str(path), "exec"), module.__dict__)
    return module


def patch_rows(ctx: Context, home: Path) -> list[str]:
    """Apply ~/.manga-agent/plugins.json: a row with a known id changes its config or disables it, a new row mounts a plugin file."""
    path, problems = home / ".manga-agent" / "plugins.json", []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return problems
    except (OSError, ValueError) as exc:
        return [f"plugins.json: {exc}"]
    folder = (home / ".manga-agent" / "plugins").resolve()
    for entry in (data.get("rows") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict) or not ID_RE.match(str(entry.get("id", "")).lower()):
            problems.append(f"plugins.json: bad row {entry!r}"[:200])
            continue
        if isinstance(entry.get("mcp"), dict):
            continue  # an MCP server row; the session mounts it as mcp:ID
        row_id, config = entry["id"], entry.get("config") if isinstance(entry.get("config"), dict) else None
        if row_id in ctx.rows and not entry.get("plugin"):
            row = ctx.rows[row_id]
            row.config = config if config is not None else row.config
            row.disabled = bool(entry.get("disabled", row.disabled))
            continue
        if row_id in ctx.rows and ctx.rows[row_id].source == "builtin":
            # Same rule as plugin_write (H12): a plugins.json edit must never silently swap the module
            # behind a built-in row like shell or web-fetch; switch it off instead.
            problems.append(f"plugins.json: {row_id} is a built-in row and cannot be replaced by a plugin file")
            continue
        target = (folder / str(entry.get("plugin", ""))).resolve()
        if folder not in target.parents or target.suffix != ".py" or not target.is_file():
            problems.append(f"plugins.json: {row_id} needs a .py file inside ~/.manga-agent/plugins")
            continue
        try:
            module = load_module(target, f"manga_agent_row_{row_id.replace('-', '_').replace('.', '_')}")
        except Exception as exc:
            problems.append(f"{row_id}: {type(exc).__name__}: {exc}"[:300])
            continue
        ctx.rows[row_id] = Row(row_id, module, config or {}, bool(entry.get("disabled")), "user")
    return problems
