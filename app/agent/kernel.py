"""A small plugin kernel after Cordis (DeepSeek Harness): plugins inject services, register reversible effects and chain waterfall events."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
            return last if index == len(chain) else (lambda value: chain[index](value, step(index + 1)))
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


def load_module_bytes(source: bytes, label: str, filename: str = "<plugin>"):
    """Execute plugin source from already-read bytes; the file is never re-read (H8).

    The trust digest is computed over these exact bytes, so what was pinned is
    what runs — a writer swapping the file between the digest and the load
    cannot slip untrusted code under a trusted digest.
    """
    spec = importlib.util.spec_from_file_location(label, filename)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def load_module(path: Path, label: str):
    """Run a plugin file from its source every time, so a reload never picks up a stale cached .pyc."""
    return load_module_bytes(path.read_bytes(), label, str(path))


def _plugins_manifest(home: Path) -> tuple[bytes | None, list[dict], list[str]]:
    """Read ~/.manga-agent/plugins.json and every .py row it mounts, once each.

    Returns (raw, rows, problems): raw is the plugins.json bytes (None when the
    file is absent), rows is one dict per row with id/config/disabled/plugin /
    target/src (src is the .py bytes, None when unreadable), problems lists
    manifest errors. Bytes are read exactly once; the trust digest and the
    executed code come from these same bytes.
    """
    path = home / ".manga-agent" / "plugins.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, [], []
    except OSError as exc:
        return None, [], [f"plugins.json: {exc}"]
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return raw, [], [f"plugins.json: {exc}"]
    folder = (home / ".manga-agent" / "plugins").resolve()
    rows: list[dict] = []
    problems: list[str] = []
    for entry in (data.get("rows") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict) or not ID_RE.match(str(entry.get("id", "")).lower()):
            problems.append(f"plugins.json: bad row {entry!r}"[:200])
            continue
        if isinstance(entry.get("mcp"), dict):
            continue  # an MCP server row; the session mounts it as mcp:ID
        item: dict = {
            "id": entry["id"],
            "config": entry.get("config") if isinstance(entry.get("config"), dict) else None,
            "disabled": entry.get("disabled"),
            "plugin": None,
            "target": None,
            "src": None,
            "problem": None,
        }
        plugin_name = entry.get("plugin")
        if plugin_name:
            item["plugin"] = str(plugin_name)
            target = (folder / str(plugin_name)).resolve()
            if folder not in target.parents or target.suffix != ".py" or not target.is_file():
                item["problem"] = f"plugins.json: {item['id']} needs a .py file inside ~/.manga-agent/plugins"
            else:
                try:
                    item["src"] = target.read_bytes()
                except OSError as exc:
                    item["problem"] = f"plugins.json: {item['id']}: cannot read {plugin_name}: {exc}"
                item["target"] = target
        rows.append(item)
    return raw, rows, problems


def _manifest_digest(raw: bytes, rows: list[dict]) -> str:
    """Digest of the plugins.json bytes plus every .py row it mounts.

    The raw bytes pin the rows, their configs and disabled flags; the per-row
    sources pin the code itself, so editing a .py without touching plugins.json
    also invalidates the pin.
    """
    h = hashlib.sha256()
    h.update(b"plugins.json\0" + raw + b"\0")
    for item in rows:
        h.update(str(item["id"]).encode() + b"\0" + (item["plugin"] or "").encode() + b"\0")
        h.update(item["src"] if item["src"] is not None else b"<unreadable>\0")
    return h.hexdigest()


def plugins_digest(home: Path) -> str | None:
    """The pin for ~/.manga-agent/plugins.json, or None when there is no file to pin.

    Mirrors the workspace-plugin trust flow: the session requires
    trust.trusted(workspace, "plugins.json", digest) before mounting any row,
    and trust_plugins() allow()s the digest explicitly (default-deny).
    """
    raw, rows, _ = _plugins_manifest(home)
    if raw is None:
        return None
    return _manifest_digest(raw, rows)


def patch_rows(ctx: Context, home: Path, trust=None, workspace: Path | None = None) -> list[str]:
    """Apply ~/.manga-agent/plugins.json: a row with a known id changes its config or disables it, a new row mounts a plugin file.

    Every row mounts only when the manifest digest is pinned in the trust
    store (C4) — the same trust flow as workspace plugins: trust.trusted()
    for the check, an explicit allow() (e.g. via trust_plugins()) to pin. An
    unpinned or changed plugins.json refuses to mount any row, loudly: the
    refusal is returned in problems (surfaced by /plugins) and logged.
    """
    raw, rows, problems = _plugins_manifest(home)
    if raw is None:
        return problems
    digest = _manifest_digest(raw, rows)
    if trust is None or workspace is None or not trust.trusted(workspace, "plugins.json", digest):
        msg = ("plugins.json is not pinned in the trust store: refusing to mount its rows "
               "(review ~/.manga-agent/plugins.json, then pin it via trust_plugins()).")
        problems.append(msg)
        logger.warning("Refusing to mount plugins.json rows: digest {} is not trusted", digest[:16])
        # Switching a row off only tightens, so it holds even while the file is not pinned.
        for item in rows:
            if item["id"] in ctx.rows and not item["plugin"] and item["disabled"] is True:
                ctx.rows[item["id"]].disabled = True
        return problems
    for item in rows:
        if item["problem"]:
            problems.append(item["problem"])
            continue
        row_id = item["id"]
        if row_id in ctx.rows and not item["plugin"]:
            row = ctx.rows[row_id]
            row.config = item["config"] if item["config"] is not None else row.config
            if item["disabled"] is not None:
                row.disabled = bool(item["disabled"])
            continue
        if item["src"] is None:
            problems.append(item["problem"] or f"plugins.json: {row_id}: no code to mount")
            continue
        try:
            module = load_module_bytes(item["src"], f"manga_agent_row_{row_id.replace('-', '_').replace('.', '_')}",
                                       str(item["target"]))
        except Exception as exc:
            problems.append(f"{row_id}: {type(exc).__name__}: {exc}"[:300])
            continue
        ctx.rows[row_id] = Row(row_id, module, item["config"] or {}, bool(item["disabled"]), "user")
    return problems
