"""Capability seams in the DeepSeek Harness way: each service has a contract, built-in providers and plugin ones, and profile.json picks."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

from app.agent import sandbox, websearch

# What a provider of each service is called with and returns.
CONTRACTS = {
    "web.search": "fn(query: str, count: int, recency: str) -> list of (title, url, snippet); tried in the chosen order",
    "web.fetch": "fn(url: str) -> (body: bytes, content_type: str, encoding: str | None)",
    "shell": "fn(command: str, policy: sandbox.Policy, cwd: Path, tty: bool) -> a job like sandbox.Job (code, read(), out, stop())",
    "model": "fn(provider, api_key, model, messages, tools=None, **extra) -> {text, calls, reasoning, usage}",
    "compact": "fn(session, messages) -> the summary text that replaces the older conversation",
}
DEFAULTS = {"web.search": ["tinyfish", "tavily", "duckduckgo"], "web.fetch": ["http"], "shell": ["local"], "model": ["default"], "compact": ["default"]}
# Argument names an MCP tool may use for what a seam passes it.
MCP_ARGS = {"query": ("query", "q", "search_query", "search_term"), "count": ("count", "max_results", "num_results", "limit"),
            "url": ("url", "uri", "link"), "command": ("command", "cmd", "script")}


def _builtin() -> dict[str, dict[str, Callable]]:
    # Built when asked, so a test or plugin that patches a module function is seen.
    return {
        "web.search": dict(websearch._default_providers()),
        "web.fetch": {"http": None},
        "shell": {"local": lambda command, policy, cwd, tty=False: sandbox.Job(command, policy, cwd, tty=tty)},
        "model": {"default": None},
        "compact": {"default": None},
    }


class _Output:
    def __init__(self, text: str):
        self.data = bytearray(text.encode("utf-8"))

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")

    def join(self, timeout: float | None = None) -> None:
        return None


class DoneJob:
    """A command an MCP tool already ran to the end, shaped like a finished sandbox.Job."""

    def __init__(self, command: str, output: str, code: int):
        self.command, self.started, self.outside, self.cursor = command, time.time(), True, 0
        self.out, self._code = _Output(output), code

    @property
    def code(self) -> int:
        return self._code

    def read(self) -> str:
        data = bytes(self.out.data[self.cursor:])
        self.cursor += len(data)
        return data.decode("utf-8", errors="replace")

    def write(self, text: str) -> None:
        raise OSError("a command run by an MCP tool takes no input")

    def stop(self) -> None:
        return None


def _mcp_args(schema: dict, values: dict) -> dict:
    props = (schema or {}).get("properties") or {}
    out = {}
    for key, value in values.items():
        name = next((n for n in MCP_ARGS.get(key, (key,)) if n in props), None)
        if name is not None or not props:
            out[name or key] = value
    return out


def mcp_provider(service: str, tool: dict, call: Callable) -> Callable | None:
    """A provider for a seam made from an MCP tool; call(arguments) returns (text, ok)."""
    schema = tool.get("inputSchema") or {}

    def checked(values: dict) -> str:
        text, ok = call(_mcp_args(schema, values))
        if not ok:
            raise websearch.SearchError(text[:300])
        return text
    if service == "web.search":
        return lambda query, count, recency="": checked({"query": query, "count": count})
    if service == "web.fetch":
        return lambda url: (checked({"url": url}).encode("utf-8"), "text/markdown; charset=utf-8", "utf-8")
    if service == "shell":
        def run(command, policy, cwd, tty=False):
            text, ok = call(_mcp_args(schema, {"command": command}))
            return DoneJob(command, text, 0 if ok else 1)
        return run
    return None


class NoProvider(OSError):
    """Every provider of a service was switched off in plugins.json."""


def builtin_rows() -> list:
    """The built-in providers as plugins of their own, so plugins.json can switch any of them off or replace it."""
    from app.agent import kernel

    def late(service: str, name: str) -> Callable | None:
        if _builtin()[service][name] is None:
            return None
        return lambda *args, **kwargs: _builtin()[service][name](*args, **kwargs)
    return [kernel.Row(f"{service.replace('.', '-')}-{name}", kernel.provider(f"{service}/{name}", late(service, name)))
            for service, rows in _builtin().items() for name in rows]


class Services:
    """The providers a session uses: the kernel's (built-in rows and plugin ones), those from api.provide, and the user's choice."""

    def __init__(self, chosen: dict | None = None, extra: dict[str, dict[str, Callable]] | None = None, ctx=None, mcp=None):
        self.extra = {name: dict(rows) for name, rows in (extra or {}).items() if name in CONTRACTS}
        self.chosen = {name: ([value] if isinstance(value, str) else list(value)) for name, value in (chosen or {}).items()
                       if name in CONTRACTS and isinstance(value, (str, list))}
        # mcp gives (tools(), call(name, arguments)) so a chosen provider can be an MCP tool such as mcp__brave__search.
        self.ctx, self.mcp = ctx, mcp

    def providers(self, name: str) -> dict[str, Callable]:
        if self.ctx is None:
            mounted = _builtin()[name]
        else:
            mounted = {key.split("/", 1)[1]: value for key, (_, value) in self.ctx.services.items() if key.startswith(name + "/")}
        wanted = [p for p in self.chosen.get(name, []) if p.startswith("mcp__")]
        if wanted and self.mcp is not None:
            tools = self.mcp.tools()
            for tool_name in wanted:
                made = mcp_provider(name, tools[tool_name][1], lambda args, t=tool_name: self.mcp.call(t, args)) if tool_name in tools else None
                if made is not None:
                    mounted[tool_name] = made
        return {**mounted, **self.extra.get(name, {})}

    def chosen_fn(self, name: str) -> Callable | None:
        """The provider picked for a core seam (model, compact), or None to keep the harness's own."""
        return self.providers(name)[self.active(name)[0]]

    def active(self, name: str) -> list[str]:
        """The chosen providers that exist, in order; otherwise the defaults that exist, otherwise any."""
        known = self.providers(name)
        picked = [p for p in self.chosen.get(name, []) if p in known] or [p for p in DEFAULTS[name] if p in known] or list(known)
        if not picked:
            raise NoProvider(f"no provider for {name} is mounted; check ~/.manga-agent/plugins.json")
        return picked

    def missing(self, name: str) -> list[str]:
        known = self.providers(name)
        return [p for p in self.chosen.get(name, []) if p not in known]

    def is_local_shell(self) -> bool:
        try:
            return self.active("shell")[0] == "local"
        except NoProvider:
            return False

    def search(self, query: str, count: int = 8, recency: str = "") -> str:
        known = self.providers("web.search")
        return websearch.search(query, count, recency, providers=[(p, known[p]) for p in self.active("web.search")])

    def fetcher(self) -> Callable | None:
        """The page downloader to use, or None for the built-in HTTP reader."""
        return self.providers("web.fetch")[self.active("web.fetch")[0]]

    def shell(self, command: str, policy: sandbox.Policy, cwd: Path, tty: bool = False):
        return self.providers("shell")[self.active("shell")[0]](command, policy, cwd, tty)

    def describe(self) -> list[str]:
        rows = []
        for name in CONTRACTS:
            try:
                active = self.active(name)
            except NoProvider as exc:
                rows.append(f"{name}: {exc}")
                continue
            other = [p for p in self.providers(name) if p not in active]
            line = f"{name}: {', '.join(active)}" + (f" (còn có: {', '.join(other)})" if other else "")
            if self.missing(name):
                line += f" — không có provider {', '.join(self.missing(name))}, dùng mặc định"
            rows.append(line)
        return rows
