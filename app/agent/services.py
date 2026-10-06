"""Capability seams in the DeepSeek Harness way: each service has a contract, built-in providers and plugin ones, and profile.json picks."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.agent import sandbox, websearch

# What a provider of each service is called with and returns.
CONTRACTS = {
    "web.search": "fn(query: str, count: int, recency: str) -> list of (title, url, snippet); tried in the chosen order",
    "web.fetch": "fn(url: str) -> (body: bytes, content_type: str, encoding: str | None)",
    "shell": "fn(command: str, policy: sandbox.Policy, cwd: Path, tty: bool) -> a job like sandbox.Job (code, read(), out, stop())",
}
DEFAULTS = {"web.search": ["tinyfish", "tavily", "duckduckgo"], "web.fetch": ["http"], "shell": ["local"]}


def _builtin() -> dict[str, dict[str, Callable]]:
    # Built when asked, so a test or plugin that patches a module function is seen.
    return {
        "web.search": dict(websearch._default_providers()),
        "web.fetch": {"http": None},
        "shell": {"local": lambda command, policy, cwd, tty=False: sandbox.Job(command, policy, cwd, tty=tty)},
    }


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

    def __init__(self, chosen: dict | None = None, extra: dict[str, dict[str, Callable]] | None = None, ctx=None):
        self.extra = {name: dict(rows) for name, rows in (extra or {}).items() if name in CONTRACTS}
        self.chosen = {name: ([value] if isinstance(value, str) else list(value)) for name, value in (chosen or {}).items()
                       if name in CONTRACTS and isinstance(value, (str, list))}
        self.ctx = ctx

    def providers(self, name: str) -> dict[str, Callable]:
        if self.ctx is None:
            mounted = _builtin()[name]
        else:
            mounted = {key.split("/", 1)[1]: value for key, (_, value) in self.ctx.services.items() if key.startswith(name + "/")}
        return {**mounted, **self.extra.get(name, {})}

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
