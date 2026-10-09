"""Model Context Protocol client: stdio and streamable HTTP servers, their tools offered to the agent."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import requests

from app.logging_config import logger

try:
    import tomllib
except ImportError:  # Python 3.10 has no TOML reader, so Codex's config is skipped there.
    tomllib = None

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "manga-translator-agent", "version": "1.0"}
REQUEST_TIMEOUT = 120
# M33: bounds on what an MCP server can make us buffer. A hostile server otherwise grows our
# memory without limit: one endless JSON-RPC line, an infinite tools/list pagination, or a
# multi-gigabyte tool result.
MAX_LINE_CHARS = 10_000_000
MAX_RESULT_CHARS = 20_000
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class MCPError(RuntimeError):
    """A server that failed to start or answered with an error."""


def _expand(value):
    """${VAR} and ${VAR:-default} in config strings, as Claude Code's .mcp.json allows."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def expand_config(config: dict) -> dict:
    """The config as the server will actually see it, with ${VAR} substituted (M31). The trust
    digest must cover this, not the raw config: the environment decides what command, URL and
    token really run, so a digest of the unexpanded text would miss env-driven changes."""
    return _expand(config)


def _json_servers(path: Path, key: str = "mcpServers") -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    servers = data.get(key) if isinstance(data, dict) else None
    return servers if isinstance(servers, dict) else {}


def _plugin_rows(path: Path) -> dict:
    """MCP servers written as rows of the plugin tree: {"id": "github", "mcp": {...}, "disabled": false}."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = data.get("rows") if isinstance(data, dict) else None
    return {str(r["id"]): {**r["mcp"], **({"disabled": True} if r.get("disabled") else {})}
            for r in rows or [] if isinstance(r, dict) and isinstance(r.get("mcp"), dict) and r.get("id")}


def configured(workspace: Path, home: Path | None = None) -> list[dict]:
    """Every configured server with where it came from; workspace ones need the user's trust before they start."""
    home = home if home is not None else Path.home()
    rows: dict[str, dict] = {}

    def add(servers: dict, source: str, scope: str) -> None:
        for name, config in servers.items():
            if isinstance(config, dict) and re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", str(name)) and name not in rows:
                rows[name] = {"name": name, "config": config, "source": source, "scope": scope}

    add(_json_servers(workspace / ".mcp.json"), ".mcp.json", "workspace")
    add(_json_servers(home / ".manga-agent" / "mcp.json"), "~/.manga-agent/mcp.json", "user")
    add(_plugin_rows(home / ".manga-agent" / "plugins.json"), "~/.manga-agent/plugins.json", "plugin")
    add(_json_servers(home / ".claude.json"), "~/.claude.json", "user")
    try:
        codex = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8")).get("mcp_servers", {}) if tomllib else {}
    except (OSError, ValueError):
        codex = {}
    add(codex if isinstance(codex, dict) else {}, "~/.codex/config.toml", "user")
    return list(rows.values())


class _StdioTransport:
    def __init__(self, config: dict, cwd: Path):
        command = config.get("command")
        if not command:
            raise MCPError("stdio server has no command")
        # L4: validate the config shape up front with a clear error instead of degrading noisily
        # (a string "args" used to be iterated character-by-character into argv).
        if not isinstance(command, str):
            raise MCPError(f"stdio server command must be a string, got {type(command).__name__}")
        args = config.get("args") or []
        if isinstance(args, str) or not isinstance(args, (list, tuple)):
            raise MCPError(f"stdio server args must be a list, got {type(args).__name__}")
        env_cfg = config.get("env") or {}
        if not isinstance(env_cfg, dict):
            raise MCPError(f"stdio server env must be a dict, got {type(env_cfg).__name__}")
        # C4: a stdio server used to see the whole os.environ, including API keys and secrets.
        # It now starts from the scrubbed clean_env() (no secret-looking names); only variables
        # the user explicitly set in the server's "env" config are added back on top.
        from app.agent import sandbox
        env = {**sandbox.clean_env(), **{str(k): str(v) for k, v in env_cfg.items()}}
        # Windows needs the full name of launchers such as npx.cmd.
        command = shutil.which(command, path=env.get("PATH")) or command
        self.proc = subprocess.Popen([command, *[str(a) for a in args]], cwd=config.get("cwd") or cwd,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1)
        self.waiting: dict[int, dict] = {}
        self.lock = threading.Condition()
        threading.Thread(target=self._read, daemon=True, name="mcp-stdio").start()

    def _read(self) -> None:
        # M33: a malicious server must not be able to grow our memory without bound; one line
        # (one JSON-RPC message) past MAX_LINE_CHARS kills the server instead of us.
        while True:
            chunks, size = [], 0
            while True:
                chunk = self.proc.stdout.readline(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_LINE_CHARS or chunk.endswith("\n"):
                    break
            if not chunks:
                break
            line = "".join(chunks)
            if size > MAX_LINE_CHARS:
                logger.warning("MCP server sent a line past {} chars; closing it", MAX_LINE_CHARS)
                self.close()
                break
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if "method" in message and "id" in message:
                # The server asks something back (ping, roots); answer so it never waits on us.
                result = {} if message["method"] == "ping" else {"roots": []} if message["method"] == "roots/list" else None
                reply = {"jsonrpc": "2.0", "id": message["id"]}
                reply.update({"result": result} if result is not None else {"error": {"code": -32601, "message": "not supported"}})
                self.send(reply)
                continue
            with self.lock:
                if message.get("id") in self.waiting:
                    self.waiting[message["id"]] = message
                    self.lock.notify_all()
        with self.lock:
            self.lock.notify_all()

    def send(self, message: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise MCPError(f"server stopped: {exc}") from exc

    def request(self, message: dict, timeout: float) -> dict:
        with self.lock:
            self.waiting[message["id"]] = {}
        self.send(message)
        with self.lock:
            if not self.lock.wait_for(lambda: self.waiting[message["id"]] or self.proc.poll() is not None, timeout):
                self.waiting.pop(message["id"], None)
                raise MCPError("server did not answer in time")
            reply = self.waiting.pop(message["id"])
        if not reply:
            raise MCPError("server exited")
        return reply

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


class _HttpTransport:
    def __init__(self, config: dict):
        self.url = config.get("url")
        if not self.url:
            raise MCPError("http server has no url")
        # M21: validate the URL against SSRF (private/link-local IPs, DNS rebinding). A malicious
        # or compromised MCP config pointing at http://169.254.169.254/ would otherwise get the
        # bearer token forwarded to cloud metadata.
        from app.security import validate_url
        try:
            self.url = validate_url(str(self.url))
        except ValueError as exc:
            raise MCPError(f"refusing unsafe MCP server URL: {exc}") from exc
        self.headers = {str(k): str(v) for k, v in (config.get("headers") or {}).items()}
        token = config.get("bearer_token") or (os.environ.get(config["bearer_token_env_var"]) if config.get("bearer_token_env_var") else None)
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.session_id: str | None = None

    def _post(self, message: dict, timeout: float) -> requests.Response:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL_VERSION, **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        # M32: never follow redirects. A 307/308 would re-POST the JSON-RPC body (and our
        # Authorization / custom headers) to wherever the server points, which is credential
        # forwarding to an unvalidated URL by another name.
        response = requests.post(self.url, json=message, headers=headers, timeout=timeout, stream=True,
                                 allow_redirects=False)
        if response.headers.get("Mcp-Session-Id"):
            self.session_id = response.headers["Mcp-Session-Id"]
        if 300 <= response.status_code < 400:
            response.close()
            raise MCPError(f"server redirected to {response.headers.get('location', '?')[:200]}; refusing to follow")
        if response.status_code >= 400:
            raise MCPError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response

    def send(self, message: dict) -> None:
        self._post(message, 30).close()

    def request(self, message: dict, timeout: float) -> dict:
        response = self._post(message, timeout)
        # M32: a per-read timeout does not bound a trickling server; the whole answer must arrive
        # within a multiple of the request timeout or the thread is held forever.
        deadline = time.monotonic() + max(timeout, 30) * 3
        try:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                return response.json()
            data: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if time.monotonic() > deadline:
                    raise MCPError("server trickled the answer past the deadline")
                if line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and data:
                    event = json.loads("\n".join(data))
                    data = []
                    if event.get("id") == message["id"]:
                        return event
            raise MCPError("stream ended without an answer")
        except ValueError as exc:
            raise MCPError("server sent an unreadable answer") from exc
        finally:
            response.close()

    def close(self) -> None:
        if self.session_id:
            try:
                requests.delete(self.url, headers={"Mcp-Session-Id": self.session_id, **self.headers}, timeout=5)
            except requests.RequestException:
                pass


class Server:
    """One running MCP server and the tools it offers."""

    def __init__(self, name: str, config: dict, cwd: Path):
        self.name = name
        config = _expand(config)
        kind = str(config.get("type") or config.get("transport") or ("http" if config.get("url") else "stdio")).lower()
        if kind in ("http", "streamable-http", "streamable_http"):
            self.transport = _HttpTransport(config)
        elif kind == "stdio":
            self.transport = _StdioTransport(config, cwd)
        else:
            raise MCPError(f"transport {kind!r} is not supported; use stdio or http")
        self.ids = itertools.count(1)
        self.tools: list[dict] = []
        try:
            self._call("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO}, 60)
            self.transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            cursor = None
            seen_cursors: set = set()
            while True:
                page = self._call("tools/list", {"cursor": cursor} if cursor else {}, 60)
                self.tools += [t for t in page.get("tools") or [] if isinstance(t, dict) and t.get("name")]
                cursor = page.get("nextCursor")
                # M33: a hostile server must not spin us in an endless pagination loop.
                if not cursor or cursor in seen_cursors or len(self.tools) > 500:
                    break
                seen_cursors.add(cursor)
        except Exception:
            self.close()
            raise

    def _call(self, method: str, params: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
        reply = self.transport.request({"jsonrpc": "2.0", "id": next(self.ids), "method": method, "params": params}, timeout)
        if reply.get("error"):
            raise MCPError(str(reply["error"].get("message") or reply["error"])[:500])
        return reply.get("result") or {}

    def call_tool(self, tool: str, arguments: dict) -> tuple[str, bool]:
        result = self._call("tools/call", {"name": tool, "arguments": arguments})
        # M33: truncate incrementally; a hostile server must not make us buffer a multi-gigabyte
        # result just to slice it down to MAX_RESULT_CHARS afterwards.
        parts, size = [], 0
        for item in result.get("content") or []:
            if size >= MAX_RESULT_CHARS:
                break
            if item.get("type") == "text":
                text = str(item.get("text", ""))
            elif item.get("type") == "resource":
                resource = item.get("resource") or {}
                text = str(resource.get("text") or f"[resource {resource.get('uri', '')}]")
            else:
                text = f"[{item.get('type', 'content')} omitted]"
            parts.append(text[:MAX_RESULT_CHARS - size])
            size += len(parts[-1])
        if result.get("structuredContent") and not parts:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False)[:MAX_RESULT_CHARS])
        text = "\n".join(parts) or "(no output)"
        return text[:MAX_RESULT_CHARS], not result.get("isError")

    def close(self) -> None:
        try:
            self.transport.close()
        except Exception:
            logger.opt(exception=True).debug("MCP server {} did not close cleanly", self.name)


def tool_name(server: str, tool: str) -> str:
    """The name the model sees: mcp__server__tool plus a short hash of the exact (server, tool) pair.

    The readable part alone is ambiguous (H16): server names may contain dots, so "my.server" and
    "my_server" normalize identically; "__" is both the separator and legal inside names; and names
    truncate at the 64 characters most APIs allow. The hash suffix makes every distinct pair a
    distinct key, so a hostile server can never silently shadow another server's tool under the
    same name — including a built-in tool mapped onto an MCP tool by profile.json "replace"."""
    suffix = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:8]
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{server}__{tool}")
    return f"{clean[:54]}__{suffix}"
