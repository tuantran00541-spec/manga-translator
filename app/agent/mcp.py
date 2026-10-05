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

import requests

from app.logging_config import logger

try:
    import tomllib
except ImportError:  # Python 3.10 has no TOML reader, so Codex's config is skipped there.
    tomllib = None

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "manga-translator-agent", "version": "1.0"}
REQUEST_TIMEOUT = 120
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


def _json_servers(path: Path, key: str = "mcpServers") -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    servers = data.get(key) if isinstance(data, dict) else None
    return servers if isinstance(servers, dict) else {}


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
        env = {**os.environ, **{str(k): str(v) for k, v in (config.get("env") or {}).items()}}
        # Windows needs the full name of launchers such as npx.cmd.
        command = shutil.which(command, path=env.get("PATH")) or command
        self.proc = subprocess.Popen([command, *[str(a) for a in config.get("args") or []]], cwd=config.get("cwd") or cwd,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1)
        self.waiting: dict[int, dict] = {}
        self.lock = threading.Condition()
        threading.Thread(target=self._read, daemon=True, name="mcp-stdio").start()

    def _read(self) -> None:
        for line in self.proc.stdout:
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
        response = requests.post(self.url, json=message, headers=headers, timeout=timeout, stream=True)
        if response.headers.get("Mcp-Session-Id"):
            self.session_id = response.headers["Mcp-Session-Id"]
        if response.status_code >= 400:
            raise MCPError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response

    def send(self, message: dict) -> None:
        self._post(message, 30).close()

    def request(self, message: dict, timeout: float) -> dict:
        response = self._post(message, timeout)
        try:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                return response.json()
            data: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
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
            while True:
                page = self._call("tools/list", {"cursor": cursor} if cursor else {}, 60)
                self.tools += [t for t in page.get("tools") or [] if isinstance(t, dict) and t.get("name")]
                cursor = page.get("nextCursor")
                if not cursor or len(self.tools) > 500:
                    break
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
        parts = []
        for item in result.get("content") or []:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "resource":
                resource = item.get("resource") or {}
                parts.append(str(resource.get("text") or f"[resource {resource.get('uri', '')}]"))
            else:
                parts.append(f"[{item.get('type', 'content')} omitted]")
        if result.get("structuredContent") and not parts:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        text = "\n".join(parts) or "(no output)"
        return text[:MAX_RESULT_CHARS], not result.get("isError")

    def close(self) -> None:
        try:
            self.transport.close()
        except Exception:
            logger.opt(exception=True).debug("MCP server {} did not close cleanly", self.name)


def tool_name(server: str, tool: str) -> str:
    """The name the model sees: mcp__server__tool, kept to the 64 characters most APIs allow."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{server}__{tool}")
    return clean[:64]
