"""Model Context Protocol client: stdio and streamable HTTP servers, their tools offered to the agent."""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
from urllib.parse import urljoin

import requests
from fastapi import HTTPException

from app.logging_config import logger
from app.security import validate_url

from app.agent import sandbox

try:
    import tomllib
except ImportError:  # Python 3.10 has no TOML reader, so Codex's config is skipped there.
    tomllib = None

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "chaos", "version": "1.0"}
REQUEST_TIMEOUT = 120
MAX_RESULT_CHARS = 20_000
# A fresh nextCursor on every tools/list page would hang startup; cap the pages.
MAX_TOOL_LIST_PAGES = 50
# Redirects are followed manually so each hop goes through the SSRF guard again.
_MAX_HTTP_REDIRECTS = 5
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The harness judges read-only by tool name; a server's own readOnlyHint never skips approval.
_READONLY_NAME_PREFIXES = ("get_", "list_", "read_", "fetch_", "search_", "describe_", "query_", "show_", "lookup_")
_READONLY_NAMES = frozenset({"get", "list", "read", "fetch", "search", "describe", "query", "show", "status", "version", "ping", "help"})


def harness_read_only(tool: str) -> bool:
    """True when the harness itself judges an MCP tool read-only from its name.

    The server's readOnlyHint annotation is deliberately NOT consulted here:
    a server controls its annotations, so a destructive tool may self-certify
    as read-only and would otherwise run approval-free in auto/edits mode.
    """
    name = str(tool or "").strip().lower()
    return name in _READONLY_NAMES or name.startswith(_READONLY_NAME_PREFIXES)


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


def _plugin_rows(path: Path) -> dict:
    """MCP servers written as rows of the plugin tree: {"id": "github", "mcp": {...}, "disabled": false}."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = data.get("rows") if isinstance(data, dict) else None
    return {str(r["id"]): {**r["mcp"], **({"disabled": True} if r.get("disabled") else {})}
            for r in rows or [] if isinstance(r, dict) and isinstance(r.get("mcp"), dict) and r.get("id")}


NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,40}")
SECRET_VALUE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}$")


def user_file(home: Path) -> Path:
    return home / ".manga-agent" / "mcp.json"


def _pairs(value, what: str) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and k.strip() for k in value):
        raise ValueError(f"{what} must be an object of names and values")
    return {k.strip(): str(v) for k, v in value.items()}


def clean_config(config) -> dict:
    """One server's settings checked and reduced to what the client uses: a command with args and env, or a URL with headers.

    ``env_allowlist`` names host variables a stdio server genuinely needs (e.g. its own API key);
    they are restored after the harness process env is scrubbed (see _StdioTransport).
    """
    if not isinstance(config, dict):
        raise ValueError("a server's settings must be a JSON object")
    kind = str(config.get("type") or config.get("transport") or ("http" if config.get("url") else "stdio")).lower()
    out: dict = {"disabled": True} if config.get("disabled") or config.get("enabled") is False else {}
    if kind in ("http", "streamable-http", "streamable_http", "sse"):
        url = str(config.get("url") or "").strip()
        if not re.match(r"^https?://[^\s/]+", url):
            raise ValueError("an http server needs a url starting with http:// or https://")
        out.update(type="http", url=url)
        if headers := _pairs(config.get("headers"), "headers"):
            out["headers"] = headers
        for key in ("bearer_token_env_var",):
            if isinstance(config.get(key), str) and config[key].strip():
                out[key] = config[key].strip()
        return out
    if kind != "stdio":
        raise ValueError(f"transport {kind!r} is not supported; use stdio or http")
    command, args = config.get("command"), config.get("args") or []
    if isinstance(command, list):
        command, args = (command[0] if command else ""), [*command[1:], *args]
    command = str(command or "").strip()
    if command and not args and " " in command:
        # A whole command line pasted into one field is split the way a shell would.
        parts = shlex.split(command, posix=os.name != "nt")
        command, args = parts[0], parts[1:]
    if not command:
        raise ValueError("a stdio server needs a command, such as npx or uvx")
    if not isinstance(args, list):
        raise ValueError("args must be a list")
    out.update(command=command, args=[str(a) for a in args])
    if env := _pairs(config.get("env"), "env"):
        out["env"] = env
    if isinstance(config.get("env_allowlist"), list):
        allowed = [n.strip() for n in config["env_allowlist"] if isinstance(n, str) and _ENV_NAME_RE.fullmatch(n.strip())]
        if allowed:
            out["env_allowlist"] = allowed
    if isinstance(config.get("cwd"), str) and config["cwd"].strip():
        out["cwd"] = config["cwd"].strip()
    return out


def parse_pasted(text: str, name: str = "") -> dict[str, dict]:
    """Servers from JSON as MCP docs show it: {"mcpServers": {...}}, VS Code's {"servers": {...}}, a bare {"name": {...}} or one server's settings."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        try:
            # Docs often show only the inner "name": {...} lines, sometimes with a trailing comma.
            data = json.loads("{" + text.rstrip(",") + "}")
        except ValueError as exc:
            raise ValueError("that is not JSON; paste the block from the server's docs") from exc
    if not isinstance(data, dict):
        raise ValueError("paste a JSON object")
    for key in ("mcpServers", "mcp_servers", "servers", "mcp"):
        if isinstance(data.get(key), dict):
            data = data[key]
            break
    if any(k in data for k in ("command", "url", "type", "transport")):
        if not name.strip():
            raise ValueError("this is one server's settings; give it a name")
        data = {name.strip(): data}
    servers = {}
    for key, config in data.items():
        if not NAME_RE.fullmatch(str(key)):
            raise ValueError(f"server name {key!r}: use letters, digits, '.', '_' or '-' (at most 40)")
        servers[str(key)] = clean_config(config)
    if not servers:
        raise ValueError("no server found in that JSON")
    return servers


def _read_user(home: Path) -> dict:
    try:
        data = json.loads(user_file(home).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"{user_file(home)} cannot be read: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _write_user(home: Path, data: dict) -> None:
    path = user_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Settings may hold tokens, so only this user can read the file (the sandbox keeps commands out of it too).
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def save_user(home: Path, servers: dict[str, dict]) -> None:
    """Add or replace servers in ~/.manga-agent/mcp.json, keeping everything else in it."""
    data = _read_user(home)
    table = data.setdefault("mcpServers", {})
    if not isinstance(table, dict):
        table = data["mcpServers"] = {}
    table.update(servers)
    _write_user(home, data)


def remove_user(home: Path, name: str) -> None:
    data = _read_user(home)
    table = data.get("mcpServers") if isinstance(data.get("mcpServers"), dict) else {}
    if name not in table:
        raise KeyError(name)
    table.pop(name)
    _write_user(home, data)


def set_user_enabled(home: Path, name: str, enabled: bool) -> None:
    data = _read_user(home)
    table = data.get("mcpServers") if isinstance(data.get("mcpServers"), dict) else {}
    if name not in table or not isinstance(table[name], dict):
        raise KeyError(name)
    table[name].pop("enabled", None)
    if enabled:
        table[name].pop("disabled", None)
    else:
        table[name]["disabled"] = True
    _write_user(home, data)


def public(row: dict) -> dict:
    """A configured server as the page may show it: values of env and headers are hidden unless they only name a variable."""
    config = row["config"]
    hide = lambda pairs: {k: (v if SECRET_VALUE.match(str(v)) else "•••") for k, v in (pairs or {}).items()} if isinstance(pairs, dict) else {}
    kind = "http" if config.get("url") else "stdio"
    where = config.get("url") if kind == "http" else " ".join([str(config.get("command") or ""), *[str(a) for a in config.get("args") or []]])
    return {"name": row["name"], "scope": row["scope"], "source": row["source"], "type": kind, "target": str(where)[:300],
            "env": hide(config.get("env")), "headers": hide(config.get("headers")),
            "disabled": bool(config.get("disabled") or config.get("enabled") is False), "editable": row["source"] == "~/.manga-agent/mcp.json"}


def probe(name: str, config: dict, cwd: Path) -> list[str]:
    """Start the server, list its tools and stop it again; raises MCPError with the server's own words when it fails."""
    server = Server(name, config, cwd)
    try:
        return [str(t["name"]) for t in server.tools]
    finally:
        server.close()


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
    add(_plugin_rows(home / ".manga-agent" / "plugins.json"), "~/.manga-agent/plugins.json", "user")
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
        # Scrubbed so the server never inherits the harness's provider keys and tokens.
        env = sandbox.clean_env()
        # A server names genuinely-needed host variables in env_allowlist; the rest stays hidden.
        for name in config.get("env_allowlist") or ():
            if isinstance(name, str) and _ENV_NAME_RE.fullmatch(name) and name in os.environ:
                env[name] = os.environ[name]
        # Config-declared env layers on top, so declared values still reach the server.
        env.update({str(k): str(v) for k, v in (config.get("env") or {}).items()})
        # Windows needs the full name of launchers such as npx.cmd.
        command = shutil.which(command, path=env.get("PATH")) or command
        self.proc = subprocess.Popen([command, *[str(a) for a in config.get("args") or []]], cwd=config.get("cwd") or cwd,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.waiting: dict[int, dict] = {}
        self.lock = threading.Condition()
        # The last lines a server wrote to stderr say why it failed to start or stopped.
        self.errors: collections.deque[str] = collections.deque(maxlen=20)
        threading.Thread(target=self._read, daemon=True, name="mcp-stdio").start()
        threading.Thread(target=self._read_errors, daemon=True, name="mcp-stderr").start()

    def _read_errors(self) -> None:
        for line in self.proc.stderr:
            if line.strip():
                self.errors.append(line.rstrip()[:300])

    def why(self) -> str:
        tail = " | ".join(list(self.errors)[-5:])
        return f": {tail}" if tail else ""

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
                raise MCPError("server did not answer in time" + self.why())
            reply = self.waiting.pop(message["id"])
        if not reply:
            raise MCPError(f"server exited (code {self.proc.poll()}){self.why()}")
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
        # The configured URL goes through the same SSRF guard as the harness's web tools.
        try:
            validate_url(self.url)
        except HTTPException as exc:
            raise MCPError(f"refusing MCP server URL {self.url!r}: {exc.detail}") from exc
        self.headers = {str(k): str(v) for k, v in (config.get("headers") or {}).items()}
        token = config.get("bearer_token") or (os.environ.get(config["bearer_token_env_var"]) if config.get("bearer_token_env_var") else None)
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.session_id: str | None = None

    def _check_url(self, url: str, what: str) -> None:
        try:
            validate_url(url)
        except HTTPException as exc:
            raise MCPError(f"refusing MCP server {what} {url!r}: {exc.detail}") from exc

    def _post(self, message: dict, timeout: float) -> requests.Response:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL_VERSION, **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        # Follow redirects manually (re-POSTing) so every hop is re-checked by the SSRF guard.
        url, redirects = self.url, 0
        while True:
            try:
                response = requests.post(url, json=message, headers=headers, timeout=timeout, stream=True,
                                         allow_redirects=False)
            except requests.RequestException as exc:
                raise MCPError(f"cannot reach {url} ({type(exc).__name__})") from exc
            if response.status_code not in _REDIRECT_STATUS_CODES:
                break
            location = response.headers.get("Location")
            response.close()
            redirects += 1
            if not location:
                raise MCPError(f"MCP server at {url} redirected without a Location header")
            if redirects > _MAX_HTTP_REDIRECTS:
                raise MCPError(f"MCP server at {self.url} redirected too many times (>{_MAX_HTTP_REDIRECTS})")
            url = urljoin(url, location)
            self._check_url(url, "redirect")
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
            cursor, pages = None, 0
            while True:
                page = self._call("tools/list", {"cursor": cursor} if cursor else {}, 60)
                self.tools += [t for t in page.get("tools") or [] if isinstance(t, dict) and t.get("name")]
                pages += 1
                cursor = page.get("nextCursor")
                if not cursor or len(self.tools) > 500:
                    break
                if pages >= MAX_TOOL_LIST_PAGES:
                    logger.warning("MCP server %s still paginating tools/list after %d pages; stopping",
                                   name, MAX_TOOL_LIST_PAGES)
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
