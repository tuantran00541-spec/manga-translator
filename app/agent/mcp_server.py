"""The harness as an MCP server over stdio: its tools, its plugins' tools and the MCP servers it connects, behind its own rules and sandbox."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "manga-agent", "version": "1"}
# Tools that need a person in the loop or the agent's own conversation are not offered to another client.
HIDDEN = {"ask_user", "exit_plan_mode", "goal_done", "todo_write", "memory", "task", "spawn_agent", "fan_out", "wait_agent", "send_input",
          "close_agent", "oracle", "schedule_create", "schedule_list", "schedule_delete", "context_notes", "new_context", "plugin_write",
          "plugin_remove", "skill", "job_input", "delegate", "tool_search"}


def _never(*args, **kwargs):
    raise RuntimeError("the MCP server does not call a model")


class Bridge:
    """One headless agent session whose tools are served; calls go through the same rules, sandbox and plugin waterfalls."""

    def __init__(self, folder: Path, write: bool = False, network: bool = False, home: Path | None = None):
        from app.agent import sandbox
        from app.agent.session import AgentSessionManager
        from app.agent.tools import Workspace
        from app.ai_providers import PROVIDERS

        policy = sandbox.Policy("workspace-write" if write else "read-only", network)
        self.manager = AgentSessionManager(None, home=home)
        self.session = self.manager.create(PROVIDERS["openai"], "", "", Workspace(folder, policy), "auto", complete=_never)
        self.write, self.network = write, network

    def _kind(self, name: str) -> str:
        """read, edit, exec or net for a built-in, plugin or connected MCP tool."""
        from app.agent.tools import KIND

        session = self.session
        if name in KIND:
            return KIND[name]
        if name in session.registry.tools:
            return session.registry.tools[name].kind
        if name in session.mcp_tools:
            read_only = (session.mcp_tools[name][1].get("annotations") or {}).get("readOnlyHint")
            return session._role(name) or ("read" if read_only else "exec")
        return "exec"

    def _offered(self, spec: dict) -> bool:
        name = spec["name"]
        kind = self._kind(name)
        if name in HIDDEN or (kind in ("edit", "exec") and not self.write) or (kind == "net" and not self.network):
            return False
        if name in self.session.registry.tools and self.session.registry.tools[name].always_ask:
            return False
        return True

    def tools(self) -> list[dict]:
        self.session._ensure_mcp()
        rows = []
        for spec in self.session.specs():
            if not self._offered(spec):
                continue
            schema = json.loads(json.dumps(spec.get("parameters") or {"type": "object", "properties": {}}))
            (schema.get("properties") or {}).pop("outside_sandbox", None)
            kind = self._kind(spec["name"])
            rows.append({"name": spec["name"], "description": spec.get("description", ""), "inputSchema": schema,
                         "annotations": {"readOnlyHint": kind == "read", "destructiveHint": kind in ("edit", "exec"), "openWorldHint": kind == "net"}})
        return rows

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        if not any(t["name"] == name for t in self.tools()):
            return f"Unknown tool {name!r}", False
        args = {k: v for k, v in (arguments or {}).items() if k != "outside_sandbox"}
        return self.session._run_call({"id": f"mcp-{uuid.uuid4().hex[:8]}", "name": name, "args": args})

    def close(self) -> None:
        self.manager.close_all()


def handle(bridge: Bridge, message: dict) -> dict | None:
    """One JSON-RPC message in, one reply out (None for a notification)."""
    if "id" not in message:
        return None
    method, params = message.get("method"), message.get("params") or {}
    try:
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": SERVER_INFO}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": bridge.tools()}
        elif method == "tools/call":
            text, ok = bridge.call(str(params.get("name") or ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": text}], "isError": not ok}
        else:
            return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"method {method} is not supported"}}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"[:500]}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def serve(folder: Path, write: bool = False, network: bool = False) -> None:
    # stdout carries the protocol only; anything else the app prints goes to stderr.
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    bridge = Bridge(folder, write, network)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                reply = handle(bridge, json.loads(line))
            except ValueError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            if reply is not None:
                out.write(json.dumps(reply, ensure_ascii=False) + "\n")
    finally:
        bridge.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="manga mcp", description="Serve the agent's tools to another MCP client over stdio")
    parser.add_argument("--folder", default=".", help="the project folder the tools work in (default: here)")
    parser.add_argument("--write", action="store_true", help="also offer edit and command tools (still sandboxed)")
    parser.add_argument("--network", action="store_true", help="also offer web tools and let commands use the network")
    args = parser.parse_args(argv)
    serve(Path(args.folder).expanduser().resolve(), args.write, args.network)


if __name__ == "__main__":
    main()
