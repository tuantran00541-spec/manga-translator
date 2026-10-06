"""MCP stdio server exposing the pptxdeck plugin tools (make_deck, check_deck_tool).

JSON-RPC 2.0 over stdio. Implements the MCP 2024-11-05 lifecycle:
initialize, tools/list, tools/call. The two tools delegate to the
mounted plugin (~/.manga-agent/plugins/pptxdeck.py) so behaviour stays
identical to the in-agent tools.

Usage: python3 mcp_server.py   (reads line-delimited JSON-RPC on stdin)
"""
import importlib.util
import json
import os
import sys

PLUGIN = os.path.expanduser("~/.manga-agent/plugins/pptxdeck.py")
FALLBACK = os.path.join(os.path.dirname(__file__), "..", "..",
                       ".agents", "plugins", "pptxdeck_v4.py")
PROTOCOL_VERSION = "2024-11-05"

def _load_plugin():
    path = PLUGIN if os.path.exists(PLUGIN) else os.path.abspath(FALLBACK)
    spec = importlib.util.spec_from_file_location("pptxdeck_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = _load_plugin()

TOOLS = [
    {
        "name": "make_deck",
        "description": "Build a .pptx deck from a JSON spec (theme, transition, slides "
                       "with layouts title|bullets|cards|two_panel|chart|numbered_rows|"
                       "compare|process). Returns the saved path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec": {"type": "string", "description": "JSON string of the deck spec"},
                "path": {"type": "string", "description": "output .pptx path (parent dir must exist)"},
            },
            "required": ["spec", "path"],
        },
    },
    {
        "name": "check_deck_tool",
        "description": "QC a .pptx: overflow, overlap, font floors, contrast, margins, "
                       "word count, off-slide shapes. Pass png prefix for render checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": ".pptx file"},
                "png": {"type": "string", "description": "optional pdftoppm prefix"},
            },
            "required": ["path"],
        },
    },
]

def _call(name, args):
    if name == "make_deck":
        return mod.make_deck(None, args)
    if name == "check_deck_tool":
        return mod.check_deck(None, args)
    raise ValueError(f"unknown tool {name}")

def _handle(msg):
    mtype = msg.get("method")
    mid = msg.get("id")
    if mtype == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "deck", "version": "1.0.0"},
        }}
    if mtype in ("notifications/initialized",):
        return None
    if mtype == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if mtype == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            out = _call(name, args)
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": str(out)}],
                "isError": False}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": f"error: {e}"}],
                "isError": True}}
    if mtype == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"unknown method {mtype}"}}
    return None

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write("bad json: %s\n" % line)
            continue
        resp = _handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
