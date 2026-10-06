"""One-off: TinyFish search and fetch as a stdio MCP server, over the same official endpoints the harness already calls."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.agent import webread, websearch  # noqa: E402

KEY = os.environ.get("TINYFISH_API_KEY", "")
TOOLS = [
    {"name": "search", "description": "Search the web with TinyFish; returns title, URL and snippet per result.",
     "inputSchema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}, "count": {"type": "integer"}}},
     "annotations": {"readOnlyHint": True, "openWorldHint": True}},
    {"name": "fetch", "description": "Read one web page as markdown through TinyFish's real browser (works on JavaScript pages).",
     "inputSchema": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}},
     "annotations": {"readOnlyHint": True, "openWorldHint": True}},
]


def call(name: str, args: dict) -> tuple[str, bool]:
    if not KEY:
        return "TINYFISH_API_KEY is not set", False
    try:
        if name == "search":
            rows = websearch._tinyfish(str(args["query"]), min(int(args.get("count") or 8), 20), KEY)
            return "\n\n".join(f"{i}. {t}\n{u}\n{s}" for i, (t, u, s) in enumerate(rows, 1)) or "No results", True
        if name == "fetch":
            text = webread.browsed(str(args["url"]), KEY)
            return (text[:60000] or "The page came back empty"), bool(text)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:500], False
    return f"Unknown tool {name}", False


def main() -> None:
    out = sys.stdout
    sys.stdout = sys.stderr
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        if "id" not in message:
            continue
        method, params = message.get("method"), message.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion") or "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "tinyfish", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            text, ok = call(str(params.get("name")), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": text}], "isError": not ok}
        elif method == "ping":
            result = {}
        else:
            out.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"no method {method}"}}) + "\n")
            out.flush()
            continue
        out.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
