"""MCP server "deck": expose make_deck and check_deck (criterion C) as MCP tools.

Stdout carries JSON-RPC only (rule: no other prints). Implementation lives in
.agents/skills/pptx/deck_tools.py so it is testable standalone too."""
from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "deck", "version": "1"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _import_tools():
    import os
    import importlib.util
    import sys as _sys
    root = os.environ.get("DECK_TOOLS_DIR") or os.path.abspath(os.path.join(os.getcwd(), ".agents", "skills", "pptx"))
    # Mỗi lần gọi tool, nạp lại deck_tools.py và phụ thuộc của nó (helpers) từ đầu —
    # nếu đã cache trước đó (ví dụ phiên trước khi helper có layout mới), sẽ dùng phiên
    # bản cũ, báo lỗi ImportError giả (ngày 5: slide_timeline đã thêm vào helpers.py
    # nhưng module cache chưa cập nhật). Xóa cache trước khi import, không ảnh hưởng
    # gì đến các module khác.
    for mod_name in ('deck_tools', 'helpers'):
        _sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location("deck_tools", os.path.join(root, "deck_tools.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_deck(params: dict) -> tuple[str, bool]:
    mod = _import_tools()
    return mod.make_deck(params)


def check_deck(params: dict) -> tuple[str, bool]:
    mod = _import_tools()
    return mod.check_deck(params)


TOOLS = [
    {
        "name": "make_deck",
        "description": "Build a .pptx deck from a JSON spec of slides (title, agenda, cards, two-column, process, barchart, quote, closing) using the SKILL.md helper recipes. Writes the file under deck-lab/ and returns the path and slide count.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "File name (without .pptx) under deck-lab/, e.g. 'a9_tieu.de'"},
                "slides": {"type": "array", "description": "Slide specs in order; see SKILL.md section 4 for the 8 kinds",
                           "items": {"type": "object",
                                     "properties": {
                                         "kind": {"type": "string", "enum": ["cover", "agenda", "cards", "twocol", "process", "barchart", "quote", "closing"],
                                                  "description": "Which of the 8 SKILL.md layouts"},
                                         "title": {"type": "string", "description": "Main heading text"},
                                         "subtitle": {"type": "string", "description": "Secondary line (cover) or source line (charts)"},
                                         "items": {"type": "array", "items": {"type": "string"},
                                                    "description": "Agenda rows or process step labels"},
                                         "cards": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                                     "description": "Cards: up to 3 rows of [big, label, description]"},
                                         "columns": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                                      "description": "Two-column comparison: [header, point1, point2, point3] x 2"},
                                         "data": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                                   "description": "Bar chart: [[label, value], ...]"}},
                                        "required": ["kind"]}},
                "accent": {"type": "string", "description": "Optional hex accent color, default #0B5ED7"},
            },
            "required": ["name", "slides"],
        },
    },
    {
        "name": "check_deck",
        "description": "Run the 8-point SKILL.md self-check (transitions, out-of-bounds, font floor, empty boxes, word cap, contrast, overflow estimate, overlap) on a .pptx; reports each defect with the slide number and the exact field/value that failed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the .pptx file to check"},
            },
            "required": ["path"],
        },
    },
]


def handle(msg: dict) -> dict | None:
    if "id" not in msg:
        return None
    method, params = msg.get("method"), msg.get("params") or {}
    try:
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": SERVER_INFO}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = str(params.get("name") or "")
            args = params.get("arguments") or {}
            if name == "make_deck":
                text, ok = make_deck(args)
            elif name == "check_deck":
                text, ok = check_deck(args)
            else:
                return {"jsonrpc": "2.0", "id": msg["id"],
                        "error": {"code": -32602, "message": f"unknown tool {name!r}"}}
            result = {"content": [{"type": "text", "text": text}], "isError": not ok}
        else:
            return {"jsonrpc": "2.0", "id": msg["id"],
                    "error": {"code": -32601, "message": f"method {method} not supported"}}
    except Exception as exc:  # surface as JSON-RPC error, keep stdout clean
        return {"jsonrpc": "2.0", "id": msg.get("id"),
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {str(exc)[:300]}"}}
    return {"jsonrpc": "2.0", "id": msg["id"], "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
