"""Plugin: expose make_deck and check_deck as agent tools (criterion C).

Implementation lives in .agents/skills/pptx/deck_tools.py (shared with the MCP
server in .mcp.json), so this plugin and the MCP server stay behaviorally
identical. Every tool returns a string whose errors name the exact field and
reason, not just a key.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path.cwd()
_TOOLS_DIR = ROOT / ".agents" / "skills" / "pptx"

inject = ["tools"]


def _load():
    # Ngày 5: xóa cache sys.modules['deck_tools']/'helpers' trước khi import lại,
    # tránh lỗi ImportError giả khi file helper đã thêm layout mới (ví dụ
    # slide_timeline) nhưng module phiên trước vẫn cache bản cũ.
    for mod_name in ("deck_tools", "helpers"):
        sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location("deck_tools", str(_TOOLS_DIR / "deck_tools.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string",
                 "description": "File name under deck-lab/ WITHOUT .pptx; e.g. 'a13_sudulieu'. Must be a non-empty string; error otherwise."},
        "slides": {
            "type": "array",
            "description": "Ordered list of slide specs; at least 1 required. Each item must be an object with 'kind' (one of the 15 SKILL.md layouts) plus the fields that kind needs.",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["cover", "agenda", "cards", "twocol", "panels", "process",
                                      "barchart", "quote", "closing", "diagram", "picture",
                                      "table", "bento", "timeline", "kpi"],
                             "description": "Which SKILL.md layout to build."},
                    "title": {"type": "string", "description": "Main heading / action title text."},
                    "subtitle": {"type": "string", "description": "Secondary line: cover subtitle, or source line for charts/tables/pictures/kpis."},
                    "items": {"type": "array", "items": {"type": "string"},
                               "description": "Agenda rows or process step labels (strings)."},
                    "cards": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                "description": "Card rows: up to 3, each [big, label, description]."},
                    "columns": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                 "description": "Two-column/panels comparison: [header, point1, point2, point3] x 2."},
                    "data": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                              "description": "Bar chart rows: [[label, value], ...]."},
                    "top": {"type": "string", "description": "Diagram: top conclusion box text."},
                    "branches": {"type": "array", "items": {"type": "string"},
                                  "description": "Diagram: 2-4 supporting-argument labels."},
                    "evidence": {"type": "array", "items": {"type": "string"},
                                  "description": "Diagram: 2-4 evidence labels, one per branch."},
                    "caption": {"type": "string", "description": "Picture: the caption beside the image area."},
                    "header": {"type": "array", "items": {"type": "string"},
                               "description": "Table: column header labels."},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                              "description": "Table data rows, one list per row, same length as header."},
                    "cells": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                              "description": "Bento grid cells: up to 6, each [big, label, description]; cell 0 is the wide hero cell."},
                    "events": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                               "description": "Timeline stops: up to 5, each [when, what, short description]."},
                    "kpis": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                              "description": "KPI cells: up to 4, each [value, label, delta]. delta starts with '+' (up/teal arrow) or '-' (down/red arrow); a plain number defaults to the up arrow."},
                },
                "required": ["kind"],
            },
        },
        "accent": {"type": "string", "description": "Optional hex accent color, default #0B5ED7."},
    },
    "required": ["name", "slides"],
}

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string",
                 "description": "Path to the .pptx to check, relative to the workspace root or absolute; e.g. 'deck-lab/a13_sudulieu.pptx'."},
    },
    "required": ["path"],
}


def _make_deck(session, args):
    text, ok = _load().make_deck(args or {})
    return text


def _check_deck(session, args):
    text, ok = _load().check_deck(args or {})
    return text


def apply(ctx, config):
    tools = ctx.get("tools")
    tools.register(
        {"name": "make_deck",
         "description": "Build a .pptx deck in deck-lab/ from a JSON spec of slides using the 15 SKILL.md layouts (cover/agenda/cards/twocol/panels/process/barchart/quote/closing/diagram/picture/table/bento/timeline/kpi). Returns the file path, slide count, and any hard errors. Errors name the exact slide index and field that failed.",
         "parameters": _MAKE_SCHEMA, "kind": "exec"},
        _make_deck)
    tools.register(
        {"name": "check_deck",
         "description": "Run the SKILL.md self-check on a .pptx (missing transition, out-of-bounds shape, <12pt text, empty textbox, >70 words per slide, contrast <4.5:1, overflow estimate, text overlap, text spilling onto the block below). Reports each defect with the slide number and the failing value; distinguates hard errors from render-confirm warnings.",
         "parameters": _CHECK_SCHEMA, "kind": "read"},
        _check_deck)
    print("deck plugin mounted: make_deck + check_deck registered (15 layouts)")
