"""MCP client test: drives deck-lab/mcp_server.py over stdio JSON-RPC.

Sends initialize, notifications/initialized, tools/list, and a tools/call
that builds a real .pptx, then verifies the file exists on disk.
"""
import json
import os
import subprocess
import sys

SERVER = os.path.abspath("mcp_server.py")
OUT = os.path.abspath("pv4/mcp_smoke.pptx")

msgs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "deck-lab-test", "version": "0"},
    }},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "make_deck",
        "arguments": {
            "path": OUT,
            "spec": json.dumps({
                "theme": "dark",
                "transition": "wipe",
                "slides": [
                    {"layout": "title", "title": "MCP SMOKE", "subtitle": "deck server",
                     "desc": "Built through JSON-RPC 2.0 over stdio, not the in-agent tool.",
                     "footer": "deck-lab mcp check"},
                    {"layout": "process", "title": "The Loop", "steps": [
                        {"title": "initialize", "desc": "Client and server agree on the protocol version."},
                        {"title": "list", "desc": "tools/list returns make_deck and check_deck_tool."},
                        {"title": "call", "desc": "tools/call builds a real pptx from a JSON spec."},
                        {"title": "verify", "desc": "The file exists on disk and passes QC."}]},
                    {"layout": "numbered_rows", "title": "Why MCP", "rows": [
                        {"title": "Portable", "desc": "Any MCP host (Claude, VS Code, this agent) drives the same builder."},
                        {"title": "Stateless", "desc": "Each call carries the full spec; no session to babysit."},
                        {"title": "Checkable", "desc": "check_deck_tool rides the same transport, so build and QC stay paired."}]},
                ],
            }),
        },
    }},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
        "name": "check_deck_tool",
        "arguments": {"path": OUT},
    }},
]

proc = subprocess.Popen(
    [sys.executable, SERVER],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True,
    cwd=os.path.abspath(os.path.join(os.path.dirname(SERVER), "..")),
)
payload = "\n".join(json.dumps(m) for m in msgs) + "\n"
out, err = proc.communicate(input=payload, timeout=120)
proc.wait(timeout=10)

print("=== server stdout ===")
# Tolerate non-JSON lines that a legacy build may have leaked (pre-fix).
responses = []
for l in out.strip().splitlines():
    if not l.strip():
        continue
    try:
        responses.append(json.loads(l))
    except json.JSONDecodeError:
        print("[non-json line ignored]", l[:60])
for r in responses:
    rid = r.get("id")
    if rid == 1:
        info = r["result"]["serverInfo"]
        print(f"initialize -> {info['name']} v{info['version']} (protocol {r['result']['protocolVersion']})")
    elif rid == 2:
        tools = [t["name"] for t in r["result"]["tools"]]
        print(f"tools/list -> {tools}")
        assert "make_deck" in tools and "check_deck_tool" in tools
    elif rid in (3, 4):
        text = r["result"]["content"][0]["text"]
        is_err = r["result"].get("isError", False)
        print(f"tools/call (id {rid}) [error={is_err}]: {text}")
        assert not is_err, text

print(f"\npptx on disk: {os.path.exists(OUT)} ({os.path.getsize(OUT) if os.path.exists(OUT) else 0} bytes)")
assert os.path.exists(OUT)
if proc.returncode:
    print("stderr:", err)
print("\n=== MCP SMOKE PASSED ===")
