"""Open MCP: tool filters and kinds, prompts as commands, resources, images, changed tool lists and a restart."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from app.agent import context, mcp, sandbox
from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()

SERVER = f'''
import json, os, sys
tools = [{{"name": "look", "description": "Look."}}, {{"name": "deploy", "description": "Deploy."}},
         {{"name": "picture"}}, {{"name": "grow"}}, {{"name": "die"}}]
def send(msg):
    sys.stdout.write(json.dumps(msg) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {{}}
    reply = lambda result: send({{"jsonrpc": "2.0", "id": mid, "result": result}})
    if method == "initialize":
        reply({{"protocolVersion": "2025-06-18", "capabilities": {{"tools": {{}}, "prompts": {{}}, "resources": {{}}}}}})
    elif method == "tools/list":
        reply({{"tools": tools}})
    elif method == "prompts/list":
        reply({{"prompts": [{{"name": "review", "description": "Review a file.", "arguments": [{{"name": "path"}}]}}]}})
    elif method == "prompts/get":
        reply({{"messages": [{{"role": "user", "content": {{"type": "text", "text": "Review " + params["arguments"].get("path", "?")}}}}]}})
    elif method == "resources/list":
        reply({{"resources": [{{"uri": "note://1", "name": "first note"}}]}})
    elif method == "resources/read":
        reply({{"contents": [{{"uri": params["uri"], "text": "note body"}}]}})
    elif method == "tools/call":
        name = params["name"]
        if name == "picture":
            reply({{"content": [{{"type": "image", "data": "{PNG}", "mimeType": "image/png"}}]}})
        elif name == "grow":
            tools.append({{"name": "new_tool"}})
            reply({{"content": [{{"type": "text", "text": "grown"}}]}})
            send({{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}})
        elif name == "die":
            os._exit(3)
        else:
            reply({{"content": [{{"type": "text", "text": name + " ran pid " + str(os.getpid())}}]}})
'''


class _Prov:
    id = "stub"
    label = "stub"


def _session(tmp_path, extra=None, scope="user"):
    home, root = tmp_path / "home", tmp_path / "ws"
    root.mkdir(parents=True)
    script = tmp_path / "srv.py"
    script.write_text(SERVER, encoding="utf-8")
    config = {"command": sys.executable, "args": [str(script)], **(extra or {})}
    if scope == "user":
        mcp.save_user(home, {"fake": mcp.clean_config(config)})
    else:
        (root / ".mcp.json").write_text(json.dumps({"mcpServers": {"fake": config}}), encoding="utf-8")
    mgr = AgentSessionManager(home / ".manga-agent", home=home, trust=context.TrustStore(None))
    s = mgr.create(_Prov(), "k", "m", Workspace(root, sandbox.Policy("workspace-write", False)), "edits",
                   complete=lambda *a, **k: {"text": ""})
    if scope == "workspace":
        s.trust_mcp("fake")
    s._ensure_mcp()
    return s, home


def test_allow_and_deny_lists_decide_which_tools_the_model_gets(tmp_path):
    s, home = _session(tmp_path, {"tools": {"deny": ["deploy"]}})
    try:
        assert "mcp__fake__look" in s.mcp_tools and "mcp__fake__deploy" not in s.mcp_tools
        mcp.set_user_tool(home, "fake", "deploy", True)
        mcp.set_user_tool(home, "fake", "look", False)
        s._refresh_mcp()
        assert "mcp__fake__deploy" in s.mcp_tools and "mcp__fake__look" not in s.mcp_tools
        assert {t["name"] for t in s.mcp_servers["fake"].all_tools} >= {"look", "deploy"}
    finally:
        s.close()


def test_kinds_in_your_own_config_set_approval_but_a_projects_cannot(tmp_path):
    s, _ = _session(tmp_path, {"kinds": {"look": "read"}})
    try:
        assert not s._needs_approval({"name": "mcp__fake__look", "args": {}})
        assert s._needs_approval({"name": "mcp__fake__deploy", "args": {}})
    finally:
        s.close()
    project, _ = _session(tmp_path / "p", {"kinds": {"deploy": "read"}}, scope="workspace")
    try:
        assert "mcp__fake__deploy" in project.mcp_tools
        assert project._needs_approval({"name": "mcp__fake__deploy", "args": {}}), "a project's file cannot mark its own tool read"
    finally:
        project.close()


def test_prompts_become_slash_commands_and_fill_their_first_argument(tmp_path):
    s, _ = _session(tmp_path)
    sent = []
    s.send = lambda text, scheduled=False: sent.append(text) or True
    try:
        assert "/mcp:fake:review" in s.command("/help")["message"]
        assert s.command("/mcp:fake:review app.py") == {"sent": True}
        s.command("/mcp:fake:review path=b.py")
        assert sent == ["Review app.py", "Review b.py"]
    finally:
        s.close()


def test_resources_are_listed_and_read_and_count_as_outside_content(tmp_path):
    s, _ = _session(tmp_path)
    try:
        assert "mcp_resource" in {r["name"] for r in s.library_specs()}
        out, ok = s._run_call({"id": "chain-r1", "name": "mcp_resource", "args": {"server": "fake"}})
        assert ok and "note://1" in out
        out, ok = s._run_call({"id": "chain-r2", "name": "mcp_resource", "args": {"server": "fake", "uri": "note://1"}})
        assert ok and out == "note body" and s.tainted
    finally:
        s.close()


def test_an_image_is_saved_for_view_image(tmp_path):
    s, _ = _session(tmp_path)
    try:
        out, _ = s.mcp_servers["fake"].call_tool("picture", {})
        path = Path(out.split("saved to ")[1].split(";")[0])
        assert path.read_bytes() == base64.b64decode(PNG) and "view_image" in out
    finally:
        s.close()


def test_a_changed_tool_list_is_read_again_before_the_next_turn(tmp_path):
    s, _ = _session(tmp_path)
    try:
        s.mcp_servers["fake"].call_tool("grow", {})
        server = s.mcp_servers["fake"]
        for _ in range(100):
            if server.tools_stale:
                break
            __import__("time").sleep(0.02)
        s._sync_mcp_tools()
        assert "mcp__fake__new_tool" in s.mcp_tools
    finally:
        s.close()


def test_a_server_that_died_is_started_again_for_the_next_call(tmp_path):
    s, _ = _session(tmp_path)
    try:
        server = s.mcp_servers["fake"]
        with pytest.raises(mcp.MCPError):
            server.call_tool("die", {})
        text, ok = server.call_tool("look", {})
        assert ok and text.startswith("look ran") and server.restarts == 1
    finally:
        s.close()


def test_clean_config_keeps_the_harness_settings_and_drops_bad_ones():
    out = mcp.clean_config({"command": "x", "tools": {"allow": ["a"], "deny": ["b", 3]}, "kinds": {"a": "read", "b": "root"}, "timeout": 30})
    assert out["tools"] == {"allow": ["a"], "deny": ["b"]} and out["kinds"] == {"a": "read"} and out["timeout"] == 30
    assert "timeout" not in mcp.clean_config({"command": "x", "timeout": 1})
