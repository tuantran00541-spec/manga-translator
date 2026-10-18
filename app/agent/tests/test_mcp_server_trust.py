"""Regression tests for the --trust-plugins narrowing of the C4 plugin trust gate.

The C4 fix made user-scope plugins (and plugins.json rows) default-deny: they only
execute after an explicit trust pin. The harness-as-MCP-server path
(app/agent/mcp_server.py) had no way to express that trust from its CLI, so a
legitimate flow -- serving a trusted plugin's tools to another MCP client -- was
killed. The --trust-plugins flag restores it as an explicit user ceremony while
keeping default-deny.

These tests prove both halves:
1. Attack stays blocked: a planted ~/.manga-agent/plugins/*.py is NOT served by default.
2. Legitimate flow works: with trust_plugins=True (the --trust-plugins flag), it IS served.
"""
import textwrap

import pytest

from app.agent.mcp_server import Bridge

GREETER = textwrap.dedent('''
    inject = ["tools"]

    def apply(ctx, config):
        ctx.get("tools").register({"name": "greet", "description": "Greet someone.",
                                   "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}},
                                  lambda session, args: "hi " + args.get("name", ""), kind="read")
''')


@pytest.fixture
def plugin_home(tmp_path):
    home = tmp_path / "home"
    (home / ".manga-agent" / "plugins").mkdir(parents=True)
    (home / ".manga-agent" / "plugins" / "greeter.py").write_text(GREETER, encoding="utf-8")
    return home


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def tool_names(bridge):
    return {t["name"] for t in bridge.tools()}


def test_untrusted_plugin_tool_not_served_by_default(plugin_home, folder):
    """Attack stays blocked: a planted user-scope plugin stays untrusted and its
    tool is not served unless the user explicitly passes --trust-plugins."""
    bridge = Bridge(folder, home=plugin_home)
    try:
        assert "greet" not in tool_names(bridge), "untrusted plugin must not be served by default"
        text, ok = bridge.call("greet", {"name": "Kai"})
        assert not ok and "Unknown tool" in text
    finally:
        bridge.close()


def test_trusted_plugin_tool_served_with_explicit_trust(plugin_home, folder):
    """Legitimate flow: the explicit --trust-plugins ceremony serves the plugin's tool."""
    bridge = Bridge(folder, home=plugin_home, trust_plugins=True)
    try:
        assert "greet" in tool_names(bridge)
        text, ok = bridge.call("greet", {"name": "Kai"})
        assert (text, ok) == ("hi Kai", True)
    finally:
        bridge.close()


def test_trust_does_not_survive_plugin_change_without_retrust(plugin_home, folder):
    """Changing the plugin file invalidates the pin: the new code is not served
    until trust is given again (mirrors the in-session re-gating)."""
    bridge = Bridge(folder, home=plugin_home, trust_plugins=True)
    try:
        assert "greet" in tool_names(bridge)
    finally:
        bridge.close()
    (plugin_home / ".manga-agent" / "plugins" / "greeter.py").write_text(
        GREETER.replace('"hi " + args.get("name", "")', '"pwned " + args.get("name", "")'), encoding="utf-8")
    bridge2 = Bridge(folder, home=plugin_home)
    try:
        assert "greet" not in tool_names(bridge2), "changed plugin must not be served without fresh trust"
    finally:
        bridge2.close()
