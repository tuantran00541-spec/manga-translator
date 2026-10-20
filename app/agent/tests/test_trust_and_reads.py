"""User plugins trusted by their bytes, look-only commands that write or run, MCP reads after untrusted content, nested .env files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent import context, registry, rules, sandbox
from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace


class Prov:
    id = "stub"
    label = "stub"


def session_in(tmp_path, folder, mode="edits", store=None):
    home = tmp_path / "home"
    root = tmp_path / folder
    root.mkdir(parents=True, exist_ok=True)
    manager = AgentSessionManager(store or home / ".manga-agent", home=home)
    return manager.create(Prov(), "k", "m", Workspace(root, sandbox.Policy("workspace-write", False)), mode, complete=lambda *a, **k: {"text": ""})


def test_a_user_plugin_waits_for_trust_then_loads_in_every_workspace_until_its_bytes_change(tmp_path):
    plugins = tmp_path / "home" / ".manga-agent" / "plugins"
    plugins.mkdir(parents=True)
    source = plugins / "hello.py"
    source.write_text("def register(api):\n    api.tool({'name': 'hello', 'description': 'hi'}, lambda s, a: 'hi', kind='read')\n")
    first = session_in(tmp_path, "one")
    assert [p["state"] for p in first.registry.plugins] == ["untrusted"] and "hello" not in first.registry.tools
    first.trust_plugins()
    assert "hello" in first.registry.tools
    assert "hello" in session_in(tmp_path, "two").registry.tools
    source.write_text(source.read_text() + "# changed\n")
    assert "hello" not in session_in(tmp_path, "three").registry.tools


def test_look_only_programs_that_write_a_file_or_run_another_program_are_not_look_only():
    for command in ("rg --pre ./x.sh foo .", "rg --pre=sh foo", "sort -o out.txt in.txt", "sort -ro out.txt in.txt", "sort --output=o in",
                    "uniq in.txt out.txt", "tree -o out.txt", "date -s 2020-01-01"):
        assert not rules.safe_readonly(command), command
    for command in ("rg foo .", "sort -n in.txt", "sort --reverse in.txt", "uniq -c in.txt", "tree -L 2", "date -u", "cat a | sort | uniq"):
        assert rules.safe_readonly(command), command


def test_an_mcp_tool_named_like_a_read_asks_once_the_session_read_untrusted_content(tmp_path):
    session = session_in(tmp_path, "one")
    session.mcp_tools["mcp__web__fetch_url"] = ("web", {"name": "fetch_url"})
    call = {"name": "mcp__web__fetch_url", "args": {"url": "https://example.com/?q=secret"}}
    assert not session._needs_approval(call)
    session.tainted = True
    assert session._needs_approval(call)
    session.profile["roles"]["mcp__web__fetch_url"] = "read"
    assert not session._needs_approval(call)


def test_dotenv_files_in_sub_folders_are_kept_from_commands_too(tmp_path):
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / ".env").write_text("KEY=1")
    (tmp_path / ".env.example").write_text("KEY=")
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / ".env").write_text("KEY=2")
    found = sandbox.workspace_secrets(tmp_path)
    assert str((tmp_path / "api" / ".env").resolve()) in found
    assert not any(p.endswith(".env.example") or "node_modules" in p for p in found)


def test_the_trust_store_keys_user_plugins_by_home_not_by_workspace(tmp_path):
    trust = context.TrustStore(None)
    home, path = tmp_path / "home", tmp_path / "home" / ".manga-agent" / "plugins" / "a.py"
    registry.trust_user_plugin(trust, home, path, b"x")
    assert registry.user_trusted(trust, home, path, b"x") and not registry.user_trusted(trust, home, path, b"y")
