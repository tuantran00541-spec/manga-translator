"""The Windows low-integrity sandbox and the Docker engine, each tested on a machine that has it."""
from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import tempfile
import time

import pytest

from app.agent import sandbox

WINDOWS = platform.system() == "Windows"


@pytest.fixture
def home(tmp_path, monkeypatch):
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setattr(Path, "home", lambda: folder)
    monkeypatch.delenv("MANGA_AGENT_SANDBOX", raising=False)
    return folder


def test_engine_choice_is_saved_checked_and_shown(home, monkeypatch):
    assert sandbox.settings() == {"engine": "auto", "image": sandbox.DOCKER_IMAGE}
    assert sandbox.save_settings("docker", "node:20-slim") == {"engine": "docker", "image": "node:20-slim"}
    with pytest.raises(ValueError):
        sandbox.save_settings("vm")
    with pytest.raises(ValueError):
        sandbox.save_settings("docker", "img; rm -rf /")
    monkeypatch.setattr(sandbox, "docker_ready", lambda: True)
    assert sandbox.backend() == "docker" and sandbox.full()
    args, shell = sandbox.wrap("ls", sandbox.Policy("read-only", False), home, "/scratch")
    assert not shell and args[:2] == ["docker", "run"] and f"{home}:/workspace:ro" in args and "--network" in args and args[-4:] == ["node:20-slim", "sh", "-c", "ls"]
    assert "/workspace" in sandbox.Policy().describe(home)
    monkeypatch.setattr(sandbox, "docker_ready", lambda: False)
    assert sandbox.backend() != "docker", "a chosen engine that does not answer falls back to the OS sandbox"


def test_the_engine_endpoint_saves_the_choice(home, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers import agent

    monkeypatch.setattr(agent, "is_loopback", lambda host: host == "testclient")

    client = TestClient(app, base_url="http://127.0.0.1", headers={"X-Manga-Agent": "1"})
    assert client.post("/api/agent/sandbox/engine", json={"engine": "docker", "image": "python:3.11"}).json()["image"] == "python:3.11"
    assert client.get("/api/agent/config").json()["engine"]["engine"] == "docker"
    assert client.post("/api/agent/sandbox/engine", json={"engine": "vm"}).status_code == 400


def _run(command: str, root: Path, mode: str = "workspace-write", timeout: int = 60):
    return sandbox.run(command, sandbox.Policy(mode, False), root, timeout)


@pytest.mark.skipif(not WINDOWS, reason="the low-integrity sandbox is Windows only")
def test_windows_commands_write_only_inside_the_workspace(tmp_path, home):
    assert sandbox.backend() == "low-integrity" and not sandbox.full()
    root, outside = tmp_path / "ws", Path(tempfile.mkdtemp(prefix="outside-"))
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "old.txt").write_text("old")
    code, out = _run("echo made> new.txt && echo changed> pkg\\old.txt && python -c \"print(6*7)\" && git --version", root)
    assert code == 0, out
    assert "42" in out and "git version" in out
    assert (root / "new.txt").read_text().strip() == "made" and (root / "pkg" / "old.txt").read_text().strip() == "changed"
    code, out = _run(f'echo bad> "{outside / "x.txt"}"', root)
    assert code != 0 and not (outside / "x.txt").exists(), out
    code, out = _run(f'python -c "open(r\'{home / "y.txt"}\', \'w\')"', root)
    assert code != 0 and not (home / "y.txt").exists(), out
    code, out = _run('python -c "import tempfile; f = tempfile.NamedTemporaryFile(delete=False); f.write(b\'1\'); print(\'temp ok\')"', root)
    assert code == 0 and "temp ok" in out, out


@pytest.mark.skipif(not WINDOWS, reason="the low-integrity sandbox is Windows only")
def test_windows_read_only_commands_read_but_cannot_write_a_fresh_workspace_or_outside(tmp_path, home):
    root, outside = tmp_path / "ws", Path(tempfile.mkdtemp(prefix="outside-"))
    root.mkdir()
    (root / "a.txt").write_text("first")
    code, out = _run('python -c "print(open(\'a.txt\').read())"', root, mode="read-only")
    assert code == 0 and "first" in out, out
    code, out = _run("echo again> b.txt", root, mode="read-only")
    assert code != 0 and not (root / "b.txt").exists(), out
    code, out = _run(f'echo bad> "{outside / "x.txt"}"', root, mode="read-only")
    assert code != 0 and not (outside / "x.txt").exists(), out


@pytest.mark.skipif(not WINDOWS, reason="the low-integrity sandbox is Windows only")
def test_windows_exit_codes_pass_through_and_a_timeout_kills_the_whole_command(tmp_path, home):
    root = tmp_path / "ws"
    root.mkdir()
    assert _run("exit /b 7", root)[0] == 7
    started = time.time()
    code, _ = _run('python -c "import time; open(\'alive.txt\', \'w\').close(); time.sleep(30); open(\'late.txt\', \'w\').close()"', root, timeout=5)
    assert code is None and time.time() - started < 20
    time.sleep(1)
    assert (root / "alive.txt").exists() and not (root / "late.txt").exists()


def _docker() -> bool:
    return sandbox.docker_ready()


@pytest.mark.skipif(not _docker(), reason="needs a Docker daemon running Linux containers")
def test_docker_commands_see_only_the_workspace_and_no_network(tmp_path, home, monkeypatch):
    monkeypatch.setenv("MANGA_AGENT_SANDBOX", "docker")
    root = tmp_path / "ws"
    root.mkdir()
    (root / "in.txt").write_text("hello")
    assert sandbox.backend() == "docker"
    code, out = _run("cat in.txt && echo made > out.txt && pwd", root, timeout=300)
    assert code == 0 and "hello" in out and "/workspace" in out and (root / "out.txt").read_text().strip() == "made", out
    code, out = _run(f"ls {home}", root)
    assert code != 0, out
    code, out = _run("python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"", root)
    assert code != 0, out
    code, out = _run("echo x > nope.txt", root, mode="read-only")
    assert code != 0 and not (root / "nope.txt").exists(), out
    code, _ = _run("sleep 60", root, timeout=3)
    time.sleep(2)
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    assert code is None and "manga-agent-" not in names


SCRIPT = "from pathlib import Path\nPath('made.txt').write_text('ok')\nprint('wrote', Path('made.txt').read_text())\ntry:\n    tools.read_file(path='x')\nexcept RuntimeError as e:\n    print('no bridge')\n"


@pytest.mark.skipif(not WINDOWS, reason="the low-integrity sandbox is Windows only")
def test_windows_runs_created_tool_scripts_without_the_tool_bridge(tmp_path, home):
    from app.agent import codemode
    root = tmp_path / "ws"
    root.mkdir()
    code, out, calls = codemode.run(SCRIPT, sandbox.Policy("workspace-write", False), root, 60, lambda n, a: ("", False))
    assert code == 0 and "wrote ok" in out and "no bridge" in out and (root / "made.txt").read_text() == "ok", out


@pytest.mark.skipif(not _docker(), reason="needs a Docker daemon running Linux containers")
def test_docker_runs_created_tool_scripts_without_the_tool_bridge(tmp_path, home, monkeypatch):
    from app.agent import codemode
    monkeypatch.setenv("MANGA_AGENT_SANDBOX", "docker")
    root = tmp_path / "ws"
    root.mkdir()
    code, out, calls = codemode.run(SCRIPT, sandbox.Policy("workspace-write", False), root, 300, lambda n, a: ("", False))
    assert code == 0 and "wrote ok" in out and "no bridge" in out and (root / "made.txt").read_text() == "ok", out


def _created_tool_session(tmp_path, home):
    from app.agent.session import AgentSessionManager
    from app.agent.tools import Workspace
    from app.ai_providers import PROVIDERS
    root = tmp_path / "ws"
    root.mkdir()
    (home / ".manga-agent").mkdir(exist_ok=True)
    (home / ".manga-agent" / "profile.json").write_text('{"tools": "create"}')
    session = AgentSessionManager(None, home=home).create(PROVIDERS["openai"], "k", "m", Workspace(root, sandbox.Policy("workspace-write", False)), "bypass",
                                                          complete=lambda *a, **k: {"text": "x", "calls": [], "usage": {}})
    session._new_request("go")
    session._create({"name": "put", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                     "code": "import os\ndef put(text):\n    open('made.txt', 'w').write(text)\n    open(os.path.join(os.environ['AGENT_STATE'], 's'), 'a').write(text)\n"
                             "    return open(os.path.join(os.environ['AGENT_STATE'], 's')).read()\n"})
    return session, root


@pytest.mark.skipif(not WINDOWS, reason="the low-integrity sandbox is Windows only")
def test_windows_runs_created_tools_with_state_and_change_tracking(tmp_path, home):
    session, root = _created_tool_session(tmp_path, home)
    session._run_call({"id": "a", "name": "put", "args": {"text": "1"}})
    out, ok = session._run_call({"id": "b", "name": "put", "args": {"text": "2"}})
    assert ok and "12" in out and "[changed: made.txt]" in out and (root / "made.txt").read_text() == "2", out


@pytest.mark.skipif(not _docker(), reason="needs a Docker daemon running Linux containers")
def test_docker_runs_created_tools_with_state_and_change_tracking(tmp_path, home, monkeypatch):
    monkeypatch.setenv("MANGA_AGENT_SANDBOX", "docker")
    session, root = _created_tool_session(tmp_path, home)
    session._run_call({"id": "a", "name": "put", "args": {"text": "1"}})
    out, ok = session._run_call({"id": "b", "name": "put", "args": {"text": "2"}})
    assert ok and "12" in out and "[changed: made.txt]" in out and (root / "made.txt").read_text() == "2", out


def _bridge_check(tmp_path, home):
    skill = home / ".manga-agent" / "skills" / "style"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: style\ndescription: House style.\n---\nUse tabs.\n")
    session, root = _created_tool_session(tmp_path, home)
    session.reload_skills()
    out = session._session_tool({"id": "c", "name": "create", "args": {"name": "mix", "keep": False, "args": {},
                                 "code": "def mix():\n    return tools.put(text='z') + '|' + str('Use tabs.' in tools.skill('style'))\n"}})
    assert "z|True" in out, out


@pytest.mark.skipif(not WINDOWS, reason="the low-integrity sandbox is Windows only")
def test_windows_chains_reach_the_agent_through_the_bridge(tmp_path, home):
    _bridge_check(tmp_path, home)


@pytest.mark.skipif(not _docker(), reason="needs a Docker daemon running Linux containers")
def test_docker_chains_reach_the_agent_through_the_bridge(tmp_path, home, monkeypatch):
    monkeypatch.setenv("MANGA_AGENT_SANDBOX", "docker")
    _bridge_check(tmp_path, home)


def test_chains_reach_the_agent_through_the_bridge_here(tmp_path, home):
    _bridge_check(tmp_path, home)
