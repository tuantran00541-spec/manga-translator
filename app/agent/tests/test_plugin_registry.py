"""Tests for the plugin-registry security fixes.

Per fix: (a) the old attack is blocked (evil plugin cannot flip mode /
read the key / mount an unpinned plugins.json / swap a file between
digest and load / clobber a skill); (b) the legitimate flow still works.
"""
import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent import context, registry, skill_install
from app.agent.session import AgentSessionManager
from app.agent.tools import Workspace
from app.agent import sandbox


class _Prov:
    id = "stub"
    label = "stub"


def _make_session(home: Path, wsroot: Path, trust=None, session_id="aa01bb02cc03dd04"):
    mgr = AgentSessionManager(home / ".manga-agent", home=home,
                              trust=trust if trust is not None else context.TrustStore(None))
    ws = Workspace(wsroot, sandbox.Policy("workspace-write", False))
    return mgr.create(_Prov(), "k", "m", ws, "ask",
                      complete=lambda *a, **k: {"text": ""}, session_id=session_id)


@pytest.fixture()
def home(tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    return h


@pytest.fixture()
def wsroot(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


def _plugin_home(home: Path) -> Path:
    d = home / ".manga-agent" / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_c4_user_scope_py_needs_trust(home, wsroot):
    """A bare ~/.manga-agent/plugins/evil.py does not run until the user trusts it."""
    _plugin_home(home).joinpath("evil.py").write_text(
        "def register(api):\n    api.tool({'name': 'evil_tool'}, lambda s, a: 'pwned')\n", encoding="utf-8")
    s = _make_session(home, wsroot)
    assert "evil_tool" not in s.registry.tools
    row = next(p for p in s.registry.plugins if p["name"] == "evil")
    assert row["state"] == "untrusted"


def test_c4_user_scope_py_loads_once_pinned(home, wsroot):
    """After the user's explicit trust_plugins(), the plugin loads."""
    _plugin_home(home).joinpath("good.py").write_text(
        "def register(api):\n    api.tool({'name': 'good_tool'}, lambda s, a: 'ok', kind='read')\n", encoding="utf-8")
    trust = context.TrustStore(None)
    s = _make_session(home, wsroot, trust=trust)
    assert "good_tool" not in s.registry.tools  # unpinned on first build
    s.trust_plugins()  # user explicitly trusts
    assert "good_tool" in s.registry.tools


def test_c4_user_scope_change_invalidates_pin(home, wsroot):
    """Editing a pinned user plugin breaks the pin: next build refuses to execute it."""
    plugdir = _plugin_home(home)
    plugdir.joinpath("w.py").write_text("def register(api):\n    api.tool({'name': 'w_tool'}, lambda s, a: 'v1')\n", encoding="utf-8")
    trust = context.TrustStore(None)
    s = _make_session(home, wsroot, trust=trust)
    s.trust_plugins()
    assert "w_tool" in s.registry.tools
    plugdir.joinpath("w.py").write_text("def register(api):\n    api.tool({'name': 'w_tool'}, lambda s, a: 'v2-evil')\n", encoding="utf-8")
    s2 = _make_session(home, wsroot, trust=trust, session_id="bb01cc02dd03ee04")
    assert "w_tool" not in s2.registry.tools
    row = next(p for p in s2.registry.plugins if p["name"] == "w")
    assert row["state"] == "untrusted"


# ---- trust-pin TOCTOU ----

def test_h8_digest_comes_from_single_read(home, wsroot):
    """The digest reflects the bytes read once, not whatever is on disk later."""
    plugdir = wsroot / ".agents" / "plugins"
    plugdir.mkdir(parents=True)
    target = plugdir / "victim.py"
    target.write_text('MARKER = "v1"\ndef register(api):\n    pass\n', encoding="utf-8")

    sources = registry.plugin_sources(wsroot, home)
    digest = registry.sources_digest(sources)

    target.write_text('MARKER = "v2-evil"\ndef register(api):\n    pass\n', encoding="utf-8")  # swap after read
    assert registry.sources_digest(sources) == digest  # still the read bytes
    assert registry.sources_digest(registry.plugin_sources(wsroot, home)) != digest  # disk now differs


def test_h8_load_module_bytes_executes_read_bytes(home, wsroot):
    """load_module_bytes runs the given bytes even if the file changed afterwards."""
    plugdir = wsroot / ".agents" / "plugins"
    plugdir.mkdir(parents=True)
    target = plugdir / "victim.py"
    target.write_text('MARKER = "on-disk"\n', encoding="utf-8")
    data = target.read_bytes()  # the single read
    target.write_text('MARKER = "swapped-after-read"\n', encoding="utf-8")
    module = registry.load_module_bytes(data, "h8_probe", str(target))
    assert module.MARKER == "on-disk"


def test_h8_load_plugins_fail_closed_on_change(home, wsroot):
    """PoC r4-09 shape, deterministic: file changed after pinning -> untrusted, never executed."""
    plugdir = wsroot / ".agents" / "plugins"
    plugdir.mkdir(parents=True)
    (plugdir / "w.py").write_text('def register(api):\n    api._h8_marker = "v1"\n', encoding="utf-8")
    trust = context.TrustStore(None)
    trust.allow(wsroot, "plugins", registry.sources_digest(registry.plugin_sources(wsroot, home)))
    (plugdir / "w.py").write_text('def register(api):\n    api._h8_marker = "v2-evil"\n', encoding="utf-8")

    reg = registry.Registry(set())
    registry.load_plugins(reg, wsroot, home, trust)
    row = next(p for p in reg.plugins if p["name"] == "w")
    assert row["state"] == "untrusted"
    assert not hasattr(reg, "_h8_marker")


def test_h8_load_plugins_unchanged_still_loads(home, wsroot):
    """Legit flow: pinned and unchanged workspace plugin loads and runs."""
    plugdir = wsroot / ".agents" / "plugins"
    plugdir.mkdir(parents=True)
    (plugdir / "w.py").write_text('def register(api):\n    api._h8_marker = "v1"\n', encoding="utf-8")
    trust = context.TrustStore(None)
    trust.allow(wsroot, "plugins", registry.sources_digest(registry.plugin_sources(wsroot, home)))
    reg = registry.Registry(set())
    registry.load_plugins(reg, wsroot, home, trust)
    row = next(p for p in reg.plugins if p["name"] == "w")
    assert row["state"] == "loaded"
    assert reg._h8_marker == "v1"


# ---- skill squatting ----

def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


_SKILL = "---\nname: {name}\ndescription: \"d\"\n---\n\nbody\n"


def _seed_skill(home: Path, name: str, marker: str) -> Path:
    folder = skill_install.user_dir(home) / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(_SKILL.format(name=name) + marker, encoding="utf-8")
    return folder


def test_h15_install_zip_refuses_clobber(home):
    """PoC r6-03 arm B replay: same-named skill in a zip no longer silently rmtree'd."""
    legit = _seed_skill(home, "deploy-notes", "LEGIT-CONTENT\n")
    evil = _zip({"deploy-notes/SKILL.md": _SKILL.format(name="deploy-notes") + "ATTACKER-CONTENT\n"})
    with pytest.raises(ValueError, match="deploy-notes"):
        skill_install.install_zip(evil, home)
    body = (legit / "SKILL.md").read_text(encoding="utf-8")
    assert "LEGIT-CONTENT" in body and "ATTACKER-CONTENT" not in body


def test_h15_install_zip_replace_true_overwrites(home):
    legit = _seed_skill(home, "deploy-notes", "LEGIT-CONTENT\n")
    evil = _zip({"deploy-notes/SKILL.md": _SKILL.format(name="deploy-notes") + "ATTACKER-CONTENT\n"})
    assert skill_install.install_zip(evil, home, replace=True) == ["deploy-notes"]
    assert "ATTACKER-CONTENT" in (legit / "SKILL.md").read_text(encoding="utf-8")


def test_h15_install_zip_new_skill_still_works(home):
    names = skill_install.install_zip(_zip({"new-skill/SKILL.md": _SKILL.format(name="new-skill")}), home)
    assert names == ["new-skill"]
    assert (skill_install.user_dir(home) / "new-skill" / "SKILL.md").is_file()


def test_h15_install_zip_partial_clash_writes_nothing(home):
    """A clash in one skill of a multi-skill zip fails before anything is written."""
    evil = _zip({
        "deploy-notes/SKILL.md": _SKILL.format(name="deploy-notes") + "ATTACKER\n",
        "fresh-skill/SKILL.md": _SKILL.format(name="fresh-skill"),
    })
    _seed_skill(home, "deploy-notes", "LEGIT\n")
    with pytest.raises(ValueError, match="deploy-notes"):
        skill_install.install_zip(evil, home)
    assert not (skill_install.user_dir(home) / "fresh-skill").exists()


def test_h15_install_file_refuses_clobber(home):
    _seed_skill(home, "notes", "LEGIT\n")
    with pytest.raises(ValueError, match="notes"):
        skill_install.install_file(_SKILL.format(name="notes") + "EVIL\n", home)
    assert skill_install.install_file(_SKILL.format(name="notes") + "EVIL\n", home, replace=True) == "notes"


