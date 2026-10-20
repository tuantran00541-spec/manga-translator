"""Regression tests for the misc security fixes.

For each fix: (a) a test asserting the old attack is blocked on the
fixed code, and (b) a check that the legitimate behavior still works.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import stat
import threading
from pathlib import Path

import pytest

from app.agent import memory, roles, watch
from app.agent.session import AgentSessionManager
from app.agent import sandbox
from app.agent.tools import Workspace

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---- fixtures ----

class _Prov:
    id = "stub"
    label = "stub"


@pytest.fixture()
def agent_home(tmp_path):
    home = tmp_path / "home"
    (home / ".manga-agent").mkdir(parents=True)
    return home


@pytest.fixture()
def ws_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "app.py").write_text("print('v1')\n", encoding="utf-8")
    return root


@pytest.fixture()
def ask_session(agent_home, ws_root):
    """A real AgentSession in ask mode, like the r5-15 PoC builds it."""
    mgr = AgentSessionManager(agent_home / ".manga-agent", home=agent_home)
    ws = Workspace(ws_root, sandbox.Policy("workspace-write", False))
    session = mgr.create(_Prov(), "k", "m", ws, "ask",
                         complete=lambda *a, **k: {"text": ""},
                         session_id="misc-test-01")
    yield session
    session.close()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---- tools.branch run_command needs approval ----

class TestH11BranchApproval:
    def _run_branch_check(self, session, command, decision):
        asked = []

        def decide(call):
            asked.append(call)
            return {"decision": decision}

        session._wait_for_decision = decide
        out, ok = session._branch({"branches": {"a": "1", "b": "2"},
                                   "check": f"tools.run_command(command={command!r})"})
        return out, ok, asked

    def test_branch_run_command_asks_for_approval(self, ask_session, agent_home):
        """the branch check's run_command must go through
        _run_call, so in ask mode the user is asked; a denial runs nothing."""
        out, ok, asked = self._run_branch_check(ask_session, "touch h11_marker", "deny")
        assert asked, "branch run_command never reached the approval path"
        assert asked[0]["name"] == "run_command"
        markers = list((agent_home / ".manga-agent" / "outputs").rglob("h11_marker"))
        assert not markers, "branch run_command executed despite the denial"
        assert "refused" in out.lower() or "deny" in out.lower() or "denied" in out.lower()

    def test_branch_tools_still_work_when_allowed(self, ask_session, agent_home):
        """Legitimate behavior: an allowed branch check still runs on the copy."""
        out, ok, asked = self._run_branch_check(ask_session, "touch h11_ok_marker", "allow")
        assert asked, "approval path was not consulted"
        markers = list((agent_home / ".manga-agent" / "outputs").rglob("h11_ok_marker"))
        assert markers, "allowed branch run_command did not execute"
        # The swap is unwound: the session still points at the real workspace.
        assert ask_session.workspace.root.name == "ws"

    def test_branch_read_file_routed_through_run_call(self, ask_session):
        """A read-only branch tool also passes through _run_call (rules/taint/hooks)."""
        seen = []
        orig = ask_session._run_call

        def spy(call):
            seen.append(call)
            return orig(call)

        ask_session._run_call = spy
        asked = []
        ask_session._wait_for_decision = lambda call: (asked.append(call), {"decision": "allow"})[1]
        out, ok = ask_session._run_branch_tool(
            Workspace(ask_session.workspace.root, ask_session.workspace.policy),
            "read_file", {"path": "app.py"})
        assert ok and "v1" in out
        assert seen and seen[0]["name"] == "read_file"
        # branch code is tool code: it reaches _run_call through the chain library
        assert seen[0]["id"].startswith("chain-")


# ---- state files must be owner-only (0600) ----

class TestL2PrivateStateFiles:
    def test_write_private_is_owner_only(self, tmp_path):
        """The shared primitive: content intact, mode 0600, atomic replace."""
        from app.agent.session import AgentSession
        target = tmp_path / "state.txt"
        target.write_text("old", encoding="utf-8")
        AgentSession._write_private(target, "secret payload")
        assert target.read_text(encoding="utf-8") == "secret payload"
        assert _mode(target) == 0o600
        assert not list(tmp_path.glob("state.txt.part-*")), "temp file leaked"

    def test_offload_file_is_owner_only(self, ask_session):
        """_offload wrote 0644/0660; now 0600."""
        call = {"id": "abc123", "name": "run_command"}
        big = "x" * (13_000)
        note = ask_session._offload(call, big)
        assert "full output saved to" in note
        path = Path(note.split("full output saved to ")[1].split(":")[0])
        assert path.is_file()
        assert _mode(path) == 0o600
        assert path.read_text(encoding="utf-8") == big

    def test_archive_file_is_owner_only(self, ask_session):
        """_archive wrote 0644/0660; now 0600."""
        items = [{"role": "user", "content": "hello"},
                 {"role": "assistant", "content": "token=sk-canary world"}]
        note = ask_session._archive(items, "the test")
        path = Path(note.split("saved in ")[1].split(";")[0])
        assert path.is_file()
        assert _mode(path) == 0o600
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved[1]["content"] == "token=sk-canary world"

    def test_offload_short_output_not_written(self, ask_session):
        """Legitimate behavior: short outputs stay inline, no file created."""
        out = ask_session._offload({"id": "x", "name": "run_command"}, "short")
        assert out == "short"


# ---- memory.add refuses secrets ----

class TestL3MemoryRedaction:
    def test_add_refuses_secret_looking_note(self, agent_home, ws_root):
        """a note carrying a secret-shaped value is refused, not persisted."""
        with pytest.raises(ValueError, match="[Ss]ecret"):
            memory.add(agent_home, ws_root, "user", "deploy uses key sk-abcdefghijklmnopqrstuvwx do not rotate")
        memfile = agent_home / ".manga-agent" / "memory" / "user.md"
        assert not memfile.is_file() or "sk-abcdefghijklmnopqrstuvwx" not in memfile.read_text(encoding="utf-8")

    def test_add_refuses_known_env_secret_value(self, agent_home, ws_root, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-value-long-enough-123")
        with pytest.raises(ValueError, match="[Ss]ecret"):
            memory.add(agent_home, ws_root, "user", "search needs TAVILY_API_KEY=tvly-value-long-enough-123")

    def test_add_stores_clean_note(self, agent_home, ws_root):
        """Legitimate behavior: clean notes are stored and reappear in the prompt."""
        n = memory.add(agent_home, ws_root, "user", "the user prefers concise answers")
        assert n == 1
        assert "concise answers" in memory.prompt(agent_home, ws_root)


# ---- role-usage.json: atomic writes, fail-closed reads, no budget race ----

class TestM8RoleBudget:
    CFG = {"daily_requests": 10, "daily_tokens": 0}

    @pytest.fixture(autouse=True)
    def _fresh_process(self):
        """Each test starts with no in-memory last-known state, like a fresh process."""
        roles._GOOD.clear()
        yield
        roles._GOOD.clear()

    def _usage_file(self, home):
        return home / ".manga-agent" / "role-usage.json"

    def test_truncated_file_keeps_last_known_instead_of_resetting(self, agent_home):
        """a crash-torn file must not silently reset budgets."""
        assert roles.add(agent_home, "thinker", 9, 100) is True
        usage = self._usage_file(agent_home)
        assert usage.is_file()
        raw = usage.read_text(encoding="utf-8")
        usage.write_text(raw[:len(raw) // 2], encoding="utf-8")  # kill -9 mid-write
        # Fail closed to the last known state: still 9 used, not a reset to 0.
        assert roles.today(agent_home)["thinker"]["requests"] == 9
        assert roles.within(agent_home, "thinker", self.CFG) is True

    def test_corrupt_file_denies_with_no_trustworthy_copy(self, agent_home):
        """With no backup and no in-memory state, a corrupt file denies (fail closed)."""
        assert roles.add(agent_home, "thinker", 3, 10) is True
        roles._GOOD.clear()  # fresh process: nothing trustworthy left
        self._usage_file(agent_home).write_text("{not json", encoding="utf-8")
        assert roles.within(agent_home, "thinker", self.CFG) is False
        assert roles.today(agent_home) == {}

    def test_backup_recovers_last_good_state(self, agent_home):
        """A second consecutive good write leaves a backup the fail-closed read uses."""
        assert roles.add(agent_home, "thinker", 9, 100) is True
        assert roles.add(agent_home, "thinker", 0, 5) is True  # backs up the 9-request state
        self._usage_file(agent_home).write_text("torn", encoding="utf-8")
        # 9 of 10 used -> still within budget, from the backup, not a reset
        assert roles.within(agent_home, "thinker", self.CFG) is True
        assert roles.today(agent_home)["thinker"]["requests"] == 9

    def test_accounting_stays_correct(self, agent_home):
        """Legitimate behavior: budget accounting works end to end."""
        assert roles.within(agent_home, "thinker", self.CFG) is True  # fresh home
        assert roles.add(agent_home, "thinker", 9, 100) is True
        assert roles.within(agent_home, "thinker", self.CFG) is True
        assert roles.add(agent_home, "thinker", 1, 10) is True
        assert roles.today(agent_home)["thinker"] == {"requests": 10, "tokens": 110}
        assert roles.within(agent_home, "thinker", self.CFG) is False

    def test_atomic_add_never_exceeds_budget(self, agent_home):
        """the atomic check-and-count cannot be raced past the limit."""
        assert roles.add(agent_home, "thinker", 9, 100) is True
        barrier = threading.Barrier(2)
        results = []

        def use_one():
            barrier.wait(timeout=10)
            results.append(roles.add(agent_home, "thinker", 1, 10, self.CFG))

        threads = [threading.Thread(target=use_one) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        used = roles.today(agent_home)["thinker"]["requests"]
        assert used <= 10, f"budget race: used={used}"
        assert sorted(results) == [False, True], f"exactly one atomic add may succeed: {results}"

    def test_writes_are_valid_json_under_concurrency(self, agent_home):
        """Atomic tmp+replace: concurrent writers never leave a torn file."""
        barrier = threading.Barrier(4)

        def writer(n):
            barrier.wait(timeout=10)
            for _ in range(10):
                roles.add(agent_home, f"role{n}", 1, 1)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        raw = self._usage_file(agent_home).read_text(encoding="utf-8")
        data = json.loads(raw)  # raises if torn
        assert isinstance(data.get("days"), dict)


# ---- watch.changes() caps retained old-bytes ----

class TestM9WatchCap:
    def test_changes_enforces_total_cap(self, tmp_path):
        """300 x 300KB of changed files must not balloon past 64MB."""
        root = tmp_path / "ws"
        root.mkdir()
        w = watch.Watch(root)
        w.refresh()
        blob = b"x" * (300 * 1024)
        for i in range(300):
            (root / f"gen_{i:04d}.bin").write_bytes(blob)
        changed = w.changes()
        assert len(changed) == 300
        kept = sum(len(b) for b in w.content.values())
        assert kept <= watch.MAX_TOTAL_BYTES, f"kept {kept:,} > cap {watch.MAX_TOTAL_BYTES:,}"

    def test_changes_still_reports_old_bytes(self, tmp_path):
        """Legitimate behavior: small changes still report old bytes for /undo."""
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.txt").write_text("before", encoding="utf-8")
        w = watch.Watch(root)
        w.refresh()
        (root / "a.txt").write_text("after", encoding="utf-8")
        (root / "b.txt").write_text("new", encoding="utf-8")
        changed = {p.name: old for p, old in w.changes()}
        assert changed["a.txt"] == b"before"
        assert changed["b.txt"] is None  # new file


# ---- export gate: untranslated content never silently drops ----

class TestCExportGate:
    @pytest.fixture()
    def runner(self, monkeypatch):
        import app.ai_mode.job as jobmod
        import app.region_policy as rp

        manifest = {"pages": [{"page_index": 0, "text_objects": [
            # Budget-skipped slice: cleaned already, translation never attempted.
            {"id": "b1", "translation": "", "ocr_text": "おはよう",
             "region": {"x": 0, "y": 0, "w": 10, "h": 10}},
            # Translator examined it and left it blank on purpose (watermark).
            {"id": "b2", "translation": "", "ocr_text": "©2026",
             "translation_attempted": True,
             "region": {"x": 0, "y": 0, "w": 10, "h": 10}},
            # Normally translated control.
            {"id": "b3", "translation": "hello", "ocr_text": "こんにちは",
             "region": {"x": 0, "y": 0, "w": 10, "h": 10}},
        ]}]}
        saved = {}

        @contextlib.contextmanager
        def fake_lock(chapter_id):
            yield

        monkeypatch.setattr(jobmod, "get_manifest_lock", fake_lock)
        monkeypatch.setattr(jobmod, "load_manifest_raw", lambda chapter_id: manifest)
        monkeypatch.setattr(jobmod, "save_manifest_raw",
                            lambda chapter_id, m: saved.update(manifest=m))
        monkeypatch.setattr(rp, "text_object_in_preserve_region", lambda page, obj: False)
        runner = jobmod.AIModeRunner.__new__(jobmod.AIModeRunner)

        class FakeJob:
            chapter_id = "c-chapter"

        runner.job = FakeJob()
        return runner, manifest, saved

    def _objs(self, manifest):
        return {o["id"]: o for o in manifest["pages"][0]["text_objects"]}

    def test_budget_skipped_marked_untranslated_not_reviewed(self, runner):
        """a never-attempted object must NOT become
        reviewed non_story; it gets the blocking 'untranslated' disposition."""
        r, manifest, _ = runner
        assert r._settle_blanks() == 2  # b3 is translated, skipped
        b1 = self._objs(manifest)["b1"]
        assert b1["editorial_disposition"] == "untranslated"
        assert b1.get("editorial_reviewed") is not True

    def test_attempted_blank_stays_non_story(self, runner):
        """Legitimate behavior: translator-deliberate blanks still settle reviewed."""
        r, manifest, _ = runner
        r._settle_blanks()
        b2 = self._objs(manifest)["b2"]
        assert b2["editorial_disposition"] == "non_story"
        assert b2["editorial_reviewed"] is True

    def test_translated_object_untouched(self, runner):
        r, manifest, _ = runner
        r._settle_blanks()
        b3 = self._objs(manifest)["b3"]
        assert "editorial_disposition" not in b3

    def _gate_manifest(self, objects):
        return {"pages": [{
            "page_index": 0,
            "boxes": [{"id": "box1", "confidence": 0.95, "semantic_type": "speech_bubble",
                       "x1": 0, "y1": 0, "x2": 10, "y2": 10}],
            "text_objects": objects,
        }]}

    def test_gate_blocks_untranslated_object(self):
        """Fail closed: the export gate 409s on an untranslated object."""
        from app.editorial_gate import UNTRANSLATED_DISPOSITION, editorial_preflight
        manifest = self._gate_manifest([{
            "id": "b1", "translation": "", "ocr_text": "おはよう",
            "region": {"x": 0, "y": 0, "w": 10, "h": 10},
            "source_boxes": ["box1"],
            "editorial_disposition": UNTRANSLATED_DISPOSITION,
        }])
        preflight = editorial_preflight(manifest)
        kinds = [b["kind"] for b in preflight["blockers"]]
        assert "untranslated_object" in kinds, f"no blocker: {kinds}"

    def test_gate_passes_fully_translated_chapter(self):
        """Legitimate behavior: a fully translated chapter is unaffected."""
        from app.editorial_gate import editorial_preflight
        manifest = self._gate_manifest([{
            "id": "b3", "translation": "hello", "ocr_text": "こんにちは",
            "region": {"x": 0, "y": 0, "w": 10, "h": 10},
            "source_boxes": ["box1"],
        }])
        preflight = editorial_preflight(manifest)
        kinds = [b["kind"] for b in preflight["blockers"]]
        assert "untranslated_object" not in kinds


# ---- self-update verifies the ZIP before unpack/exec ----

RUN_PY = REPO_ROOT / "run.py"


@pytest.fixture()
def runmod():
    spec = importlib.util.spec_from_file_location("runmod_misc_test", str(RUN_PY))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


class TestDSelfUpdate:
    def _archive(self, tmp_path, content=b"fake zip bytes"):
        archive = tmp_path / "source.zip"
        archive.write_bytes(content)
        return archive

    def test_no_pin_lets_the_update_through(self, runmod, tmp_path, monkeypatch):
        """No pin configured: the update goes ahead over HTTPS as before."""
        monkeypatch.delenv("MANGA_UPDATE_SHA256", raising=False)
        assert runmod._verify_archive(self._archive(tmp_path), "main", {}) is None

    def test_refuses_on_mismatch(self, runmod, tmp_path, monkeypatch):
        monkeypatch.delenv("MANGA_UPDATE_SHA256", raising=False)
        with pytest.raises(SystemExit, match="[Kk]hông khớp|mismatch"):
            runmod._verify_archive(self._archive(tmp_path), "main", {"update_sha256": "0" * 64})

    def test_accepts_matching_pin(self, runmod, tmp_path, monkeypatch):
        """Legitimate behavior: a correct pin lets the verified update through."""
        monkeypatch.delenv("MANGA_UPDATE_SHA256", raising=False)
        archive = self._archive(tmp_path)
        digest = runmod._sha256_file(archive)
        assert len(digest) == 64
        assert runmod._verify_archive(archive, "main", {"update_sha256": digest}) is None

    def test_env_pin_wins_over_marker(self, runmod, tmp_path, monkeypatch):
        archive = self._archive(tmp_path)
        digest = runmod._sha256_file(archive)
        monkeypatch.setenv("MANGA_UPDATE_SHA256", digest)
        assert runmod._verify_archive(archive, "main", {"update_sha256": "0" * 64}) is None

    def test_rejects_malformed_pin(self, runmod, tmp_path, monkeypatch):
        monkeypatch.delenv("MANGA_UPDATE_SHA256", raising=False)
        with pytest.raises(SystemExit):
            runmod._verify_archive(self._archive(tmp_path), "main", {"update_sha256": "not-a-hash"})

    def test_update_verifies_before_unpack(self, runmod):
        """Structural: _update calls the verify hook after download and before
        unpack_archive/_run_installer on the ZIP path."""
        import inspect
        src = inspect.getsource(runmod._update)
        i_dl = src.index("urlretrieve")
        i_verify = src.index("_verify_archive")
        i_unpack = src.index("unpack_archive")
        i_install = src.rindex("_run_installer")  # the ZIP path's installer run
        assert i_dl < i_verify < i_unpack < i_install
        assert "sha256" in inspect.getsource(runmod._sha256_file).lower()
