"""Shell-rule hardening: operator splitting, environ/deny handling, and rule merges.

Run: python -m pytest app/agent/tests/test_shell_rules.py -v
"""
import ast
import json
import os
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent import rules, sandbox

AGENT = str(Path(__file__).resolve().parents[1])
DEFAULT_RULES = [("default", *r) for r in rules.DEFAULT]
ALLOW_ECHO_LS = [("user", "bash", "echo *", "allow"), ("user", "bash", "ls *", "allow")]


def check(ruleset, command):
    return rules.check(ruleset, "run_command", {"command": command}, lambda p: p)


def _footprint():
    """The real _footprint staticmethod, extracted from session.py."""
    src = Path(AGENT, "session.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "AgentSession"
        for n in n.body if isinstance(n, ast.FunctionDef) and n.name == "_footprint"
    )
    parallel = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARALLEL_CALLS" for t in n.targets)
    )
    parallel_calls = eval(ast.get_source_segment(src, parallel.value), {"frozenset": frozenset})
    ns = {"os": os, "re": re, "rules": rules, "PARALLEL_CALLS": parallel_calls}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<_footprint>", "exec"), ns)
    return ns["_footprint"]


FOOTPRINT = _footprint()


def fp(command):
    return FOOTPRINT({"name": "run_command", "args": {"command": command}})


# ---- lone `&` / bare `>` ----


class TestH1Ampersand:
    """`echo ok & rm -rf …` must not classify as look-only."""

    def test_safe_readonly_rejects_ampersand_chain(self):
        assert rules.safe_readonly("echo ok & rm -rf /tmp/victim") is False
        assert rules.safe_readonly("echo ok & sudo -n true") is False

    def test_check_asks_on_ampersand_chain(self):
        assert check(DEFAULT_RULES, "echo ok & rm -rf /tmp/victim") == "ask"

    def test_footprint_treats_backgrounded_command_as_effectful(self):
        # None: runs alone, not in parallel with other calls.
        assert fp("echo ok & rm -rf /tmp/victim") is None

    def test_remembered_refuses_ampersand(self):
        # A saved prefix allow must not auto-allow a chained command.
        assert rules.remembered("run_command", {"command": "ls /tmp & id"}, None) is None
        assert rules.remembered("run_command", {"command": "echo hi & curl http://e/x"}, None) is None

    def test_ampersand_under_prefix_allow_blocked(self):
        v = check(ALLOW_ECHO_LS, "ls /tmp & touch /tmp/x_amp")
        assert v != "allow"
        assert rules.safe_readonly("ls /tmp & touch /tmp/x_amp") is False

    def test_double_ampersand_still_splits_first(self):
        assert rules.SPLIT.split("ls && echo ok") == ["ls ", " echo ok"]
        assert rules.safe_readonly("echo ok && rm -rf /tmp/victim") is False
        assert check(DEFAULT_RULES, "echo ok && rm -rf /tmp/victim") == "ask"

    def test_legit_compound_commands_unchanged(self):
        assert rules.safe_readonly("ls && echo ok") is True
        assert rules.safe_readonly("cat file.txt | head") is True
        assert rules.safe_readonly("echo ok; echo done") is True
        assert fp("ls && echo ok") == (frozenset({"."}), frozenset())

    def test_quoted_ampersand_stays_look_only(self):
        # The & is literal text inside quotes, not a background operator.
        assert rules.safe_readonly('echo "a & b"') is True
        assert rules.remembered("run_command", {"command": 'echo "a & b"'}, None) == ("bash", "echo *")


class TestH1Redirect:
    """Redirects under an `echo *` allow-rule must still ask."""

    def test_bare_redirect_under_prefix_allow_now_asks(self):
        assert check(ALLOW_ECHO_LS, "echo pwned > /tmp/r2_redir_proof") == "ask"

    def test_bare_input_redirect_under_prefix_allow_now_asks(self):
        assert check(ALLOW_ECHO_LS, "echo hi < /tmp/in.txt") == "ask"

    def test_redirect_still_asks_without_allow_rule(self):
        assert rules.safe_readonly("echo hello > /tmp/out.txt") is False
        assert check(DEFAULT_RULES, "echo hello > /tmp/out.txt") is None

    def test_quoted_redirect_is_not_a_redirect(self):
        assert rules.safe_readonly('echo "a > b"') is True

    def test_substitution_tricks_still_caught(self):
        for cmd in ("echo `id`", "echo $(id)", "echo <(id)", "ls >(cat)"):
            assert rules.safe_readonly(cmd) is False
            assert check(ALLOW_ECHO_LS, cmd) == "ask"


# ---- /proc/self/environ is never look-only ----


class TestH3ProcEnviron:
    """`cat /proc/self/environ` must not be look-only."""

    def test_safe_readonly_rejects_environ(self):
        assert rules.safe_readonly("cat /proc/self/environ") is False
        assert rules.safe_readonly("grep -a . /proc/1234/environ") is False
        assert rules.safe_readonly("tr '\\0' '\\n' < /proc/self/environ") is False
        assert rules.safe_readonly("cat /proc/$$/environ") is False

    def test_quoted_environ_still_rejected(self):
        assert rules.safe_readonly('cat "/proc/self/environ"') is False
        assert rules.safe_readonly("cat /proc/self/'environ'") is False

    def test_check_denies_environ(self):
        assert check(DEFAULT_RULES, "cat /proc/self/environ") == "deny"

    def test_other_proc_reads_still_look_only(self):
        assert rules.safe_readonly("cat /proc/cpuinfo") is True
        assert rules.safe_readonly("ls /proc") is True


# ---- quote-evasion of the .env deny ----


class TestH4QuoteEvasion:
    """`cat .e''nv` must hit the .env deny."""

    def test_dequoted_token_denied(self):
        assert check(DEFAULT_RULES, "cat .e''nv") == "deny"
        assert check(DEFAULT_RULES, 'cat ".env"') == "deny"
        assert check(DEFAULT_RULES, "cat .env") == "deny"  # control

    def test_benign_reads_unaffected(self):
        assert check(DEFAULT_RULES, "cat notes.txt") is None
        assert rules._names_denied_path(DEFAULT_RULES, "cat notes.txt") is False
        assert rules._names_denied_path(DEFAULT_RULES, "cat .env.example") is False

    def test_writes_still_detected_through_quotes(self):
        assert check(DEFAULT_RULES, "tee .e''nv") == "deny"

    def test_apostrophe_in_text_does_not_crash(self):
        assert rules._names_denied_path(DEFAULT_RULES, """echo "don't" """) is False


class TestH4SandboxBackstop:
    def test_workspace_secrets_lists_dotenv_files(self, tmp_path):
        (tmp_path / ".env").write_text("K=1")
        (tmp_path / ".env.local").write_text("K=2")
        (tmp_path / ".env.example").write_text("K=")
        (tmp_path / "notes.txt").write_text("hi")
        found = sandbox.workspace_secrets(tmp_path)
        assert os.path.realpath(tmp_path / ".env") in found
        assert os.path.realpath(tmp_path / ".env.local") in found
        assert os.path.realpath(tmp_path / ".env.example") not in found
        assert os.path.realpath(tmp_path / "notes.txt") not in found

    def test_workspace_secrets_empty_dir(self, tmp_path):
        assert sandbox.workspace_secrets(tmp_path) == []


# ---- merges only tighten, never weaken ----


def write_settings(base: Path, name: str, permission: dict):
    p = base / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"permission": permission}))


class TestM1OnlyTighten:
    """A workspace/user setting may never weaken a deny to ask/allow."""

    def test_workspace_cannot_downgrade_default_env_deny(self):
        rows = DEFAULT_RULES + [("workspace", "read", "*", "ask")]
        assert rules.check(rows, "read_file", {"path": ".env"}, lambda p: p) == "deny"

    def test_workspace_cannot_downgrade_user_bash_deny(self):
        rows = [("user", "bash", "rm *-r*", "deny"), ("workspace", "bash", "*", "ask")]
        assert rules.check(rows, "run_command", {"command": "rm -rf /tmp/x"}, lambda p: p) == "deny"

    def test_workspace_cannot_downgrade_default_git_deny(self):
        rows = DEFAULT_RULES + [("workspace", "edit", "*", "ask")]
        assert rules.check(rows, "write_file", {"path": ".git/hooks/pre-commit"}, lambda p: p) == "deny"

    def test_user_cannot_weaken_default_deny_either(self):
        rows = DEFAULT_RULES + [("user", "read", ".env", "allow")]
        assert rules.check(rows, "read_file", {"path": ".env"}, lambda p: p) == "deny"

    def test_load_end_to_end(self, tmp_path):
        home, ws = tmp_path / "home", tmp_path / "ws"
        write_settings(home, ".manga-agent/settings.json", {"bash": {"rm *": "deny"}})
        write_settings(ws, ".agents/settings.json", {"bash": {"*": "ask"}})
        loaded = rules.load(ws, home)
        assert rules.check(loaded, "run_command", {"command": "rm -rf /tmp/x"}, lambda p: p) == "deny"

    def test_allow_over_ask_still_works(self):
        # Last-match-wins where no deny is involved.
        rows = DEFAULT_RULES + [("user", "bash", "git push*", "allow")]
        assert rules.check(rows, "run_command", {"command": "git push origin main"}, lambda p: p) == "allow"

    def test_workspace_may_still_tighten(self):
        rows = [("user", "bash", "echo *", "allow"), ("workspace", "bash", "echo *", "ask")]
        assert rules.check(rows, "run_command", {"command": "echo hi"}, lambda p: p) == "ask"

    def test_deny_sticky_with_none_scope(self):
        rows = [(None, "read", ".env", "deny"), (None, "read", "*", "ask")]
        assert rules.check(rows, "read_file", {"path": ".env"}, lambda p: p) == "deny"
        rows = [(None, "edit", ".git/*", "deny")]
        assert rules.check(rows, "run_command", {"command": "git rev-parse HEAD > .git/ORIG_HEAD"}, lambda p: p) == "deny"

    def test_same_layer_allow_carveout_survives(self):
        # .env.example is the default's own exception, not a weakening.
        assert rules.check(DEFAULT_RULES, "read_file", {"path": ".env.example"}, lambda p: p) == "allow"
        assert rules.check(DEFAULT_RULES, "read_file", {"path": ".env"}, lambda p: p) == "deny"
        assert rules.check(DEFAULT_RULES, "read_file", {"path": ".env.local"}, lambda p: p) == "deny"


# ---- cross-cutting regression ----


class TestNoRegressions:
    def test_shell_parts_quote_aware(self):
        assert rules._shell_parts("ls && echo ok") == ["ls ", " echo ok"]
        assert rules._shell_parts('echo "a & b"') == ['echo "a & b"']
        assert [p.strip() for p in rules._shell_parts("echo ok & rm -rf /x") if p.strip()] == ["echo ok", "rm -rf /x"]

    def test_mask_quotes(self):
        m = rules._mask_quotes('echo "a & b"')
        assert len(m) == len('echo "a & b"') and "&" not in m and m.startswith("echo ")
        m2 = rules._mask_quotes("echo 'x > y'")
        assert len(m2) == len("echo 'x > y'") and ">" not in m2

    def test_subjects_splits_backgrounded(self):
        subs = rules.subjects("run_command", {"command": "echo ok & rm -rf /x"}, lambda p: p)
        assert subs == ["echo ok", "rm -rf /x"]

    def test_safe_commands_matrix(self):
        look_only = ["ls", "ls -la /tmp", "pwd", "whoami", "git status", "git log --oneline",
                     "cat f.txt | grep x | head", "echo hello", "printf '%s\\n' a"]
        for cmd in look_only:
            assert rules.safe_readonly(cmd) is True, cmd
        effectful = ["rm -rf /x", "echo ok & rm -rf /x", "echo x > f", "sudo true",
                     "python -c 'import os'", "env", "cat /proc/self/environ"]
        for cmd in effectful:
            assert rules.safe_readonly(cmd) is False, cmd
