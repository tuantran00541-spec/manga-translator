"""sandbox.py hardening: clean_env() scrubs secret-ish names with suffixes or
embedded keywords, and describe() is honest about the 'none' backend."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from app.agent import sandbox

# Secret-ish names the old (_|$) regex let through.
POC_LEAKY = ["R3_AGNES_KEY2", "R3_OPENAI_API_KEY2", "R3_KEY2", "R3_SECRET1",
             "R3_PASSWORD2", "R3_MYSECRET", "R3_OAUTH_TOKEN2", "R3_BEARER2"]
# Control group the old regex already scrubbed.
POC_CONTROLS = ["R3_OPENAI_API_KEY", "R3_TINYFISH_API_KEY", "R3_GITHUB_TOKEN",
                "R3_AWS_SECRET_ACCESS_KEY", "R3_HF_TOKEN"]
# Benign vars the fix must not over-scrub.
BENIGN = ["PATH", "HOME", "USER", "LANG", "TERM", "PYTHONPATH", "SHELL", "PWD",
          "HOSTNAME", "DISPLAY", "TMPDIR", "DEBIAN_FRONTEND", "XDG_SESSION_TYPE",
          "LOGNAME", "TZ", "EDITOR", "JAVA_HOME", "GOPATH"]
CANARY = "test-canary-value"


@pytest.fixture(autouse=True)
def _canary_env(monkeypatch):
    for name in POC_LEAKY + POC_CONTROLS:
        monkeypatch.setenv(name, CANARY)
    for name in BENIGN:
        if name not in __import__("os").environ:
            monkeypatch.setenv(name, "benign")


def test_h2_poc_leaky_names_scrubbed():
    env = sandbox.clean_env()
    leaked = [n for n in POC_LEAKY if n in env]
    assert not leaked, f"secret-ish names survived clean_env(): {leaked}"


def test_h2_control_secret_names_still_scrubbed():
    env = sandbox.clean_env()
    leaked = [n for n in POC_CONTROLS if n in env]
    assert not leaked, f"control secret names survived clean_env(): {leaked}"


def test_h2_benign_vars_preserved():
    env = sandbox.clean_env()
    missing = [n for n in BENIGN if n not in env]
    assert not missing, f"benign vars were over-scrubbed: {missing}"


def test_h2_leaky_names_do_not_reach_confined_command(tmp_path):
    """End to end: a confined command's environment has none of these names."""
    code, out = sandbox.run("env", sandbox.Policy("workspace-write", False), tmp_path, 30)
    assert code == 0, f"confined 'env' failed: {out[:200]}"
    leaked = [n for n in POC_LEAKY + POC_CONTROLS if f"{n}={CANARY}" in out]
    assert not leaked, f"secret-ish vars visible to confined command: {leaked}"


def test_m4_none_backend_describe_says_unsandboxed(monkeypatch):
    """Backend 'none' must be reported honestly, not as a sandbox."""
    monkeypatch.setattr(sandbox, "backend", lambda: "none")
    desc = sandbox.Policy("workspace-write", False).describe(Path("/tmp"))
    assert "unsandboxed" in desc.lower(), f"none-backend describe() not honest: {desc!r}"
    assert "run in a sandbox" not in desc.lower(), f"still claims a sandbox: {desc!r}"


@pytest.mark.parametrize("kind", ["landlock", "seatbelt", "docker", "low-integrity"])
def test_m4_real_backends_do_not_claim_unsandboxed(monkeypatch, kind):
    monkeypatch.setattr(sandbox, "backend", lambda: kind)
    desc = sandbox.Policy("workspace-write", False).describe(Path("/tmp"))
    assert "unsandboxed" not in desc.lower(), f"{kind} backend claims unsandboxed: {desc!r}"


def test_m4_full_access_describe_unchanged(monkeypatch):
    """The explicit full-access message stays honest."""
    monkeypatch.setattr(sandbox, "backend", lambda: "none")
    desc = sandbox.Policy("full-access", False).describe(Path("/tmp"))
    assert "without a sandbox" in desc
