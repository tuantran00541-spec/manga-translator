"""Where agent commands may write and whether they reach the network, enforced by the OS where it can."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile

MODES = ("read-only", "workspace-write", "full-access")


@dataclass(frozen=True)
class Policy:
    mode: str = "workspace-write"
    network: bool = False

    def describe(self, root: Path) -> str:
        if self.mode == "full-access":
            return "Commands run without a sandbox: full file and network access."
        where = "nowhere except a private temp folder" if self.mode == "read-only" else f"only inside {root} and the temp folder"
        net = "allowed" if self.network else "blocked"
        return f"Commands run in a sandbox: they can read everything, write {where}; network {net}."


@functools.lru_cache(maxsize=1)
def backend() -> str:
    """The OS mechanism that confines commands here, or 'none'."""
    system = platform.system()
    if system == "Linux":
        from app.agent import landlock_run

        return "landlock" if landlock_run.abi() >= 1 else "none"
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    return "none"


def writable_roots(policy: Policy, root: Path, scratch: str) -> list[str]:
    """Folders a confined command may write: its own scratch folder and /dev, plus the workspace and temp when allowed."""
    roots = [scratch, "/dev"]
    if policy.mode == "workspace-write":
        roots = [str(root), os.path.realpath(tempfile.gettempdir())] + roots
    return roots


def _seatbelt_profile(policy: Policy, root: Path, scratch: str) -> str:
    forms = ["(version 1)", "(allow default)", "(deny file-write*)",
             '(allow file-write* (literal "/dev/null") (literal "/dev/tty") (subpath "/dev/fd"))']
    paths = [os.path.realpath(p) for p in writable_roots(policy, root, scratch) if p != "/dev"]
    forms.append("(allow file-write* " + " ".join(f'(subpath "{p}")' for p in paths) + ")")
    if not policy.network:
        forms.append('(deny network-outbound (remote ip))')
    return " ".join(forms)


def wrap(command: str, policy: Policy, root: Path, scratch: str) -> tuple[list[str] | str, bool]:
    """The process to start for a shell command under the policy, and whether it goes through a shell string."""
    kind = backend()
    if policy.mode == "full-access" or kind == "none":
        return command, True
    if kind == "landlock":
        # The launcher runs by file path, so the user's project never sees this app on its import path.
        args = [sys.executable, str(Path(__file__).with_name("landlock_run.py"))]
        for path in writable_roots(policy, root, scratch):
            args += ["--write", path]
        if not policy.network:
            args.append("--no-network")
        return args + ["--", "/bin/sh", "-c", command], False
    return ["sandbox-exec", "-p", _seatbelt_profile(policy, root, scratch), "/bin/sh", "-c", command], False


def run(command: str, policy: Policy, root: Path, timeout: int, stdin: str | None = None,
        split: bool = False) -> tuple[int | None, str]:
    """Run a shell command under the policy; a None exit code means it timed out, and split returns stderr alone."""
    # Each run gets its own temp folder, the only place a read-only command may write.
    scratch = tempfile.mkdtemp(prefix="agent-run-")
    try:
        target, shell = wrap(command, policy, root, os.path.realpath(scratch))
        env = {**os.environ, "TMPDIR": scratch, "TMP": scratch, "TEMP": scratch} if policy.mode == "read-only" else None
        return _execute(target, shell, root, timeout, stdin, split, env)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _execute(target, shell: bool, root: Path, timeout: int, stdin: str | None, split: bool,
             env: dict | None) -> tuple[int | None, str]:
    posix = os.name == "posix"
    proc = subprocess.Popen(target, shell=shell, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE if split else subprocess.STDOUT,
                            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            text=True, errors="replace", start_new_session=posix, env=env)
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
        return proc.returncode, (err if split else out) or ""
    except subprocess.TimeoutExpired:
        # The whole process group goes, so a test runner's children do not outlive it.
        if posix:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        else:
            proc.kill()
        out, err = proc.communicate()
        return None, (err if split else out) or ""
