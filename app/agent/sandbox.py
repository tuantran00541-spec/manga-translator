"""Where agent commands may write and whether they reach the network, enforced by the OS where it can."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

MODES = ("read-only", "workspace-write", "full-access")
MAX_OUTPUT_BYTES = 8_000_000
RUNAWAY_BYTES = 200_000_000
# Where people keep credentials; confined commands cannot read them, whatever the model is told or tries.
HOME_SECRETS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".netrc", ".git-credentials", ".npmrc", ".pypirc", ".claude.json",
                ".config/gcloud", ".config/gh", ".config/git", ".config/opencode", ".config/google-chrome", ".config/chromium",
                ".config/BraveSoftware", ".local/share/keyrings", ".local/share/python_keyring", ".password-store", ".mozilla",
                ".claude/.credentials.json", ".codex/auth.json", ".codex/config.toml", ".manga-agent/mcp.json", ".cache/huggingface/token",
                "Library/Keychains", "Library/Application Support/Google/Chrome")
SYSTEM_SECRETS = ("/run/user", "/run/secrets", "/var/run/secrets")
# More folders to keep unreadable, added by whoever stores private data (the agent's own saved sessions).
EXTRA_DENY: list[str] = []
SECRET_NAME = re.compile(r"(?i)(^|_)(api_?key|key|token|secret|passw(or)?d|credentials?|auth|cookie|session_?id|private)(_|$)")
DROP_ENV = {"SSH_AUTH_SOCK", "DBUS_SESSION_BUS_ADDRESS", "GPG_AGENT_INFO", "GNOME_KEYRING_CONTROL", "KRB5CCNAME"}


def clean_env() -> dict[str, str]:
    """The environment a confined command gets: no API keys, tokens or agent sockets."""
    return {k: v for k, v in os.environ.items() if k not in DROP_ENV and not SECRET_NAME.search(k)}


def secret_paths() -> list[str]:
    """Existing credential folders and files, as real paths."""
    home = Path.home()
    found = [home / name for name in HOME_SECRETS] + [Path(p) for p in SYSTEM_SECRETS] + [Path(p) for p in EXTRA_DENY]
    return sorted({os.path.realpath(p) for p in found if p.exists()})


@dataclass(frozen=True)
class Policy:
    mode: str = "workspace-write"
    network: bool = False

    def describe(self, root: Path) -> str:
        if self.mode == "full-access":
            return "Commands run without a sandbox: full file and network access."
        where = "nowhere except a private temp folder" if self.mode == "read-only" else f"only inside {root} and the temp folder"
        net = "allowed" if self.network else "blocked"
        return f"Commands run in a sandbox: they can read everything except credential folders, write {where}; network {net}."


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
    forms += [f'(deny file-read* (subpath "{p}"))' for p in secret_paths()]
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
        for path in secret_paths():
            args += ["--deny-read", path]
        if not policy.network:
            args.append("--no-network")
        return args + ["--", "/bin/sh", "-c", command], False
    return ["sandbox-exec", "-p", _seatbelt_profile(policy, root, scratch), "/bin/sh", "-c", command], False


def _env_for(policy: Policy, scratch: str) -> dict | None:
    env = None if policy.mode == "full-access" else clean_env()
    if policy.mode == "read-only":
        env = {**(env or {}), "TMPDIR": scratch, "TMP": scratch, "TEMP": scratch}
    return env


class Job:
    """A command left running in the background, under the same policy as any other."""

    def __init__(self, command: str, policy: Policy, root: Path, tty: bool = False):
        self.command, self.started, self.outside = command, time.time(), policy.mode == "full-access"
        self.scratch = tempfile.mkdtemp(prefix="agent-job-")
        target, shell = wrap(command, policy, root, os.path.realpath(self.scratch))
        env = _env_for(policy, self.scratch)
        if tty and os.name == "posix":
            import pty
            master, slave = pty.openpty()
            env = {**(env or os.environ), "TERM": "dumb"}
            self.proc = subprocess.Popen(target, shell=shell, cwd=root, stdout=slave, stderr=slave, stdin=slave, start_new_session=True, env=env)
            os.close(slave)
            self.stdout, self.stdin = os.fdopen(master, "rb", buffering=0), os.fdopen(os.dup(master), "wb", buffering=0)
        else:
            self.proc = subprocess.Popen(target, shell=shell, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                                         start_new_session=os.name == "posix", env=env)
            self.stdout, self.stdin = self.proc.stdout, self.proc.stdin
        self.out = _Capture(self.stdout, lambda: _kill_group(self.proc))
        self.out.start()
        self.cursor = 0

    def write(self, text: str) -> None:
        """Send text to the job's input, as if typed."""
        try:
            self.stdin.write(text.encode("utf-8"))
            self.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise OSError("the job no longer reads input") from exc

    def read(self) -> str:
        """Output produced since the last read."""
        data = bytes(self.out.data[self.cursor:])
        self.cursor += len(data)
        return data.decode("utf-8", errors="replace")

    @property
    def code(self) -> int | None:
        return self.proc.poll()

    def stop(self) -> None:
        _kill_group(self.proc)
        self.proc.wait()
        self.out.join(5)
        for stream in (self.stdin, self.stdout):
            try:
                stream.close()
            except OSError:
                pass
        shutil.rmtree(self.scratch, ignore_errors=True)


def run(command: str, policy: Policy, root: Path, timeout: int, stdin: str | None = None,
        split: bool = False) -> tuple[int | None, str]:
    """Run a shell command under the policy; a None exit code means it timed out, and split returns stderr alone."""
    # Each run gets its own temp folder, the only place a read-only command may write.
    scratch = tempfile.mkdtemp(prefix="agent-run-")
    try:
        target, shell = wrap(command, policy, root, os.path.realpath(scratch))
        return _execute(target, shell, root, timeout, stdin, split, _env_for(policy, scratch))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class _Capture(threading.Thread):
    """Reads a pipe to the end, keeping at most MAX_OUTPUT_BYTES and noting how much was dropped."""

    def __init__(self, stream, on_runaway):
        super().__init__(daemon=True)
        self.stream, self.on_runaway = stream, on_runaway
        self.data, self.total = bytearray(), 0

    def run(self) -> None:
        while True:
            try:
                chunk = self.stream.read1(65536) if hasattr(self.stream, "read1") else self.stream.read(65536)
            except OSError:  # A terminal reports EIO once the program behind it exits.
                return
            if not chunk:
                return
            self.total += len(chunk)
            if len(self.data) < MAX_OUTPUT_BYTES:
                self.data += chunk[:MAX_OUTPUT_BYTES - len(self.data)]
            if self.total > RUNAWAY_BYTES:
                self.on_runaway()
                return

    def text(self) -> str:
        cut = f"\n[output cut: {self.total - len(self.data)} bytes more]" if self.total > len(self.data) else ""
        return self.data.decode("utf-8", errors="replace") + cut


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass


def _execute(target, shell: bool, root: Path, timeout: int, stdin: str | None, split: bool,
             env: dict | None) -> tuple[int | None, str]:
    posix = os.name == "posix"
    proc = subprocess.Popen(target, shell=shell, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE if split else subprocess.STDOUT,
                            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            start_new_session=posix, env=env)
    out, err = _Capture(proc.stdout, lambda: _kill_group(proc)), _Capture(proc.stderr, lambda: _kill_group(proc)) if split else None
    for reader in (out, err):
        if reader:
            reader.start()
    if stdin is not None:
        def feed() -> None:
            try:
                proc.stdin.write(stdin.encode("utf-8"))
                proc.stdin.close()
            except OSError:
                pass
        threading.Thread(target=feed, daemon=True).start()
    try:
        code: int | None = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        code = None
    # Whatever the command left running, including after a clean exit, goes with it.
    _kill_group(proc)
    proc.wait()
    for reader in (out, err):
        if reader:
            reader.join(5)
    for stream in (proc.stdout, proc.stderr):
        if stream:
            stream.close()
    return code, (err if split else out).text()
