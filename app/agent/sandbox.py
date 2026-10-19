"""Where agent commands may write and whether they reach the network, enforced by the OS where it can."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import json
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
import uuid

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
SECRET_NAME = re.compile(r"(?i)(api_?key|key|token|secret|passw(or)?d|credentials?|auth|cookie|session_?id|private|bearer)")
# Substring match without word boundaries: real secret names carry suffixes
# (AGNES_KEY2) or embed the keyword mid-token (MYSECRET), which a boundary
# would miss. Everyday vars (PATH, HOME, ...) contain none of these.
# Engines that confine reads, writes and the network; low-integrity confines writes only.
FULL = ("landlock", "seatbelt", "docker")
DOCKER_IMAGE = "python:3.12-slim"
DOCKER_CHECK_S = 60
_docker_seen: list = [0.0, False]
DROP_ENV = {"SSH_AUTH_SOCK", "DBUS_SESSION_BUS_ADDRESS", "GPG_AGENT_INFO", "GNOME_KEYRING_CONTROL", "KRB5CCNAME"}


def clean_env() -> dict[str, str]:
    """The environment a confined command gets: no API keys, tokens or agent sockets."""
    return {k: v for k, v in os.environ.items() if k not in DROP_ENV and not SECRET_NAME.search(k)}


def secret_paths() -> list[str]:
    """Existing credential folders and files, as real paths."""
    home = Path.home()
    found = [home / name for name in HOME_SECRETS] + [Path(p) for p in SYSTEM_SECRETS] + [Path(p) for p in EXTRA_DENY]
    return sorted({os.path.realpath(p) for p in found if p.exists()})


# The only dotenv files left readable; every other .env* is a secret.
DOTENV_PUBLIC = {".env.example", ".env.sample"}


def workspace_secrets(root: Path) -> list[str]:
    """Workspace dotenv files (.env, .env.local, ...), as real paths: denied read at the
    sandbox layer too, so a quote-evasion that slips past the rule scan still cannot read them."""
    hits, top = [], str(root)
    for folder, dirs, names in os.walk(top):
        # Four levels down covers app/.env and services/api/.env without walking dependency trees.
        if folder[len(top):].count(os.sep) >= ENV_DEPTH:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in ENV_SKIP]
        hits += [os.path.join(folder, n) for n in names if n.startswith(".env") and n not in DOTENV_PUBLIC]
    return sorted({os.path.realpath(p) for p in hits if os.path.isfile(p)})


ENV_DEPTH = 4
ENV_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "site-packages", "data", "models"}


@dataclass(frozen=True)
class Policy:
    mode: str = "workspace-write"
    network: bool = False

    def describe(self, root: Path) -> str:
        if self.mode == "full-access":
            return "Commands run without a sandbox: full file and network access."
        kind = backend()
        if kind == "none":
            # No confinement is possible on this machine: saying anything else would lie to the model.
            return ("Commands run UNSANDBOXED with the full privileges of this user: no sandbox backend is available "
                    "on this machine, so there is no file confinement, no network block, and no credential-folder "
                    "protection — treat every command as running with your user's full access.")
        net = "allowed" if self.network else "blocked"
        if kind == "docker":
            mount = "read-only" if self.mode == "read-only" else "writable"
            return (f"Commands run in a Linux container ({settings()['image']}): the workspace is mounted {mount} at /workspace and nothing else "
                    f"of this machine is visible; use /workspace paths in commands; network {net}.")
        if kind == "low-integrity":
            where = f"only inside {root} and a private temp folder"
            if self.mode == "read-only":
                where += " (read-only is not enforced inside a workspace an earlier session opened for writing; do not write there)"
            return f"Commands run at low integrity: they can write {where}, but can still read files and use the network."
        where = "nowhere except a private temp folder" if self.mode == "read-only" else f"only inside {root} and the temp folder"
        return f"Commands run in a sandbox: they can read everything except credential folders, write {where}; network {net}."


def settings() -> dict:
    """The user's engine choice (auto or docker) and container image, from ~/.manga-agent/sandbox.json or MANGA_AGENT_SANDBOX."""
    data = {}
    try:
        data = json.loads((Path.home() / ".manga-agent" / "sandbox.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    engine = os.environ.get("MANGA_AGENT_SANDBOX") or (data.get("engine") if isinstance(data, dict) else "") or "auto"
    image = data.get("image") if isinstance(data, dict) and isinstance(data.get("image"), str) and data["image"].strip() else DOCKER_IMAGE
    return {"engine": engine if engine in ("auto", "docker") else "auto", "image": image.strip()}


def save_settings(engine: str, image: str = "") -> dict:
    if engine not in ("auto", "docker"):
        raise ValueError("engine must be auto or docker")
    if image and not re.fullmatch(r"[\w./:@-]{1,200}", image):
        raise ValueError("image must be a plain image name such as python:3.12-slim")
    path = Path.home() / ".manga-agent" / "sandbox.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"engine": engine, "image": image or DOCKER_IMAGE}), encoding="utf-8")
    return settings()


def docker_ready() -> bool:
    """True when a Docker daemon running Linux containers answers; checked at most once a minute."""
    if time.time() - _docker_seen[0] < DOCKER_CHECK_S:
        return _docker_seen[1]
    ok = False
    if shutil.which("docker"):
        try:
            done = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=10)
            ok = done.returncode == 0 and done.stdout.strip() == "linux"
        except (OSError, subprocess.TimeoutExpired):
            ok = False
    _docker_seen[:] = [time.time(), ok]
    return ok


def backend() -> str:
    """The mechanism that confines commands here: the chosen container engine when it answers, else the OS's own, or 'none'."""
    if settings()["engine"] == "docker" and docker_ready():
        return "docker"
    return _os_backend()


def full() -> bool:
    """True when commands are kept from secrets and the network, not only from writing."""
    return backend() in FULL


@functools.lru_cache(maxsize=1)
def _os_backend() -> str:
    system = platform.system()
    if system == "Linux":
        from app.agent import landlock_run

        return "landlock" if landlock_run.abi() >= 1 else "none"
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    if system == "Windows":
        return "low-integrity"
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
    forms += [f'(deny file-read* (subpath "{p}"))' for p in secret_paths() + workspace_secrets(root)]
    if not policy.network:
        forms.append('(deny network-outbound (remote ip))')
    return " ".join(forms)


def wrap(command: str, policy: Policy, root: Path, scratch: str) -> tuple[list[str] | str, bool]:
    """The process to start for a shell command under the policy, and whether it goes through a shell string."""
    kind = backend()
    if policy.mode == "full-access" or kind == "none":
        return command, True
    if kind == "docker":
        image = settings()["image"]
        mount = f"{root}:/workspace" + (":ro" if policy.mode == "read-only" else "")
        args = ["docker", "run", "--rm", "-i", "--init", "--name", f"manga-agent-{uuid.uuid4().hex[:12]}", "--pids-limit", "512",
                "-v", mount, "-v", f"{scratch}:/tmp", "-w", "/workspace", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp"]
        if not policy.network:
            args += ["--network", "none"]
        if hasattr(os, "getuid"):
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return args + [image, "sh", "-c", command], False
    if kind == "low-integrity":
        from app.agent import win_low

        win_low.label(scratch)
        if policy.mode == "workspace-write" and str(root) not in _labelled:
            win_low.label(str(root))
            _labelled.add(str(root))
        return [sys.executable, str(Path(__file__).with_name("win_low.py")), "--", command], False
    if kind == "landlock":
        # The launcher runs by file path, so the user's project never sees this app on its import path.
        args = [sys.executable, str(Path(__file__).with_name("landlock_run.py"))]
        for path in writable_roots(policy, root, scratch):
            args += ["--write", path]
        for path in secret_paths() + workspace_secrets(root):
            args += ["--deny-read", path]
        if not policy.network:
            args.append("--no-network")
        return args + ["--", "/bin/sh", "-c", command], False
    return ["sandbox-exec", "-p", _seatbelt_profile(policy, root, scratch), "/bin/sh", "-c", command], False


_labelled: set[str] = set()


def _env_for(policy: Policy, scratch: str) -> dict | None:
    env = None if policy.mode == "full-access" else clean_env()
    if policy.mode == "read-only" or (env is not None and backend() == "low-integrity"):
        env = {**(env or {}), "TMPDIR": scratch, "TMP": scratch, "TEMP": scratch}
    # A package install that asks a question, such as tzdata's time zone, would otherwise wait forever on a job's open input.
    base = env if env is not None else os.environ
    return {**base, "DEBIAN_FRONTEND": base.get("DEBIAN_FRONTEND", "noninteractive")}


class Job:
    """A command left running in the background, under the same policy as any other."""

    def __init__(self, command: str, policy: Policy, root: Path, tty: bool = False, scratch: str | None = None):
        self.command, self.started, self.outside, self.tty = command, time.time(), policy.mode == "full-access", tty and os.name == "posix"
        # A given scratch folder is the caller's to keep, so stop leaves it in place.
        self.own_scratch = scratch is None
        self.scratch = scratch or tempfile.mkdtemp(prefix="agent-job-")
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

    def close_input(self) -> None:
        """End the job's input, so a program that asks a question (a password, a y/n) gets end-of-file instead of waiting forever."""
        if self.tty or self.stdin is None:
            return
        try:
            self.stdin.close()
        except OSError:
            pass
        self.stdin = None

    def write(self, text: str) -> None:
        """Send text to the job's input, as if typed."""
        if self.stdin is None:
            raise OSError("this command's input was closed when it started; start it with start_job to type into it")
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

    @property
    def alive(self) -> bool:
        """Whether the command, or anything it started such as a server, is still running."""
        return self.proc.poll() is None or (os.name == "posix" and _group_alive(self.proc.pid))

    def detach(self) -> None:
        """Leave the job running after this process exits, its output drained so a later write cannot kill it."""
        if os.name == "posix":
            subprocess.Popen(["cat"], stdin=self.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

    def stop(self) -> None:
        _kill_group(self.proc)
        self.proc.wait()
        self.out.join(5)
        for stream in (self.stdin, self.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        if self.own_scratch:
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


def _group_alive(pgid: int) -> bool:
    """Whether a process group has a member that is not a zombie; a container with no init to reap them keeps zombies around."""
    procs = Path("/proc")
    if not procs.is_dir():
        try:
            os.killpg(pgid, 0)
            return True
        except OSError:
            return False
    for entry in procs.iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
    return False


def _kill_group(proc: subprocess.Popen) -> None:
    args = proc.args if isinstance(proc.args, list) else []
    if args[:2] == ["docker", "run"] and "--name" in args:
        # Killing the docker client leaves the container running, so it is removed by name.
        try:
            subprocess.run(["docker", "rm", "-f", args[args.index("--name") + 1]], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
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
