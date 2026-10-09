"""The fixed world-access API for Seed capabilities.

This module is part of Seed's immutable core. Model-written capability code
may not import anything except a small whitelist of pure-computation stdlib
modules; every contact with the outside world goes through a bound
``Harness`` instance, which enforces:

- workspace containment for all file paths,
- the OS sandbox (Landlock / Seatbelt, same as the main harness) for any
  spawned process, with secrets scrubbed from the environment,
- an audit entry for every world-touching call.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from app.agent import sandbox

FETCH_MAX_BYTES = 1_000_000


class Pipe:
    """A subprocess the capability converses with line by line.

    The intended use is an MCP server speaking JSON-RPC over stdio:
    the model writes the server file with ``harness.write``, then
    ``harness.spawn``s it and talks to it through this handle.
    """

    def __init__(self, proc: subprocess.Popen, scratch: str):
        self._proc = proc
        self._scratch = scratch
        self._queue: queue.Queue = queue.Queue()
        self._pump = threading.Thread(target=self._drain, daemon=True)
        self._pump.start()

    def _drain(self) -> None:
        try:
            for raw in self._proc.stdout:
                self._queue.put(raw.decode("utf-8", errors="replace").rstrip("\n"))
        except Exception:
            pass
        finally:
            self._queue.put(None)

    def write_line(self, text: str) -> None:
        if self._proc.poll() is not None:
            raise EOFError("the process has ended")
        self._proc.stdin.write((text + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def read_line(self, timeout: int = 30) -> str:
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no line within {timeout}s")
        if line is None:
            raise EOFError("the process has ended")
        return line

    @property
    def running(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        try:
            self._proc.kill()
        except OSError:
            pass
        shutil.rmtree(self._scratch, ignore_errors=True)


def _wrap_argv(argv: list[str], policy: "sandbox.Policy", root: Path, scratch: str) -> list[str]:
    """The argv version of ``sandbox.wrap``: run a program (no shell) under the policy."""
    kind = sandbox.backend()
    if policy.mode == "full-access" or kind == "none":
        return list(argv)
    if kind == "landlock":
        from app.agent import landlock_run

        args = [sys.executable, str(Path(landlock_run.__file__))]
        for path in sandbox.writable_roots(policy, root, scratch):
            args += ["--write", path]
        for path in sandbox.secret_paths():
            args += ["--deny-read", path]
        for path in sandbox.git_guard_paths(root):
            args += ["--deny-write", path]
        if not policy.network:
            args.append("--no-network")
        return args + ["--", *argv]
    profile = sandbox._seatbelt_profile(policy, root, scratch)
    return ["sandbox-exec", "-p", profile, *argv]


class Harness:
    """The world, as seen by capability code. Bound to one session."""

    def __init__(self, root: Path, audit):
        self._root = root.resolve()
        self._audit = audit
        self._policy = sandbox.Policy("workspace-write", False)

    # -- files -----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        candidate = (self._root / path).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise PermissionError(f"path escapes the workspace: {path!r}")
        return candidate

    def read(self, path: str) -> str:
        """Read a text file inside the workspace."""
        return self._resolve(path).read_text(encoding="utf-8")

    def write(self, path: str, text: str) -> str:
        """Write a text file inside the workspace (creates parent folders)."""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self._audit("harness.write", path)
        return f"wrote {len(text)} characters to {path}"

    def ls(self, pattern: str = "*") -> list[str]:
        """File paths inside the workspace matching a glob pattern."""
        return sorted(
            str(p.relative_to(self._root))
            for p in self._root.glob(pattern)
            if p.is_file()
        )

    # -- execution --------------------------------------------------------

    def run(self, cmd: str, timeout: int = 30) -> str:
        """Run a shell command in the OS sandbox: writes only inside the
        workspace (plus a scratch dir), no network, secrets scrubbed from
        the environment. Returns ``exit=N`` followed by the output."""
        code, out = sandbox.run(cmd, self._policy, self._root, timeout)
        self._audit("harness.run", cmd[:200])
        return f"exit={code}\n{out}"

    def spawn(self, argv: list[str], timeout: int = 30) -> Pipe:
        """Start a program under the same sandbox as :meth:`run` and talk
        to it line by line. ``argv[0]`` must live inside the workspace
        (this is how a model-written MCP server gets launched)."""
        if not argv:
            raise ValueError("argv is empty")
        binary = self._resolve(argv[0])
        scratch = tempfile.mkdtemp(prefix="seed-spawn-")
        target = _wrap_argv([str(binary), *argv[1:]], self._policy, self._root, scratch)
        env = {**sandbox.clean_env(), "TMPDIR": scratch}
        proc = subprocess.Popen(
            target, cwd=self._root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, start_new_session=True,
        )
        self._audit("harness.spawn", " ".join(argv)[:200])
        return Pipe(proc, scratch)

    # -- network -----------------------------------------------------------

    def fetch(self, url: str, timeout: int = 20) -> str:
        """Fetch a URL over plain HTTPS (size-capped). No sandboxing here —
        this is fixed harness code, and every call is audited."""
        request = urllib.request.Request(url, headers={"User-Agent": "seed-harness/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(FETCH_MAX_BYTES + 1)
        text = data[:FETCH_MAX_BYTES].decode("utf-8", errors="replace")
        self._audit("harness.fetch", url[:200])
        return text

    # -- misc ---------------------------------------------------------------

    def log(self, message: str) -> str:
        """Write a note into the session's audit log."""
        self._audit("harness.log", message)
        return "logged"
