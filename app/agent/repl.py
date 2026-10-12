"""A long-lived sandboxed Python process that runs created tools one after another, so imports and the memo dict last the session."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.agent import sandbox

# Runs each script in a fresh __main__ with its output on fd 1 and 2, then puts back the folder, path and streams it found.
WORKER = r'''import builtins, json, os, sys, time, traceback
HOME = sys.argv[1]
INBOX = os.path.join(HOME, "inbox")
builtins.memo = {}
real_out, real_err = os.dup(1), os.dup(2)
while True:
    names = sorted(n for n in os.listdir(INBOX) if n.endswith(".run"))
    if not names:
        time.sleep(0.01)
        continue
    path = os.path.join(INBOX, names[0])
    try:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        os.remove(path)
    except (OSError, ValueError):
        continue
    code, cwd, saved_path = 0, os.getcwd(), list(sys.path)
    with open(run["out"], "wb", buffering=0) as out:
        os.dup2(out.fileno(), 1)
        os.dup2(out.fileno(), 2)
        sys.modules.pop("agent_tools", None)
        try:
            with open(run["script"], encoding="utf-8") as f:
                source = f.read()
            exec(compile(source, run["script"], "exec"), {"__name__": "__main__", "__file__": run["script"], "__builtins__": builtins})
        except SystemExit as exc:
            if exc.code is not None and not isinstance(exc.code, int):
                print(exc.code)
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except BaseException:
            traceback.print_exc()
            code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(real_out, 1)
            os.dup2(real_err, 2)
            os.chdir(cwd)
            sys.path[:] = saved_path
    with open(run["done"] + ".tmp", "w") as f:
        f.write(str(code))
    os.replace(run["done"] + ".tmp", run["done"])
'''


class _Out:
    """A run's output file read the way sandbox.Job's capture is read."""

    def __init__(self, path: Path):
        self.path = path

    @property
    def total(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    @property
    def data(self) -> bytes:
        try:
            with open(self.path, "rb") as f:
                return f.read(sandbox.MAX_OUTPUT_BYTES)
        except OSError:
            return b""

    def join(self, timeout: float | None = None) -> None:
        return None

    def text(self) -> str:
        data, total = self.data, self.total
        cut = f"\n[output cut: {total - len(data)} bytes more]" if total > len(data) else ""
        return data.decode("utf-8", errors="replace") + cut


class Run:
    """One created-tool call inside the kernel, shaped like a sandbox.Job so waiting, backgrounding and stopping work the same."""

    def __init__(self, kernel: "Kernel", folder: Path, run_id: str, note: str):
        self.kernel, self.started, self.cursor, self.note = kernel, time.time(), 0, note
        self.command = "created tool"
        self.out = _Out(folder / f"{run_id}.out")
        self.done = folder / f"{run_id}.done"
        self.final: int | None = None

    @property
    def code(self) -> int | None:
        if self.final is not None:
            return self.final
        try:
            return int(self.done.read_text() or 1)
        except ValueError:
            return 1
        except OSError:
            pass
        worker = self.kernel.job
        if worker is None or worker.code is not None:
            return worker.code if worker is not None and worker.code else -9
        if self.out.total > sandbox.RUNAWAY_BYTES:
            # Output past the runaway limit stops the tool, as it would stop a command.
            self.kernel.close()
            return -9
        return None

    def read(self) -> str:
        data = self.out.data[self.cursor:]
        self.cursor += len(data)
        return data.decode("utf-8", errors="replace")

    def write(self, text: str) -> None:
        raise OSError("a created tool does not read input")

    def stop(self) -> None:
        # A run cut short takes the kernel with it, since only killing the process stops its code.
        if self.code is None:
            self.kernel.close()
        self.final = self.code if self.code is not None else -9
        for path in (self.out.path, self.done):
            path.unlink(missing_ok=True)


class Kernel:
    """The session's Python process for created tools; it restarts, with memo cleared, when it dies or the sandbox policy changes."""

    def __init__(self, state: Path):
        self.state, self.folder = state, state / "kernel"
        self.job: sandbox.Job | None = None
        self.key: tuple | None = None
        self.current: Run | None = None
        self.lock = threading.RLock()

    def start(self, run_id: str, script: str, policy: sandbox.Policy, root: Path, where: str, docker: bool) -> Run | None:
        """Hand one script to the kernel; None when a call is still running there, so the caller starts a fresh process instead."""
        with self.lock:
            if self.current is not None and self.current.code is None:
                return None
            key, note = (policy.mode, policy.network, docker), ""
            if self.job is not None and (self.job.code is not None or self.key != key):
                note = "[the tool process restarted, so memo was cleared]\n"
                self._stop()
            if self.job is None:
                self._spawn(policy, root, where, docker)
                self.key = key
            inside = (lambda *p: "/".join((where, *p))) if docker else (lambda *p: os.path.join(where, *p))
            runs = self.folder / "runs"
            ticket = {"script": script, "out": inside("kernel", "runs", f"{run_id}.out"), "done": inside("kernel", "runs", f"{run_id}.done")}
            (self.folder / "inbox" / f"{run_id}.tmp").write_text(json.dumps(ticket), encoding="utf-8")
            (self.folder / "inbox" / f"{run_id}.tmp").replace(self.folder / "inbox" / f"{run_id}.run")
            self.current = Run(self, runs, run_id, note)
            return self.current

    def _spawn(self, policy: sandbox.Policy, root: Path, where: str, docker: bool) -> None:
        for sub in ("inbox", "runs"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        (self.folder / "worker.py").write_text(WORKER, encoding="utf-8")
        if docker:
            command = f"python3 {where}/kernel/worker.py {where}/kernel"
        elif os.name == "nt":
            command = subprocess.list2cmdline([sys.executable, str(self.folder / "worker.py"), str(self.folder)])
        else:
            command = f"{shlex.quote(sys.executable)} {shlex.quote(str(self.folder / 'worker.py'))} {shlex.quote(str(self.folder))}"
        self.job = sandbox.Job(command, policy, root, scratch=str(self.state))

    def _stop(self) -> None:
        if self.job is not None:
            self.job.stop()
        self.job, self.key, self.current = None, None, None
        for sub in ("inbox", "runs"):
            for path in (self.folder / sub).glob("*"):
                path.unlink(missing_ok=True)

    def close(self) -> None:
        with self.lock:
            self._stop()
