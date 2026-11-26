from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.agent import sandbox

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

WORKER = r'''import builtins, json, os, sys, time, traceback
HOME = sys.argv[1]
INBOX = os.path.join(HOME, "inbox")
builtins.memo = {}
real_out, real_err = os.dup(1), os.dup(2)
orig_stdout, orig_stderr = sys.stdout, sys.stderr
while True:
    try:
        names = sorted(n for n in os.listdir(INBOX) if n.endswith(".run"))
    except OSError:
        time.sleep(0.05)
        continue
    if not names:
        time.sleep(0.01)
        continue
    path = os.path.join(INBOX, names[0])
    try:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
    except ValueError:
        try:
            os.remove(path)
        except OSError:
            pass
        time.sleep(0.05)
        continue
    except OSError:
        time.sleep(0.01)
        continue
    try:
        os.remove(path)
    except OSError:
        pass
    if not isinstance(run, dict) or not all(isinstance(run.get(k), str) for k in ("script", "out", "done")):
        continue
    code, cwd, saved_path = 0, os.getcwd(), list(sys.path)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        with open(run["out"], "wb", buffering=0) as out:
            try:
                os.dup2(out.fileno(), 1)
                os.dup2(out.fileno(), 2)
                sys.stdout, sys.stderr = orig_stdout, orig_stderr
                sys.modules.pop("agent_tools", None)
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
                for stream in (sys.stdout, sys.stderr, orig_stdout, orig_stderr):
                    try:
                        stream.flush()
                    except Exception:
                        pass
                sys.stdout, sys.stderr = orig_stdout, orig_stderr
                try:
                    os.dup2(real_out, 1)
                    os.dup2(real_err, 2)
                except OSError:
                    pass
                try:
                    os.chdir(cwd)
                except OSError:
                    pass
                sys.path[:] = saved_path
    except OSError:
        code = 1
    try:
        with open(run["done"] + ".tmp", "w", encoding="utf-8") as f:
            f.write(str(code))
        os.replace(run["done"] + ".tmp", run["done"])
    except OSError:
        pass
'''


class _Out:
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

    def read_from(self, offset: int) -> bytes:
        remaining = sandbox.MAX_OUTPUT_BYTES - offset
        if remaining <= 0:
            return b""
        try:
            with open(self.path, "rb") as f:
                f.seek(offset)
                return f.read(remaining)
        except OSError:
            return b""

    def join(self, timeout: float | None = None) -> None:
        return None

    def text(self) -> str:
        data, total = self.data, self.total
        cut = f"\n[output cut: {total - len(data)} bytes more]" if total > len(data) else ""
        return data.decode("utf-8", errors="replace") + cut


class Run:
    def __init__(self, kernel: "Kernel", folder: Path, run_id: str, note: str):
        self.kernel, self.started, self.cursor, self.note = kernel, time.time(), 0, note
        self.command = "created tool"
        self.out = _Out(folder / f"{run_id}.out")
        self.done = folder / f"{run_id}.done"
        self.final: int | None = None

    def _read_done_code(self) -> int | None:
        try:
            raw = self.done.read_text(encoding="utf-8").strip()
            return int(raw) if raw else 1
        except ValueError:
            return 1
        except OSError:
            return None

    @property
    def code(self) -> int | None:
        if self.final is not None:
            return self.final
        done_code = self._read_done_code()
        if done_code is not None:
            self.final = done_code
            return self.final
        worker = self.kernel.job
        if worker is None or worker.code is not None:
            done_code = self._read_done_code()
            if done_code is not None:
                self.final = done_code
            else:
                self.final = worker.code if worker is not None and worker.code else -9
            return self.final
        if self.out.total > sandbox.RUNAWAY_BYTES:
            self.kernel.close()
            self.final = -9
            return self.final
        return None

    def read(self) -> str:
        data = self.out.read_from(self.cursor)
        self.cursor += len(data)
        return data.decode("utf-8", errors="replace")

    def write(self, text: str) -> None:
        raise OSError("a created tool does not read input")

    def stop(self) -> None:
        current = self.code
        if current is None:
            self.kernel.close()
            current = self._read_done_code()
        self.final = current if current is not None else -9
        for path in (self.out.path, self.done):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class Kernel:
    def __init__(self, state: Path):
        self.state, self.folder = state, state / "kernel"
        self.job: sandbox.Job | None = None
        self.key: tuple | None = None
        self.current: Run | None = None
        self.lock = threading.RLock()

    def start(
        self,
        run_id: str,
        script: str,
        policy: sandbox.Policy,
        root: Path,
        where: str,
        docker: bool,
    ) -> Run | None:
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        with self.lock:
            if self.current is not None and self.current.code is None:
                return None
            key = (policy.mode, policy.network, docker, str(root), where)
            note = ""
            if self.job is not None and (self.job.code is not None or self.key != key):
                note = "[the tool process restarted, so memo was cleared]\n"
                self._stop()
            if self.job is None:
                self._spawn(policy, root, where, docker)
                self.key = key
            inside = (lambda *p: "/".join((where, *p))) if docker else (lambda *p: os.path.join(where, *p))
            runs = self.folder / "runs"
            ticket = {
                "script": script,
                "out": inside("kernel", "runs", f"{run_id}.out"),
                "done": inside("kernel", "runs", f"{run_id}.done"),
            }
            tmp = self.folder / "inbox" / f"{run_id}.tmp"
            tmp.write_text(json.dumps(ticket), encoding="utf-8")
            tmp.replace(self.folder / "inbox" / f"{run_id}.run")
            self.current = Run(self, runs, run_id, note)
            return self.current

    def _spawn(self, policy: sandbox.Policy, root: Path, where: str, docker: bool) -> None:
        for sub in ("inbox", "runs"):
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        worker_path = self.folder / "worker.py"
        worker_path.write_text(WORKER, encoding="utf-8")
        if docker:
            command = (
                f"python3 {shlex.quote(f'{where}/kernel/worker.py')} "
                f"{shlex.quote(f'{where}/kernel')}"
            )
        elif os.name == "nt":
            command = subprocess.list2cmdline([sys.executable, str(worker_path), str(self.folder)])
        else:
            command = (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(worker_path))} "
                f"{shlex.quote(str(self.folder))}"
            )
        self.job = sandbox.Job(command, policy, root, scratch=str(self.state))
        self.job.close_input()

    def _stop(self) -> None:
        if self.job is not None:
            self.job.stop()
        self.job, self.key, self.current = None, None, None
        for sub in ("inbox", "runs"):
            folder = self.folder / sub
            if not folder.is_dir():
                continue
            for path in folder.glob("*"):
                try:
                    if not path.is_symlink() and path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        with self.lock:
            self._stop()
