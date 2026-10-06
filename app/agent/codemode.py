"""Code mode: a Python script runs in the sandbox and calls the agent's read-only tools as functions, so one step can do many lookups."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading

from app.agent import sandbox

BOOT = '''import json, os, sys
_out = os.fdopen(int(os.environ["AGENT_TOOL_OUT"]), "w", buffering=1)
_in = os.fdopen(int(os.environ["AGENT_TOOL_IN"]), "r")


class _Tools:
    """Each attribute is a tool: tools.read_file(path="a.py") returns its text and raises on an error."""

    def __getattr__(self, name):
        def call(**args):
            _out.write(json.dumps({"name": name, "args": args}) + "\\n")
            reply = json.loads(_in.readline() or '{"ok": false, "output": "the agent stopped answering"}')
            if not reply.get("ok"):
                raise RuntimeError(reply.get("output"))
            return reply["output"]
        return call


tools = _Tools()
source = open(sys.argv[1], encoding="utf-8").read()
exec(compile(source, "script.py", "exec"), {"tools": tools, "__name__": "__main__"})
'''
MAX_CALLS = 200


def run(code: str, policy: sandbox.Policy, root: Path, timeout: int, call_tool) -> tuple[int | None, str, int]:
    """(exit code or None on timeout, output, tool calls made); call_tool(name, args) returns (output, ok)."""
    scratch = tempfile.mkdtemp(prefix="agent-script-")
    try:
        boot, script = Path(scratch) / "boot.py", Path(scratch) / "script.py"
        boot.write_text(BOOT, encoding="utf-8")
        script.write_text(code, encoding="utf-8")
        request_r, request_w = os.pipe()
        reply_r, reply_w = os.pipe()
        command = f"{shlex.quote(sys.executable)} {shlex.quote(str(boot))} {shlex.quote(str(script))}"
        target, shell = sandbox.wrap(command, policy, root, os.path.realpath(scratch))
        env = dict(sandbox._env_for(policy, scratch) or os.environ)
        env.update({"AGENT_TOOL_OUT": str(request_w), "AGENT_TOOL_IN": str(reply_r), "PYTHONUNBUFFERED": "1"})
        proc = subprocess.Popen(target, shell=shell, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=os.name == "posix", env=env, pass_fds=(request_w, reply_r))
        os.close(request_w)
        os.close(reply_r)
        output = sandbox._Capture(proc.stdout, lambda: sandbox._kill_group(proc))
        output.start()
        calls = [0]

        def serve() -> None:
            with os.fdopen(request_r, "r", encoding="utf-8") as requests, os.fdopen(reply_w, "w", encoding="utf-8", buffering=1) as replies:
                for line in requests:
                    calls[0] += 1
                    try:
                        request = json.loads(line)
                        if calls[0] > MAX_CALLS:
                            result, ok = f"at most {MAX_CALLS} tool calls per script", False
                        else:
                            result, ok = call_tool(str(request.get("name") or ""), request.get("args") if isinstance(request.get("args"), dict) else {})
                    except ValueError:
                        result, ok = "bad request", False
                    try:
                        replies.write(json.dumps({"ok": ok, "output": result}) + "\n")
                    except (BrokenPipeError, OSError):
                        return

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        try:
            code_out: int | None = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code_out = None
        sandbox._kill_group(proc)
        proc.wait()
        output.join(5)
        server.join(5)
        proc.stdout.close()
        return code_out, output.text(), calls[0]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
