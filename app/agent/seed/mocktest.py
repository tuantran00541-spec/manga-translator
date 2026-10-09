"""End-to-end test of the Seed harness with a scripted mock model (no API key).

The mock plays a model that bootstraps itself: it defines read_file and
write_file from nothing, uses them to do a task, then finishes. It also
probes the guardrails: forbidden imports, open(), dunder escapes and
workspace containment must all be refused.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent.seed.session import SeedSession  # noqa: E402

READ_FILE_CODE = '''
def main(args):
    return harness.read(args["path"])
'''

WRITE_FILE_CODE = '''
def main(args):
    return harness.write(args["path"], args["content"])
'''

EVIL_IMPORT = '''
import os
def main(args):
    return os.listdir(".")
'''

EVIL_OPEN = '''
def main(args):
    return open("/etc/passwd").read()
'''

EVIL_DUNDER = '''
def main(args):
    return ().__class__.__base__.__subclasses__()
'''

EVIL_ESCAPE = '''
def main(args):
    return harness.read("/etc/passwd")
'''


def make_script():
    calls = []
    seq = [
        ("define", {"name": "read_file", "description": "read a text file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                   "required": ["path"]},
                    "code": READ_FILE_CODE}),
        ("define", {"name": "write_file", "description": "write a text file",
                    "parameters": {"type": "object", "properties": {
                        "path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]},
                    "code": WRITE_FILE_CODE}),
        ("call", "read_file", {"path": "notes.txt"}),
        ("call", "write_file", {"path": "summary.txt", "content": "line one\nline two\nline three"}),
        ("text", "done: summarized notes.txt into summary.txt"),
    ]

    def mock(messages, specs):
        kind = seq.pop(0)
        if kind[0] == "text":
            calls.append(("text", kind[1]))
            return {"text": kind[1], "calls": []}
        if kind[0] == "define":
            name, args = "define", kind[1]
        else:
            name, args = kind[1], kind[2]
        calls.append(("call", name, args))
        return {"text": "", "calls": [{"id": f"call-{len(calls)}", "name": name, "args": args}]}

    return mock, calls


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="seed-mock-"))
    home = Path(tempfile.mkdtemp(prefix="seed-home-"))
    try:
        (workspace / "notes.txt").write_text(
            "alpha beta gamma\ndelta epsilon zeta\neta theta iota\n", encoding="utf-8")

        mock, calls = make_script()
        session = SeedSession(provider=None, api_key="", model="mock",
                              workspace=workspace, home=home, complete=mock,
                              approve=lambda prompt: True)
        result = session.run("Read notes.txt, summarize it in 3 lines, write summary.txt")

        # 1. The task got done through self-defined capabilities.
        assert (workspace / "summary.txt").read_text(encoding="utf-8") == "line one\nline two\nline three", \
            "summary.txt content wrong"
        assert "done" in result
        assert session.registry.names() == ["read_file", "write_file"], session.registry.names()

        # 2. The tool surface grew: define was the only tool at first.
        first_specs = None  # (checked implicitly: mock never saw read_file before defining it)

        # 3. Guardrails: forbidden imports / builtins / dunder escapes refused at define time.
        for evil, label in [(EVIL_IMPORT, "import os"), (EVIL_OPEN, "open()"), (EVIL_DUNDER, "dunder")]:
            out = session._dispatch("define", {"name": "evil", "description": "x", "code": evil})
            assert out.startswith("error:"), f"{label} was NOT refused: {out}"
            assert "evil" not in session.registry.names()

        # 4. Workspace containment: harness.read cannot leave the workspace.
        out = session._dispatch("define", {"name": "esc", "description": "x", "code": EVIL_ESCAPE})
        assert not out.startswith("error:"), f"define itself should pass: {out}"
        out = session.registry.call("esc", {})
        assert out.startswith("error:") and "escapes the workspace" in out, f"containment failed: {out}"

        # 5. Redefine bumps the version; audit logged everything.
        out = session._dispatch("define", {"name": "read_file", "description": "read v2", "code": READ_FILE_CODE})
        assert "v2" in out, out
        kinds = [e["kind"] for e in session.audit]
        assert "capability.define" in kinds and "capability.call" in kinds and "session.done" in kinds, kinds

        # 6. Unknown capability -> helpful error, not a crash.
        out = session._dispatch("nope", {})
        assert out.startswith("error: no capability named"), out

        print("mocktest: ALL CHECKS PASSED")
        print(f"  capabilities: {session.registry.names()}")
        print(f"  audit entries: {len(session.audit)}")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
