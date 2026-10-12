"""Run the agent without the web page: `python -m app.agent.cli exec "task"` or `resume ID "more"`, with --json for one event per line."""
from __future__ import annotations

import argparse
import dataclasses
import ipaddress
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

from app.agent import client, sandbox
from app.agent.session import AgentSessionManager, MODES
from app.agent.tools import Workspace
from app.ai_providers import resolve_provider


def _show(event: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(event, ensure_ascii=False), flush=True)
    elif event["type"] == "assistant" and event.get("text"):
        print(event["text"], flush=True)
    elif event["type"] == "tool":
        print(f"[{event['name']}] {'ok' if event['ok'] else 'failed'}", file=sys.stderr, flush=True)
    elif event["type"] in ("notice", "error"):
        print(event["text"], file=sys.stderr, flush=True)


def _private_base(base: str) -> str:
    """An http base URL on a loopback or private address, for a relay that holds the key (benchmarks); anything else is refused."""
    parsed = urlparse(base)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise SystemExit("--allow-private-base needs an IP address in the base URL") from exc
    if parsed.scheme != "http" or not (address.is_private or address.is_loopback) or parsed.username or parsed.query:
        raise SystemExit("--allow-private-base takes only http://<private or loopback IP>[:port]/path")
    return base.rstrip("/")


def run(args: argparse.Namespace) -> int:
    home = Path.home()
    manager = AgentSessionManager(home / ".manga-agent", home=home)
    # Several keys may be given comma-separated; a limited key moves on to the next one by itself, once round the list.
    keys = [k.strip() for k in os.environ.get(args.key_env, "").split(",") if k.strip()]
    if not keys:
        print(f"Set {args.key_env} to the provider's API key (several comma-separated).", file=sys.stderr)
        return 2
    if args.allow_private_base:
        provider = dataclasses.replace(resolve_provider(args.provider, label=args.provider, protocol="openai", api_base="https://relay.invalid"), api_base=_private_base(args.base))
        client.PRIVATE_BASES.add(provider.api_base.rstrip("/"))
    else:
        provider = resolve_provider(args.provider, label=args.provider, protocol="openai", api_base=args.base)
    if args.command == "resume":
        data = manager.saved(args.session)
        mode, network = (data.get("sandbox") or ["workspace-write", False])[:2]
        workspace = Workspace(data["workspace"], sandbox.Policy(mode, bool(network)))
        session = manager.create(provider, keys, args.model or data["model"], workspace, args.mode, session_id=args.session)
        session.restore(data)
    else:
        workspace = Workspace(args.workspace, sandbox.Policy(args.sandbox, args.network))
        session = manager.create(provider, keys, args.model, workspace, args.mode)
    seen = len(session.events)
    session.set_deadline(args.timeout_min * 60)
    # A line starting with / is a slash command, such as /goal TEXT to keep going until the goal is done.
    session.command(args.prompt) if args.prompt.startswith("/") else session.send(args.prompt)
    deadline = time.time() + args.timeout_min * 60
    switches = 0
    while True:
        snap = session.snapshot(seen)
        for event in snap["events"]:
            _show(event, args.json)
            seen = max(seen, event["seq"])
            if event["type"] == "key_limit":
                switches += 1
                session.choose_key("next" if switches < len(keys) else "stop")
        if snap["status"] == "waiting":
            # Nobody is here to answer: refuse, and say how to allow it next time.
            session.decide("deny", "Running without a person to approve; the user can rerun with --mode auto or bypass.")
        elif snap["status"] == "idle":
            break
        if time.time() > deadline:
            session.stop()
            print("Timed out.", file=sys.stderr)
            break
        time.sleep(0.2)
    # The learning review runs after the turn; a process that exits at once would cut it off.
    session.wait_learning()
    final = session.snapshot(seen)
    for event in final["events"]:
        _show(event, args.json)
    result = {"type": "result", "session": session.id, "status": final["status"], "usage": final["usage"], "stats": final["stats"]}
    if args.json:
        print(json.dumps(result), flush=True)
    else:
        print(f"session {session.id} · {final['usage']['prompt_tokens']} in, {final['usage']['completion_tokens']} out · ${final['stats']['cost']}", file=sys.stderr)
    if not args.stop_jobs:
        # Servers and other background jobs the agent left running outlive the command, as they would in a shell.
        for name, job in list(session.jobs.items()):
            if hasattr(job, "detach") and job.alive:
                job.detach()
                session.jobs.pop(name)
                print(f"left running: {name} (process group {job.proc.pid}): {job.command[:120]}", file=sys.stderr)
    manager.close_all()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chaos", description="Chaos, the Manga Translator coding agent, run from a terminal")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("exec", "resume"):
        one = sub.add_parser(name)
        if name == "resume":
            one.add_argument("session")
        one.add_argument("prompt")
        one.add_argument("--base", required=True, help="OpenAI-compatible API base URL")
        one.add_argument("--model", default="" if name == "resume" else None, required=name == "exec")
        one.add_argument("--key-env", default="AGENT_API_KEY", help="environment variable holding the API key")
        one.add_argument("--provider", default="cli")
        one.add_argument("--workspace", default=".")
        one.add_argument("--mode", choices=MODES, default="auto")
        one.add_argument("--sandbox", choices=sandbox.MODES, default="workspace-write")
        one.add_argument("--network", action="store_true")
        one.add_argument("--timeout-min", type=int, default=60)
        one.add_argument("--allow-private-base", action="store_true", help="allow an http base URL on a private or loopback IP (a local relay)")
        one.add_argument("--json", action="store_true", help="print one JSON event per line")
        one.add_argument("--stop-jobs", action="store_true", help="stop background jobs the agent left running when the command ends")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
