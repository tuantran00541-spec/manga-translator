"""Run a Seed session from the command line.

The model starts with NO tools — only `define` — and must write every
capability it needs. Example:

    SEED_API_KEY=... python -m app.agent.seed.demo \
        --provider gemini --model gemini-2.0-flash \
        --workspace /tmp/seed-ws \
        --task "Read notes.txt, summarize it in 3 lines, write summary.txt"
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed: the tool-less harness demo")
    parser.add_argument("--provider", default="gemini", help="provider id from app.ai_providers.PROVIDERS, or 'custom'")
    parser.add_argument("--api-base", default=os.getenv("SEED_API_BASE", ""),
                        help="base URL for a custom OpenAI-compatible provider (or SEED_API_BASE env)")
    parser.add_argument("--model", default="", help="model name (default: provider default)")
    parser.add_argument("--api-key", default=os.getenv("SEED_API_KEY", ""), help="or SEED_API_KEY env")
    parser.add_argument("--workspace", default=".", help="folder the session works in")
    parser.add_argument("--task", required=True, help="the task to give the model")
    parser.add_argument("--yes", action="store_true", help="auto-approve persisted capabilities")
    parser.add_argument("--max-steps", type=int, default=40,
                        help="max agent steps; 0 = unlimited (endurance mode)")
    args = parser.parse_args(argv)

    if not args.api_key:
        print("need --api-key or SEED_API_KEY", file=sys.stderr)
        return 2

    from app.ai_providers import PROVIDERS, resolve_provider
    from app.agent.seed.session import SeedSession

    if args.provider == "custom" or args.api_base:
        if not args.api_base:
            print("custom provider needs --api-base or SEED_API_BASE", file=sys.stderr)
            return 2
        provider = resolve_provider("seed-custom", protocol="openai", api_base=args.api_base,
                                    label="seed-custom")
    else:
        provider = PROVIDERS.get(args.provider)
        if provider is None:
            print(f"unknown provider {args.provider!r}; known: {', '.join(sorted(PROVIDERS))}", file=sys.stderr)
            return 2
    model = args.model or provider.default_qc_model

    approve = (lambda prompt: True) if args.yes else None
    if approve is None:
        def approve(prompt: str) -> bool:
            answer = input(f"{prompt} [y/N] ").strip().lower()
            return answer in ("y", "yes")

    session = SeedSession(provider, args.api_key, model, args.workspace,
                          approve=approve, max_steps=args.max_steps)
    print(f"Seed session {session.session_id} — model starts with ONE tool: define.\n")
    result = session.run(args.task)
    print("\n--- final ---\n" + result)
    print(f"\nCapabilities defined: {session.registry.names() or '(none)'}")
    print(f"Audit entries: {len(session.audit)} (see ~/.seed/audit-{session.session_id}.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
