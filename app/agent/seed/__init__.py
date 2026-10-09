"""Seed: the tool-less agent harness.

The harness ships with NO domain tools — no read_file, no shell, no web
search. The model starts with exactly one tool, ``define``, and must write
every capability it needs as Python code against the fixed ``harness`` API.

The only fixed things (the immutable core):
  1. ``define`` — the seed tool that registers a capability.
  2. ``registry`` — static checks + restricted execution of capability code.
  3. ``harness`` — the world-access API below (workspace containment,
     OS-sandboxed shell, audit log).
  4. The audit log and the approval gate for persisted capabilities.

Everything else — tools, plugins, MCP clients — is a *capability*:
the same single concept. "Everything is everything."
"""
