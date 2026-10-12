"""After enough tool work the agent reviews the session and writes or updates its own skills (Hermes Agent's learning loop)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.agent import skills

PROMPT = """You maintain the skill library of a coding agent. Read the session below and decide whether a future session would do this class of task better with a skill.
A skill is how to do a CLASS of task for this user: the steps in order with the commands and tool calls that worked, decision points, the user's preferences for the result, and pitfalls (a general rule plus one clause of why).
Write one when: the user corrected the approach, style or format; a non-trivial technique, fix, workaround or debugging path emerged; or a learned skill below proved wrong, incomplete or outdated (then rewrite that one in full).
Prefer updating a learned skill below over making a new one. Name at class level (lowercase-hyphens), never after one PR, error string, date or today's task.
Do not capture: environment failures the user can fix (missing binaries, unset keys), claims that a tool is broken, transient errors that a retry fixed, one-off task narratives, unresolved failures, secrets, keys or tokens.
Reply with one line of JSON: {"skills": [{"name": "...", "description": "when to use it, one sentence", "body": "markdown instructions"}]} with at most 2 items, or {"skills": []} when nothing is worth keeping."""
FOLDER = Path(".manga-agent") / "skills"
MAX_LEARNED = 30
MAX_BODY = 12_000
SECRET_LIKE = re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,})|Bearer\s+[A-Za-z0-9._-]{20,}")


def learned(home: Path) -> dict[str, Path]:
    """Skills this loop wrote, by name; only these may be rewritten by it."""
    rows = {}
    for path in sorted((home / FOLDER).glob("*/SKILL.md")):
        try:
            fields, _ = skills.frontmatter(path.read_text(encoding="utf-8", errors="replace")[:20_000])
        except OSError:
            continue
        if fields.get("learned", "").lower() == "true":
            rows[path.parent.name] = path
    return rows


def review(complete, provider, key: str, model: str, transcript: str, home: Path) -> tuple[list[dict], dict]:
    """(skills to write, usage) for the session shown; nothing on any failure, since writing nothing is always safe."""
    current = learned(home)
    shown = "\n\n".join(f"### {name}\n{path.read_text(encoding='utf-8', errors='replace')[:4000]}" for name, path in list(current.items())[:MAX_LEARNED]) or "(none yet)"
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Learned skills:\n{shown[:60000]}\n\nSession:\n{transcript[-80000:]}"}]
    try:
        reply = complete(provider, key, model, messages, tools=None)
        text = reply["text"]
        start, end = text.find("{"), text.rfind("}")
        items = json.loads(text[start:end + 1]).get("skills") or []
    except Exception:
        return [], {}
    return [i for i in items[:2] if isinstance(i, dict)], reply.get("usage") or {}


def write(home: Path, items: list[dict], secrets: tuple[str, ...] = ()) -> list[str]:
    """Write each valid item as a learned skill; never touches a skill the user or an installer put there."""
    current, done = learned(home), []
    for item in items:
        name = str(item.get("name") or "").strip().lower()
        description = " ".join(str(item.get("description") or "").split())[:300]
        body = str(item.get("body") or "").strip()
        if not skills.NAME_RE.match(name) or not description or not body or len(body) > MAX_BODY:
            continue
        if SECRET_LIKE.search(body + description) or any(s and len(s) >= 8 and s in body for s in secrets):
            continue
        folder = home / FOLDER / name
        if (folder / "SKILL.md").exists() and name not in current:
            continue
        if name not in current and len(current) >= MAX_LEARNED:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {json.dumps(description, ensure_ascii=False)}\nlearned: true\n---\n\n{body}\n", encoding="utf-8")
        done.append(f"{'Updated' if name in current else 'Learned'} {name}")
        current[name] = folder / "SKILL.md"
    return done
