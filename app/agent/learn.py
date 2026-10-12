"""After enough tool work the agent reviews the session, keeps its mistakes and fixes as notes, and writes its own skills (Hermes Agent's learning loop)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.agent import skills

PROMPT = """You maintain the skill library of a coding agent. Read the session below and decide whether a future session would do this class of task better with a skill.
A skill is how to do a CLASS of task for this user: the steps in order with the commands and tool calls that worked, decision points, the user's preferences for the result, and pitfalls (a general rule plus one clause of why).
Write one when: the user corrected the approach, style or format; a non-trivial technique, fix, workaround or debugging path emerged; or a learned skill below proved wrong, incomplete or outdated (then rewrite that one in full).
Prefer updating a learned skill below over making a new one. Name at class level (lowercase-hyphens), never after one PR, error string, date or today's task.
Do not capture: environment failures the user can fix (missing binaries, unset keys), claims that a tool is broken, transient errors that a retry fixed, one-off task narratives, unresolved failures, secrets, keys or tokens.
Also keep this session's lesson notes: every mistake the agent made and how it was found and fixed (wrong assumption, failed command, bug it introduced, test that caught it), each as: what went wrong, why, the fix, the rule that avoids it next time.
The notes so far for this session are shown; return the full updated notes in markdown without a title, or "" when the session had no mistakes worth noting.
One-off chains (scripts the agent ran once without keeping) that ended cleanly are listed; name in keep_tools the ones general enough to reuse as tools.
Reply with one line of JSON: {"skills": [{"name": "...", "description": "when to use it, one sentence", "body": "markdown instructions"}], "lessons": "markdown", "keep_tools": ["name"]} with at most 2 skills; use [] when nothing is worth keeping."""
CONSOLIDATE = """You keep a coding agent's lesson notes: one markdown file per past session listing mistakes and their fixes, plus an optional SUMMARY.md.
Read them and decide. If they repeat each other or a summary would help a future session more than many files, reply {"action": "summarize", "summary": "markdown"}: merge every lasting lesson into one summary grouped by topic, each as a rule plus why, keeping specifics (commands, paths, error text) that make it usable, dropping one-off noise and anything already contradicted. Otherwise reply {"action": "keep"}.
Reply with one line of JSON."""
LESSONS = Path(".manga-agent") / "lessons"
CONSOLIDATE_AT = 10
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


def _ask(complete, provider, key: str, model: str, system: str, user: str) -> tuple[dict, dict]:
    """The model's one-line JSON answer and usage; an empty answer on any failure."""
    try:
        reply = complete(provider, key, model, [{"role": "system", "content": system}, {"role": "user", "content": user}], tools=None)
        text = reply["text"]
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        return (data if isinstance(data, dict) else {}), reply.get("usage") or {}
    except Exception:
        return {}, {}


def review(complete, provider, key: str, model: str, transcript: str, home: Path, notes: Path | None = None,
           chains: dict | None = None) -> tuple[list[dict], str | None, dict, list[str]]:
    """(skills to write, this session's updated lesson notes or None, usage, one-off chains to keep as tools) for the session shown."""
    current = learned(home)
    shown = "\n\n".join(f"### {name}\n{path.read_text(encoding='utf-8', errors='replace')[:4000]}" for name, path in list(current.items())[:MAX_LEARNED]) or "(none yet)"
    so_far = notes.read_text(encoding="utf-8", errors="replace")[:30000] if notes and notes.is_file() else "(none yet)"
    data, usage = _ask(complete, provider, key, model, PROMPT,
                       f"Learned skills:\n{shown[:60000]}\n\nLesson notes for this session so far:\n{so_far}\n\n"
                       f"One-off chains that ended cleanly:\n{_chains(chains)}\n\nSession:\n{transcript[-80000:]}")
    items = data.get("skills") if isinstance(data.get("skills"), list) else []
    lessons = data.get("lessons") if isinstance(data.get("lessons"), str) else None
    keep = [str(n) for n in data.get("keep_tools") or [] if str(n) in (chains or {})] if isinstance(data.get("keep_tools"), list) else []
    return [i for i in items[:2] if isinstance(i, dict)], lessons, usage, keep


def _chains(chains: dict | None) -> str:
    return "\n\n".join(f"### {name}\n```python\n{tool['code'][:3000]}\n```" for name, tool in (chains or {}).items()) or "(none)"


def project_root(paths: list[Path], fallback: Path) -> Path:
    """The project the work was really in: the git root (else folder) most of the touched paths sit under."""
    def root_of(path: Path) -> Path:
        path = path if path.is_dir() else path.parent
        for folder in (path, *path.parents):
            if (folder / ".git").exists():
                return folder
        return path
    counts: dict[Path, int] = {}
    for path in paths:
        try:
            found = root_of(path.resolve())
        except OSError:
            continue
        counts[found] = counts.get(found, 0) + 1
    return max(counts, key=lambda p: (counts[p], p == root_of(fallback.resolve()))) if counts else root_of(fallback.resolve())


def project_folder(home: Path, root: Path) -> Path:
    """Where one project's lesson notes live: its folder name plus a hash of its full path."""
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", root.name)[:40] or "root"
    return home / LESSONS / f"{slug}-{hashlib.sha1(str(root).encode()).hexdigest()[:12]}"


def session_file(folder: Path, session_id: str) -> Path:
    return folder / f"{re.sub(r'[^A-Za-z0-9_-]', '_', session_id)[:80] or 'session'}.md"


def _sessions(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.md") if p.name != "SUMMARY.md")


def write_lessons(folder: Path, root: Path, session_id: str, text: str, secrets: tuple[str, ...] = ()) -> bool:
    """Replace this session's lesson notes in its project's folder; False when they are empty or look like they hold a secret."""
    text = text.strip()[:MAX_BODY * 2]
    if not text or not session_id or SECRET_LIKE.search(text) or any(s and len(s) >= 8 and s in text for s in secrets):
        return False
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "PROJECT").write_text(f"{root}\n", encoding="utf-8")
    session_file(folder, session_id).write_text(f"# Lessons from session {session_id} in {root}\n\n{text}\n", encoding="utf-8")
    return True


def consolidate(complete, provider, key: str, model: str, folder: Path, secrets: tuple[str, ...] = ()) -> tuple[str, dict]:
    """Once CONSOLIDATE_AT session notes pile up in a project, let the model merge them into SUMMARY.md (originals go to archive/) or leave them be."""
    files = _sessions(folder)
    state = folder / ".state.json"
    try:
        last = int(json.loads(state.read_text()).get("kept_at", 0))
    except (OSError, ValueError, AttributeError):
        last = 0
    if len(files) < max(CONSOLIDATE_AT, last + CONSOLIDATE_AT):
        return "", {}
    summary = folder / "SUMMARY.md"
    shown = "\n\n".join(f"## FILE {p.name}\n{p.read_text(encoding='utf-8', errors='replace')[:8000]}" for p in files)
    old = summary.read_text(encoding="utf-8", errors="replace")[:30000] if summary.is_file() else "(none yet)"
    data, usage = _ask(complete, provider, key, model, CONSOLIDATE, f"SUMMARY.md now:\n{old}\n\nSession notes:\n{shown[-150000:]}")
    text = str(data.get("summary") or "").strip()
    if data.get("action") == "summarize" and text and not SECRET_LIKE.search(text) and not any(s and len(s) >= 8 and s in text for s in secrets):
        summary.write_text(f"# Lessons summary\n\n{text}\n", encoding="utf-8")
        archive = folder / "archive"
        archive.mkdir(exist_ok=True)
        for path in files:
            path.replace(archive / path.name)
        state.write_text(json.dumps({"kept_at": 0}))
        return f"Merged {len(files)} lesson files into SUMMARY.md", usage
    if data.get("action") == "keep":
        # Asked again only after another CONSOLIDATE_AT sessions add notes.
        state.write_text(json.dumps({"kept_at": len(files)}))
        return f"Kept {len(files)} lesson files as they are", usage
    return "", usage


def lessons_prompt(home: Path, workspace: Path) -> str:
    """A pointer to past lesson notes for the system prompt, worded the same whatever they hold so the cached prefix stays put."""
    base = home / LESSONS
    if not base.is_dir() or not any(base.glob("*/PROJECT")):
        return ""
    here = project_folder(home, project_root([], workspace))
    lead = f"Mistakes and fixes from past sessions in this project are in {here} (SUMMARY.md first, then one file per session). " if here.is_dir() else ""
    return (lead + f"Notes for other projects sit in {base}, one folder per project with a PROJECT file naming its path; "
            "when you work in one of those projects, read its notes before work where an old mistake could repeat.")


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
