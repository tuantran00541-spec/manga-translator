"""After enough tool work the agent reviews the session, keeps its mistakes and fixes as notes, and writes its own skills (Hermes Agent's learning loop)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from app.agent import skills

PROMPT = """You keep the skill library of a coding agent. Read the session below and decide whether a future session would handle this kind of task better with a skill.
A skill tells how to do a KIND of task for this user: the steps in order with the commands and tool calls that worked, the points where a choice is made, how the user wants the result, and the traps (a general rule plus a short why).
Write one when the user corrected the approach, style or format; when a non-obvious technique, fix, workaround or way of debugging came out of the work; or when a learned skill below proved wrong, incomplete or out of date.
Make the smallest edit the evidence supports: update one learned skill below rather than add a new one, delete one only when this session shows it wrong, and leave the rest alone. Name skills after the kind of task (lowercase-hyphens), never after one PR, error text, date or today's task.
Leave out: environment problems the user can fix (missing programs, unset keys), claims that a tool is broken, errors a retry cleared, one-off stories, failures left unsolved, and secrets, keys or tokens.
Also keep this session's lesson notes: each mistake the agent made and how it was found and fixed (a wrong assumption, a failed command, a bug it caused, a test that caught it), written as what went wrong, why, the fix, and the rule that avoids it next time.
The notes this session already has are shown; return only new lessons (never repeat or reword one), and in drop give the exact text of any earlier lesson this session proved wrong.
One-off chains (scripts run once without keeping) that ended cleanly are listed; put in keep_tools the ones general enough to reuse.
Reply with one line of JSON: {"edits": [{"action": "create|update|delete", "name": "...", "description": "when to use it, one sentence", "body": "full markdown instructions", "reason": "the evidence"}], "lessons": ["new lesson"], "drop": ["old lesson text"], "keep_tools": ["name"]} with at most 2 edits; use [] when nothing is worth keeping."""
CONSOLIDATE = """You keep a coding agent's lesson notes: one markdown file per past session listing mistakes and their fixes, and maybe a SUMMARY.md.
Read them and decide. When they repeat each other, or one summary would serve a future session better than many files, reply {"action": "summarize", "summary": "markdown"}: fold every lesson that still holds into one summary grouped by topic, each as a rule with its why, keeping the specifics (commands, paths, error text) that make it usable and dropping one-off noise and anything later contradicted. Otherwise reply {"action": "keep"}.
Reply with one line of JSON."""
LESSONS = Path(".manga-agent") / "lessons"
HISTORY = Path(".manga-agent") / "learn-history.jsonl"
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


class Changes:
    """Every file one review touches, with what it held before, so the review can be undone as a whole."""

    def __init__(self):
        self.before: dict[Path, str | None] = {}

    def track(self, path: Path) -> None:
        if path not in self.before:
            self.before[path] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None

    def commit(self, home: Path, trigger: str, session_id: str, summary: list[str]) -> str:
        """Append this review to the history with each file's before and after; returns its id, or "" when nothing changed."""
        rows = []
        for path, before in self.before.items():
            after = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
            if after != before:
                rows.append({"path": str(path), "before": before, "after": after})
        if not rows:
            return ""
        entry = {"id": hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:10], "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "trigger": trigger, "session": session_id, "summary": summary, "changes": rows}
        log = home / HISTORY
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]


def history(home: Path) -> list[dict]:
    """Past reviews, oldest first, each marked undone once rolled back."""
    rows, undone = [], set()
    try:
        lines = (home / HISTORY).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "undo" in row:
            undone.add(row["undo"])
        else:
            rows.append(row)
    return [{**row, "undone": row["id"] in undone} for row in rows]


def undo(home: Path, entry_id: str = "") -> str:
    """Put back every file the given review (by default the latest one not undone) changed; a file edited since is left alone."""
    rows = [r for r in history(home) if not r["undone"] and (not entry_id or r["id"] == entry_id)]
    if not rows:
        return ""
    row, kept = rows[-1], []
    for change in row["changes"]:
        path = Path(change["path"])
        now = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        if now != change["after"]:
            kept.append(path.name)
            continue
        if change["before"] is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(change["before"], encoding="utf-8")
    with open(home / HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"undo": row["id"]}) + "\n")
    return row["id"] + (f" (left alone, changed since: {', '.join(kept)})" if kept else "")


def worth_review(items: list[dict], chains: dict | None) -> bool:
    """Evidence a review could learn from: a failed call, a user correction, a non-zero exit, a change made, or a one-off chain to keep."""
    if chains:
        return True
    asks = 0
    for item in items:
        if item.get("from_user"):
            asks += 1
            if asks > 1:
                return True
        elif item["role"] == "tool" and (item.get("failed") or re.search(r"\[exit code [1-9]|Traceback \(most recent", item.get("content") or "")):
            return True
        elif item["role"] == "assistant" and any(c.get("name") not in LOOK_ONLY for c in item.get("calls") or []):
            return True
    return False


LOOK_ONLY = frozenset({"read_file", "list_dir", "search", "glob", "symbols", "changes", "todo_write", "think"})


def review(complete, provider, key: str, model: str, transcript: str, home: Path, notes: Path | None = None,
           chains: dict | None = None) -> tuple[list[dict], dict, dict, list[str]]:
    """(skill edits, {"add": new lessons, "drop": wrong ones}, usage, one-off chains to keep as tools) for the session shown."""
    current = learned(home)
    shown = "\n\n".join(f"### {name}\n{path.read_text(encoding='utf-8', errors='replace')[:4000]}" for name, path in list(current.items())[:MAX_LEARNED]) or "(none yet)"
    so_far = notes.read_text(encoding="utf-8", errors="replace")[:30000] if notes and notes.is_file() else "(none yet)"
    data, usage = _ask(complete, provider, key, model, PROMPT,
                       f"Learned skills:\n{shown[:60000]}\n\nLesson notes for this session so far:\n{so_far}\n\n"
                       f"One-off chains that ended cleanly:\n{_chains(chains)}\n\nSession:\n{transcript[-80000:]}")
    items = data.get("edits") if isinstance(data.get("edits"), list) else data.get("skills") if isinstance(data.get("skills"), list) else []
    add = data.get("lessons")
    add = [add] if isinstance(add, str) else add if isinstance(add, list) else []
    drop = data.get("drop") if isinstance(data.get("drop"), list) else []
    lessons = {"add": [str(x).strip() for x in add if str(x).strip()], "drop": [str(x).strip() for x in drop if str(x).strip()]}
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


def write_lessons(folder: Path, root: Path, session_id: str, add: str | list[str], secrets: tuple[str, ...] = (),
                  drop: list[str] = (), changes: Changes | None = None) -> bool:
    """Append new lessons to this session's notes and remove ones proven wrong, never rewriting the rest; False when nothing changed or a secret showed up."""
    new = [add] if isinstance(add, str) else list(add)
    new = [t.strip() for t in new if t and t.strip()]
    text = "\n".join(new)
    if not session_id or SECRET_LIKE.search(text) or any(s and len(s) >= 8 and s in text for s in secrets):
        return False
    path = session_file(folder, session_id)
    old = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else f"# Lessons from session {session_id} in {root}\n"
    lines = [line for line in old.rstrip("\n").split("\n") if not any(d and d in line for d in drop)]
    body = "\n".join(lines)
    for item in new:
        if item not in body:
            body += "\n\n" + item if not item.startswith("- ") else "\n" + item
    body = body.rstrip() + "\n"
    if body == old or len(body) > MAX_BODY * 2 or body.count("\n") < 2:
        return False
    folder.mkdir(parents=True, exist_ok=True)
    if changes is not None:
        changes.track(path)
    (folder / "PROJECT").write_text(f"{root}\n", encoding="utf-8")
    path.write_text(body, encoding="utf-8")
    return True


def consolidate(complete, provider, key: str, model: str, folder: Path, secrets: tuple[str, ...] = (), changes: Changes | None = None) -> tuple[str, dict]:
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
        archive = folder / "archive"
        if changes is not None:
            for path in [summary, *files, *(archive / p.name for p in files)]:
                changes.track(path)
        summary.write_text(f"# Lessons summary\n\n{text}\n", encoding="utf-8")
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


WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{2,}")
STOP = frozenset("the and for with from that this then than when what into onto are was were not but use used run ran its it's has have had you your can".split())


def _stems(text: str) -> set[str]:
    """Words of a text cut to four letters, so fail, fails and failure match, without common filler words."""
    return {w.lower()[:4] for w in WORD.findall(text) if w.lower() not in STOP}


def _lesson_items(path: Path) -> list[str]:
    """One file's lessons: each bullet with its continuation lines, or each paragraph, without headings."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    items: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            items.append("")
        elif line.lstrip().startswith(("- ", "* ")) or not items or not items[-1]:
            items.append(line.strip())
        else:
            items[-1] += " " + line.strip()
    return [i for i in items if i]


def digest(home: Path, workspace: Path, request: str, limit: int = 8, max_chars: int = 3000) -> str:
    """This project's past lessons most related to the request (by shared words, newest first on ties), for one message at a cold start."""
    folder = project_folder(home, project_root([], workspace))
    files = [folder / "SUMMARY.md", *sorted(_sessions(folder), key=lambda p: p.stat().st_mtime, reverse=True)]
    terms = _stems(request)
    seen, rows = set(), []
    for rank, path in enumerate(files):
        for item in _lesson_items(path):
            if item not in seen:
                seen.add(item)
                rows.append((len(terms & _stems(item)), -rank, item))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    picked, size = [], 0
    for score, _, item in rows:
        if len(picked) >= limit or (score == 0 and len(picked) >= 3) or size + len(item) > max_chars:
            break
        picked.append(item if item.startswith(("- ", "* ")) else f"- {item}")
        size += len(item)
    if not picked:
        return ""
    return ("<lessons>\nMistakes and fixes from earlier sessions in this project, most related to this request first "
            f"(all notes: {folder}):\n" + "\n".join(picked) + "\n</lessons>")


def write(home: Path, items: list[dict], secrets: tuple[str, ...] = (), changes: Changes | None = None) -> list[str]:
    """Apply each valid create, update or delete to the learned skills, counting versions; never touches a skill the user or an installer put there."""
    current, done = learned(home), []
    for item in items:
        name = str(item.get("name") or "").strip().lower()
        if str(item.get("action") or "").lower() == "delete":
            if name in current:
                if changes is not None:
                    changes.track(current[name])
                current.pop(name).unlink()
                done.append(f"Deleted {name}")
            continue
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
        version = 1
        if name in current:
            try:
                version = int(skills.frontmatter(current[name].read_text(encoding="utf-8", errors="replace")[:20_000])[0].get("version") or 1) + 1
            except (OSError, ValueError):
                version = 2
        if changes is not None:
            changes.track(folder / "SKILL.md")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {json.dumps(description, ensure_ascii=False)}\nlearned: true\nversion: {version}\n---\n\n{body}\n", encoding="utf-8")
        done.append(f"{'Updated' if name in current else 'Learned'} {name}" + (f" (v{version})" if version > 1 else ""))
        current[name] = folder / "SKILL.md"
    return done
