#!/usr/bin/env python3
"""Per-file automated code review gate powered by Google Gemini Decision 1.

Usage:
  python scripts/decision_review.py                      # review HEAD~1..HEAD
  python scripts/decision_review.py --rev ecc62f0c..HEAD # review commit range
  python scripts/decision_review.py --staged             # review staged changes
  python scripts/decision_review.py --working            # review unstaged + staged changes
  python scripts/decision_review.py --files a.py b.py    # review specific files
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from google import genai

MODEL = "gemini-decision-1"
MAX_FULL_FILE_CHARS = 140_000
MAX_DIFF_CHARS = 45_000
CODE_EXTENSIONS = frozenset({".py", ".js", ".html", ".css"})

FULL_FILE_QUESTIONS: dict[str, dict[str, Any]] = {
    "has_correctness_bug": {
        "type": "noul",
        "instructions": (
            "Does this source file contain a concrete correctness bug "
            "(logic error, unhandled None/empty/shape crash, race condition, off-by-one, or broken invariant)?"
        ),
    },
    "has_security_or_data_loss_risk": {
        "type": "noul",
        "instructions": (
            "Does this file contain a security vulnerability (path traversal, XSS, SSRF, command/SQL injection, "
            "secret leak, auth bypass) or a risk of silent user-data loss or state corruption?"
        ),
    },
    "has_perf_or_memory_issue": {
        "type": "noul",
        "instructions": (
            "Does this file contain a meaningful performance or memory bottleneck "
            "(unnecessary full-image copies, O(N^2) loops, unbounded caches/memory growth, or blocking I/O on async event loop)?"
        ),
    },
    "has_overfitting_or_magic_hack": {
        "type": "noul",
        "instructions": (
            "Does this file rely heavily on brittle hardcoded magic numbers, fragile regex/DOM assumptions, "
            "or narrow heuristics likely to fail on unseen inputs?"
        ),
    },
    "needs_refactor": {
        "type": "noul",
        "instructions": (
            "Is this file overly monolithic, tangled, duplicated, or hard to maintain enough that it genuinely needs refactoring?"
        ),
    },
    "verdict": {
        "type": "choice",
        "instructions": "As a strict senior staff engineer auditing this entire file, what is your verdict?",
        "criteria": {
            "approve": "Clean, solid, production-ready code with no concrete defects.",
            "approve_with_nits": "Generally solid; only minor polish, cleanup, or low-risk caveats.",
            "request_changes": "Contains real bugs, fragile architecture, or notable risks that should be fixed.",
            "block": "Critical crash, security vulnerability, or data-corruption hazard.",
        },
    },
    "code_quality": {
        "type": "score",
        "instructions": "Rate the overall engineering quality of this file strictly (0=broken, 4=production-grade).",
        "criteria": [
            "Broken or dangerous: concrete bugs, crashes, or security holes.",
            "Weak: fragile hacks, poor error handling, or severe technical debt.",
            "Acceptable: functional but has notable edge-case gaps, bloat, or roughness.",
            "Solid: clean, idiomatic, defensive, and well-structured with minor nits at most.",
            "Production-grade: crisp, modular, well-typed, defensive, and exemplary.",
        ],
    },
    "maintainability": {
        "type": "score",
        "instructions": "Rate the readability, modularity, and long-term maintainability of this file (0=unmaintainable, 4=exemplary).",
        "criteria": [
            "Unmaintainable tangle or god-module.",
            "Hard to follow; oversized functions, hidden coupling, or scattered state.",
            "Moderately maintainable; some bloat or duplicated logic.",
            "Clean, well-decomposed, and easy to reason about.",
            "Exemplary modular design with crystal-clear boundaries.",
        ],
    },
    "review_focus": {
        "type": "choice",
        "instructions": "Which risk area (if any) is the #1 priority to inspect or improve in this file?",
        "criteria": {
            "none": "No notable risk area.",
            "edge_cases_and_bugs": "Null/empty checks, type/shape mismatches, error handling, logic bugs.",
            "security_and_data": "Input validation, path/URL safety, XSS, auth, persistence/concurrency.",
            "thresholds_and_heuristics": "Magic numbers, brittle heuristics, generalization risk.",
            "performance_and_memory": "Async blocking, redundant compute/inference, large copies, memory leaks.",
            "architecture_and_bloat": "God-file size, tight coupling, duplicated code, maintainability.",
        },
    },
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "has_correctness_bug": {
        "type": "noul",
        "instructions": (
            "Does the diff introduce or leave a concrete correctness bug in this file "
            "(logic error, off-by-one, unhandled None/empty/shape crash, wrong coordinate space, or broken invariant)?"
        ),
    },
    "has_regression_risk": {
        "type": "noul",
        "instructions": (
            "Could the diff break existing callers, API/manifest contracts, tests, "
            "or valid edge cases outside the intended change?"
        ),
    },
    "has_overfitting_or_magic_hack": {
        "type": "noul",
        "instructions": (
            "Does the change rely on brittle hardcoded assumptions or narrow magic thresholds "
            "that are likely to overfit or fail to generalize to unseen real-world inputs?"
        ),
    },
    "has_perf_or_memory_issue": {
        "type": "noul",
        "instructions": (
            "Does the diff introduce a meaningful performance or memory issue "
            "(unnecessary full-image copies, O(N^2) loops, redundant inference, unbounded caches, or hot-path blocking)?"
        ),
    },
    "has_security_or_data_loss_risk": {
        "type": "noul",
        "instructions": (
            "Does the diff introduce a security issue (path traversal, injection, XSS) "
            "or risk of silent user-data overwrite/loss or stale-state corruption?"
        ),
    },
    "verdict": {
        "type": "choice",
        "instructions": "As a strict senior staff engineer, what is your review verdict for this file's changes?",
        "criteria": {
            "approve": "Clean, safe, well-scoped change with no concrete defects.",
            "approve_with_nits": "Safe to merge; only minor polish, readability, or non-blocking caveats.",
            "request_changes": "Contains a real bug, regression risk, or generalization flaw that should be fixed.",
            "block": "Critical crash, data corruption/loss, or security vulnerability.",
        },
    },
    "code_quality": {
        "type": "score",
        "instructions": "Rate the engineering quality of this change strictly (0=broken, 4=production-grade).",
        "criteria": [
            "Broken or dangerous: concrete bugs, crashes, or severe regressions.",
            "Weak: fragile hacks, missing edge-case guards, or hard-to-maintain code.",
            "Acceptable: works for common cases but has notable edge-case gaps or roughness.",
            "Solid: clean, idiomatic, handles edge cases well, at most minor nits.",
            "Production-grade: crisp, defensive, well-structured, and cleanly bounded.",
        ],
    },
    "generalization_score": {
        "type": "score",
        "instructions": "Rate how well the logic generalizes across diverse unseen inputs/chapters (0=overfit, 4=principled).",
        "criteria": [
            "Hardcoded or overfit to a single specific input.",
            "Narrowly tuned with brittle assumptions.",
            "Moderately general, may struggle on unusual inputs.",
            "Broadly generalizable across diverse real-world inputs.",
            "Principled, invariant-based design that generalizes robustly.",
        ],
    },
    "review_focus": {
        "type": "choice",
        "instructions": "Which risk area (if any) warrants the closest follow-up inspection in this file?",
        "criteria": {
            "none": "No notable risk area.",
            "edge_cases_and_types": "Empty arrays/crops, grayscale/RGBA, None values, coordinate bounds.",
            "thresholds_and_heuristics": "Magic numbers, precision/recall tradeoffs, generalization.",
            "performance_and_memory": "Extra model passes, large copies, cache eviction, latency.",
            "state_and_contracts": "Manifest/API schemas, concurrency/staleness, UI state sync.",
        },
    },
}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return proc.stdout


def _changed_files(repo: Path, args: argparse.Namespace) -> tuple[list[str], list[str], bool]:
    if args.all_code:
        roots = args.roots or ["app"]
        found: list[str] = []
        for root_name in roots:
            base = repo / root_name
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if (
                    p.is_file()
                    and p.suffix.lower() in CODE_EXTENSIONS
                    and "__pycache__" not in p.parts
                    and p.stat().st_size >= 20
                ):
                    found.append(p.relative_to(repo).as_posix())
        return found, [], True
    if args.files:
        return list(args.files), ["diff", "HEAD"], False
    if args.staged:
        diff_args = ["diff", "--cached"]
    elif args.working:
        diff_args = ["diff", "HEAD"]
    else:
        diff_args = ["diff", args.rev]
    names = [line.strip() for line in _git(repo, *diff_args, "--name-only").splitlines() if line.strip()]
    return names, diff_args, False


def _build_payload(repo: Path, rel_path: str, diff_args: list[str], *, full_audit: bool = False) -> str:
    full_path = repo / rel_path
    if full_path.is_file():
        source = full_path.read_text(encoding="utf-8", errors="replace")
        if len(source) > MAX_FULL_FILE_CHARS:
            source = source[:MAX_FULL_FILE_CHARS] + "\n... [file truncated for review budget] ..."
    else:
        source = "[file deleted or not present on disk]"
    if full_audit:
        return f"FILE: {rel_path}\n=== SOURCE CODE ===\n{source}\n"
    diff_text = _git(repo, *diff_args, "--", rel_path)
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + "\n... [diff truncated for review budget] ..."
    return (
        f"FILE: {rel_path}\n"
        f"=== GIT DIFF ===\n{diff_text or '(no diff — full file review)'}\n"
        f"=== CURRENT FILE CONTENT ===\n{source}\n"
    )


def _review_file(
    client: genai.Client,
    repo: Path,
    rel_path: str,
    diff_args: list[str],
    *,
    full_audit: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = _build_payload(repo, rel_path, diff_args, full_audit=full_audit)
    questions = FULL_FILE_QUESTIONS if full_audit else QUESTIONS
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            res = client.decisions.create(
                model=MODEL,
                input=payload,
                questions=questions,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 3:
                raise
            time.sleep(1.5 * (2 ** attempt))
    else:
        raise last_exc  # pragma: no cover

    dt = time.perf_counter() - t0
    ans = res.answers
    usage = getattr(res, "usage", None)
    tokens = getattr(usage, "total_input_tokens", None) if usage is not None else None
    if full_audit:
        return {
            "file": rel_path,
            "verdict": ans["verdict"].choice,
            "verdict_conf": round(float(ans["verdict"].confidence), 3),
            "quality": round(float(ans["code_quality"].score), 2),
            "maintainability": round(float(ans["maintainability"].score), 2),
            "bug_p": round(float(ans["has_correctness_bug"].noul), 3),
            "sec_p": round(float(ans["has_security_or_data_loss_risk"].noul), 3),
            "perf_p": round(float(ans["has_perf_or_memory_issue"].noul), 3),
            "overfit_p": round(float(ans["has_overfitting_or_magic_hack"].noul), 3),
            "refactor_p": round(float(ans["needs_refactor"].noul), 3),
            "focus": ans["review_focus"].choice,
            "focus_conf": round(float(ans["review_focus"].confidence), 3),
            "tokens": tokens,
            "elapsed_s": round(dt, 2),
        }
    return {
        "file": rel_path,
        "verdict": ans["verdict"].choice,
        "verdict_conf": round(float(ans["verdict"].confidence), 3),
        "quality": round(float(ans["code_quality"].score), 2),
        "generalization": round(float(ans["generalization_score"].score), 2),
        "bug_p": round(float(ans["has_correctness_bug"].noul), 3),
        "regr_p": round(float(ans["has_regression_risk"].noul), 3),
        "overfit_p": round(float(ans["has_overfitting_or_magic_hack"].noul), 3),
        "perf_p": round(float(ans["has_perf_or_memory_issue"].noul), 3),
        "sec_p": round(float(ans["has_security_or_data_loss_risk"].noul), 3),
        "focus": ans["review_focus"].choice,
        "focus_conf": round(float(ans["review_focus"].confidence), 3),
        "tokens": tokens,
        "elapsed_s": round(dt, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score changed or all codebase files with gemini-decision-1.")
    parser.add_argument("--repo", default=".", help="Git repository root.")
    parser.add_argument("--rev", default="HEAD~1..HEAD", help="Git revision range (default: HEAD~1..HEAD).")
    parser.add_argument("--staged", action="store_true", help="Review staged diff.")
    parser.add_argument("--working", action="store_true", help="Review working tree diff against HEAD.")
    parser.add_argument("--all-code", action="store_true", help="Audit every non-empty code file under --roots.")
    parser.add_argument("--roots", nargs="*", help="Directories for --all-code (default: app).")
    parser.add_argument("--files", nargs="*", help="Explicit relative file paths to review.")
    parser.add_argument("--full-audit", action="store_true", help="Force full-file audit rubric even with --files/--working.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API requests.")
    parser.add_argument("--json-out", default="", help="Optional path to save full JSON report.")
    parser.add_argument("--fail-on-flag", action="store_true", help="Exit 1 if any file is request_changes/block.")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
        return 2

    repo = Path(args.repo).resolve()
    files, diff_args, full_audit = _changed_files(repo, args)
    if args.full_audit:
        full_audit = True
    if not files:
        print("No files found.")
        return 0

    print(f"Scoring {len(files)} files with {MODEL} (workers={args.workers}, full_audit={full_audit})...", flush=True)
    client = genai.Client(http_options={"api_version": "v1beta"})
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_review_file, client, repo, f, diff_args, full_audit=full_audit): f for f in files}
        for fut in as_completed(futs):
            rel = futs[fut]
            done += 1
            try:
                row = fut.result()
                results.append(row)
                if full_audit:
                    print(
                        f"({done:3d}/{len(files)}) [{row['verdict']:17s}] Q={row['quality']:.2f} M={row['maintainability']:.2f} "
                        f"bug={row['bug_p']:.2f} sec={row['sec_p']:.2f} perf={row['perf_p']:.2f} "
                        f"ovf={row['overfit_p']:.2f} ref={row['refactor_p']:.2f} focus={row['focus']} :: {rel}",
                        flush=True,
                    )
                else:
                    print(
                        f"({done:3d}/{len(files)}) [{row['verdict']:17s}] Q={row['quality']:.2f}/4 G={row['generalization']:.2f}/4 "
                        f"bug={row['bug_p']:.2f} regr={row['regr_p']:.2f} ovf={row['overfit_p']:.2f} "
                        f"perf={row['perf_p']:.2f} sec={row['sec_p']:.2f} focus={row['focus']} :: {rel}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"({done:3d}/{len(files)}) [ERROR            ] {rel}: {exc}", file=sys.stderr, flush=True)
                results.append({"file": rel, "verdict": "error", "error": str(exc)})

    results.sort(key=lambda r: (
        {"block": 0, "request_changes": 1, "approve_with_nits": 2, "approve": 3, "error": 4}.get(r.get("verdict", ""), 5),
        r.get("quality", 99.0),
        r["file"],
    ))
    elapsed = time.perf_counter() - started
    valid = [r for r in results if r.get("verdict") != "error"]
    avg_q = sum(r["quality"] for r in valid) / max(1, len(valid))
    second_key = "maintainability" if full_audit else "generalization"
    avg_2 = sum(r[second_key] for r in valid) / max(1, len(valid))
    total_toks = sum(int(r.get("tokens") or 0) for r in valid)
    flagged = [r for r in valid if r["verdict"] in ("request_changes", "block")]

    print(
        f"\nSUMMARY: {len(valid)}/{len(files)} files scored in {elapsed:.1f}s | "
        f"avg_quality={avg_q:.2f}/4.00 | avg_{second_key}={avg_2:.2f}/4.00 | "
        f"flagged={len(flagged)} | input_tokens={total_toks:,}",
        flush=True,
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return 1 if (args.fail_on_flag and flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
