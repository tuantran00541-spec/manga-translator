# Repository Cleanup and Comment Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove confirmed repository junk and normalize production comments/docstrings to concise single-line forms without changing behavior.

**Architecture:** Work from the current `main` tree, create a small allowlist of empty assets, use syntax-aware source transformations for `app/` and `scripts/`, then verify with existing local and GitHub Actions gates.

**Tech Stack:** Python, FastAPI, vanilla JavaScript/CSS, GitHub Actions, GitHub Contents/Git Data APIs.

**Spec:** [docs/superpowers/specs/2026-09-22-repository-cleanup-design.md](../specs/2026-09-22-repository-cleanup-design.md)

---

### Task 1: Audit and allowlist

- [ ] Reconfirm current `main` tree and parent SHA.
- [ ] Identify only zero-byte or generated artifacts with no runtime purpose.
- [ ] Scan production Python/JS/CSS for multiline comments/docstrings and tool directives.
- [ ] Record preserved paths: models, tests, benchmark/evidence assets, docs, workflows, and manual correction behavior.

### Task 2: Remove confirmed junk

- [ ] Remove only allowlisted empty font placeholders.
- [ ] Update the font catalog and metadata so every listed asset exists and is non-empty.
- [ ] Keep the catalog within the requested 60–80-font range.

### Task 3: Normalize production comments

- [ ] Collapse multiline module/function/class docstrings to one concise line.
- [ ] Collapse multiline CSS/JS block comments to one line.
- [ ] Preserve shebangs, encoding markers, lint/type/coverage directives, licenses, and safety-critical notes.
- [ ] Do not alter executable statements or test/benchmark evidence.

### Task 4: Verify

- [ ] Run compile/syntax and focused font checks.
- [ ] Run the existing release gate and live browser smoke workflow.
- [ ] Inspect the resulting diff for accidental behavior or asset changes.
