# Repository Cleanup and Comment Normalization Design

## Goal

Remove tracked junk from `main` and keep production source comments/docstrings concise without changing runtime behavior.

## Scope

- Preserve application logic, font matching/default-auto behavior, model paths, tests, benchmark evidence, audit artifacts, and CI workflows.
- Remove only confirmed empty/generated artifacts; do not infer dead source files from names alone.
- In `app/` and `scripts/`, collapse multiline Python docstrings and CSS/JS block comments to one concise line while preserving tool directives and required safety notes.
- Keep tests, docs, benchmark outputs, and workflow evidence unchanged unless a tracked artifact is proven to be junk.

## Verification

Run Python compilation, focused font/catalog checks, existing pytest/release checks, and the live UI smoke workflow.
