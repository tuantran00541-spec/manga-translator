# Chapter Readability Remediation Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans and test-driven development. Do not delegate without user authorization. This is a staged roadmap; each subsystem must have a detailed, reviewed implementation plan before its code is changed.

**Goal:** Prevent silent loss of chapter content and produce an export whose completeness and review status are honest.

**Architecture:** Preserve original images and independently track detection, cleanup, translation, layout and export states. Ambiguous machine decisions must remain reviewable instead of becoming human approvals; published artifacts remain revision-checked and recoverable.

**Tech Stack:** Python, FastAPI, Pillow, NumPy, OpenCV, ONNX Runtime, SQLite, pytest, browser extension.

**Spec:** `docs/superpowers/specs/2026-11-01-core-audit-remediation-design.md`.

## Global Constraints

- Do not filter detected text as watermark before cleaning or translation; an empty model response is not proof of non-story text.
- Keep originals and existing chapter data recoverable; support old manifests and series memory without destructive migration.
- Do not modify main, push, merge, or commit copyrighted source images or audit output to a code branch.
- Preserve desktop and extension workflows; do not silently change externally exposed service behavior.
- Write a failing regression before each fix; never change destructive-cleaning defaults solely from synthetic evidence.
- Existing uncommitted code, plans and probes are user work; do not discard or overwrite them.

## Work sequence and acceptance gates

### 0. Establish the baseline

**Files:** `requirements.txt`, `requirements-test.txt`, existing tests and model configuration.

- [ ] Use an isolated environment with the pinned dependencies, including Playwright and keyring; report unavailable models, browser binaries, network or credentials rather than changing production configuration.
- [ ] Run `python -m pytest tests/test_ai_mode.py tests/test_ai_seams.py tests/test_ai_seam_overlaps.py tests/test_editorial_gate.py tests/test_export_gate.py tests/test_export_pages.py tests/test_vision_translation.py tests/test_cleanup_safety_regressions.py tests/test_render_regions.py tests/test_ocr_render_metadata.py tests/test_kiuyha_detector.py -q` and record pass/fail/error counts outside the code branch.
- [ ] Run `python -m pytest tests/test_gateway_auth_billing.py tests/test_gateway_upstream_retry.py tests/test_host_guard.py tests/test_download_pinning.py tests/test_upload_archive.py -q`; distinguish environment failures from application defects.
- [ ] Preserve the existing regression `test_blade_chapter_two_slice_nine_lobes_are_not_joined` and record its failure before modifying behavior.

### 1. Prevent missing text from becoming an approved blank

**Files:** `app/routers/translation.py`, `app/ai_mode/job.py`, `app/editorial_gate.py`, `tests/test_ai_mode.py`, `tests/test_vision_translation.py`, `tests/test_editorial_gate.py`.

**Contract:** Absent IDs, explicit empty answers, failures and budget exhaustion remain distinct machine outcomes. None implies human review, watermark or non-story. Unresolved visible story candidates block normal export or retain original pixels and explicitly require review.

- [ ] Add tests covering a mixed successful/empty response and an all-empty response; assert the attempt state persists in both, no `editorial_reviewed` flag is manufactured, and export remains blocked until translation or an explicit human disposition.
- [ ] Run the focused tests and confirm the current code fails on the stated assertions.
- [ ] Save machine attempt state even when no translation commits; preserve explicit human decisions and invalidate stale artifacts when inputs change.
- [ ] Replace automatic blank-to-`non_story` settlement with an unresolved review state; preserve the distinction between omitted and explicitly empty results and keep the source recoverable.
- [ ] Test retry, force export, review and existing manifests without treating old machine-set approvals as reliable human evidence; devise a non-destructive compatibility strategy before migration.
- [ ] Run the three affected test files and `tests/test_export_gate.py`; require all new and existing compatible tests to pass.

### 2. Detection, filtering, seam ownership and cleaning

**Files:** `app/ai_mode/job.py`, `app/ai_mode/decider.py`, `app/ai_mode/seams.py`, `app/page_processing.py`, `app/detector/kiuyha_detector.py`, `app/detector/boxes.py`, `app/detector/mask_builder.py`, and corresponding tests.

- [ ] Add regressions for wrongly classified credit/textless slices, watermark-like dialogue, detector boxes with empty masks, a truly blank slice, and a human-preserved region; assert no machine-only classification silently suppresses detected text or authorizes a rectangle erase.
- [ ] Add a slice containing visible dialogue but zero detector candidates; assert a whole-slice missed-text check or explicit review catches it rather than treating the empty candidate list as evidence of completeness.
- [ ] Test the existing working-tree empty-mask changes through detection, NMS, persistence, OCR, inpaint and editorial preflight; resolve failures without deleting the current edits.
- [ ] Add paired fixtures for separate double-bubble lobes and true split title/sentence lines, including the existing Blade regression and seam-crossing cases; retain both translations and source masks when evidence is ambiguous.
- [ ] Before implementing merge policy, document which mask/letter/bubble evidence distinguishes a confident join from an uncertain case; do not tune coordinate thresholds solely to pass the Blade test.
- [ ] Compare original, mask, clean and rendered crops for representative real chapters; record text coverage and damage outside masks. Do not enable a new clean default without that comparison.
- [ ] Run `python -m pytest tests/test_kiuyha_detector.py tests/test_cleanup_safety_regressions.py tests/test_ai_seams.py tests/test_ai_seam_overlaps.py tests/test_region_policy.py -q`.

### 3. Render legibility and slice integrity

**Files:** `app/render/page_renderer.py`, `app/render/text_renderer.py`, `app/routers/render_commit.py`, `tests/test_render_regions.py`, `tests/test_ocr_render_metadata.py`, `tests/test_source_size.py`.

- [ ] Add a regression in which an OCR size hint only fits the tight box after a severe reduction, while a larger valid box supports a more readable size; assert the wider attempt is selected before the hint is reduced below the allowed share.
- [ ] Add cases for explicit size, automatic size, minimum readable size, long words, multiline text, stroke, bold, vertical alignment and boxes crossing slice edges; assert no pixels leave the allowed area and failed attempts do not mutate the image.
- [ ] Trace the chosen layout and drawn glyph bounds, including the same wrap algorithm during fitting and drawing; fix only behavior demonstrated by regressions.
- [ ] Retain the caller-image mutation contract for edge renders and test multiple objects in both drawing orders.
- [ ] Run `python -m pytest tests/test_render_regions.py tests/test_ocr_render_metadata.py tests/test_source_size.py tests/test_ai_seams.py -q`.

### 4. Translation ordering, budgeting and honest completion

**Files:** `app/ai_mode/job.py`, `app/translation/context.py`, `app/translation/vision.py`, `app/series_memory.py`, `app/routers/export.py`, relevant AI and series-memory tests.

- [ ] Simulate slice 1 completing after slice 2; assert the chosen ordering policy never lets later text overwrite earlier context and clearly documents any throughput tradeoff.
- [ ] Test budget exhaustion and a failed QC sheet: record which slices and checks ran, return partial/review status, and never claim that unchecked work passed.
- [ ] Test multiple simultaneous calls near budget and distinguish estimated, unknown and provider-reported cost; ensure no unknown value is reported as zero.
- [ ] Test the same series in two target languages, including existing saved series memory; agree a backwards-compatible namespace before changing persisted data.
- [ ] Test unboxed detections and retry exhaustion without automatic removal of original story content; ensure untranslated content is visible in review and normal export preflight.
- [ ] Run `python -m pytest tests/test_ai_mode.py tests/test_ai_verify.py tests/test_ai_render_restore.py tests/test_vision_translation.py tests/test_series_memory.py tests/test_export_gate.py -q`.

### 5. Gateway, local API and network boundaries

**Files:** `gateway/store.py`, `gateway/app.py`, `app/main.py`, `app/downloader/generic_js.py`, `app/downloader/http.py`, `app/agent/skill_install.py`, corresponding gateway and security tests.

- [ ] Reproduce a request finishing after job close: create a funded job, begin a request, finish the job, then charge the request; assert the debit appears exactly once in the ledger and returned status reflects actual settlement.
- [ ] Reproduce two `Store` instances sharing a database while a request is reserved; assert creating the second instance does not release the first instance's hold. Specify crash recovery and multi-worker ownership before changing the schema.
- [ ] Test finish/expiry concurrent with requests, duplicate callbacks, underestimated upstream usage and process restart; reconcile holds, balances and ledger without losing historical transactions.
- [ ] Define the local API trust boundary and test cross-origin browser mutations, direct local clients and extension calls; keep legitimate desktop and extension flows working.
- [ ] Test browser navigation, redirects, subresources and alternate channels against private network and DNS rebinding in an isolated test network; do not assume the HTTP downloader's connection pinning also protects Chromium.
- [ ] Test compressed skill archives with oversized `SKILL.md`, oversized extracted files, many entries, duplicate names and traversal; enforce compressed and decompressed limits before allocation or writes.
- [ ] Run `python -m pytest tests/test_gateway_auth_billing.py tests/test_gateway_upstream_retry.py tests/test_host_guard.py tests/test_download_pinning.py tests/test_upload_archive.py app/agent/tests -q` plus new focused regressions.

### 6. Full chapter acceptance

**Files:** existing CI workflows and test fixtures; audit output must go to the approved evidence location, not this code branch.

- [ ] Re-run both Blade of the Discarded chapters through GitHub Actions with approved credentials and model configuration; retain per-stage status, manifests, masks, costs and artifacts on the evidence branch.
- [ ] Compare every exported slice to the source for missing or duplicated pages and text; inspect especially chapter 2 slices 5, 9 and 15 and chapter 1 slice 16.
- [ ] Review OCR/translation completeness, reading order, names, watermarks, art damage, text clipping, seams and correct target language; an automated score alone is not acceptance.
- [ ] Confirm export rejects unresolved text, stale artifacts and invalid stitch coverage; exercise explicit force export and document its warning behavior.
- [ ] Run the full available suite; list skipped or unavailable integration checks and never label the chapter production-ready until real visual review passes.
