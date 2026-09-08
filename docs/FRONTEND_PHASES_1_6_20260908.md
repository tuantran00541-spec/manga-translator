# Frontend phases 1–6 implementation baseline — 2026-09-08

## Scope

This document records the implementation baseline for steps 1–6 of `frontend-completion-prompt-en.md`. It is intentionally limited to the frontend workflow/persistence/navigation/accessibility/state/action-hierarchy work. Responsive layout optimization, mask encoding benchmarks, navigator lifecycle/windowing, undo/redo, CSS ownership cleanup, and the final release gate remain steps 7–12.

No merge or deployment is authorized by this work.

## Source-of-truth pin

- Repository: `tuantran00541-spec/manga-translator`
- Audit commit: `2d57a058b8d71cc4c5a902fb09c45140d7fe7c59`
- Revalidated `main` HEAD before implementation: `8f8faa9be7fb6133510b1ac06415e5dacabd72ae`
- `main` was 55 commits ahead of the audit commit.
- The intervening frontend source delta was limited to small CSS changes in `tokens.css` and `workbench.css`; the audited JavaScript control-flow paths remained present.
- The `main` Release Gate for the revalidated HEAD was green before the feature branch was created.
- Implementation branch: `frontend/phases-1-6`

## Revalidation of findings

| Finding | Revalidated status | Phase 1–6 handling |
| --- | --- | --- |
| F01 save barrier/in-flight writes | Retained | Added object-aware write tracking, ordered revisions, and `awaitSaved`. |
| F02 page switch after failed/timeout save | Retained | Added guarded page switching with no timeout-as-success path. |
| F03 top-level navigation bypasses review drafts | Retained | Added a shared exit guard for stage links, Home, editor exits, slice masks, and stitched marks. |
| F04 Tab/Shift+Tab hijacked by focus mode | Retained | Restored native sequential Tab behavior outside the modal Settings dialog; focus mode remains an explicit button. |
| F05 unflagged page displayed as verified | Retained | Unflagged review items are normalized to `Chưa duyệt`; no reviewer approval is invented. |
| F06 responsive width/header policy | Retained | Deferred to step 7 as specified. |
| F07 mouse-only preview/editor creation | Retained | Added Pointer Event paths for touch/pen with capture, cancellation, and drawing-only `touch-action`. Existing mouse behavior remains compatible. |
| F08 synchronous mask encoding | Retained | Deferred to step 8 benchmark/optimization. |
| F09 stale chapter activation | Partially resolved before this branch | URL/resume already had independent generation guards. This branch centralizes URL/resume/upload under one activation generation so upload cannot win late. |
| F10 label-derived operation ownership | Retained | Added explicit operation state keyed by scope and network operation; review locks also consume explicit state. The pre-existing review observer remains a compatibility fallback until its source-level cleanup is safe to land. |
| F11 competing primary actions/terminology | Retained | Chapter export remains primary; current-page render is presented as `Xem trước trang`. Chapter context prefers a meaningful name and keeps the technical ID in a tooltip. |
| F12 Settings unavailable before Review | Retained | AI settings are mounted independently of Review and provider status is loaded from Settings. |
| F13 navigator/shell churn | Retained | Deferred to step 9. |
| F14 overlay rebuild during resize | Retained | Deferred to step 9 profiling. |
| F15 undo/recovery model | Retained | Deferred to step 10. |

## Current contracts used by phases 1–6

The implementation does not add or assume new backend APIs. It coordinates existing contracts:

- Open URL chapter: `POST /api/chapter`
- Resume chapter: `GET /api/chapter/{chapter_id}`
- Upload chapter: `POST /api/chapter/upload`
- Persist text/style/geometry object updates: `POST /api/text_object/update`
- Workflow checkpoint: `POST /api/workflow_checkpoint`
- Manual repaint: `POST /api/repaint_mask`
- AI provider status/settings: `GET /api/visual_qc/settings`
- Existing OCR, translation, processing, render and export routes retain their current ownership and are observed only to expose explicit frontend operation state.

### Identifier rules

- `chapter_id` is the chapter ownership key.
- `page_index` is the canonical processed-page/slice index sent to backend routes.
- `source_page` identifies the original source page.
- `slice_index` identifies a slice within a split source page.
- UI labels should prefer source-page/slice vocabulary while backend calls continue using canonical `page_index`.

## Draft and operation ownership

- Text/style draft ownership: editor text-object persistence.
- Geometry draft ownership: `editor-box-transform.js` geometry persistence.
- Review slice-mask ownership: active review brush canvas plus preserved per-page draft snapshots.
- Stitched-mask ownership: stitched review inspector.
- Chapter activation ownership: shared frontend activation generation covering URL load, upload and resume.
- Operation ownership: explicit frontend operation registry with scope such as `save`, `review`, `preview`, and `editor`.

## Phase implementation summary

### Phase 1 — baseline/contracts/fixtures

Revalidated `main` against the audit commit and added `scripts/fixtures/frontend_phase1_6_manifest.json` with processed, stylized, skipped, split-source and warning cases plus a 120-item long-chapter target.

### Phase 2 — reliable persistence barrier

`frontend-coordinator.js` tracks each text-object update by chapter/page/object. Requests for the same object are serialized by revision so an older request cannot finish after a newer request has already been sent and become authoritative. `awaitSaved` waits for legacy dirty flushes, tracked in-flight requests and geometry persistence, and surfaces the most recent failed revision.

### Phase 3 — guarded navigation and chapter activation

Page switching now waits on the save barrier and remains on the current page if persistence fails. Top-level exits share review-draft/editor-save checks. URL, upload and resume use one request generation so reordered responses cannot replace the most recently requested chapter.

### Phase 4 — keyboard and pointer input

Normal Tab and Shift+Tab navigation is restored. The Settings modal keeps its existing focus containment. Touch/pen creation paths use Pointer Events, pointer capture and `pointercancel`; native scrolling remains available while drawing mode is inactive.

### Phase 5 — explicit state and review semantics

Network-backed operations publish explicit scoped state. Review controls consume that state in addition to card-owned repaint state. An empty issue list is no longer rendered as human verification; it is `Chưa duyệt` until a supported approval contract exists.

### Phase 6 — command hierarchy and Settings availability

Chapter export is the primary editor completion action, while current-page render is a secondary preview action. Settings can initialize AI configuration without visiting Review first. Header chapter context favors a user-facing chapter name/source-derived label instead of repeating a raw identifier.

## Regression coverage

`scripts/frontend_coordinator_sanity.js` exercises the coordinator in an isolated Node VM:

1. A second revision for the same text object does not start its network write until the first settles.
2. `awaitSaved` does not resolve while a relevant write is already in flight.
3. Failed persistence remains visible to the barrier.
4. A later successful revision clears the prior failure.
5. Page switching waits for persistence before invoking the existing switch path.
6. Review repaint work registers and clears explicit operation state.

The Release Gate runs this regression after JavaScript syntax checks on pull requests.

## Known backend/validation gaps after phase 6

- No supported persisted reviewer-approval contract was found, so this branch deliberately does not manufacture an `approved/verified` state.
- DeepSeek status is available independently in Settings, while its full key-management controls still use the existing Review-owned implementation; consolidating all provider credential UI without duplication is a follow-up UI cleanup, not a new backend requirement.
- The legacy review busy observer still exists as a compatibility fallback even though explicit operation state is now authoritative for coordinator locks. Removing the fallback source code should be done with browser regression coverage rather than by assuming label changes are harmless.
- Steps 7–12 still require real-browser responsive, geometry, encoding/performance, lifecycle, history/recovery and final end-to-end validation. No performance improvement is claimed by phases 1–6.
