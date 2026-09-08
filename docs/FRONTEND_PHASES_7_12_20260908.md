# Frontend completion phases 7-12 — 2026-09-08

Base for this implementation: `frontend/phases-1-6` at `2fe4efa98b75da821b50e8a61f96291aa887011c`, itself based on `main` `8f8faa9be7fb6133510b1ac06415e5dacabd72ae`.

This document records the implementation and validation scope for steps 7-12 of the September 8 frontend completion plan. It does not claim detector/inpaint accuracy changes; canonical image coordinates and backend processing contracts are unchanged.

## Phase 7 — responsive policy

- One explicit layout policy now owns the final panel state:
  - `wide` >=1280px: page navigator + inspector visible by default;
  - `medium` 1000-1279px: navigator collapsed by default, inspector retained;
  - `drawer` 760-999px: canvas remains interactive and one non-modal auxiliary drawer can be open;
  - `compact` <760px: canvas + one bottom sheet for pages or tools.
- `ResizeObserver` measures the actual application header and writes `--app-header-height`; sticky command bars and drawers no longer rely on a fixed 48px top offset after the header wraps.
- At wide widths the navigator is 176px and inspector 292px. The 1366px browser gate requires at least 800px for the canvas column.
- Panel grid column animation is intentionally removed. This avoids repeatedly rebuilding overlays during a 200ms width transition and honors the plan's requirement to preserve geometry during panel changes.
- Escape closes a compact/drawer panel and returns focus to its toggle. These drawers are intentionally non-modal; the canvas remains interactive and no modal semantics/focus trap are applied.

## Phase 8 — mask snapshot and encoding

- `frontendReleaseEncodeCanvas()` prefers an OffscreenCanvas worker path using a transferred `ImageBitmap` and `convertToBlob({type:'image/png'})`.
- The worker closes transferred bitmaps after drawing. The main thread bounds simultaneous worker encodes; asynchronous `HTMLCanvasElement.toBlob()` is the first fallback. Synchronous base64 encoding remains only as the compatibility fallback when both async paths are unavailable.
- Existing Review and stitched-review code can still restore its saved snapshot representation. Before those legacy capture functions run, the completion layer asynchronously encodes the mask and temporarily supplies an object URL through the old snapshot call. This bypasses synchronous base64 encoding during the normal page/source-page switch path without changing the old restore contract.
- Object URLs are replaced/revoked and the worker is terminated on page lifecycle disposal.
- Browser validation decodes the PNG and compares a sampled lossless checksum against the source RGBA canvas. Dimensions and canonical mask boundaries remain unchanged.

## Phase 9 — persistent shell/navigator and lifecycle

- `page-navigator.js` now caches navigators per chapter/context, reconfigures callbacks and items, and reuses the same DOM element after stage page renders detach it.
- Active-page changes keep direct references to the previous/current button rather than scanning every navigator item. A 5,000-item synthetic regression therefore keeps active-state writes bounded to at most two item references instead of N item scans.
- The navigator exposes `dispose()`, `setOnSelect()`, and chapter-level disposal. Thumbnail `IntersectionObserver` state is disconnected when disposed.
- The completion renderer wrapper preserves the existing stage grid identity, moves only the newly rendered children into it, restores panel ownership, and preserves navigator focus/scroll context where applicable.
- No list windowing is added. The implemented direct-reference active update plus retained navigator DOM removes the measured/identified O(N) active scan; browser evidence is used as the gate before introducing extra virtualization complexity.

## Phase 10 — bounded undo, deletion recovery, and draft restoration

- Editor text/style/geometry history is bounded to 60 records per recent chapter.
- Review brush history is separate and bounded to six PNG snapshots / 32 MiB per chapter. It only represents local draft marks; it never claims to roll back committed inpaint output.
- `Ctrl/Cmd+Z` and `Ctrl/Cmd+Shift+Z` operate on the current editor/review history. Visible Undo/Redo controls are also inserted for discoverability.
- Deletion recovery snapshots the object before a successful delete response. Undo recreates the object and reapplies text/style/geometry through the supported create/update APIs. Because the backend has no undelete-with-same-ID contract, a recovered object may receive a new ID. Redo deletes that recovered object without recursively adding another history item.
- Unsaved editor object state is stored by chapter under `mt_editor_draft:<chapter_id>` and capped to 80 changed objects. On chapter resume it is reapplied before editor rendering and then persisted through the normal save coordinator.
- The draft store is cleared once the editor reports saved and no text/geometry save is still in flight.
- `beforeunload` is armed only while editor save state, local draft storage, slice mask drafts, or stitched mask drafts indicate genuinely unsaved work.

## Phase 11 — ownership cleanup

- `frontend-release.css` is the final owner for responsive panel widths, measured sticky offsets, drawer/sheet layout, canvas overflow containment, and completion-history controls.
- `frontend-release.js` is the final owner for responsive panel state after the older shell helpers run. Existing shell APIs remain compatible, but every shell setup/sync ends by applying the completion policy.
- The completion layer owns its `ResizeObserver`, mutation observer, mask object URLs, worker, history caches, and pagehide disposal.
- Existing hidden-state, focus-visible, tokens, SVG icon, reduced-motion, and thumbnail-lazy-loading behavior remains in place.
- No arbitrary CSS-file reduction was attempted; unused legacy rules are not deleted without a separate usage proof.

## Phase 12 — release gate

Release Gate now includes:

1. existing Python compile/source/inpaint/browser safety checks;
2. JavaScript syntax checks;
3. the Phase 1-6 persistence/navigation regression;
4. `scripts/frontend_release_sanity.js` for navigator lifecycle, responsive ownership, history, encoding, and load-order contracts;
5. a headless Chromium gate at 390, 768, 1024, 1280, 1366, and 1920 CSS pixels;
6. canonical geometry round-trip assertions at every target width;
7. a 1366px minimum canvas-width assertion;
8. PNG mask encode/decode pixel-equivalence benchmark;
9. before/after layout screenshots and a JSON benchmark report uploaded as the `frontend-browser-evidence-*` artifact.

The browser harness uses a cold Chromium profile per run and the repository's real `app.css`, completion CSS, completion JavaScript, and worker. It is a deterministic frontend fixture rather than a live backend chapter, so it measures layout/encoding/frontend behavior separately from detector/inpaint server latency.

### Measured results

This section is filled from the passing GitHub Actions browser artifact before merge to `main`. A merge is blocked if the browser gate, Release Gate, or backend foundation gate fails.

## Remaining backend contract limitation

The repository still has no supported persisted reviewer-approval/verification contract. The frontend therefore continues to use `Chưa duyệt` rather than inventing a durable `Đã xác minh` state. This is a documented backend dependency, not a frontend release blocker because no false verification state is shown.
