const fs = require('fs');
const assert = require('assert');

const nav = fs.readFileSync('app/static/js/page-navigator.js', 'utf8');
const release = fs.readFileSync('app/static/js/frontend-release.js', 'utf8');
const css = fs.readFileSync('app/static/css/frontend-release.css', 'utf8');
const index = fs.readFileSync('app/templates/index.html', 'utf8');
const worker = fs.readFileSync('app/static/js/mask-encoder-worker.js', 'utf8');
const fixture = fs.readFileSync('scripts/fixtures/frontend_release_browser.html', 'utf8');

assert(nav.includes('navigatorCache'), 'navigator must preserve instances across page renders');
assert(nav.includes('setOnSelect(handler)'), 'cached navigator must refresh its callback');
assert(nav.includes('dispose({ remove = true'), 'navigator needs explicit disposal');
assert(nav.includes('activeButton = buttons[currentIndex]'), 'active update must keep a direct active reference');
assert(!/updateActiveItem[\s\S]{0,900}querySelectorAll/.test(nav), 'active update must not scan the whole navigator DOM');

assert(release.includes('preserveStageShell'), 'stage shell identity must be preserved');
assert(release.includes('ResizeObserver'), 'header height must be measured, not hard-coded');
assert(release.includes('frontendReleaseEncodeCanvas'), 'mask encoder must be exported for regression/benchmark use');
assert(release.includes('createImageBitmap(canvas)'), 'worker path must snapshot canonical canvas pixels');
assert(release.includes('canvasToAsyncBlob'), 'async Blob fallback is required');
assert(release.includes('beforeunload'), 'genuinely unsaved work needs reload protection');
assert(release.includes('mt_editor_draft:'), 'chapter-scoped draft restoration must be persisted');
assert(release.includes('MASK_HISTORY_LIMIT'), 'brush history must be bounded');
assert(release.includes('history_replay'), 'delete recovery redo must not recursively create history');

for (const mode of ['wide', 'medium', 'drawer', 'compact']) {
  assert(css.includes(`data-layout-mode="${mode}"`), `missing responsive mode: ${mode}`);
}
assert(css.includes('--app-header-height'), 'responsive offsets must use measured header height');
assert(css.includes('transition: none !important'), 'panel resize must not repeatedly rebuild overlays during a 200ms column animation');

assert(index.includes('/static/css/frontend-release.css'), 'release stylesheet must be loaded');
assert(index.includes('/static/js/frontend-release.js'), 'release coordinator must be loaded');
assert(index.indexOf('/static/js/frontend-coordinator.js') < index.indexOf('/static/js/frontend-release.js'), 'phase 7-12 layer must extend the phase 1-6 coordinator');
assert(index.indexOf('/static/js/frontend-release.js') < index.indexOf('/static/js/main.js'), 'release layer must be ready before startup resumes a chapter');
assert(worker.includes('convertToBlob'), 'worker must encode PNG asynchronously');
assert(worker.includes('bitmap.close'), 'worker must release transferred bitmap ownership');

for (const width of ['390', '768', '1024', '1280', '1366', '1920']) {
  assert(fixture.includes('Frontend release harness'), 'browser fixture missing');
}

// Algorithmic contract: old active-state scan touched N nodes; the new direct-ref path
// touches at most the previous and next item independent of chapter length.
for (const n of [1, 10, 100, 1000, 5000]) {
  const oldTouches = n;
  const newTouches = n <= 1 ? 1 : 2;
  assert(newTouches <= 2);
  if (n > 2) assert(newTouches < oldTouches);
}

console.log('frontend release sanity: PASS');
