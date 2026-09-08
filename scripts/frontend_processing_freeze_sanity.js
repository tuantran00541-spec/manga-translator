const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const guardPath = 'app/static/js/frontend-observer-guard.js';
const guardSource = fs.readFileSync(guardPath, 'utf8');
const apiSource = fs.readFileSync('app/static/js/api.js', 'utf8');
const mainSource = fs.readFileSync('app/static/js/main.js', 'utf8');
const dependencies = fs.readFileSync('app/dependencies.py', 'utf8');
const optimized = fs.readFileSync('app/optimized_pipeline.py', 'utf8');
const index = fs.readFileSync('app/templates/index.html', 'utf8');

const observations = [];
class NativeMutationObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe(target, options) {
    observations.push({ target, options });
  }
  disconnect() {}
}

global.window = { MutationObserver: NativeMutationObserver };
vm.runInThisContext(guardSource, { filename: guardPath });

const GuardedMutationObserver = window.MutationObserver;
assert.notStrictEqual(
  GuardedMutationObserver,
  NativeMutationObserver,
  'page-view observer guard must wrap the native observer',
);

const pageView = { id: 'page-view' };
const nested = { id: 'review-card' };
const pageObserver = new GuardedMutationObserver(() => {});
pageObserver.observe(pageView, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['class', 'hidden'],
});

assert.strictEqual(observations.length, 1);
assert.strictEqual(observations[0].target, pageView);
assert.strictEqual(observations[0].options.childList, true);
assert.strictEqual(observations[0].options.subtree, false, 'page-view must not observe deep progress/text churn');
assert.strictEqual(observations[0].options.attributes, false, 'page-view must not react to deep class/hidden churn');
assert.strictEqual(observations[0].options.characterData, false);
assert.strictEqual('attributeFilter' in observations[0].options, false);

const otherOptions = { attributes: true, attributeFilter: ['class'] };
const otherObserver = new GuardedMutationObserver(() => {});
otherObserver.observe(nested, otherOptions);
assert.strictEqual(observations.length, 2);
assert.strictEqual(observations[1].target, nested);
assert.strictEqual(observations[1].options, otherOptions, 'non-page-view observers must remain untouched');

assert(
  apiSource.includes('const PROCESS_BATCH_SIZE = 16;'),
  'processing throughput contract regressed: batch size must remain 16',
);
assert(
  !mainSource.includes('RESPONSIVE_PROCESS_BATCH_SIZE')
    && !mainSource.includes('responsiveProcessSelectedPagesOnce'),
  'startup must not override the proven 16-page processing path',
);
assert(
  !dependencies.includes('runtime_responsiveness')
    && !optimized.includes('responsive_process_workers'),
  'processing worker/ORT throttling must not be reintroduced',
);
assert(
  mainSource.includes('PROCESS_PROGRESS_POLL_MS = 1000')
    && mainSource.includes('process_revision')
    && mainSource.includes('/api/chapter/${encodeURIComponent(chapterId)}')
    && mainSource.includes('Đang xử lý ${completed}/${indices.length}'),
  'preview must surface durable per-page progress while a 16-page request is still running',
);
assert(
  !mainSource.includes('window.fetch =')
    && !mainSource.includes('window.processSelectedPages ='),
  'progress reporting must observe the existing process path instead of monkey-patching fetch or duplicating processing',
);

const guardTag = '/static/js/frontend-observer-guard.js';
const coordinatorTag = '/static/js/frontend-coordinator.js';
const releaseTag = '/static/js/frontend-release.js';
assert(index.includes(guardTag), 'observer guard must be loaded');
assert(
  index.indexOf(guardTag) < index.indexOf(coordinatorTag)
    && index.indexOf(guardTag) < index.indexOf(releaseTag),
  'observer guard must load before both page-view observer owners',
);

console.log('frontend processing freeze sanity: PASS');