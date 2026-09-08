const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function classList() {
  const set = new Set();
  return {
    add: (...xs) => xs.forEach((x) => set.add(x)),
    remove: (...xs) => xs.forEach((x) => set.delete(x)),
    toggle: (x, force) => {
      if (force === undefined) force = !set.has(x);
      if (force) set.add(x); else set.delete(x);
      return force;
    },
    contains: (x) => set.has(x),
  };
}

const listeners = new Map();
global.window = global;
window.location = { href: 'http://localhost/', hash: '' };
window.history = { replaceState() {} };
window.addEventListener = (type, fn) => {
  const list = listeners.get(type) || [];
  list.push(fn);
  listeners.set(type, list);
};
window.removeEventListener = () => {};
window.dispatchEvent = () => true;
global.CustomEvent = class CustomEvent { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
global.MutationObserver = class MutationObserver { observe() {} disconnect() {} };

global.document = {
  body: { dataset: { appStage: 'editor' }, classList: classList() },
  addEventListener(type, fn) {
    const list = listeners.get(`document:${type}`) || [];
    list.push(fn);
    listeners.set(`document:${type}`, list);
  },
  removeEventListener() {},
  getElementById() { return null; },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  createElement() { return { classList: classList(), style: {}, append() {}, appendChild() {}, setAttribute() {}, addEventListener() {}, querySelector() { return null; } }; },
};

global.queueMicrotask = global.queueMicrotask || ((fn) => Promise.resolve().then(fn));
window.currentChapterId = 'chapter-a';
window.currentManifest = { chapter_id: 'chapter-a', pages: [{}, {}, {}] };
window.editorState = { activePageIndex: 0 };
window.hasPendingGeom = () => false;
window.isGeomSaving = () => false;
window.flushAllPendingPersists = async () => {};
let legacySwitchCalls = [];
window.switchEditorPage = async (index) => { legacySwitchCalls.push(index); window.editorState.activePageIndex = index; };

const pending = [];
let nativeCalls = 0;
window.fetch = (input, init) => {
  nativeCalls += 1;
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  pending.push({ input, init, resolve, reject, promise });
  return promise;
};

const sourcePath = process.argv[2] || 'app/static/js/frontend-coordinator.js';
vm.runInThisContext(fs.readFileSync(sourcePath, 'utf8'), { filename: sourcePath });

const tick = () => new Promise((resolve) => setImmediate(resolve));
const updateInit = (translation) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ chapter_id: 'chapter-a', page_index: 0, id: 'obj-1', translation }),
});

(async () => {
  const p1 = window.fetch('/api/text_object/update', updateInit('one'));
  const p2 = window.fetch('/api/text_object/update', updateInit('two'));
  await tick();
  assert.strictEqual(nativeCalls, 1, 'second revision must wait for first network write');
  pending[0].resolve({ ok: true, status: 200 });
  await tick();
  assert.strictEqual(nativeCalls, 2, 'second revision starts only after first settles');
  pending[1].resolve({ ok: true, status: 200 });
  await Promise.all([p1, p2]);

  const p3 = window.fetch('/api/text_object/update', updateInit('three'));
  await tick();
  let barrierDone = false;
  const barrier = window.awaitSaved(0).then(() => { barrierDone = true; });
  await tick();
  assert.strictEqual(barrierDone, false, 'save barrier must wait for already in-flight writes');
  pending[2].resolve({ ok: true, status: 200 });
  await p3;
  await barrier;
  assert.strictEqual(barrierDone, true);

  const p4 = window.fetch('/api/text_object/update', updateInit('bad'));
  await tick();
  pending[3].resolve({ ok: false, status: 500 });
  await p4;
  let failed = false;
  try { await window.awaitSaved(0); } catch (_) { failed = true; }
  assert.strictEqual(failed, true, 'save barrier must surface latest failed revision');

  const p5 = window.fetch('/api/text_object/update', updateInit('recovery'));
  await tick();
  pending[4].resolve({ ok: true, status: 200 });
  await p5;
  await window.awaitSaved(0);

  const p6 = window.fetch('/api/text_object/update', updateInit('before-switch'));
  await tick();
  const switchPromise = window.switchEditorPage(2);
  await tick();
  assert.deepStrictEqual(legacySwitchCalls, [], 'page switch must not run while save is pending');
  pending[5].resolve({ ok: true, status: 200 });
  await p6;
  await switchPromise;
  assert.deepStrictEqual(legacySwitchCalls, [2], 'page switch runs after save barrier resolves');

  const repaint = window.fetch('/api/repaint_mask', { method: 'POST', body: new FormData() });
  await tick();
  assert.strictEqual(window.frontendOperationState.isBusy({ scope: 'review' }), true, 'review operation must be explicit');
  pending[6].resolve({ ok: true, status: 200 });
  await repaint;
  await tick();
  assert.strictEqual(window.frontendOperationState.isBusy({ scope: 'review' }), false, 'review operation must clear explicitly');

  console.log('frontend coordinator sanity: PASS');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
