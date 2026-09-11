const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const source = fs.readFileSync('app/static/js/editor.js', 'utf8');
const match = source.match(/function _sourceBoxSet[\s\S]*?(?=async function ensureAutoTextObjects)/);
assert(match, 'editor auto-sync helpers must be present');

const context = {};
vm.runInNewContext(`${match[0]}\nglobalThis.helpers = { _pageNeedsAutoSync, _autoSyncChangedPage };`, context);
const { _pageNeedsAutoSync, _autoSyncChangedPage } = context.helpers;
const box = { id: 'box-1', x1: 10, y1: 12, x2: 60, y2: 44, ocr_text: 'old OCR' };

assert.strictEqual(_pageNeedsAutoSync({ boxes: [{ ...box, ocr_eligible: false }], text_objects: [] }), false,
  'server-ineligible boxes must not trigger repeated ensure calls');
assert.strictEqual(_pageNeedsAutoSync({ boxes: [{ ...box, x2: 10 }], text_objects: [] }), false,
  'invalid geometry must not trigger repeated ensure calls');
assert.strictEqual(_pageNeedsAutoSync({
  boxes: [box],
  text_objects: [{ auto_generated: true, source_boxes: ['box-1'], ocr_text: '', auto_ocr_text: 'old OCR', region: { x1: 10, y1: 12, x2: 60, y2: 44 }, auto_geometry: { x1: 10, y1: 12, x2: 60, y2: 44 } }],
}), false, 'a manually cleared OCR value must remain user-owned');
assert.strictEqual(_pageNeedsAutoSync({
  boxes: [{ ...box, ocr_text: 'new OCR' }],
  text_objects: [{ auto_generated: true, source_boxes: ['box-1'], ocr_text: 'old OCR', auto_ocr_text: 'old OCR', region: { x1: 10, y1: 12, x2: 60, y2: 44 }, auto_geometry: { x1: 10, y1: 12, x2: 60, y2: 44 } }],
}), true, 'a stale machine-owned OCR value must still synchronize');

assert.strictEqual(_autoSyncChangedPage({ auto_text_objects: { changed_pages: [2] } }, 2), true);
assert.strictEqual(_autoSyncChangedPage({ auto_text_objects: { changed_pages: [] } }, 2), false,
  'an unchanged ensure response must not schedule another editor render');

console.log('editor auto-sync sanity: PASS');
