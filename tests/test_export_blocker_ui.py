import shutil
import subprocess

import pytest


def test_blocked_export_lists_actionable_slices_without_forcing_download():
    if not shutil.which("node"):
        pytest.skip("node is unavailable")
    script = r'''
const fs = require('fs');
const source = fs.readFileSync('app/static/js/editor.js', 'utf8');
const start = source.indexOf('function buildChapterExportButton() {');
const end = source.indexOf('window.buildChapterExportButton =', start);
class El {
  constructor(tag) { this.tagName = tag; this.children = []; this.dataset = {}; this.events = {}; this.hidden = false; }
  append(...els) { this.children.push(...els); }
  appendChild(el) { this.children.push(el); return el; }
  replaceChildren(...els) { this.children = els; }
  addEventListener(name, fn) { this.events[name] = fn; }
  setAttribute() {}
}
global.document = {createElement: tag => new El(tag), body: new El('body')};
global.window = {currentChapterId: 'chapter', currentManifest: {pages: [{}, {}, {}, {}, {}]}, editorState: {}, renderReview: () => { window.reviewOpened = true; }};
global.buildExportOptions = () => new El('details');
global.exportHref = () => 'unused';
let calls = [];
global.fetch = async url => { calls.push(url); return {status: 409, ok: false, json: async () => ({detail: {
  code: 'editorial_preflight', message: 'Còn lỗi.', blockers: [{page_index: 4, kind: 'unresolved_ai_region', region: {x1: 10, y1: 20, x2: 80, y2: 90}}]
}})}; };
const build = new Function(source.slice(start, end) + '; return buildChapterExportButton;')();
const wrap = build();
const button = wrap.children.find(e => e.tagName === 'button');
(async () => {
  await button.events.click();
  const text = e => [e.textContent || '', ...e.children.flatMap(text)];
  const contents = text(wrap).join(' ');
  if (calls.length !== 1 || calls[0].includes('force=true')) throw Error('must not force export');
  if (!contents.includes('5') || !contents.includes('10') || !contents.includes('20')) throw Error('missing slice or region');
  if (!text(wrap).some(t => /xem|kiểm tra/i.test(t))) throw Error('missing review action');
  const buttons = e => [...(e.tagName === 'button' ? [e] : []), ...e.children.flatMap(buttons)];
  const review = buttons(wrap).find(e => /Xem lát 5/.test(e.textContent || ''));
  review.events.click();
  if (window.editorState.activePageIndex !== 4 || window.initialReviewCanonicalPageIndex !== 4 || !window.reviewOpened) throw Error('review did not open slice');
})().catch(e => { console.error(e); process.exit(1); });
'''
    subprocess.run(["node", "-e", script], check=True)


def test_explicit_review_destination_does_not_restore_previous_scroll():
    from pathlib import Path
    source = Path("app/static/js/review-workspace.js").read_text(encoding="utf-8")
    assert 'previousState?.chapterId === chapterId && !Number.isInteger(requestedPageIndex)' in source
