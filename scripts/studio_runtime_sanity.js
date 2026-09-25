const fs = require('fs');
const assert = require('assert');
const html = fs.readFileSync('app/templates/index.html', 'utf8');
const css = fs.readFileSync('app/static/css/app.css', 'utf8');
const studio = fs.readFileSync('app/static/css/studio.css', 'utf8');
const shell = fs.readFileSync('app/static/js/ui-shell.js', 'utf8');
const editor = fs.readFileSync('app/static/js/editor.js', 'utf8');
const editorInspector = fs.readFileSync('app/static/js/editor-inspector.js', 'utf8');
const editorPersistence = fs.readFileSync('app/static/js/editor-persistence.js', 'utf8');
const transforms = fs.readFileSync('app/static/js/editor-box-transform.js', 'utf8');
const reviewWorkspace = fs.readFileSync('app/static/js/review-workspace.js', 'utf8');
const stitchInspector = fs.readdirSync('app/static/js/review-stitch').sort().map((name) => fs.readFileSync(`app/static/js/review-stitch/${name}`, 'utf8')).join('\n');
const review = fs.readFileSync('app/static/js/review.js', 'utf8');
const preview = fs.readFileSync('app/static/js/preview.js', 'utf8');
const theme = fs.readFileSync('app/static/js/theme.js', 'utf8');
const chapterOcr = fs.readFileSync('app/static/js/chapter-ocr.js', 'utf8');
const chapterQc = fs.readFileSync('app/static/js/chapter-qc.js', 'utf8');
const visionTranslation = fs.readFileSync('app/translation/vision.py', 'utf8');
const translationRouter = fs.readFileSync('app/routers/translation.py', 'utf8');
assert(css.includes('./studio.css'), 'studio stylesheet must be the runtime surface');
for (const legacy of ['frontend-release.css', 'frontend-coordinator.js', 'frontend-release.js', 'frontend-release-encoder-guard.js']) assert(!html.includes(legacy), `${legacy} must not be loaded at runtime`);
assert(!shell.includes('wrapRenderer("renderEditor"'), 'Editor must not be shell-wrapped');
assert(!shell.includes('function wrapRenderer'), 'stage renderers must own their lifecycle without wrapper chains');
assert(!editor.includes('cancelTextObjectPersist();\n  if (typeof window.cancelGeomPersist'), 'Editor render must not cancel pending edits');
assert(studio.includes('.translation-workspace-body'), 'studio must own editor workspace layout');
assert(studio.includes('.page-navigator-item.active'), 'studio must own navigator active state');
assert(editor.includes('window.setAppStage?.("editor")'), 'editor must own its stage lifecycle');
assert(editor.includes('void flushAllPendingPersists().catch'), 'page switches must persist without blocking navigation');
assert(editor.includes('window.installEditorBoxTransforms?.(imgWrap)'), 'editor must explicitly install overlay transforms');
assert(!transforms.includes('new MutationObserver'), 'box transforms must not keep a document-wide mutation observer');
assert(!shell.includes('event.key === "Tab" && !event.ctrlKey'), 'shell must preserve native Tab navigation');
assert(shell.includes('event.key === "f" || event.key === "F"'), 'focus mode must use the dedicated F shortcut');
assert(reviewWorkspace.includes('window.cleanupReviewWorkspace = () =>'), 'Review must expose one lifecycle cleanup');
assert(reviewWorkspace.includes('review-canvas-only'), 'Review must mount the single-document canvas layout');
assert(stitchInspector.includes('new AbortController()'), 'stitched Review interactions must have abortable ownership');
assert(
  stitchInspector.includes('window.addEventListener("keydown", (e) => {')
    && /\}, \{ signal \}\);\n\s*window\.addEventListener\("keyup"/.test(stitchInspector),
  'stitched Review key listener must be abortable'
);
assert(
  stitchInspector.includes('function chunkHasPaint(')
    && stitchInspector.includes('512 / Math.max(width, h)'),
  'stitched mask scans must use a bounded probe canvas'
);
assert(!stitchInspector.includes('subCtx.getImageData(0, 0, w, subH)'), 'stitched slice detection must not scan full-resolution pixel buffers');
assert(!editor.includes('function renderEditorPanel'), 'editor core must not retain inspector rendering');
assert(editorInspector.includes('function renderEditorPanel'), 'editor inspector must own panel rendering');
assert(!editor.includes('let _textPersistChain = Promise.resolve()'), 'editor core must not retain persistence state');
assert(editorPersistence.includes('let _textPersistChain = Promise.resolve()'), 'text saves must preserve request order');
const coreIndex = html.indexOf('/static/js/editor.js');
const inspectorIndex = html.indexOf('/static/js/editor-inspector.js');
const persistenceIndex = html.indexOf('/static/js/editor-persistence.js');
const transformIndex = html.indexOf('/static/js/editor-box-transform.js');
assert(coreIndex < inspectorIndex && inspectorIndex < persistenceIndex && persistenceIndex < transformIndex, 'editor modules must load in ownership order');
assert(editor.includes('imgWrap.addEventListener("pointerdown"'), 'editor drawing must support pointer and touch input');
assert(shell.includes('window.createAIProviderSettings?.()'), 'AI provider settings must mount before Review is opened');
assert(!review.includes('  refreshSrcData();\n\n  img.addEventListener("load"'), 'Review must not decode full source pixels on every page mount');
assert(!review.includes('setupGeminiQCSettings'), 'legacy Gemini-only settings path must be removed');
assert(!reviewWorkspace.includes('mountGeminiSettings'), 'multi-provider settings must not retain a Gemini-only mount path');
assert(reviewWorkspace.includes('ai-custom-provider-form'), 'Settings must expose a custom provider form');
assert(reviewWorkspace.includes('Mã này đã dành riêng'), 'custom provider form must reject built-in provider IDs');
assert(reviewWorkspace.includes('provider_api_base'), 'custom provider settings must save its OpenAI-compatible API root');
assert(reviewWorkspace.includes('remove_config=true'), 'custom provider settings must support removing its configuration');
assert(reviewWorkspace.includes('syncAIProviderSelects'), 'custom providers must be synchronized into feature selectors');
assert(editor.includes('syncAIProviderSelects'), 'vision translation selector must include configured custom providers');
assert(chapterQc.includes('syncAIProviderSelects'), 'visual QC selector must include configured custom providers');
assert(visionTranslation.includes('{"type": "image_url"'), 'vision translation must send image inputs to OpenAI-compatible providers');
assert(translationRouter.includes('def _resolve_translation_provider'), 'translation API must resolve configured custom providers');
assert(!review.includes('/api/visual_qc/key'), 'Review must use provider-scoped credential endpoints only');
assert(preview.includes('drawLayer.addEventListener("pointerdown"'), 'preview exclusion drawing must support touch input');
assert(preview.includes('window.setAppStage?.("preview")'), 'preview must own its stage lifecycle');
assert(reviewWorkspace.includes('window.setAppStage?.("review")'), 'Review must own its stage lifecycle');
assert(!preview.includes('window.addEventListener("mousemove", onMouseMove)'), 'preview must not use mouse-only global drawing');
assert(html.includes('class="app-sidebar"'), 'canonical sidebar shell must be present');
assert(html.includes('id="home-view"') && html.includes('id="import-view"'), 'Home and Import must be separate views');
assert(html.includes('id="theme-select"'), 'theme selector must be visible in the application header');
assert(theme.includes('window.setAppTheme'), 'theme controller must expose one canonical setter');
assert(shell.includes('let shellMounted = false'), 'shell listeners must have an idempotent mount guard');
assert(!html.includes('workbench-topbar'), 'legacy horizontal topbar must be removed');
assert(!html.includes('workbench-stage-link'), 'legacy horizontal workflow links must be removed');
assert(!chapterOcr.includes('observeWorkspaceRoot'), 'OCR must not scan the whole workspace with an observer');
assert(!chapterQc.includes('observeWorkspaceRoot'), 'chapter QC must not scan the whole workspace with an observer');
assert(!chapterQc.includes('new MutationObserver'), 'chapter QC must use explicit lifecycle synchronization');
assert(chapterQc.includes('window.syncChapterQCWorkspace = renderPanel'), 'chapter QC must expose explicit workspace synchronization');
assert((reviewWorkspace.match(/new MutationObserver/g) || []).length === 0, 'Review canvas must not keep DOM observers');
assert(!reviewWorkspace.includes('busyObserver'), 'Review busy state must not infer lifecycle from DOM mutations');
assert(!reviewWorkspace.includes('createPageNavigator({'), 'Review must not recreate the old thumbnail page navigator');
assert(!reviewWorkspace.includes('context-inspector review-inspector'), 'Review must not recreate the old right-hand inspector');
assert(!reviewWorkspace.includes('review-sticky-toolbar'), 'Review must not create a second outer toolbar');
assert(stitchInspector.includes('review-document-toolbar-compact'), 'Review must use one compact document toolbar');
assert(stitchInspector.includes('review-floating-inspector'), 'Review text properties must use an on-demand floating inspector');
const reviewTestWindow = { __currentChapterId: "test-chapter" };
reviewTestWindow.currentChapterId = reviewTestWindow.__currentChapterId;
globalThis.window = reviewTestWindow;
globalThis.document = { querySelectorAll: () => [] };
(async () => {
  const overlays = await import(require('url').pathToFileURL(require('path').resolve('app/static/js/review-stitch/overlays.js')).href);
  reviewTestWindow.reviewPageIndexAtSourceY = overlays.reviewPageIndexAtSourceY;
  reviewTestWindow.__testEnsureObjects = overlays.ensureObjects;
  assert.strictEqual(typeof reviewTestWindow.reviewPageIndexAtSourceY, 'function', 'Review must expose its source-Y page mapping for navigation');
  assert.strictEqual(
    reviewTestWindow.reviewPageIndexAtSourceY([
      { item: { canonicalIndex: 0 }, sourceY1: 0, sourceY2: 1600 },
      { item: { canonicalIndex: 1 }, sourceY1: 1600, sourceY2: 3200 },
      { item: { canonicalIndex: 2 }, sourceY1: 3200, sourceY2: 4800 },
    ], 2400),
    1,
    'scrolling to the center of the second page must select that page for stage navigation'
  );
  let attempts = 0;
  reviewTestWindow.ensureAutoTextObjects = async () => {
    attempts += 1;
    if (attempts === 1) throw new Error('transient OCR failure');
  };
  const testShell = { _descriptors: [{ item: { canonicalIndex: 0 } }] };
  await reviewTestWindow.__testEnsureObjects(testShell);
  await reviewTestWindow.__testEnsureObjects(testShell);
  assert.strictEqual(attempts, 2, 'failed automatic text-object sync must retry when Review renders again');
  console.log('studio runtime sanity: PASS');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
