(() => {
"use strict";
const HISTORY_LIMIT = 60;
const MASK_HISTORY_LIMIT = 6;
const MASK_HISTORY_BYTES = 32 * 1024 * 1024;
const DRAFT_LIMIT = 80;
const MASK_WORKER_URL = window.MT_MASK_WORKER_URL || "/static/js/mask-encoder-worker.js";
const shellRegistry = new Map();
const panelPrefs = new Map();
const maskObjectUrls = new Map();
const historyByChapter = new Map();
const maskHistoryByChapter = new Map();
const restoredDraftChapters = new Set();
let knownChapterId = null;
let encoderWorker = null;
let encoderSeq = 0;
let encoderActive = 0;
let historyApplying = false;
let headerObserver = null;
let viewObserver = null;
let lastEditorRecord = { key: null, at: 0 };
const encoderPending = new Map();
const encoderMetrics = {
worker: 0,
asyncBlob: 0,
syncFallback: 0,
failures: 0,
lastDurationMs: 0,
lastBytes: 0,
lastWidth: 0,
lastHeight: 0,
};
const toast = (message, type = "info") => window.showToast?.(message, type);
const clone = (value) => value == null ? value : JSON.parse(JSON.stringify(value));
const chapterId = () => String(window.currentChapterId || window.currentManifest?.chapter_id || "");
const stage = () => document.body?.dataset?.appStage || "landing";
const getPath = (input) => {
try { return new URL(typeof input === "string" ? input : input?.url, location.href).pathname; }
catch (_) { return ""; }
};
function layoutMode() {
const width = window.innerWidth || document.documentElement.clientWidth || 1280;
if (width >= 1280) return "wide";
if (width >= 1000) return "medium";
if (width >= 760) return "drawer";
return "compact";
}
function defaultPanels(mode) {
if (mode === "wide") return { nav: true, inspector: true };
if (mode === "medium") return { nav: false, inspector: true };
return { nav: false, inspector: false };
}
function currentGrid() {
return document.querySelector("#page-view .workbench-stage-grid");
}
function updateHeaderHeight() {
const header = document.getElementById("site-header");
const height = Math.max(48, Math.ceil(header?.getBoundingClientRect().height || 48));
document.documentElement.style.setProperty("--app-header-height", `${height}px`);
}
function applyResponsivePanels() {
const mode = layoutMode();
document.body.dataset.layoutMode = mode;
updateHeaderHeight();
const grid = currentGrid();
if (!grid) return;
const nav = grid.querySelector(":scope > .page-navigator");
const inspector = grid.querySelector(":scope > .context-inspector");
if (!nav || !inspector) return;
const key = `${stage()}:${mode}`;
const state = panelPrefs.get(key) || defaultPanels(mode);
panelPrefs.set(key, state);
const focus = document.body.classList.contains("focus-mode");
const navOpen = !focus && Boolean(state.nav);
const inspectorOpen = !focus && Boolean(state.inspector);
nav.hidden = !navOpen;
inspector.hidden = !inspectorOpen;
grid.dataset.navOpen = String(navOpen);
grid.dataset.inspectorOpen = String(inspectorOpen);
grid.dataset.layoutMode = mode;
const pageButton = document.getElementById("toggle-page-panel");
const inspectorButton = document.getElementById("toggle-inspector-panel");
if (pageButton) {
pageButton.setAttribute("aria-controls", nav.id || `${stage()}-page-panel`);
pageButton.setAttribute("aria-expanded", String(navOpen));
}
if (inspectorButton) {
inspectorButton.setAttribute("aria-controls", inspector.id || `${stage()}-inspector-panel`);
inspectorButton.setAttribute("aria-expanded", String(inspectorOpen));
}
}
function toggleReleasePanel(name) {
const mode = layoutMode();
const key = `${stage()}:${mode}`;
const current = { ...(panelPrefs.get(key) || defaultPanels(mode)) };
current[name] = !current[name];
if ((mode === "drawer" || mode === "compact") && current[name]) {
current[name === "nav" ? "inspector" : "nav"] = false;
}
panelPrefs.set(key, current);
if (document.body.classList.contains("focus-mode") && current[name]) {
document.body.classList.remove("focus-mode");
const focus = document.getElementById("toggle-focus-mode");
focus?.setAttribute("aria-pressed", "false");
focus?.classList.remove("active", "ui-btn-primary");
focus?.classList.add("ui-btn-ghost");
}
applyResponsivePanels();
}
document.addEventListener("click", (event) => {
const page = event.target?.closest?.("#toggle-page-panel");
const inspector = event.target?.closest?.("#toggle-inspector-panel");
if (!page && !inspector) return;
event.preventDefault();
event.stopImmediatePropagation();
toggleReleasePanel(page ? "nav" : "inspector");
}, true);
window.addEventListener("keydown", (event) => {
if (event.key !== "Escape" || document.body.classList.contains("settings-open")) return;
const mode = layoutMode();
if (mode !== "drawer" && mode !== "compact") return;
const key = `${stage()}:${mode}`;
const current = panelPrefs.get(key) || defaultPanels(mode);
if (!current.nav && !current.inspector) return;
event.preventDefault();
event.stopImmediatePropagation();
const closing = current.nav ? "nav" : "inspector";
panelPrefs.set(key, { ...current, [closing]: false });
applyResponsivePanels();
document.getElementById(closing === "nav" ? "toggle-page-panel" : "toggle-inspector-panel")?.focus();
}, true);
function workerReady() {
return typeof Worker !== "undefined"
&& typeof createImageBitmap === "function"
&& typeof OffscreenCanvas !== "undefined";
}
function ensureEncoderWorker() {
if (encoderWorker || !workerReady()) return encoderWorker;
try {
encoderWorker = new Worker(MASK_WORKER_URL);
encoderWorker.onmessage = (event) => {
const { id, ok, blob, error } = event.data || {};
const pending = encoderPending.get(id);
if (!pending) return;
encoderPending.delete(id);
if (ok && blob instanceof Blob) pending.resolve(blob);
else pending.reject(new Error(error || "Mask worker encode failed"));
};
encoderWorker.onerror = (event) => {
const error = new Error(event?.message || "Mask worker failed");
for (const pending of encoderPending.values()) pending.reject(error);
encoderPending.clear();
encoderWorker?.terminate();
encoderWorker = null;
};
} catch (_) {
encoderWorker = null;
}
return encoderWorker;
}
function canvasToAsyncBlob(canvas) {
return new Promise((resolve, reject) => {
const native = HTMLCanvasElement.prototype.toBlob;
if (typeof native !== "function") {
reject(new Error("canvas.toBlob unavailable"));
return;
}
native.call(canvas, (blob) => blob ? resolve(blob) : reject(new Error("canvas.toBlob returned null")), "image/png");
});
}
function dataUrlFallback(canvas) {
const dataUrl = HTMLCanvasElement.prototype.toDataURL.call(canvas, "image/png");
const encoded = dataUrl.split(",", 2)[1] || "";
const binary = atob(encoded);
const bytes = new Uint8Array(binary.length);
for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
return new Blob([bytes], { type: "image/png" });
}
async function encodeCanvas(canvas) {
if (!(canvas instanceof HTMLCanvasElement) || !canvas.width || !canvas.height) {
throw new Error("Canvas mask không hợp lệ");
}
const started = performance.now?.() || Date.now();
encoderActive += 1;
try {
let blob = null;
const worker = encoderActive <= 2 ? ensureEncoderWorker() : null;
if (worker) {
let bitmap = null;
try {
bitmap = await createImageBitmap(canvas);
const id = `mask-${Date.now()}-${++encoderSeq}`;
blob = await new Promise((resolve, reject) => {
encoderPending.set(id, { resolve, reject });
worker.postMessage({ id, bitmap, width: canvas.width, height: canvas.height }, [bitmap]);
});
encoderMetrics.worker += 1;
} catch (error) {
try { bitmap?.close?.(); } catch (_) {}
encoderMetrics.failures += 1;
}
}
if (!blob) {
try {
blob = await canvasToAsyncBlob(canvas);
encoderMetrics.asyncBlob += 1;
} catch (_) {
blob = dataUrlFallback(canvas);
encoderMetrics.syncFallback += 1;
}
}
encoderMetrics.lastDurationMs = (performance.now?.() || Date.now()) - started;
encoderMetrics.lastBytes = blob.size;
encoderMetrics.lastWidth = canvas.width;
encoderMetrics.lastHeight = canvas.height;
return blob;
} finally {
encoderActive = Math.max(0, encoderActive - 1);
}
}
window.frontendReleaseEncodeCanvas = encodeCanvas;
window.canvasToBlob = encodeCanvas;
function maskKey(canvas) {
if (!canvas) return "";
const chapter = chapterId();
if (canvas.classList.contains("stitched-brush-canvas")) {
const source = canvas.closest(".review-stitched-shell")?.querySelector(".review-stitched-select")?.value || "unknown";
return `stitched:${chapter}:${source}`;
}
const page = canvas.closest(".review-card")?.dataset?.pageIndex || "unknown";
return `slice:${chapter}:${page}`;
}
function revokeMaskUrl(key) {
const url = maskObjectUrls.get(key);
if (url) URL.revokeObjectURL(url);
maskObjectUrls.delete(key);
}
async function withLegacyMaskSnapshot(canvas, callback) {
if (!canvas?._reviewDirty || !canvas.width || !canvas.height) return callback();
const key = maskKey(canvas);
const previousUrl = maskObjectUrls.get(key) || null;
let url = null;
const own = Object.prototype.hasOwnProperty.call(canvas, "toDataURL");
const previousMethod = canvas.toDataURL;
try {
const blob = await encodeCanvas(canvas);
url = URL.createObjectURL(blob);
maskObjectUrls.set(key, url);
canvas.toDataURL = () => url;
const result = await callback();
if (previousUrl && previousUrl !== url) URL.revokeObjectURL(previousUrl);
return result;
} catch (error) {
encoderMetrics.failures += 1;
console.warn("Async mask snapshot failed; using compatibility path:", error);
if (url) {
URL.revokeObjectURL(url);
if (previousUrl) maskObjectUrls.set(key, previousUrl);
else maskObjectUrls.delete(key);
}
return callback();
} finally {
if (own) canvas.toDataURL = previousMethod;
else delete canvas.toDataURL;
}
}
const coordinatedNavigator = window.createPageNavigator?.bind(window);
if (coordinatedNavigator) {
window.createPageNavigator = (options = {}) => {
const next = { ...options };
if (next.title === "Trang kiểm tra" && typeof next.onSelect === "function") {
const select = next.onSelect;
next.onSelect = async (index, item) => {
const canvas = document.querySelector(".review-canvas-host .review-card canvas.brush-canvas");
if (!canvas?._reviewDirty) {
if (canvas) revokeMaskUrl(maskKey(canvas));
return select(index, item);
}
return withLegacyMaskSnapshot(canvas, () => select(index, item));
};
}
return coordinatedNavigator(next);
};
}
async function replayStitchedControl(event, target, type) {
if (target.dataset.releaseMaskReplay === "1") return false;
const canvas = document.querySelector(".review-stitched-image canvas.stitched-brush-canvas");
if (!canvas?._reviewDirty) {
if (canvas) revokeMaskUrl(maskKey(canvas));
return false;
}
event.preventDefault();
event.stopImmediatePropagation();
target.dataset.releaseMaskReplay = "1";
try {
await withLegacyMaskSnapshot(canvas, () => {
if (type === "click") target.click();
else target.dispatchEvent(new Event("change", { bubbles: true }));
});
} finally {
delete target.dataset.releaseMaskReplay;
}
return true;
}
document.addEventListener("click", (event) => {
const target = event.target?.closest?.(".review-view-switch button, .review-stitched-toolbar button");
if (!target || target.dataset.releaseMaskReplay === "1") return;
void replayStitchedControl(event, target, "click");
}, true);
document.addEventListener("change", (event) => {
const target = event.target?.closest?.(".review-stitched-select");
if (!target || target.dataset.releaseMaskReplay === "1") return;
void replayStitchedControl(event, target, "change");
}, true);
const legacyHasStitched = window.hasUnsavedStitchedMarks?.bind(window);
window.hasUnsavedStitchedMarks = () => {
const current = document.querySelector(".review-stitched-image canvas.stitched-brush-canvas");
const stored = [...maskObjectUrls.keys()].some((key) => key.startsWith(`stitched:${chapterId()}:`));
return Boolean(stored || current?._reviewDirty || legacyHasStitched?.());
};
function canonicalPageIndex() {
if (stage() === "editor") return Number(window.editorState?.activePageIndex || 0);
const card = document.querySelector(".review-card[data-page-index]") || document.querySelector(".preview-card-active[data-page-index]");
return Number(card?.dataset?.pageIndex || 0);
}
function captureViewContext() {
const grid = currentGrid();
const canvas = grid?.querySelector(":scope > .workbench-canvas-column, :scope > .translation-canvas-host");
const activeNav = grid?.querySelector('.page-navigator-item[aria-current="page"]');
return {
pageIndex: canonicalPageIndex(),
scrollTop: canvas?.scrollTop || 0,
scrollLeft: canvas?.scrollLeft || 0,
navIndex: Number(activeNav?.dataset?.pageNavigatorIndex),
navFocused: Boolean(activeNav && activeNav === document.activeElement),
};
}
function syncGridAttributes(target, source) {
target.className = source.className;
for (const name of [...target.getAttributeNames()]) {
if (name.startsWith("data-") && !source.hasAttribute(name)) target.removeAttribute(name);
}
for (const name of source.getAttributeNames()) {
if (name.startsWith("data-")) target.setAttribute(name, source.getAttribute(name));
}
}
function preserveStageShell(stageName, oldGrid, context) {
const container = document.getElementById("page-view");
const newGrid = container?.querySelector(".workbench-stage-grid");
if (!oldGrid || !newGrid || oldGrid === newGrid) return newGrid;
const children = [...newGrid.children];
syncGridAttributes(oldGrid, newGrid);
oldGrid.replaceChildren(...children);
newGrid.replaceWith(oldGrid);
const canvas = oldGrid.querySelector(":scope > .workbench-canvas-column, :scope > .translation-canvas-host");
if (canvas && context?.pageIndex === canonicalPageIndex()) {
canvas.scrollTop = context.scrollTop;
canvas.scrollLeft = context.scrollLeft;
}
if (context?.navFocused && Number.isFinite(context.navIndex)) {
requestAnimationFrame(() => oldGrid.querySelector(`.page-navigator-item[data-page-navigator-index="${context.navIndex}"]`)?.focus());
}
window.setupWorkbenchPanels?.(stageName);
return oldGrid;
}
function draftStorageKey(chapter = chapterId()) {
return chapter ? `mt_editor_draft:${chapter}` : "";
}
function readDrafts(chapter = chapterId()) {
const key = draftStorageKey(chapter);
if (!key) return null;
try {
const parsed = JSON.parse(localStorage.getItem(key) || "null");
return parsed && parsed.chapterId === chapter && parsed.objects ? parsed : null;
} catch (_) { return null; }
}
function objectState(obj) {
return {
ocr_text: obj?.ocr_text || "",
translation: obj?.translation || "",
style: clone(obj?.style || window.DEFAULT_TEXT_OBJECT_STYLE || {}),
region: clone(obj?.region || null),
shape: obj?.shape || "rectangle",
source_boxes: clone(obj?.source_boxes || []),
};
}
function writeDraftObject(pageIndex, id) {
if (historyApplying || !chapterId() || typeof window.findTextObject !== "function") return;
const obj = window.findTextObject(Number(pageIndex), String(id));
if (!obj) return;
const current = readDrafts() || { chapterId: chapterId(), updatedAt: 0, objects: {} };
current.objects[`${pageIndex}:${id}`] = { pageIndex: Number(pageIndex), id: String(id), state: objectState(obj), updatedAt: Date.now() };
const keys = Object.keys(current.objects).sort((a, b) => current.objects[a].updatedAt - current.objects[b].updatedAt);
while (keys.length > DRAFT_LIMIT) delete current.objects[keys.shift()];
current.updatedAt = Date.now();
try { localStorage.setItem(draftStorageKey(), JSON.stringify(current)); } catch (_) {}
}
function clearStoredDrafts(chapter = chapterId()) {
const key = draftStorageKey(chapter);
if (!key) return;
try { localStorage.removeItem(key); } catch (_) {}
}
function applyStoredDraftsToManifest() {
const chapter = chapterId();
const saved = readDrafts(chapter);
if (!saved || !window.currentManifest?.pages) return [];
const applied = [];
for (const entry of Object.values(saved.objects || {})) {
const obj = window.findTextObject?.(Number(entry.pageIndex), String(entry.id));
if (!obj || !entry.state) continue;
Object.assign(obj, {
ocr_text: entry.state.ocr_text,
translation: entry.state.translation,
style: clone(entry.state.style),
region: clone(entry.state.region),
});
applied.push(entry);
}
if (applied.length && !restoredDraftChapters.has(chapter)) {
restoredDraftChapters.add(chapter);
queueMicrotask(() => toast(`Đã khôi phục ${applied.length} thay đổi cục bộ chưa lưu của chương này.`, "info"));
}
return applied;
}
function scheduleRestoredDraftSave(entries) {
if (!entries?.length) return;
queueMicrotask(() => {
for (const entry of entries) {
window.scheduleTextObjectPersist?.(Number(entry.pageIndex), String(entry.id));
if (entry.state?.region) window.scheduleGeomPersist?.(Number(entry.pageIndex), String(entry.id));
}
});
}
function historyState(chapter = chapterId()) {
if (!historyByChapter.has(chapter)) historyByChapter.set(chapter, { editorUndo: [], editorRedo: [] });
return historyByChapter.get(chapter);
}
function updateHistoryControls() {
const h = historyState();
document.querySelectorAll('[data-history-action="undo"]').forEach((button) => {
const scope = button.dataset.historyScope || "editor";
const state = scope === "editor" ? h : maskHistoryState();
button.disabled = !(scope === "editor" ? state.editorUndo.length : state.undo.length);
});
document.querySelectorAll('[data-history-action="redo"]').forEach((button) => {
const scope = button.dataset.historyScope || "editor";
const state = scope === "editor" ? h : maskHistoryState();
button.disabled = !(scope === "editor" ? state.editorRedo.length : state.redo.length);
});
}
function pushEditorRecord(record, coalesceKey = null) {
if (historyApplying || !record) return;
const now = Date.now();
if (coalesceKey && lastEditorRecord.key === coalesceKey && now - lastEditorRecord.at < 650) return;
lastEditorRecord = { key: coalesceKey, at: now };
const h = historyState();
h.editorUndo.push(record);
if (h.editorUndo.length > HISTORY_LIMIT) h.editorUndo.shift();
h.editorRedo.length = 0;
updateHistoryControls();
}
function recordObjectBefore(pageIndex, id, group = "edit") {
const obj = window.findTextObject?.(Number(pageIndex), String(id));
if (!obj) return;
pushEditorRecord({ type: "object", pageIndex: Number(pageIndex), id: String(id), state: objectState(obj) }, `${pageIndex}:${id}:${group}`);
}
async function applyObjectRecord(record, destination) {
const obj = window.findTextObject?.(record.pageIndex, record.id);
if (!obj) throw new Error("Vùng chữ không còn tồn tại");
destination.push({ type: "object", pageIndex: record.pageIndex, id: record.id, state: objectState(obj) });
Object.assign(obj, {
ocr_text: record.state.ocr_text,
translation: record.state.translation,
style: clone(record.state.style),
region: clone(record.state.region),
});
historyApplying = true;
try { window.renderEditor?.(); }
finally { historyApplying = false; }
window.scheduleTextObjectPersist?.(record.pageIndex, record.id);
if (record.state.region) window.scheduleGeomPersist?.(record.pageIndex, record.id);
writeDraftObject(record.pageIndex, record.id);
}
async function parseResponse(response) {
const data = window.parseApiResponse ? await window.parseApiResponse(response) : await response.json().catch(() => ({}));
if (!response.ok) throw new Error(window.getErrorMessage ? window.getErrorMessage(response.status, data) : data?.detail || `HTTP ${response.status}`);
return data;
}
async function restoreDeletedRecord(record, destination) {
const chapter = chapterId();
const createdManifest = await parseResponse(await window.fetch("/api/text_object/create", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ chapter_id: chapter, page_index: record.pageIndex, shape: record.state.shape, region: record.state.region }),
}));
const objects = createdManifest.pages?.[record.pageIndex]?.text_objects || [];
const created = objects[objects.length - 1];
if (!created) throw new Error("Không thể phục hồi vùng chữ đã xóa");
const updated = await parseResponse(await window.fetch("/api/text_object/update", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({
chapter_id: chapter,
page_index: record.pageIndex,
id: created.id,
ocr_text: record.state.ocr_text,
translation: record.state.translation,
style: record.state.style,
region: record.state.region,
}),
}));
window.currentManifest = updated;
record.recreatedId = created.id;
destination.push({ ...record });
if (window.editorState) window.editorState.selectedTextObjectId = created.id;
historyApplying = true;
try { window.renderEditor?.(); }
finally { historyApplying = false; }
}
async function deleteRestoredRecord(record, destination) {
if (!record.recreatedId) throw new Error("Không còn bản phục hồi để xóa lại");
const data = await parseResponse(await window.fetch("/api/text_object/delete", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ chapter_id: chapterId(), page_index: record.pageIndex, id: record.recreatedId, history_replay: true }),
}));
window.currentManifest = data;
destination.push({ ...record, recreatedId: null });
historyApplying = true;
try { window.renderEditor?.(); }
finally { historyApplying = false; }
}
async function undoEditor() {
const h = historyState();
const record = h.editorUndo.pop();
if (!record) return;
historyApplying = true;
try {
if (record.type === "delete") await restoreDeletedRecord(record, h.editorRedo);
else await applyObjectRecord(record, h.editorRedo);
} catch (error) {
h.editorUndo.push(record);
toast("Không thể hoàn tác: " + error.message, "error");
} finally {
historyApplying = false;
updateHistoryControls();
}
}
async function redoEditor() {
const h = historyState();
const record = h.editorRedo.pop();
if (!record) return;
historyApplying = true;
try {
if (record.type === "delete") await deleteRestoredRecord(record, h.editorUndo);
else await applyObjectRecord(record, h.editorUndo);
} catch (error) {
h.editorRedo.push(record);
toast("Không thể làm lại: " + error.message, "error");
} finally {
historyApplying = false;
updateHistoryControls();
}
}
function maskHistoryState(chapter = chapterId()) {
if (!maskHistoryByChapter.has(chapter)) maskHistoryByChapter.set(chapter, { undo: [], redo: [], bytes: 0 });
return maskHistoryByChapter.get(chapter);
}
function trimMaskHistory(state) {
while (state.undo.length > MASK_HISTORY_LIMIT || state.bytes > MASK_HISTORY_BYTES) {
const record = state.undo.shift();
state.bytes = Math.max(0, state.bytes - (record?.blob?.size || 0));
}
}
async function captureMaskRecord(canvas) {
if (historyApplying || !canvas?.width || !canvas?.height) return;
try {
const blob = await encodeCanvas(canvas);
const state = maskHistoryState();
state.undo.push({ type: "mask", key: maskKey(canvas), blob, dirty: Boolean(canvas._reviewDirty) });
state.bytes += blob.size;
state.redo.length = 0;
trimMaskHistory(state);
updateHistoryControls();
} catch (_) {}
}
async function restoreMaskBlob(record, destination) {
const canvas = document.querySelector("canvas.brush-canvas:is(.stitched-brush-canvas, .review-card canvas.brush-canvas), .review-card canvas.brush-canvas");
const candidates = [...document.querySelectorAll("canvas.brush-canvas")];
const target = candidates.find((item) => maskKey(item) === record.key) || canvas;
if (!target || maskKey(target) !== record.key) throw new Error("Hãy mở đúng trang chứa nét đánh dấu này trước khi hoàn tác");
const current = await encodeCanvas(target);
destination.push({ type: "mask", key: record.key, blob: current, dirty: Boolean(target._reviewDirty) });
const bitmap = await createImageBitmap(record.blob);
const ctx = target.getContext("2d");
ctx.clearRect(0, 0, target.width, target.height);
ctx.drawImage(bitmap, 0, 0, target.width, target.height);
bitmap.close?.();
target._reviewDirty = Boolean(record.dirty);
}
async function undoMask() {
const state = maskHistoryState();
const record = state.undo.pop();
if (!record) return;
try { await restoreMaskBlob(record, state.redo); }
catch (error) { state.undo.push(record); toast("Không thể hoàn tác vùng đánh dấu: " + error.message, "error"); }
updateHistoryControls();
}
async function redoMask() {
const state = maskHistoryState();
const record = state.redo.pop();
if (!record) return;
try { await restoreMaskBlob(record, state.undo); }
catch (error) { state.redo.push(record); toast("Không thể làm lại vùng đánh dấu: " + error.message, "error"); }
updateHistoryControls();
}
function ensureHistoryControls() {
const editorToolbar = document.querySelector(".translation-sticky-toolbar");
if (editorToolbar && !editorToolbar.querySelector(".release-history-controls")) {
const group = document.createElement("div");
group.className = "release-history-controls";
group.setAttribute("role", "group");
group.setAttribute("aria-label", "Lịch sử biên tập");
group.innerHTML = '<button type="button" class="ui-btn ui-btn-ghost ui-btn-compact" data-history-scope="editor" data-history-action="undo">Hoàn tác</button><button type="button" class="ui-btn ui-btn-ghost ui-btn-compact" data-history-scope="editor" data-history-action="redo">Làm lại</button>';
editorToolbar.prepend(group);
}
const reviewActions = document.querySelector(".review-sticky-toolbar .review-actions-group");
if (reviewActions && !reviewActions.querySelector(".release-mask-history-controls")) {
const group = document.createElement("div");
group.className = "release-history-controls release-mask-history-controls";
group.setAttribute("role", "group");
group.setAttribute("aria-label", "Lịch sử vùng đánh dấu cục bộ");
group.innerHTML = '<button type="button" class="ui-btn ui-btn-ghost ui-btn-compact" data-history-scope="review" data-history-action="undo" title="Chỉ hoàn tác nét đánh dấu cục bộ, không rollback inpaint đã commit">Hoàn tác nét</button><button type="button" class="ui-btn ui-btn-ghost ui-btn-compact" data-history-scope="review" data-history-action="redo">Làm lại nét</button>';
reviewActions.prepend(group);
}
updateHistoryControls();
}
document.addEventListener("click", (event) => {
const button = event.target?.closest?.("[data-history-action]");
if (!button) return;
event.preventDefault();
const undo = button.dataset.historyAction === "undo";
if (button.dataset.historyScope === "review") void (undo ? undoMask() : redoMask());
else void (undo ? undoEditor() : redoEditor());
});
window.addEventListener("keydown", (event) => {
if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLowerCase() !== "z") return;
if (document.body.classList.contains("settings-open")) return;
if (stage() !== "editor" && stage() !== "review") return;
event.preventDefault();
if (stage() === "review") void (event.shiftKey ? redoMask() : undoMask());
else void (event.shiftKey ? redoEditor() : undoEditor());
}, true);
function panelObjectFromTarget(target) {
const panel = target?.closest?.(".text-editor-panel[data-object-id][data-page-index]");
if (!panel) return null;
return { pageIndex: Number(panel.dataset.pageIndex), id: String(panel.dataset.objectId) };
}
document.addEventListener("beforeinput", (event) => {
const target = event.target;
if (!target?.matches?.(".text-editor-panel textarea")) return;
const ref = panelObjectFromTarget(target);
if (ref) recordObjectBefore(ref.pageIndex, ref.id, target.className.includes("ocr") ? "ocr" : "translation");
}, true);
document.addEventListener("input", (event) => {
const target = event.target;
if (!target?.closest?.(".text-editor-panel")) return;
const ref = panelObjectFromTarget(target);
if (!ref) return;
if (!target.matches("textarea")) recordObjectBefore(ref.pageIndex, ref.id, "style");
queueMicrotask(() => writeDraftObject(ref.pageIndex, ref.id));
}, true);
document.addEventListener("change", (event) => {
const target = event.target;
if (!target?.closest?.(".text-editor-panel")) return;
const ref = panelObjectFromTarget(target);
if (!ref) return;
recordObjectBefore(ref.pageIndex, ref.id, target.classList.contains("geometry-input") ? "geometry" : "style");
queueMicrotask(() => writeDraftObject(ref.pageIndex, ref.id));
}, true);
document.addEventListener("pointerdown", (event) => {
const overlay = event.target?.closest?.(".text-object-overlay[data-object-id][data-page-index]");
if (overlay && !overlay.classList.contains("drawing")) {
recordObjectBefore(Number(overlay.dataset.pageIndex), String(overlay.dataset.objectId), "geometry");
}
const canvas = event.target?.closest?.("canvas.brush-canvas");
if (canvas && !historyApplying) void captureMaskRecord(canvas);
}, true);
document.addEventListener("pointerup", (event) => {
const overlay = event.target?.closest?.(".text-object-overlay[data-object-id][data-page-index]") || document.querySelector(".text-object-overlay.transforming[data-object-id][data-page-index]");
if (overlay) queueMicrotask(() => writeDraftObject(Number(overlay.dataset.pageIndex), String(overlay.dataset.objectId)));
}, true);
document.addEventListener("click", (event) => {
if (event.target?.closest?.(".clear-brush-btn") && document.querySelector("canvas.brush-canvas")?._reviewDirty) {
void captureMaskRecord(document.querySelector("canvas.brush-canvas"));
}
}, true);
const coordinatedFetch = window.fetch.bind(window);
window.fetch = function releaseFetch(input, init) {
const path = getPath(input);
let deleted = null;
if (!historyApplying && /\/api\/text_object\/delete$/.test(path) && typeof init?.body === "string") {
try {
const payload = JSON.parse(init.body);
if (!payload.history_replay) {
const obj = window.findTextObject?.(Number(payload.page_index), String(payload.id));
if (obj) deleted = { type: "delete", pageIndex: Number(payload.page_index), id: String(payload.id), state: objectState(obj), recreatedId: null };
}
} catch (_) {}
}
const response = coordinatedFetch(input, init);
if (deleted) {
response.then((result) => {
if (result.ok) pushEditorRecord(deleted, null);
}).catch(() => {});
}
return response;
};
function maybeClearSavedDrafts() {
if (stage() !== "editor") return;
const status = document.querySelector(".editor-save-status");
const saved = status?.classList.contains("save-status-saved");
const saving = window.frontendOperationState?.isBusy?.("save") || window.hasPendingGeom?.() || window.isGeomSaving?.();
if (saved && !saving) clearStoredDrafts();
}
function installViewObserver() {
viewObserver?.disconnect();
const view = document.getElementById("page-view");
if (!view) return;
viewObserver = new MutationObserver(() => {
ensureHistoryControls();
applyResponsivePanels();
maybeClearSavedDrafts();
});
viewObserver.observe(view, { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "hidden"] });
}
function handleChapterChange() {
const current = chapterId();
if (current === knownChapterId) return;
const previous = knownChapterId;
knownChapterId = current;
shellRegistry.clear();
panelPrefs.clear();
lastEditorRecord = { key: null, at: 0 };
if (previous) window.disposePageNavigators?.(previous);
for (const key of [...maskObjectUrls.keys()]) {
if (!current || !key.includes(`:${current}:`)) revokeMaskUrl(key);
}
while (historyByChapter.size > 3) historyByChapter.delete(historyByChapter.keys().next().value);
while (maskHistoryByChapter.size > 3) maskHistoryByChapter.delete(maskHistoryByChapter.keys().next().value);
}
function wrapRenderer(name, stageName) {
const original = window[name]?.bind(window);
if (!original || original._releaseWrapped) return;
const wrapped = function releaseRenderer(...args) {
handleChapterChange();
const restored = stageName === "editor" ? applyStoredDraftsToManifest() : [];
const context = captureViewContext();
const oldEntry = shellRegistry.get(stageName);
const canReuse = oldEntry && oldEntry.chapterId === chapterId();
const result = original(...args);
const grid = preserveStageShell(stageName, canReuse ? oldEntry.grid : null, context) || currentGrid();
if (grid) shellRegistry.set(stageName, { chapterId: chapterId(), grid });
applyResponsivePanels();
ensureHistoryControls();
if (restored.length) scheduleRestoredDraftSave(restored);
return result;
};
wrapped._releaseWrapped = true;
window[name] = wrapped;
}
wrapRenderer("renderPreview", "preview");
wrapRenderer("renderReview", "review");
wrapRenderer("renderEditor", "editor");
const legacySetupPanels = window.setupWorkbenchPanels?.bind(window);
if (legacySetupPanels) {
window.setupWorkbenchPanels = (...args) => {
const result = legacySetupPanels(...args);
queueMicrotask(applyResponsivePanels);
return result;
};
}
const legacySyncPanels = window.syncWorkbenchPanels?.bind(window);
if (legacySyncPanels) {
window.syncWorkbenchPanels = (...args) => {
const result = legacySyncPanels(...args);
queueMicrotask(applyResponsivePanels);
return result;
};
}
window.addEventListener("frontend:operationchange", () => {
queueMicrotask(() => {
const currentCanvas = document.querySelector(".review-canvas-host .review-card canvas.brush-canvas, .review-stitched-image canvas.stitched-brush-canvas");
if (currentCanvas && !currentCanvas._reviewDirty) revokeMaskUrl(maskKey(currentCanvas));
maybeClearSavedDrafts();
updateHistoryControls();
});
});
function hasUnsavedWork() {
const status = document.querySelector(".editor-save-status");
const editorUnsaved = Boolean(status && !status.classList.contains("save-status-saved"));
const stored = Boolean(readDrafts());
const review = Boolean(window.frontendHasPendingReviewDrafts?.() || window.hasUnsavedStitchedMarks?.());
return editorUnsaved || stored || review;
}
window.frontendHasUnsavedWork = hasUnsavedWork;
window.addEventListener("beforeunload", (event) => {
if (!hasUnsavedWork()) return;
event.preventDefault();
event.returnValue = "";
});
function dispose() {
headerObserver?.disconnect();
headerObserver = null;
viewObserver?.disconnect();
viewObserver = null;
encoderWorker?.terminate();
encoderWorker = null;
for (const pending of encoderPending.values()) pending.reject(new Error("Trang đã đóng"));
encoderPending.clear();
for (const key of [...maskObjectUrls.keys()]) revokeMaskUrl(key);
}
window.addEventListener("pagehide", dispose, { once: true });
document.addEventListener("DOMContentLoaded", () => {
updateHeaderHeight();
const header = document.getElementById("site-header");
if (header && typeof ResizeObserver !== "undefined") {
headerObserver = new ResizeObserver(() => {
updateHeaderHeight();
applyResponsivePanels();
});
headerObserver.observe(header);
}
window.addEventListener("resize", applyResponsivePanels, { passive: true });
installViewObserver();
applyResponsivePanels();
ensureHistoryControls();
});
window.frontendReleaseDebug = {
encoderMetrics,
shellRegistry,
panelPrefs,
maskObjectUrls,
historyByChapter,
maskHistoryByChapter,
layoutMode,
applyResponsivePanels,
encodeCanvas,
hasUnsavedWork,
dispose,
};
})();
