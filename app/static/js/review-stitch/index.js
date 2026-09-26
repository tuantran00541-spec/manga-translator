import { canvasBlob, captureSnapshot, chunkHasPaint, drawBrushMask, paintPoint, paintStroke } from "./brush.js";
import { installOverlaySync, installTextDrawing, mountTextInspector, renderOverlays, reviewPageIndexAtSourceY, sourcePoint } from "./overlays.js";
import { SHORTCUTS, chapterKey, deleteSnapshot, hasSnapshotPrefix, resetChapterState, snapshotKey, state } from "./state.js";
import { orderedSlices, renderStrip } from "./strip.js";
import { mountActions, mountToolRail, setTool, syncTool } from "./tools.js";
import { applyZoom, stepZoom } from "./zoom.js";

window.hasUnsavedStitchedMarks = () => {
  const prefix = `${chapterKey()}:`;
  if (hasSnapshotPrefix(prefix)) return true;
  const shell = document.querySelector("#page-view.review-mode .review-document-shell");
  return Boolean(shell?._brushChunks?.some((chunk) => chunk.dirty && chunkHasPaint(chunk)));
};

export function mount(workspace) {
  if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchInspectorMounted === "1") return;
  resetChapterState();
  workspace._stitchAbort?.abort();
  window._reviewStitchAbort?.abort();

  const aborter = new AbortController();
  const { signal } = aborter;
  workspace._stitchAbort = aborter;
  window._reviewStitchAbort = aborter;

  const layout = workspace.querySelector(".review-workbench-grid");
  const canvasHost = layout?.querySelector(".review-canvas-host");
  if (!layout || !canvasHost) return;

  workspace.dataset.stitchInspectorMounted = "1";
  workspace.classList.add("review-single-document", "review-lettering-workspace");

  const items = orderedSlices();
  if (!items.length) return;

  canvasHost.replaceChildren();

  const compatibility = document.createElement("div");
  compatibility.className = "review-card review-card-compat";
  compatibility.hidden = true;
  canvasHost.appendChild(compatibility);

  const shell = document.createElement("section");
  shell.className = "review-stitched-shell review-document-shell";
  shell._initialCanonicalPageIndex = workspace._initialCanonicalPageIndex;

  const docbar = document.createElement("div");
  docbar.className = "review-stitched-toolbar review-document-toolbar review-document-toolbar-compact";

  const button = (icon, label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "ui-btn ui-btn-ghost ui-btn-compact";
    b.setAttribute("aria-label", label);
    if (icon) b.append(window.createUiIcon(icon));
    else b.textContent = label;
    return b;
  };

  const left = document.createElement("div");
  left.className = "review-docbar-group review-docbar-left";
  const right = document.createElement("div");
  right.className = "review-docbar-group review-docbar-right";

  const viewSwitch = document.createElement("div");
  viewSwitch.className = "review-view-switch";
  viewSwitch.setAttribute("role", "group");
  viewSwitch.setAttribute("aria-label", "Ảnh đang xem");
  const clean = button(null, "Sau inpaint");
  const rendered = button(null, "Có chữ");
  const original = button(null, "Ảnh gốc");
  viewSwitch.append(clean, rendered, original);

  const zoomDock = document.createElement("div");
  zoomDock.className = "review-zoom-dock";
  zoomDock.setAttribute("role", "group");
  zoomDock.setAttribute("aria-label", "Thu phóng");
  const zoomOut = button("minus", "Thu nhỏ");
  const zoomValue = button(null, "Vừa khung");
  zoomValue.classList.add("review-zoom-value");
  const zoomIn = button("plus", "Phóng to");
  const one = button(null, "1:1");
  one.setAttribute("aria-label", "Xem kích thước thật 1:1");
  one.title = "Kích thước thật (100%)";
  zoomDock.append(zoomOut, zoomValue, zoomIn, one);

  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "ui-btn ui-btn-primary ui-btn-compact repaint-btn";
  submit.textContent = "Làm sạch vùng";

  const clearMask = document.createElement("button");
  clearMask.type = "button";
  clearMask.className = "ui-btn ui-btn-ghost ui-btn-compact clear-brush-btn";
  clearMask.textContent = "Xóa mask";

  const sizeWrap = document.createElement("label");
  sizeWrap.className = "ui-range-field brush-size-control review-docbar-brush-size";
  sizeWrap.textContent = "Cọ ";
  const sizeOut = document.createElement("output");
  sizeOut.textContent = "48px";
  const size = document.createElement("input");
  size.type = "range";
  size.min = "4";
  size.max = "100";
  size.step = "1";
  size.value = "24";
  size.className = "brush-size-slider";
  sizeWrap.append(size, sizeOut);

  const brushBar = document.createElement("div");
  brushBar.className = "review-brush-bar";
  brushBar.setAttribute("role", "toolbar");
  brushBar.setAttribute("aria-label", "Tùy chọn cọ inpaint");
  brushBar.hidden = true;
  const undoRepaint = document.createElement("button");
  undoRepaint.type = "button";
  undoRepaint.className = "ui-btn ui-btn-ghost ui-btn-compact undo-repaint-btn";
  undoRepaint.textContent = "Bỏ repaint lát này";
  undoRepaint.title = "Xóa mọi vùng đã làm sạch bằng cọ trên lát ở giữa màn hình, trả lại kết quả xử lý tự động";
  brushBar.append(sizeWrap, clearMask, submit, undoRepaint);

  const actionsHost = document.createElement("div");
  actionsHost.className = "review-docbar-actions";

  const more = document.createElement("details");
  more.className = "review-more-menu";
  const moreToggle = document.createElement("summary");
  moreToggle.className = "ui-btn ui-btn-ghost ui-btn-compact review-more-toggle";
  moreToggle.setAttribute("aria-label", "Thêm thao tác");
  moreToggle.title = "Thêm thao tác";
  moreToggle.append(window.createUiIcon("more"));
  const morePanel = document.createElement("div");
  morePanel.className = "review-more-panel";
  more.append(moreToggle, morePanel);

  const langRow = document.createElement("label");
  langRow.className = "review-lang-row";
  const langTitle = document.createElement("span");
  langTitle.textContent = "Ngôn ngữ gốc";
  const langSelect = document.createElement("select");
  langSelect.className = "ui-select review-lang-select";
  langSelect.add(new Option("Chưa rõ", ""));
  Object.entries(window.SOURCE_LANG_LABELS || {}).forEach(([value, label]) => langSelect.add(new Option(label, value)));
  const langHint = document.createElement("small");
  langHint.className = "review-lang-hint";
  langRow.append(langTitle, langSelect, langHint);
  const moreActions = document.createElement("div");
  moreActions.className = "review-more-actions";
  morePanel.append(langRow, moreActions);

  let detectingLang = false;
  const syncLang = () => {
    const lang = window.currentSourceLang?.() || "";
    langSelect.value = lang;
    langHint.textContent = lang
      ? window.sourceLangOriginLabel?.(window.currentManifest?.source_lang_origin) || ""
      : detectingLang ? "Đang nhận diện…" : "Chưa nhận diện được, hãy chọn";
    moreToggle.classList.toggle("has-alert", !lang && !detectingLang);
  };
  langSelect.addEventListener("change", async () => {
    if (!langSelect.value) return syncLang();
    try {
      const lang = await window.setSourceLang(langSelect.value);
      window.showToast?.(`Ngôn ngữ gốc: ${window.SOURCE_LANG_LABELS?.[lang] || lang}. Chạy lại OCR toàn chương nếu cần đọc lại chữ.`, "success");
    } catch (err) {
      window.showToast?.("Không thể đổi ngôn ngữ gốc: " + err.message, "error");
    }
    syncLang();
  }, { signal });
  document.addEventListener("source-lang-changed", syncLang, { signal });
  document.addEventListener("source-lang-needed", () => { more.open = true; langSelect.focus(); }, { signal });
  morePanel.addEventListener("click", (e) => { if (e.target.closest("button")) more.open = false; }, { signal });
  // Docbar popovers (Dịch tự động, OCR, ⋯): one open at a time, closed by a click outside or Escape.
  const openMenus = () => docbar.querySelectorAll("details[open]");
  docbar.addEventListener("toggle", (e) => {
    if (!(e.target instanceof HTMLDetailsElement) || !e.target.open) return;
    openMenus().forEach((other) => { if (other !== e.target && !other.contains(e.target)) other.open = false; });
  }, { capture: true, signal });
  document.addEventListener("pointerdown", (e) => { openMenus().forEach((menu) => { if (!menu.contains(e.target)) menu.open = false; }); }, { capture: true, signal });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !openMenus().length) return;
    openMenus().forEach((menu) => { menu.open = false; });
    e.stopImmediatePropagation();
  }, { capture: true, signal });

  left.append(viewSwitch);
  right.append(actionsHost, more);
  docbar.append(left, right);

  const meta = document.createElement("div");
  meta.className = "review-stitched-meta";
  meta.hidden = true;

  const warning = document.createElement("div");
  warning.className = "review-stitched-warning";
  warning.hidden = true;

  const viewport = document.createElement("div");
  viewport.className = "review-stitched-viewport review-document-viewport";
  const stage = document.createElement("div");
  stage.className = "review-stitched-zoom-stage";
  const image = document.createElement("div");
  image.className = "review-stitched-image";
  stage.appendChild(image);
  viewport.appendChild(stage);

  shell.append(docbar, meta, warning, viewport);
  canvasHost.appendChild(shell);

  const syncDocbarHeight = () => shell.style.setProperty("--review-docbar-height", `${Math.ceil(docbar.getBoundingClientRect().height)}px`);
  const docbarObserver = new ResizeObserver(syncDocbarHeight);
  docbarObserver.observe(docbar);
  signal.addEventListener("abort", () => docbarObserver.disconnect(), { once: true });

  installOverlaySync();
  mountTextInspector(workspace, shell, signal);
  mountToolRail(shell, signal);
  shell.append(brushBar, zoomDock);
  installTextDrawing(shell, signal);
  mountActions(shell, signal, () => shell._rerender?.());
  window.mountChapterOCR?.();
  window.mountChapterQC?.();
  detectingLang = !window.currentSourceLang?.();
  syncLang();
  window.detectSourceLang?.()
    .catch((err) => console.warn("Source language detection failed", err))
    .finally(() => { detectingLang = false; syncLang(); });

  let painting = false, radius = 24, last = null, panning = false, pan = null, space = false;

  const syncVariant = () => { [[clean, "clean"], [rendered, "rendered"], [original, "original"]].forEach(([b, name]) => { b.classList.toggle("ui-btn-primary", state.variant === name); b.setAttribute("aria-pressed", String(state.variant === name)); }); const readonly = state.variant !== "clean"; submit.disabled = readonly; clearMask.disabled = readonly; size.disabled = readonly; workspace.classList.toggle("review-readonly-document", readonly); syncTool(shell); };
  shell._syncVariantUI = syncVariant;
  const syncBrushBar = () => {
    const hasMask = (shell._brushChunks || []).some((c) => c.dirty);
    brushBar.hidden = state.variant !== "clean" || !(["brush", "eraser"].includes(state.tool) || hasMask);
  };
  shell._syncBrushBar = syncBrushBar;
  const busy = (on, text = "Đang xử lý…") => { workspace.classList.toggle("review-busy", on); [submit, clearMask, undoRepaint, size, clean, rendered, original, zoomOut, zoomIn, one, zoomValue].forEach((el) => el.disabled = on); submit.textContent = on ? text : "Làm sạch vùng"; document.querySelectorAll(".review-rail-tool").forEach((b) => b.disabled = on); if (!on) syncVariant(); };
  const updateCompat = () => {
    const savedPage = Number(workspace.dataset.reviewCanonicalIndex);
    const canonical = Number.isInteger(savedPage) && savedPage >= 0
      ? savedPage
      : Number(items[0]?.canonicalIndex ?? 0);
    compatibility.dataset.pageIndex = String(canonical);
    workspace.dataset.reviewCanonicalIndex = String(canonical);
    window.setWorkflowCheckpoint?.("review", canonical);
  };
  let visiblePageFrame = 0;
  const syncVisibleReviewPage = () => {
    visiblePageFrame = 0;
    const image = shell.querySelector(".review-stitched-image");
    const imageRect = image?.getBoundingClientRect();
    const viewportRect = viewport.getBoundingClientRect();
    const scale = Number(image?.dataset.zoomScale || 1);
    if (!imageRect?.height || !scale) return;
    const sourceY = (viewportRect.top + viewport.clientHeight / 2 - imageRect.top) / scale;
    const pageIndex = reviewPageIndexAtSourceY(shell._descriptors, sourceY);
    if (!Number.isInteger(pageIndex)) return;
    if (Number(workspace.dataset.reviewCanonicalIndex) === pageIndex) return;
    workspace.dataset.reviewCanonicalIndex = String(pageIndex);
    compatibility.dataset.pageIndex = String(pageIndex);
    window.setWorkflowCheckpoint?.("review", pageIndex);
  };
  shell._syncVisibleReviewPage = syncVisibleReviewPage;
  viewport.addEventListener("scroll", () => {
    if (visiblePageFrame) return;
    visiblePageFrame = requestAnimationFrame(syncVisibleReviewPage);
  }, { passive: true, signal });
  const rerender = () => {
    captureSnapshot(shell);
    updateCompat();
    syncVariant();
    void renderStrip(shell, items, signal);
  };
  shell._rerender = rerender;
  shell._showRendered = () => { state.variant = "rendered"; rerender(); };
  signal.addEventListener("abort", () => captureSnapshot(shell), { once: true });

  image.addEventListener("pointerdown", (e) => { if (state.variant !== "clean" || !["brush", "eraser"].includes(state.tool) || e.button !== 0) { return; } const p = sourcePoint(image, e); if (!p) { return; } e.preventDefault(); painting = true; last = p; image.setPointerCapture?.(e.pointerId); paintPoint(shell, p.x, p.y, radius, state.tool === "eraser"); }, { signal });
  image.addEventListener("pointermove", (e) => { if (!painting || !last || !["brush", "eraser"].includes(state.tool)) { return; } const p = sourcePoint(image, e); if (!p) { return; } paintStroke(shell, last, p, radius, state.tool === "eraser"); last = p; }, { signal });
  const stopPaint = () => { painting = false; last = null; syncBrushBar(); }; image.addEventListener("pointerup", stopPaint, { signal }); image.addEventListener("pointercancel", stopPaint, { signal });

  window.addEventListener("keydown", (e) => {
    const tag = e.target?.tagName?.toLowerCase(), editable = e.target?.isContentEditable || tag === "input" || tag === "textarea" || tag === "select"; if (editable) return;
    if (e.code === "Space" && !e.repeat) { space = true; return; }
    if ((e.ctrlKey || e.metaKey) && e.key === "0") { e.preventDefault(); state.fitWidth = true; return applyZoom(shell); }
    if ((e.ctrlKey || e.metaKey) && e.key === "1") { e.preventDefault(); state.fitWidth = false; state.zoom = 100; return applyZoom(shell); }
    if (e.key === "[" && ["brush","eraser"].includes(state.tool)) { e.preventDefault(); radius = Math.max(4, radius - 2); size.value = String(radius); sizeOut.textContent = `${radius * 2}px`; return; }
    if (e.key === "]" && ["brush","eraser"].includes(state.tool)) { e.preventDefault(); radius = Math.min(100, radius + 2); size.value = String(radius); sizeOut.textContent = `${radius * 2}px`; return; }
    if ((e.key === "Delete" || e.key === "Backspace") && window.editorState?.selectedTextObjectId) { const pageIndex = Number(window.editorState.activePageIndex || 0), id = window.editorState.selectedTextObjectId; e.preventDefault(); window.deleteTextObject?.(pageIndex, id)?.then(() => renderOverlays(shell, signal)).catch((err) => window.showToast?.("Không thể xóa vùng chữ: " + err.message, "error")); return; }
    if (e.key === "Escape") return setTool(shell, "select");
    if (!e.ctrlKey && !e.metaKey && !e.altKey && SHORTCUTS[e.key.toLowerCase()]) { e.preventDefault(); setTool(shell, SHORTCUTS[e.key.toLowerCase()]); }
  }, { signal });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") { space = false; panning = false; pan = null; viewport.classList.remove("is-panning"); } }, { signal });
  viewport.addEventListener("pointerdown", (e) => { if (e.button !== 0 || !(space || state.tool === "hand")) { return; } panning = true; pan = { x: e.clientX, y: e.clientY, left: viewport.scrollLeft, top: viewport.scrollTop }; viewport.setPointerCapture?.(e.pointerId); viewport.classList.add("is-panning"); e.preventDefault(); }, { signal });
  viewport.addEventListener("pointermove", (e) => { if (!panning || !pan) { return; } viewport.scrollLeft = pan.left - (e.clientX - pan.x); viewport.scrollTop = pan.top - (e.clientY - pan.y); }, { signal });
  viewport.addEventListener("pointerup", () => { panning = false; pan = null; viewport.classList.remove("is-panning"); }, { signal });
  viewport.addEventListener("click", (e) => { if (state.tool !== "zoom" || e.target.closest("button,input,textarea,select")) { return; } const r = viewport.getBoundingClientRect(); stepZoom(e.altKey ? -1 : 1); applyZoom(shell, { x: e.clientX - r.left, y: e.clientY - r.top }); }, { signal });
  viewport.addEventListener("wheel", (e) => { if (!(e.ctrlKey || e.metaKey)) { return; } e.preventDefault(); const r = viewport.getBoundingClientRect(); stepZoom(e.deltaY < 0 ? 1 : -1); applyZoom(shell, { x: e.clientX - r.left, y: e.clientY - r.top }); }, { passive: false, signal });

  size.addEventListener("input", () => { radius = Number(size.value); sizeOut.textContent = `${radius * 2}px`; }, { signal });
  clearMask.addEventListener("click", () => { for (const c of shell._brushChunks || []) { if (c.canvas && c.ctx) { c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height); } c.dirty = false; } deleteSnapshot(snapshotKey()); syncBrushBar(); }, { signal });
  clean.addEventListener("click", () => { if (state.variant !== "clean") { state.variant = "clean"; rerender(); } }, { signal }); rendered.addEventListener("click", () => { if (state.variant !== "rendered") { captureSnapshot(shell); state.variant = "rendered"; rerender(); } }, { signal }); original.addEventListener("click", () => { if (state.variant !== "original") { captureSnapshot(shell); state.variant = "original"; rerender(); } }, { signal });
  zoomOut.addEventListener("click", () => { stepZoom(-1); applyZoom(shell); }, { signal }); zoomIn.addEventListener("click", () => { stepZoom(1); applyZoom(shell); }, { signal }); zoomValue.addEventListener("click", () => { state.fitWidth = true; applyZoom(shell); }, { signal }); one.addEventListener("click", () => { state.fitWidth = false; state.zoom = 100; applyZoom(shell); }, { signal });
  window.addEventListener("resize", () => { if (state.fitWidth) applyZoom(shell); }, { signal });

  submit.addEventListener("click", async () => {
    const chapterId = window.currentChapterId, chunks = shell._brushChunks || []; if (!chapterId || !chunks.some((c) => c.dirty && chunkHasPaint(c))) return window.showToast?.("Chưa có vùng nào được đánh dấu.", "error");
    const mode = typeof window.chooseRepaintMode === "function" ? await window.chooseRepaintMode() : "standard"; if (!mode || chapterId !== window.currentChapterId) { return; } busy(true, mode === "lama" ? "LaMa đang xử lý…" : "Đang xử lý…");
    try {
      let affected = 0;
      for (const desc of shell._descriptors || []) {
        if (!(shell._brushChunks || []).some((c) => chunkHasPaint(c, desc.sourceY1, desc.sourceY2))) continue;
        const mask = document.createElement("canvas"); mask.width = desc.img.naturalWidth; mask.height = desc.img.naturalHeight; drawBrushMask(mask.getContext("2d"), chunks, desc, mask.width);
        const form = new FormData(); form.append("chapter_id", chapterId); form.append("page_index", desc.item.canonicalIndex); form.append("mode", mode); form.append("mask", await canvasBlob(mask), "mask.png");
        const response = await fetch("/api/repaint_mask", { method: "POST", body: form }), parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({}))), data = await parse(response); if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
        if (window.currentManifest?.pages?.[desc.item.canonicalIndex] && data.pages?.[desc.item.canonicalIndex]) { window.currentManifest.pages[desc.item.canonicalIndex] = data.pages[desc.item.canonicalIndex]; } affected++;
      }
      for (const c of chunks) { if (c.canvas && c.ctx) { c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height); } c.dirty = false; } deleteSnapshot(snapshotKey()); window.showToast?.(affected ? "Đã làm sạch các vùng được đánh dấu." : "Không có vùng ảnh nào được cập nhật.", affected ? "success" : "info"); rerender();
    } catch (err) { window.showToast?.("Không thể xử lý vùng đánh dấu: " + err.message, "error"); } finally { busy(false); }
  }, { signal });

  undoRepaint.addEventListener("click", async () => {
    const chapterId = window.currentChapterId, pageIndex = Number(workspace.dataset.reviewCanonicalIndex), page = window.currentManifest?.pages?.[pageIndex];
    if (!chapterId || !page) return;
    if (!page.manual_mask && !page.manual_lama_mask) return window.showToast?.(`Lát ${pageIndex + 1} chưa có vùng nào làm sạch bằng cọ.`, "info");
    if (!window.confirm(`Bỏ mọi vùng đã làm sạch bằng cọ trên lát ${pageIndex + 1}? Lát sẽ trở về kết quả xử lý tự động.`)) return;
    busy(true, "Đang khôi phục…");
    try {
      const response = await fetch("/api/reset_manual_mask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ chapter_id: chapterId, page_index: pageIndex }) });
      const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({}))), data = await parse(response);
      if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
      if (data.pages?.[pageIndex]) window.currentManifest.pages[pageIndex] = data.pages[pageIndex];
      window.showToast?.(`Đã bỏ repaint trên lát ${pageIndex + 1}.`, "success"); rerender();
    } catch (err) { window.showToast?.("Không thể bỏ repaint: " + err.message, "error"); } finally { busy(false); }
  }, { signal });

  syncVariant(); setTool(shell, state.tool); rerender();
}

export function scan() { document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount); }

window.mountStitchInspector = scan;
