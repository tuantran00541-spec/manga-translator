import { canvasBlob, captureSnapshot, chunkHasPaint, drawBrushMask, paintPoint, paintStroke } from "./brush.js";
import { mountContextTypeBar } from "./context-bar.js";
import { clearSelection, descriptorFor, getSelectedTargets, installOverlaySync, installTextDrawing, mountTextInspector, ownedObjects, renderOverlays, reviewPageIndexAtSourceY, selectMany, selectObject, sourcePoint, sourceRegion, syncOverlay } from "./overlays.js";
import { SHORTCUTS, chapterKey, deleteSnapshot, hasSnapshotPrefix, resetChapterState, snapshotKey, state } from "./state.js";
import { goTo, installProofing } from "./proofing.js";
import { imageUrl, orderedSlices, renderStrip } from "./strip.js";
import { mountLayersSection, mountPageRail, mountStatusBar } from "./studio-chrome.js";
import { mountActions, mountToolRail, setTool, syncBrushTip, syncTool } from "./tools.js";
import { applyZoom, scrollToReviewPage, stepZoom } from "./zoom.js";

window.hasUnsavedStitchedMarks = () => {
  const prefix = `${chapterKey()}:`;
  if (hasSnapshotPrefix(prefix)) return true;
  const shell = document.querySelector("#page-view.review-mode .review-document-shell");
  return Boolean(shell?._brushChunks?.some((chunk) => chunk.dirty && chunkHasPaint(chunk)));
};

window.addEventListener("beforeunload", (event) => {
  if (window.hasUnsavedStitchedMarks?.()) {
    event.preventDefault();
    event.returnValue = "";
  }
});

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
  shell.className = "review-stitched-shell review-document-shell koharu-studio-shell";
  shell._initialCanonicalPageIndex = workspace._initialCanonicalPageIndex;

  const canvasPane = document.createElement("div");
  canvasPane.className = "koharu-canvas-pane";
  shell._canvasPane = canvasPane;

  const docbar = document.createElement("div");
  docbar.className = "review-stitched-toolbar review-document-toolbar review-document-toolbar-compact koharu-docbar";

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

  const railToggleBtn = button("panel-left", "Ẩn/hiện danh sách trang");
  railToggleBtn.classList.add("koharu-rail-toggle", "active");
  railToggleBtn.title = "Ẩn / hiện cột trang bên trái";
  railToggleBtn.setAttribute("aria-pressed", "true");
  railToggleBtn.addEventListener("click", () => {
    const collapsed = shell.classList.toggle("koharu-rail-collapsed");
    railToggleBtn.classList.toggle("active", !collapsed);
    railToggleBtn.setAttribute("aria-pressed", String(!collapsed));
    requestAnimationFrame(() => { if (state.fitWidth) applyZoom(shell); });
  }, { signal });

  const viewHelpers = document.createElement("div");
  viewHelpers.className = "review-view-helpers";
  viewHelpers.setAttribute("role", "group");
  viewHelpers.setAttribute("aria-label", "Hiển thị vùng chữ và đối chiếu");

  const peekBtn = button("compare", "Giữ để xem ảnh gốc (\\)");
  peekBtn.classList.add("review-helper-btn", "review-peek-btn");
  peekBtn.title = "Giữ chuột hoặc phím \\ để đối chiếu ảnh gốc tức thì";
  peekBtn.setAttribute("aria-pressed", "false");

  const previewBtn = button("type-preview", "Bật/tắt chữ dịch trên trang (T)");
  previewBtn.classList.add("review-helper-btn", "review-preview-btn");
  previewBtn.title = "Ẩn / hiện chữ dịch trực tiếp trên trang (T)";

  const overlayBtn = button("eye", "Ẩn/hiện khung vùng chữ (Q)");
  overlayBtn.classList.add("review-helper-btn", "review-overlay-btn");
  overlayBtn.title = "Ẩn / hiện khung vùng chữ (Q)";

  viewHelpers.append(peekBtn, previewBtn, overlayBtn);

  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "ui-btn ui-btn-primary ui-btn-compact repaint-btn";
  submit.textContent = "Làm sạch vùng";

  const clearMask = document.createElement("button");
  clearMask.type = "button";
  clearMask.className = "ui-btn ui-btn-ghost ui-btn-compact clear-brush-btn";
  clearMask.textContent = "Xóa vùng tô";

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
  brushBar.setAttribute("aria-label", "Tùy chọn cọ tẩy nền");
  brushBar.hidden = true;
  const undoRepaint = document.createElement("button");
  undoRepaint.type = "button";
  undoRepaint.className = "ui-btn ui-btn-ghost ui-btn-compact undo-repaint-btn";
  undoRepaint.textContent = "Khôi phục nền tự động";
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
      window.showToast?.(`Đã đổi ngôn ngữ gốc sang ${window.SOURCE_LANG_LABELS?.[lang] || lang}. Đang tự động chạy lại OCR toàn chương…`, "success");
      syncLang();
      void window.startChapterOCR?.({ auto: true, force: true });
    } catch (err) {
      window.showToast?.("Không thể đổi ngôn ngữ gốc: " + err.message, "error");
      syncLang();
    }
  }, { signal });
  document.addEventListener("source-lang-changed", syncLang, { signal });
  document.addEventListener("source-lang-needed", () => { more.open = true; langSelect.focus(); }, { signal });
  morePanel.addEventListener("click", (e) => { if (e.target.closest("button")) more.open = false; }, { signal });
  
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

  left.append(railToggleBtn, viewHelpers);
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

  const pageRail = mountPageRail(shell, items, signal, {
    imageUrl,
    onSelectSlice: (canonicalIndex) => {
      scrollToReviewPage(shell, canonicalIndex);
      pageRail.setActive(canonicalIndex);
    },
  });

  const rightSidebar = document.createElement("aside");
  rightSidebar.className = "koharu-right-sidebar";
  rightSidebar.setAttribute("aria-label", "Thuộc tính và lớp chữ");

  const sidebarTabs = document.createElement("div");
  sidebarTabs.className = "koharu-sidebar-tabs";
  sidebarTabs.setAttribute("role", "tablist");
  sidebarTabs.setAttribute("aria-label", "Bảng bên phải");

  const makeTab = (iconName, text, active) => {
    const t = document.createElement("button");
    t.type = "button";
    t.className = `koharu-sidebar-tab${active ? " is-active" : ""}`;
    t.setAttribute("role", "tab");
    t.setAttribute("aria-selected", String(active));
    t.append(window.createUiIcon(iconName, "koharu-tab-icon"));
    const span = document.createElement("span");
    span.textContent = text;
    t.appendChild(span);
    return t;
  };
  const tabProps = makeTab("sliders", "Thuộc tính", true);
  const tabStory = makeTab("list", "Soát chữ", false);
  sidebarTabs.append(tabProps, tabStory);

  const propsPane = document.createElement("div");
  propsPane.className = "koharu-props-pane";
  propsPane.setAttribute("role", "tabpanel");
  shell._propsHost = propsPane;

  const storyPane = document.createElement("div");
  storyPane.className = "koharu-story-pane";
  storyPane.setAttribute("role", "tabpanel");
  storyPane.hidden = true;
  shell._storyHost = storyPane;

  rightSidebar.append(sidebarTabs, propsPane, storyPane);

  shell._showRightTab = (tab) => {
    const showStory = tab === "story";
    tabProps.classList.toggle("is-active", !showStory);
    tabProps.setAttribute("aria-selected", String(!showStory));
    tabStory.classList.toggle("is-active", showStory);
    tabStory.setAttribute("aria-selected", String(showStory));
    propsPane.hidden = showStory;
    storyPane.hidden = !showStory;
    const proofPanel = storyPane.querySelector(".review-proof-panel");
    if (proofPanel) proofPanel.hidden = !showStory;
    if (showStory) shell._rebuildProofPanel?.();
  };
  tabProps.addEventListener("click", () => shell._showRightTab("props"), { signal });
  tabStory.addEventListener("click", () => shell._showRightTab("story"), { signal });

  canvasPane.append(docbar, meta, warning, viewport);
  shell.append(pageRail.element, canvasPane, rightSidebar);
  canvasHost.appendChild(shell);

  const syncDocbarHeight = () => {
    const h = `${Math.ceil(docbar.getBoundingClientRect().height)}px`;
    shell.style.setProperty("--review-docbar-height", h);
    canvasPane.style.setProperty("--review-docbar-height", h);
  };
  const docbarObserver = new ResizeObserver(syncDocbarHeight);
  docbarObserver.observe(docbar);
  signal.addEventListener("abort", () => docbarObserver.disconnect(), { once: true });

  installOverlaySync();
  window.refreshReviewOverlays = () => {
    renderOverlays(shell, signal);
    layers.refresh();
    pageRail.refreshAll();
    statusBar.sync();
  };
  window.selectReviewTool = (toolName) => setTool(shell, toolName);
  signal.addEventListener("abort", () => {
    delete window.refreshReviewOverlays;
    delete window.selectReviewTool;
  }, { once: true });

  mountTextInspector(workspace, shell, signal);
  const layers = mountLayersSection(propsPane, shell, signal, {
    ownedObjects,
    selectObject,
    goToObject: (s, pi, id) => {
      shell._showRightTab("props");
      void goTo(s, pi, id);
    },
    deleteOne: async (pi, id) => {
      try {
        await window.deleteTextObject?.(pi, id);
        renderOverlays(shell, signal);
        pageRail.refreshAll();
        statusBar.sync();
      } catch (err) {
        window.showToast?.("Không thể xóa vùng chữ: " + err.message, "error");
      }
    },
    onToggleLivePreview: () => {
      state.livePreview = !state.livePreview;
      syncDisplayToggles();
      layers.refresh();
    },
    onTogglePeekOriginal: () => {
      setPeekOriginal(!state.peekOriginal);
      layers.refresh();
    },
  });

  mountToolRail(shell, signal);
  canvasPane.append(brushBar);
  const statusBar = mountStatusBar(canvasPane, shell, signal, { applyZoom, stepZoom, ownedObjects });
  installTextDrawing(shell, signal);
  mountActions(shell);
  installProofing(shell, signal);
  window.mountChapterOCR?.();
  detectingLang = !window.currentSourceLang?.();
  syncLang();
  window.detectSourceLang?.()
    .catch((err) => console.warn("Source language detection failed", err))
    .finally(() => {
      detectingLang = false;
      syncLang();
      window.maybeAutoChapterOCR?.();
    });

  let painting = false, radius = state.brushRadius || 24, last = null, panning = false, pan = null, space = false, removeForceLama = false, busyFlag = false;

  const brushCursor = document.createElement("div");
  brushCursor.className = "review-brush-cursor";
  brushCursor.hidden = true;
  canvasPane.appendChild(brushCursor);

  const syncBrushCursorSize = () => {
    const scale = Number(image.dataset.zoomScale || 1);
    const d = Math.max(6, Math.round(radius * 2 * scale));
    brushCursor.style.width = `${d}px`;
    brushCursor.style.height = `${d}px`;
  };
  shell._syncBrushCursorSize = syncBrushCursorSize;

  const setBrushRadius = (next) => {
    radius = Math.max(4, Math.min(100, Math.round(Number(next) || 24)));
    state.brushRadius = radius;
    size.value = String(radius);
    sizeOut.textContent = `${radius * 2}px`;
    syncBrushTip(shell);
    syncBrushCursorSize();
  };
  shell._focusBrushSlider = () => {
    syncBrushBar();
    size.focus();
  };
  setBrushRadius(radius);

  const duplicateSelection = async () => {
    const targets = getSelectedTargets(shell);
    if (!targets.length || state.variant !== "clean") return;
    try {
      const created = [];
      for (const t of targets) {
        const newId = await window.duplicateTextObject?.(t.pageIndex, t.id);
        if (newId) created.push({ pageIndex: t.pageIndex, id: newId });
      }
      if (!created.length) return;
      renderOverlays(shell, signal);
      selectMany(shell, created);
      window.showToast?.(created.length > 1 ? `Đã nhân đôi ${created.length} vùng chữ.` : "Đã nhân đôi vùng chữ.", "success");
    } catch (err) {
      window.showToast?.("Không thể nhân đôi vùng chữ: " + err.message, "error");
    }
  };

  const deleteSelection = async () => {
    const targets = getSelectedTargets(shell);
    if (!targets.length) return;
    try {
      for (const t of targets) {
        await window.deleteTextObject?.(t.pageIndex, t.id);
      }
      clearSelection(shell);
      renderOverlays(shell, signal);
      if (targets.length > 1) window.showToast?.(`Đã xóa ${targets.length} vùng chữ.`, "info");
    } catch (err) {
      window.showToast?.("Không thể xóa vùng chữ: " + err.message, "error");
    }
  };

  mountContextTypeBar(shell, signal, {
    getSelectedTargets,
    sourceRegion,
    syncOverlay,
    duplicateSelection,
    deleteSelection,
  });

  const eyedropperBadge = document.createElement("div");
  eyedropperBadge.className = "review-eyedropper-badge";
  eyedropperBadge.hidden = true;
  const eyedropperSwatch = document.createElement("span");
  eyedropperSwatch.className = "review-eyedropper-swatch";
  const eyedropperLabel = document.createElement("span");
  eyedropperLabel.className = "review-eyedropper-hex";
  eyedropperBadge.append(eyedropperSwatch, eyedropperLabel);
  canvasPane.appendChild(eyedropperBadge);

  const _probeCanvas = document.createElement("canvas");
  _probeCanvas.width = 1;
  _probeCanvas.height = 1;
  const _probeCtx = _probeCanvas.getContext("2d", { willReadFrequently: true });

  const sampleCanvasHex = (pt) => {
    if (!pt || !_probeCtx) return null;
    const desc = (shell._descriptors || []).find((d) => pt.y >= d.sourceY1 && pt.y < d.sourceY2)
      || (shell._descriptors || [])[0];
    if (!desc) return null;
    const sliceEl = image.querySelector(`.review-strip-slice[data-page-index="${desc.item.canonicalIndex}"] > img`);
    if (!sliceEl || !sliceEl.complete || !sliceEl.naturalWidth || !sliceEl.naturalHeight) return null;
    const localX = Math.max(0, Math.min(sliceEl.naturalWidth - 1, Math.floor(pt.x)));
    const localY = Math.max(0, Math.min(sliceEl.naturalHeight - 1, Math.floor(desc.localY1 + (pt.y - desc.sourceY1))));
    try {
      _probeCtx.clearRect(0, 0, 1, 1);
      _probeCtx.drawImage(sliceEl, localX, localY, 1, 1, 0, 0, 1, 1);
      const px = _probeCtx.getImageData(0, 0, 1, 1).data;
      return `#${[px[0], px[1], px[2]].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
    } catch (_) {
      return null;
    }
  };

  const setPeekOriginal = (on) => {
    const active = Boolean(on) && state.variant !== "original";
    if (state.peekOriginal === active) return;
    state.peekOriginal = active;
    shell.classList.toggle("review-peek-active", active);
    peekBtn.classList.toggle("ui-btn-primary", active);
    peekBtn.setAttribute("aria-pressed", String(active));
    const pages = window.currentManifest?.pages || [];
    for (const desc of shell._descriptors || []) {
      const idx = Number(desc.item.canonicalIndex);
      const page = pages[idx];
      const imgEl = image.querySelector(`.review-strip-slice[data-page-index="${idx}"] > img`);
      if (!page || !imgEl) continue;
      if (active) {
        if (!imgEl.dataset.normalSrc) imgEl.dataset.normalSrc = imgEl.src;
        const origUrl = imageUrl(page, "original");
        if (origUrl && imgEl.src !== origUrl) imgEl.src = origUrl;
      } else if (imgEl.dataset.normalSrc) {
        imgEl.src = imgEl.dataset.normalSrc;
        delete imgEl.dataset.normalSrc;
      }
    }
  };

  const syncDisplayToggles = () => {
    shell.classList.toggle("review-overlays-hidden", !state.showOverlays);
    shell.classList.toggle("review-live-preview-off", !state.livePreview);
    overlayBtn.classList.toggle("active", state.showOverlays);
    overlayBtn.setAttribute("aria-pressed", String(state.showOverlays));
    previewBtn.classList.toggle("active", state.livePreview);
    previewBtn.setAttribute("aria-pressed", String(state.livePreview));
    peekBtn.disabled = state.variant === "original";
    if (state.variant === "original" && state.peekOriginal) setPeekOriginal(false);
    shell._syncContextBar?.();
  };

  peekBtn.addEventListener("pointerdown", (e) => { if (e.button === 0) { e.preventDefault(); setPeekOriginal(true); } }, { signal });
  const endPeek = () => setPeekOriginal(false);
  peekBtn.addEventListener("pointerup", endPeek, { signal });
  peekBtn.addEventListener("pointerleave", endPeek, { signal });
  peekBtn.addEventListener("pointercancel", endPeek, { signal });
  previewBtn.addEventListener("click", () => { state.livePreview = !state.livePreview; syncDisplayToggles(); }, { signal });
  overlayBtn.addEventListener("click", () => { state.showOverlays = !state.showOverlays; syncDisplayToggles(); }, { signal });

  const syncVariant = () => {
    state.variant = "clean";
    submit.disabled = false;
    clearMask.disabled = false;
    size.disabled = false;
    workspace.classList.remove("review-readonly-document");
    syncTool(shell);
    syncDisplayToggles();
    layers.refresh();
    statusBar.sync();
  };
  shell._syncVariantUI = syncVariant;
  const syncBrushBar = () => {
    const hasMask = (shell._brushChunks || []).some((c) => c.dirty);
    brushBar.hidden = !(["brush", "remove", "eraser"].includes(state.tool) || hasMask);
    if (!["brush", "remove", "eraser"].includes(state.tool)) brushCursor.hidden = true;
  };
  shell._syncBrushBar = syncBrushBar;
  const busy = (on, text = "Đang xử lý…") => {
    busyFlag = Boolean(on);
    workspace.classList.toggle("review-busy", on);
    [submit, clearMask, undoRepaint, size, ...statusBar.controls].forEach((el) => { el.disabled = on; });
    submit.textContent = on ? text : "Làm sạch vùng";
    document.querySelectorAll(".review-rail-tool").forEach((b) => { b.disabled = on; });
    if (!on) syncVariant();
  };
  const updateCompat = () => {
    const savedPage = Number(workspace.dataset.reviewCanonicalIndex);
    const canonical = Number.isInteger(savedPage) && savedPage >= 0
      ? savedPage
      : Number(items[0]?.canonicalIndex ?? 0);
    compatibility.dataset.pageIndex = String(canonical);
    workspace.dataset.reviewCanonicalIndex = String(canonical);
    window.setWorkflowCheckpoint?.("review", canonical);
    pageRail.setActive(canonical);
    layers.refresh();
    statusBar.sync();
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
    pageRail.setActive(pageIndex);
    layers.refresh();
    statusBar.sync();
  };
  shell._syncVisibleReviewPage = syncVisibleReviewPage;
  viewport.addEventListener("scroll", () => {
    if (visiblePageFrame) return;
    visiblePageFrame = requestAnimationFrame(syncVisibleReviewPage);
  }, { passive: true, signal });
  const rerender = () => {
    if (shell._descriptors?.length) {
      shell._savedScroll = { top: viewport.scrollTop, left: viewport.scrollLeft };
    }
    captureSnapshot(shell);
    updateCompat();
    syncVariant();
    void renderStrip(shell, items, signal);
  };
  shell._rerender = rerender;
  signal.addEventListener("abort", () => captureSnapshot(shell), { once: true });

  image.addEventListener("pointerdown", (e) => {
    if (state.tool === "eyedropper" && e.button === 0) {
      const p = sourcePoint(image, e);
      const hex = sampleCanvasHex(p);
      if (!hex) return;
      e.preventDefault();
      e.stopPropagation();
      navigator.clipboard?.writeText?.(hex).catch(() => {});
      const id = window.editorState?.selectedTextObjectId;
      const pageIndex = Number(window.editorState?.activePageIndex ?? 0);
      const obj = id ? window.findTextObject?.(pageIndex, id) : null;
      if (obj) {
        obj.style = Object.assign({}, window.DEFAULT_TEXT_OBJECT_STYLE, obj.style || {});
        let targetLabel = "màu chữ";
        if (e.shiftKey) {
          obj.style.strokeColor = hex;
          if (obj.style.strokeWidth === "0") obj.style.strokeWidth = "2";
          targetLabel = "màu viền";
        } else if (e.altKey) {
          obj.style.bgColor = hex;
          targetLabel = "màu nền khung";
        } else {
          obj.style.color = hex;
        }
        window.scheduleTextObjectPersist?.(pageIndex, obj.id);
        window.renderEditorPanel?.(pageIndex);
        window.showToast?.(`Đã áp ${hex} vào ${targetLabel} (đã sao chép).`, "success");
      } else {
        window.showToast?.(`Đã hút màu ${hex} (đã sao chép vào bộ nhớ tạm).`, "info");
      }
      return;
    }
    if (state.variant !== "clean" || !["brush", "remove", "eraser"].includes(state.tool) || e.button !== 0 || busyFlag) { return; }
    const p = sourcePoint(image, e); if (!p) { return; } e.preventDefault(); painting = true; last = p; removeForceLama = e.shiftKey; image.setPointerCapture?.(e.pointerId); paintPoint(shell, p.x, p.y, radius, state.tool === "eraser");
  }, { signal });
  image.addEventListener("pointermove", (e) => {
    if (state.tool === "eyedropper") {
      brushCursor.hidden = true;
      const hex = sampleCanvasHex(sourcePoint(image, e));
      if (hex) {
        const sr = canvasPane.getBoundingClientRect();
        eyedropperBadge.hidden = false;
        eyedropperSwatch.style.background = hex;
        eyedropperLabel.textContent = hex.toUpperCase();
        eyedropperBadge.style.left = `${Math.min(sr.width - 96, Math.max(8, e.clientX - sr.left + 14))}px`;
        eyedropperBadge.style.top = `${Math.max(8, e.clientY - sr.top - 32)}px`;
      } else {
        eyedropperBadge.hidden = true;
      }
      return;
    }
    if (!eyedropperBadge.hidden) eyedropperBadge.hidden = true;
    if (state.variant === "clean" && ["brush", "remove", "eraser"].includes(state.tool)) {
      const sr = canvasPane.getBoundingClientRect();
      syncBrushCursorSize();
      brushCursor.hidden = false;
      brushCursor.classList.toggle("is-remove", state.tool === "remove");
      brushCursor.classList.toggle("is-eraser", state.tool === "eraser");
      brushCursor.style.left = `${e.clientX - sr.left}px`;
      brushCursor.style.top = `${e.clientY - sr.top}px`;
    } else if (!brushCursor.hidden) {
      brushCursor.hidden = true;
    }
    if (!painting || !last || !["brush", "remove", "eraser"].includes(state.tool)) { return; } const p = sourcePoint(image, e); if (!p) { return; } paintStroke(shell, last, p, radius, state.tool === "eraser"); last = p;
  }, { signal });
  image.addEventListener("pointerleave", () => { eyedropperBadge.hidden = true; brushCursor.hidden = true; }, { signal });

  const runBrushRepaint = async (choice) => {
    const opts = typeof choice === "string" ? { mode: choice } : (choice || { mode: "standard" });
    const mode = opts.mode || "standard";
    const chapterId = window.currentChapterId, chunks = shell._brushChunks || [];
    if (!chapterId || !chunks.some((c) => c.dirty && chunkHasPaint(c))) return false;
    busy(
      true,
      mode === "ai"
        ? "AI đang vẽ lại…"
        : mode === "lama"
          ? "LaMa đang xử lý…"
          : mode === "restore"
            ? "Đang khôi phục…"
            : "Đang làm sạch nhanh…",
    );
    try {
      let affected = 0;
      for (const desc of shell._descriptors || []) {
        const fullY1 = Math.max(0, desc.sourceY1 - desc.localY1);
        const fullY2 = fullY1 + desc.img.naturalHeight;
        if (!(shell._brushChunks || []).some((c) => chunkHasPaint(c, fullY1, fullY2))) continue;
        const mask = document.createElement("canvas"); mask.width = desc.img.naturalWidth; mask.height = desc.img.naturalHeight; drawBrushMask(mask.getContext("2d"), chunks, desc, mask.width);
        const form = new FormData();
        form.append("chapter_id", chapterId);
        form.append("page_index", desc.item.canonicalIndex);
        form.append("mode", mode);
        if (mode === "ai") {
          if (opts.providerId) form.append("provider_id", opts.providerId);
          if (opts.model) form.append("model", opts.model);
          if (opts.quality) form.append("quality", opts.quality);
        }
        form.append("mask", await canvasBlob(mask), "mask.png");
        if (signal.aborted || chapterId !== window.currentChapterId) return false;
        const response = await fetch("/api/repaint_mask", { method: "POST", body: form, signal }), parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({}))), data = await parse(response); if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
        if (signal.aborted || chapterId !== window.currentChapterId) return false;
        if (window.currentManifest?.pages?.[desc.item.canonicalIndex] && data.pages?.[desc.item.canonicalIndex]) { window.currentManifest.pages[desc.item.canonicalIndex] = data.pages[desc.item.canonicalIndex]; } affected++;
      }
      if (signal.aborted || chapterId !== window.currentChapterId) return false;
      for (const c of chunks) { if (c.canvas && c.ctx) { c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height); } c.dirty = false; } deleteSnapshot(snapshotKey());
      const doneMsg = mode === "restore"
        ? "Đã khôi phục ảnh gốc ở các vùng được đánh dấu."
        : mode === "ai"
          ? "AI đã vẽ lại các vùng được đánh dấu."
          : "Đã làm sạch các vùng được đánh dấu.";
      window.showToast?.(affected ? doneMsg : "Không có vùng ảnh nào được cập nhật.", affected ? "success" : "info");
      rerender();
      return affected > 0;
    } catch (err) {
      if (!signal.aborted && chapterId === window.currentChapterId) window.showToast?.("Không thể xử lý vùng đánh dấu: " + err.message, "error");
      return false;
    } finally {
      busy(false);
    }
  };

  const stopPaint = () => {
    const wasRemove = painting && state.tool === "remove";
    const forceLama = removeForceLama;
    painting = false; last = null; removeForceLama = false; syncBrushBar();
    if (wasRemove && !busyFlag) void runBrushRepaint(forceLama ? "lama" : "standard");
  };
  image.addEventListener("pointerup", stopPaint, { signal }); image.addEventListener("pointercancel", stopPaint, { signal });

  window.addEventListener("keydown", (e) => {
    const tag = e.target?.tagName?.toLowerCase(), editable = e.target?.isContentEditable || tag === "input" || tag === "textarea" || tag === "select"; if (editable) return;
    if (e.code === "Space" && !e.repeat) { space = true; return; }
    if (e.key === "\\" && !e.repeat) { e.preventDefault(); setPeekOriginal(true); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === "0") { e.preventDefault(); state.fitWidth = true; return applyZoom(shell); }
    if ((e.ctrlKey || e.metaKey) && e.key === "1") { e.preventDefault(); state.fitWidth = false; state.zoom = 100; return applyZoom(shell); }
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === "a" || e.key === "A") && state.variant === "clean") {
      e.preventDefault();
      const visible = Number(workspace.dataset.reviewCanonicalIndex);
      const onSlice = ownedObjects(shell).filter(({ pageIndex }) => pageIndex === visible).map(({ pageIndex, obj }) => ({ pageIndex, id: obj.id }));
      const targets = onSlice.length ? onSlice : ownedObjects(shell).map(({ pageIndex, obj }) => ({ pageIndex, id: obj.id }));
      if (targets.length) {
        selectMany(shell, targets);
        window.showToast?.(`Đã chọn ${targets.length} vùng chữ.`, "info");
      }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === "d" || e.key === "D") && getSelectedTargets(shell).length && state.variant === "clean") {
      e.preventDefault();
      void duplicateSelection();
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key) && getSelectedTargets(shell).length && state.variant === "clean") {
      const targets = getSelectedTargets(shell);
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1;
      const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
      const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
      const moveAll = () => {
        for (const { pageIndex, id, obj, desc } of targets) {
          const r = obj.region, w = r.x2 - r.x1, h = r.y2 - r.y1;
          const nx1 = Math.max(0, Math.min(Math.max(0, desc.img.naturalWidth - w), r.x1 + dx));
          const ny1 = Math.max(desc.localY1, Math.min(Math.max(desc.localY1, desc.localY2 - h), r.y1 + dy));
          if (nx1 !== r.x1 || ny1 !== r.y1) {
            obj.region = { x1: Math.round(nx1), y1: Math.round(ny1), x2: Math.round(nx1 + w), y2: Math.round(ny1 + h) };
            syncOverlay(shell, pageIndex, id);
            window.scheduleGeomPersist?.(pageIndex, id);
          }
        }
      };
      if (targets.length > 1 && window.editorHistory) {
        window.editorHistory.batch(targets.map((t) => ({ pageIndex: t.pageIndex, id: t.id })), moveAll);
      } else {
        window.editorHistory?.watch(targets[0].pageIndex, targets[0].obj);
        moveAll();
      }
      shell._syncContextBar?.();
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && (e.key === "q" || e.key === "Q")) { e.preventDefault(); state.showOverlays = !state.showOverlays; syncDisplayToggles(); return; }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && (e.key === "t" || e.key === "T")) { e.preventDefault(); state.livePreview = !state.livePreview; syncDisplayToggles(); return; }
    if (e.key === "[" && ["brush", "remove", "eraser"].includes(state.tool)) { e.preventDefault(); setBrushRadius(radius - 2); return; }
    if (e.key === "]" && ["brush", "remove", "eraser"].includes(state.tool)) { e.preventDefault(); setBrushRadius(radius + 2); return; }
    if ((e.key === "Delete" || e.key === "Backspace") && getSelectedTargets(shell).length) { e.preventDefault(); void deleteSelection(); return; }
    if (e.key === "Escape") { eyedropperBadge.hidden = true; brushCursor.hidden = true; if (state.tool !== "select") return setTool(shell, "select"); return clearSelection(shell); }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && SHORTCUTS[e.key.toLowerCase()]) { e.preventDefault(); eyedropperBadge.hidden = true; setTool(shell, SHORTCUTS[e.key.toLowerCase()]); }
  }, { signal });
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") { space = false; panning = false; pan = null; viewport.classList.remove("is-panning"); }
    if (e.key === "\\") setPeekOriginal(false);
  }, { signal });
  window.addEventListener("blur", () => setPeekOriginal(false), { signal });
  viewport.addEventListener("pointerdown", (e) => { if (!((e.button === 0 && (space || state.tool === "hand")) || e.button === 1)) { return; } panning = true; pan = { x: e.clientX, y: e.clientY, left: viewport.scrollLeft, top: viewport.scrollTop }; viewport.setPointerCapture?.(e.pointerId); viewport.classList.add("is-panning"); e.preventDefault(); }, { signal });
  viewport.addEventListener("pointermove", (e) => { if (!panning || !pan) { return; } viewport.scrollLeft = pan.left - (e.clientX - pan.x); viewport.scrollTop = pan.top - (e.clientY - pan.y); }, { signal });
  viewport.addEventListener("pointerup", () => { panning = false; pan = null; viewport.classList.remove("is-panning"); }, { signal });
  viewport.addEventListener("click", (e) => { if (state.tool !== "zoom" || e.target.closest("button,input,textarea,select")) { return; } const r = viewport.getBoundingClientRect(); stepZoom(e.altKey ? -1 : 1); applyZoom(shell, { x: e.clientX - r.left, y: e.clientY - r.top }); }, { signal });
  viewport.addEventListener("wheel", (e) => {
    if (e.altKey && ["brush", "remove", "eraser"].includes(state.tool)) {
      e.preventDefault();
      setBrushRadius(radius + (e.deltaY < 0 ? 2 : -2));
      return;
    }
    if (!(e.ctrlKey || e.metaKey)) { return; }
    e.preventDefault();
    const r = viewport.getBoundingClientRect();
    stepZoom(e.deltaY < 0 ? 1 : -1);
    applyZoom(shell, { x: e.clientX - r.left, y: e.clientY - r.top });
  }, { passive: false, signal });

  size.addEventListener("input", () => setBrushRadius(Number(size.value)), { signal });
  clearMask.addEventListener("click", () => { for (const c of shell._brushChunks || []) { if (c.canvas && c.ctx) { c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height); } c.dirty = false; } deleteSnapshot(snapshotKey()); syncBrushBar(); }, { signal });
  window.addEventListener("resize", () => { if (state.fitWidth) applyZoom(shell); }, { signal });

  submit.addEventListener("click", async () => {
    const chapterId = window.currentChapterId, chunks = shell._brushChunks || []; if (!chapterId || !chunks.some((c) => c.dirty && chunkHasPaint(c))) return window.showToast?.("Chưa có vùng nào được đánh dấu.", "error");
    const mode = typeof window.chooseRepaintMode === "function" ? await window.chooseRepaintMode() : "standard"; if (!mode || chapterId !== window.currentChapterId) { return; }
    await runBrushRepaint(mode);
  }, { signal });

  undoRepaint.addEventListener("click", async () => {
    const chapterId = window.currentChapterId, pageIndex = Number(workspace.dataset.reviewCanonicalIndex), page = window.currentManifest?.pages?.[pageIndex];
    if (!chapterId || !page) return;
    if (!page.manual_mask && !page.manual_lama_mask && !page.manual_ai_applied) return window.showToast?.(`Lát ${pageIndex + 1} chưa có vùng nào làm sạch bằng cọ.`, "info");
    if (!window.confirm(`Bỏ mọi vùng đã làm sạch bằng cọ trên lát ${pageIndex + 1}? Lát sẽ trở về kết quả xử lý tự động.`)) return;
    busy(true, "Đang khôi phục…");
    try {
      const response = await fetch("/api/reset_manual_mask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ chapter_id: chapterId, page_index: pageIndex }), signal });
      const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({}))), data = await parse(response);
      if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
      if (signal.aborted || chapterId !== window.currentChapterId) return;
      if (data.pages?.[pageIndex]) window.currentManifest.pages[pageIndex] = data.pages[pageIndex];
      window.showToast?.(`Đã khôi phục nền tự động trên lát ${pageIndex + 1}.`, "success"); rerender();
    } catch (err) { if (!signal.aborted && chapterId === window.currentChapterId) window.showToast?.("Không thể khôi phục nền tự động: " + err.message, "error"); } finally { busy(false); }
  }, { signal });

  
  if (window.openLetteredChapter && window.openLetteredChapter === window.currentChapterId) {
    window.openLetteredChapter = null;
  }
  state.variant = "clean";
  syncVariant(); setTool(shell, state.tool); rerender();
}

export function scan() { document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount); }

window.mountStitchInspector = scan;
