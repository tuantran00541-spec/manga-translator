const pageCountCache = new WeakMap();

function pageCounts(pages) {
  if (!Array.isArray(pages)) return new Map();
  let counts = pageCountCache.get(pages);
  if (counts) return counts;
  counts = new Map();
  pages.forEach((page) => {
    const source = Number.isInteger(page?.source_page) ? page.source_page : -1;
    counts.set(source, (counts.get(source) || 0) + 1);
  });
  pageCountCache.set(pages, counts);
  return counts;
}

function pageLabel(pages, pageIndex) {
  const page = Array.isArray(pages) ? pages[pageIndex] : null;
  if (!page) return `Trang ${Number(pageIndex) + 1}`;
  const source = Number.isInteger(page.source_page) ? page.source_page : pageIndex;
  const total = pageCounts(pages).get(source) || 1;
  if (total <= 1) return `Trang ${source + 1}`;
  const slice = Number.isInteger(page.slice_index) ? page.slice_index : 0;
  return `Trang ${source + 1} · Lát ${slice + 1}/${total}`;
}

let previewActivePageIndex = 0;

Object.defineProperty(window, "previewActivePageIndex", {
  configurable: true,
  get: () => previewActivePageIndex,
  set: (value) => { previewActivePageIndex = value; },
});

let previewZoomScale = 1.0;
let previewDrawCleanup = null;
let previewResizeObserver = null;
let previewLastChapterId = null;
let previewRailCollapsed = false;

function cleanupPreviewDrawListeners() {
  previewResizeObserver?.disconnect();
  previewResizeObserver = null;
  if (typeof previewDrawCleanup === "function") {
    previewDrawCleanup();
    previewDrawCleanup = null;
  }
}
window.cleanupPreviewDrawListeners = cleanupPreviewDrawListeners;

function pageStateSummary(page) {
  if (page.skipped) return { code: "skipped", label: "Đã bỏ qua" };
  if (page.process_required) return { code: "review", label: "Cần xử lý" };
  if (page.preserve_regions?.length) return { code: "review", label: `${page.preserve_regions.length} vùng giữ nguyên` };
  return { code: "ready", label: "Sẵn sàng" };
}

function renderPreview() {
  const container = document.getElementById("page-view");
  if (!container || !currentManifest?.pages?.length) return;
  window.setAppStage?.("preview");

  if (window.currentChapterId && previewLastChapterId !== window.currentChapterId) {
    previewLastChapterId = window.currentChapterId;
    previewActivePageIndex = 0;
  }

  if (window.initialPreviewCanonicalPageIndex !== undefined && window.initialPreviewCanonicalPageIndex !== null) {
    const requestedIndex = parseInt(window.initialPreviewCanonicalPageIndex, 10);
    if (Number.isFinite(requestedIndex)) previewActivePageIndex = requestedIndex;
    window.initialPreviewCanonicalPageIndex = null;
  }

  cleanupPreviewDrawListeners();

  const pages = currentManifest.pages;
  previewActivePageIndex = Math.max(0, Math.min(previewActivePageIndex, pages.length - 1));
  container.innerHTML = "";
  container.className = "preview-workspace process-workspace koharu-process-workspace";

  pages.forEach((page) => {
    if (!page.preserve_regions) page.preserve_regions = [];
  });

  const activeCount = pages.filter((item) => !item.skipped).length;
  const page = pages[previewActivePageIndex];
  const stateInfo = pageStateSummary(page);

  const shell = document.createElement("section");
  shell.className = `koharu-process-shell${previewRailCollapsed ? " koharu-rail-collapsed" : ""}`;

  // 1. Left Koharu PageRail
  const pageRail = document.createElement("aside");
  pageRail.className = "koharu-page-rail";
  pageRail.setAttribute("aria-label", "Điều hướng trang xử lý");

  const railHead = document.createElement("header");
  railHead.className = "koharu-rail-head";
  const railTitleWrap = document.createElement("div");
  railTitleWrap.className = "koharu-rail-title-wrap";
  const railTitle = document.createElement("h2");
  railTitle.className = "koharu-rail-title";
  railTitle.textContent = "Trang & lát";
  const railCount = document.createElement("span");
  railCount.className = "koharu-count-pill";
  railCount.textContent = String(pages.length);
  railTitleWrap.append(railTitle, railCount);
  railHead.appendChild(railTitleWrap);

  const railSearchWrap = document.createElement("div");
  railSearchWrap.className = "koharu-rail-search-wrap";
  const railFilter = document.createElement("input");
  railFilter.type = "search";
  railFilter.className = "ui-input koharu-rail-filter";
  railFilter.placeholder = "Lọc trang…";
  railFilter.setAttribute("aria-label", "Lọc danh sách trang xử lý");
  railSearchWrap.appendChild(railFilter);

  const railList = document.createElement("div");
  railList.className = "koharu-page-list";

  const cardRefs = [];
  pages.forEach((item, index) => {
    const info = pageStateSummary(item);
    const labelText = pageLabel(pages, index);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `koharu-page-card${index === previewActivePageIndex ? " is-active" : ""}${item.skipped ? " is-skipped" : ""}`;
    card.dataset.pageIndex = String(index);
    card.dataset.filterText = `${labelText} ${index + 1} ${info.label}`.toLowerCase();

    const thumbBox = document.createElement("span");
    thumbBox.className = "koharu-page-thumb";
    if (item.original) {
      const thumbImg = document.createElement("img");
      thumbImg.src = item.original;
      thumbImg.alt = labelText;
      thumbImg.loading = "lazy";
      thumbImg.decoding = "async";
      thumbImg.draggable = false;
      thumbBox.appendChild(thumbImg);
    } else {
      thumbBox.textContent = String(index + 1);
    }

    const metaWrap = document.createElement("span");
    metaWrap.className = "koharu-page-info";
    const nameEl = document.createElement("strong");
    nameEl.className = "koharu-page-name";
    nameEl.textContent = labelText;
    const subWrap = document.createElement("span");
    subWrap.className = "koharu-page-meta-slot";
    const subState = document.createElement("span");
    subState.className = "koharu-page-sub";
    subState.textContent = info.label;
    subWrap.appendChild(subState);
    if (item.width && item.height) {
      const dims = document.createElement("span");
      dims.className = "koharu-page-dims";
      dims.textContent = `${item.width} × ${item.height}`;
      subWrap.appendChild(dims);
    }
    metaWrap.append(nameEl, subWrap);

    card.append(thumbBox, metaWrap);
    card.addEventListener("click", () => {
      previewActivePageIndex = index;
      renderPreview();
    });
    railList.appendChild(card);
    cardRefs.push(card);
  });

  railFilter.addEventListener("input", () => {
    const q = railFilter.value.trim().toLowerCase();
    cardRefs.forEach((card) => {
      card.hidden = Boolean(q && !card.dataset.filterText.includes(q));
    });
  });

  pageRail.append(railHead, railSearchWrap, railList);

  // 2. Center Koharu Canvas Pane
  const canvasPane = document.createElement("div");
  canvasPane.className = "koharu-canvas-pane koharu-process-canvas";

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";
  toolbar.className = "review-stitched-toolbar review-document-toolbar review-document-toolbar-compact koharu-docbar koharu-process-topbar process-commandbar";

  const topLeft = document.createElement("div");
  topLeft.className = "review-docbar-group review-docbar-left";
  const topRight = document.createElement("div");
  topRight.className = "review-docbar-group review-docbar-right";

  const railToggle = document.createElement("button");
  railToggle.type = "button";
  railToggle.className = `ui-btn ui-btn-ghost ui-btn-compact koharu-rail-toggle${previewRailCollapsed ? "" : " active"}`;
  railToggle.setAttribute("aria-label", "Ẩn/hiện danh sách trang");
  railToggle.setAttribute("aria-pressed", String(!previewRailCollapsed));
  railToggle.title = "Ẩn / hiện cột trang bên trái";
  railToggle.append(window.createUiIcon("panel-left"));
  railToggle.addEventListener("click", () => {
    previewRailCollapsed = shell.classList.toggle("koharu-rail-collapsed");
    railToggle.classList.toggle("active", !previewRailCollapsed);
    railToggle.setAttribute("aria-pressed", String(!previewRailCollapsed));
  });

  const sliceTitle = document.createElement("strong");
  sliceTitle.className = "koharu-process-slice-title";
  sliceTitle.textContent = pageLabel(pages, previewActivePageIndex);

  const statusPill = document.createElement("span");
  statusPill.className = "koharu-status-pill";
  statusPill.textContent = stateInfo.label;

  topLeft.append(railToggle, sliceTitle, statusPill);

  const topDrawBtn = document.createElement("button");
  topDrawBtn.type = "button";
  topDrawBtn.className = "ui-btn ui-btn-ghost ui-btn-compact excluded-toggle-btn";
  topDrawBtn.append(window.createUiIcon("rect-select"));
  const topDrawLabel = document.createElement("span");
  topDrawLabel.textContent = "Đánh dấu vùng giữ nguyên";
  topDrawBtn.appendChild(topDrawLabel);

  const topClearBtn = document.createElement("button");
  topClearBtn.type = "button";
  topClearBtn.className = "ui-btn ui-btn-ghost ui-btn-compact excluded-clear-btn";
  topClearBtn.textContent = "Xóa vùng";
  topClearBtn.title = "Xóa toàn bộ vùng giữ nguyên của lát ảnh này";

  const primaryActionBtn = document.createElement("button");
  primaryActionBtn.type = "button";
  primaryActionBtn.className = "ui-btn ui-btn-primary ui-btn-compact preview-primary-action";
  primaryActionBtn.textContent = "Bắt đầu xử lý";
  primaryActionBtn.addEventListener("click", () => {
    if (typeof window.processSelectedPages === "function") {
      void window.processSelectedPages();
    }
  });

  topRight.append(topDrawBtn, topClearBtn, primaryActionBtn);
  toolbar.append(topLeft, topRight);

  const viewport = document.createElement("div");
  viewport.className = "preview-viewport koharu-process-viewport";

  const surface = document.createElement("div");
  surface.className = "preview-canvas-surface preview-card-active koharu-process-surface";
  surface.dataset.pageIndex = String(previewActivePageIndex);

  const imgWrap = document.createElement("div");
  imgWrap.className = "preview-image-wrap";
  const img = document.createElement("img");
  img.src = page.original;
  img.alt = pageLabel(pages, previewActivePageIndex);
  img.draggable = false;

  const overlayContainer = document.createElement("div");
  overlayContainer.className = "excluded-overlay-container";
  const drawLayer = document.createElement("div");
  drawLayer.className = "excluded-draw-layer";
  imgWrap.append(img, overlayContainer, drawLayer);
  surface.appendChild(imgWrap);
  viewport.appendChild(surface);

  // Bottom 32px Koharu StatusBar
  const statusBar = document.createElement("footer");
  statusBar.className = "koharu-status-bar";
  statusBar.setAttribute("aria-label", "Thanh trạng thái và thu phóng");

  const zoomGroup = document.createElement("div");
  zoomGroup.className = "koharu-status-zoom";

  const zoomOutBtn = document.createElement("button");
  zoomOutBtn.type = "button";
  zoomOutBtn.className = "koharu-status-btn";
  zoomOutBtn.setAttribute("aria-label", "Thu nhỏ");
  zoomOutBtn.title = "Thu nhỏ";
  zoomOutBtn.append(window.createUiIcon("minus"));

  const zoomSlider = document.createElement("input");
  zoomSlider.type = "range";
  zoomSlider.min = "50";
  zoomSlider.max = "400";
  zoomSlider.step = "5";
  zoomSlider.value = "100";
  zoomSlider.className = "koharu-zoom-slider";
  zoomSlider.setAttribute("aria-label", "Mức thu phóng");

  const zoomInBtn = document.createElement("button");
  zoomInBtn.type = "button";
  zoomInBtn.className = "koharu-status-btn";
  zoomInBtn.setAttribute("aria-label", "Phóng to");
  zoomInBtn.title = "Phóng to";
  zoomInBtn.append(window.createUiIcon("plus"));

  const zoomLevelBtn = document.createElement("button");
  zoomLevelBtn.type = "button";
  zoomLevelBtn.className = "koharu-status-btn zoom-level";
  zoomLevelBtn.textContent = "100%";
  zoomLevelBtn.title = "Đặt lại thu phóng";

  const zoomResetBtn = document.createElement("button");
  zoomResetBtn.type = "button";
  zoomResetBtn.className = "koharu-status-btn zoom-reset";
  zoomResetBtn.textContent = "1:1";
  zoomResetBtn.title = "Kích thước 100%";

  zoomGroup.append(zoomOutBtn, zoomSlider, zoomInBtn, zoomLevelBtn, zoomResetBtn);

  const statusSpacer = document.createElement("div");
  statusSpacer.className = "koharu-status-spacer";

  const statusInfo = document.createElement("div");
  statusInfo.className = "koharu-status-info";
  const selectedStat = document.createElement("span");
  selectedStat.className = "koharu-status-stat preview-workspace-title";
  selectedStat.textContent = `${activeCount}/${pages.length} lát được chọn`;
  const dimsStat = document.createElement("span");
  dimsStat.className = "koharu-status-dims";
  dimsStat.textContent = page.width && page.height ? `${page.width} × ${page.height} px` : "";
  statusInfo.append(selectedStat, dimsStat);

  statusBar.append(zoomGroup, statusSpacer, statusInfo);
  canvasPane.append(toolbar, viewport, statusBar);

  // 3. Right Docked Koharu Sidebar
  const inspector = document.createElement("aside");
  inspector.className = "koharu-right-sidebar context-inspector process-inspector";
  inspector.setAttribute("aria-label", "Thuộc tính trang xử lý");

  const sidebarHead = document.createElement("div");
  sidebarHead.className = "koharu-sidebar-single-head";
  sidebarHead.append(window.createUiIcon("sliders", "koharu-tab-icon"));
  const sidebarTitle = document.createElement("strong");
  sidebarTitle.textContent = "Thuộc tính lát ảnh";
  sidebarHead.appendChild(sidebarTitle);

  const sidebarBody = document.createElement("div");
  sidebarBody.className = "koharu-process-sidebar-body";

  // Section A: Slice Status & Skip
  const statusSec = document.createElement("section");
  statusSec.className = "koharu-process-section";
  const statusSecTitle = document.createElement("h3");
  statusSecTitle.className = "koharu-section-eyebrow";
  statusSecTitle.textContent = "Trạng thái lát";

  const metaCard = document.createElement("div");
  metaCard.className = "koharu-process-meta-card";
  const metaName = document.createElement("strong");
  metaName.textContent = pageLabel(pages, previewActivePageIndex);
  const metaDims = document.createElement("span");
  metaDims.className = "koharu-process-meta-sub";
  metaDims.textContent = page.width && page.height
    ? `${page.width} × ${page.height} px · ${stateInfo.label}`
    : stateInfo.label;
  metaCard.append(metaName, metaDims);

  const skipBtn = document.createElement("button");
  skipBtn.type = "button";
  skipBtn.className = `ui-btn ${page.skipped ? "ui-btn-primary" : "ui-btn-ghost"} skip-btn koharu-full-btn`;
  skipBtn.textContent = page.skipped ? "Đã bỏ qua · Chọn để khôi phục" : "Bỏ qua lát ảnh";
  skipBtn.addEventListener("click", async () => {
    await toggleSkip(previewActivePageIndex, surface, skipBtn);
    renderPreview();
  });

  statusSec.append(statusSecTitle, metaCard, skipBtn);

  // Section B: Preserved Regions (Koharu Layers style)
  const regionsSec = document.createElement("section");
  regionsSec.className = "koharu-layers-section koharu-process-regions";
  const regionsHead = document.createElement("div");
  regionsHead.className = "koharu-layers-head";
  const regionsTitleWrap = document.createElement("div");
  regionsTitleWrap.className = "koharu-layers-title-wrap";
  regionsTitleWrap.append(window.createUiIcon("layers", "koharu-layers-icon"));
  const regionsTitle = document.createElement("strong");
  regionsTitle.textContent = "Vùng giữ nguyên";
  const regionsCountPill = document.createElement("span");
  regionsCountPill.className = "koharu-count-pill";
  regionsCountPill.textContent = String(page.preserve_regions.length);
  regionsTitleWrap.append(regionsTitle, regionsCountPill);
  regionsHead.appendChild(regionsTitleWrap);

  const regionsHint = document.createElement("p");
  regionsHint.className = "koharu-process-hint";
  regionsHint.textContent = "Khoanh những vùng tranh hoặc hiệu ứng muốn giữ nguyên không bị AI tẩy mất khi xử lý.";

  const regionsTools = document.createElement("div");
  regionsTools.className = "preview-tools koharu-process-tool-row";
  const sideDrawBtn = document.createElement("button");
  sideDrawBtn.type = "button";
  sideDrawBtn.className = "ui-btn ui-btn-ghost ui-btn-compact excluded-toggle-btn";
  sideDrawBtn.textContent = "Đánh dấu vùng";
  const sideClearBtn = document.createElement("button");
  sideClearBtn.type = "button";
  sideClearBtn.className = "ui-btn ui-btn-ghost ui-btn-compact excluded-clear-btn";
  sideClearBtn.textContent = "Xóa hết";
  regionsTools.append(sideDrawBtn, sideClearBtn);

  const regionsList = document.createElement("div");
  regionsList.className = "koharu-layers-list";

  regionsSec.append(regionsHead, regionsHint, regionsTools, regionsList);
  sidebarBody.append(statusSec, regionsSec);
  inspector.append(sidebarHead, sidebarBody);

  shell.append(pageRail, canvasPane, inspector);
  container.appendChild(shell);

  // Zoom wiring
  previewZoomScale = 1.0;
  const updateZoom = (newScale) => {
    previewZoomScale = Math.max(0.5, Math.min(4.0, Number(newScale) || 1));
    const pct = Math.round(previewZoomScale * 100);
    zoomLevelBtn.textContent = `${pct}%`;
    zoomSlider.value = String(pct);
    imgWrap.style.transform = `scale(${previewZoomScale})`;
  };
  updateZoom(1);

  zoomInBtn.addEventListener("click", () => updateZoom(previewZoomScale + 0.25));
  zoomOutBtn.addEventListener("click", () => updateZoom(previewZoomScale - 0.25));
  zoomResetBtn.addEventListener("click", () => updateZoom(1));
  zoomLevelBtn.addEventListener("click", () => updateZoom(1));
  zoomSlider.addEventListener("input", () => updateZoom(Number(zoomSlider.value) / 100));

  viewport.addEventListener("wheel", (e) => {
    if (e.ctrlKey || surface.classList.contains("draw-excluded-active")) {
      e.preventDefault();
      updateZoom(previewZoomScale + (e.deltaY < 0 ? 0.15 : -0.15));
    }
  }, { passive: false });

  // Draw-mode sync between top command bar and sidebar button
  const setDrawActive = (active) => {
    surface.classList.toggle("draw-excluded-active", active);
    topDrawBtn.classList.toggle("active", active);
    topDrawBtn.classList.toggle("ui-btn-primary", active);
    topDrawLabel.textContent = active ? "Đang đánh dấu · Xong" : "Đánh dấu vùng giữ nguyên";
    sideDrawBtn.classList.toggle("active", active);
    sideDrawBtn.classList.toggle("ui-btn-primary", active);
    sideDrawBtn.textContent = active ? "Kết thúc đánh dấu" : "Đánh dấu vùng";
  };

  const toggleDraw = () => setDrawActive(!surface.classList.contains("draw-excluded-active"));
  topDrawBtn.addEventListener("click", toggleDraw);
  sideDrawBtn.addEventListener("click", toggleDraw);

  const renderExcludedBoxes = () => {
    overlayContainer.innerHTML = "";
    const regions = page.preserve_regions || [];
    regionsCountPill.textContent = String(regions.length);
    topClearBtn.disabled = regions.length === 0;
    sideClearBtn.disabled = regions.length === 0;

    if (img.naturalWidth && img.naturalHeight) {
      dimsStat.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
    }

    // Rebuild right-sidebar layer list
    regionsList.replaceChildren();
    if (!regions.length) {
      const empty = document.createElement("div");
      empty.className = "koharu-layers-empty";
      empty.textContent = "Chưa có vùng giữ nguyên nào trên lát này.";
      regionsList.appendChild(empty);
    } else {
      regions.forEach((region, rIdx) => {
        const w = Math.max(0, Math.round((region.x2 || 0) - (region.x1 || 0)));
        const h = Math.max(0, Math.round((region.y2 || 0) - (region.y1 || 0)));
        const row = document.createElement("div");
        row.className = "koharu-layer-row";

        const main = document.createElement("div");
        main.className = "koharu-layer-main";
        main.append(window.createUiIcon("rect-select", "koharu-layer-kind-icon"));
        const copy = document.createElement("span");
        copy.className = "koharu-layer-copy";
        const nameSpan = document.createElement("span");
        nameSpan.className = "koharu-layer-name";
        nameSpan.textContent = `Vùng giữ nguyên ${rIdx + 1}`;
        const subSpan = document.createElement("span");
        subSpan.className = "koharu-layer-sub";
        subSpan.textContent = `${w} × ${h} px · (${region.x1}, ${region.y1})`;
        copy.append(nameSpan, subSpan);
        main.appendChild(copy);

        const del = document.createElement("button");
        del.type = "button";
        del.className = "koharu-layer-del";
        del.title = "Xóa vùng giữ nguyên này";
        del.setAttribute("aria-label", `Xóa vùng giữ nguyên ${rIdx + 1}`);
        del.append(window.createUiIcon("trash"));
        del.addEventListener("click", () => {
          page.preserve_regions.splice(rIdx, 1);
          void savePreserveRegions(previewActivePageIndex, page.preserve_regions);
          renderExcludedBoxes();
        });

        row.append(main, del);
        regionsList.appendChild(row);
      });
    }

    if (!img.naturalWidth || !img.clientWidth) return;
    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;
    regions.forEach((region, rIdx) => {
      const boxEl = document.createElement("div");
      boxEl.className = "excluded-region-box";
      boxEl.style.left = `${region.x1 * scaleX}px`;
      boxEl.style.top = `${region.y1 * scaleY}px`;
      boxEl.style.width = `${(region.x2 - region.x1) * scaleX}px`;
      boxEl.style.height = `${(region.y2 - region.y1) * scaleY}px`;

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "ui-icon-btn ui-btn-compact ui-btn-danger excluded-region-del";
      delBtn.append(window.createUiIcon("trash"));
      delBtn.title = "Xóa vùng giữ nguyên này";
      delBtn.setAttribute("aria-label", "Xóa vùng giữ nguyên này");
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        page.preserve_regions.splice(rIdx, 1);
        void savePreserveRegions(previewActivePageIndex, page.preserve_regions);
        renderExcludedBoxes();
      });
      boxEl.appendChild(delBtn);
      overlayContainer.appendChild(boxEl);
    });
  };

  const clearAllRegions = () => {
    page.preserve_regions = [];
    void savePreserveRegions(previewActivePageIndex, page.preserve_regions);
    renderExcludedBoxes();
  };
  topClearBtn.addEventListener("click", clearAllRegions);
  sideClearBtn.addEventListener("click", clearAllRegions);

  if (img.complete && img.naturalWidth > 0) renderExcludedBoxes();
  else img.onload = renderExcludedBoxes;

  // Preserve region drawing interaction
  let isDragging = false;
  let startPos = null;
  let tempDrawBox = null;

  const removeDrawListeners = () => {
    window.removeEventListener("pointermove", onMouseMove);
    window.removeEventListener("pointerup", onMouseUp);
    window.removeEventListener("pointercancel", onMouseUp);
  };

  const stopDrawing = () => {
    removeDrawListeners();
    isDragging = false;
    startPos = null;
    if (tempDrawBox) {
      tempDrawBox.remove();
      tempDrawBox = null;
    }
  };

  const updateTempDrawBox = (x, y) => {
    if (!tempDrawBox || !startPos) return;
    const left = Math.min(startPos.x, x);
    const top = Math.min(startPos.y, y);
    tempDrawBox.style.left = `${left}px`;
    tempDrawBox.style.top = `${top}px`;
    tempDrawBox.style.width = `${Math.abs(x - startPos.x)}px`;
    tempDrawBox.style.height = `${Math.abs(y - startPos.y)}px`;
  };

  const onMouseMove = (e) => {
    if (!isDragging || !tempDrawBox) return;
    const rect = imgWrap.getBoundingClientRect();
    updateTempDrawBox(
      (e.clientX - rect.left) / previewZoomScale,
      (e.clientY - rect.top) / previewZoomScale,
    );
  };

  const onMouseUp = () => {
    if (!isDragging || !tempDrawBox) {
      stopDrawing();
      return;
    }
    removeDrawListeners();
    isDragging = false;
    const left = parseFloat(tempDrawBox.style.left) || 0;
    const top = parseFloat(tempDrawBox.style.top) || 0;
    const w = parseFloat(tempDrawBox.style.width) || 0;
    const h = parseFloat(tempDrawBox.style.height) || 0;
    tempDrawBox.remove();
    tempDrawBox = null;
    startPos = null;
    previewDrawCleanup = null;

    if (w < 5 || h < 5 || !img.clientWidth || !img.clientHeight) return;

    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;
    page.preserve_regions.push({
      x1: Math.round(left * scaleX),
      y1: Math.round(top * scaleY),
      x2: Math.round((left + w) * scaleX),
      y2: Math.round((top + h) * scaleY),
    });
    void savePreserveRegions(previewActivePageIndex, page.preserve_regions);
    renderExcludedBoxes();
  };

  drawLayer.style.touchAction = "none";
  drawLayer.addEventListener("pointerdown", (e) => {
    if (!surface.classList.contains("draw-excluded-active")) return;
    e.preventDefault();
    stopDrawing();

    const rect = imgWrap.getBoundingClientRect();
    const x = (e.clientX - rect.left) / previewZoomScale;
    const y = (e.clientY - rect.top) / previewZoomScale;
    isDragging = true;
    startPos = { x, y };
    tempDrawBox = document.createElement("div");
    tempDrawBox.className = "excluded-region-box drawing";
    overlayContainer.appendChild(tempDrawBox);
    updateTempDrawBox(x, y);

    drawLayer.setPointerCapture?.(e.pointerId);
    window.addEventListener("pointermove", onMouseMove);
    window.addEventListener("pointerup", onMouseUp);
    window.addEventListener("pointercancel", onMouseUp);
    previewDrawCleanup = stopDrawing;
  });

  previewDrawCleanup = stopDrawing;

  if (typeof ResizeObserver === "function") {
    previewResizeObserver = new ResizeObserver(() => {
      if (!img.isConnected) {
        previewResizeObserver?.disconnect();
        return;
      }
      stopDrawing();
      renderExcludedBoxes();
    });
    previewResizeObserver.observe(img);
  }

  // Ensure active card is visible in left rail
  cardRefs[previewActivePageIndex]?.scrollIntoView({ block: "nearest" });

  if (typeof setWorkflowCheckpoint === "function") {
    setWorkflowCheckpoint("preview", previewActivePageIndex);
  }
}

function isPreviewEditingText() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName ? el.tagName.toLowerCase() : "";
  return tag === "textarea" || tag === "input" || tag === "select" || el.isContentEditable;
}

document.addEventListener("keydown", (e) => {
  const pageView = document.getElementById("page-view");
  if (!pageView || !pageView.classList.contains("preview-workspace")) return;
  if (isPreviewEditingText()) return;
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "PageUp" && e.key !== "PageDown") return;
  const pages = currentManifest ? currentManifest.pages : null;
  if (!pages || pages.length === 0) return;
  if ((e.key === "ArrowLeft" || e.key === "PageUp") && previewActivePageIndex > 0) {
    e.preventDefault();
    previewActivePageIndex -= 1;
    renderPreview();
  } else if ((e.key === "ArrowRight" || e.key === "PageDown") && previewActivePageIndex < pages.length - 1) {
    e.preventDefault();
    previewActivePageIndex += 1;
    renderPreview();
  }
});
