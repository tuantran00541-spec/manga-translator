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
let previewZoomScale = 1.0;
let previewDrawCleanup = null;
let previewResizeObserver = null;
let previewLastChapterId = null;

function cleanupPreviewDrawListeners() {
  previewResizeObserver?.disconnect();
  previewResizeObserver = null;
  if (typeof previewDrawCleanup === "function") {
    previewDrawCleanup();
    previewDrawCleanup = null;
  }
}
window.cleanupPreviewDrawListeners = cleanupPreviewDrawListeners;

function renderPreview() {
  const container = document.getElementById("page-view");
  if (!container || !currentManifest?.pages?.length) return;

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
  container.className = "preview-workspace process-workspace";

  pages.forEach((page) => {
    if (!page.excluded_regions) page.excluded_regions = [];
  });

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";
  toolbar.classList.add("process-commandbar");

  const heading = document.createElement("div");
  heading.className = "preview-workspace-heading";
  const title = document.createElement("div");
  title.className = "preview-workspace-title";
  title.textContent = `${pages.filter((item) => !item.skipped).length}/${pages.length} lát được chọn`;
  heading.appendChild(title);

  const processBtn = document.createElement("button");
  processBtn.className = "preview-primary-action";
  processBtn.textContent = "Bắt đầu xử lý";
  processBtn.addEventListener("click", processSelectedPages);
  toolbar.append(heading, processBtn);
  container.appendChild(toolbar);

  const layout = document.createElement("div");
  layout.className = "workbench-stage-grid process-workbench-grid";

  const navItems = pages.map((item, index) => ({
    key: index,
    label: pageLabel(pages, index),
    image: item.original,
    state: item.skipped ? "skipped" : (item.excluded_regions?.length ? "review" : "ready"),
    stateLabel: item.skipped ? "Bỏ qua" : (item.excluded_regions?.length ? `${item.excluded_regions.length} vùng loại trừ` : "Sẵn sàng"),
  }));
  const navigator = window.createPageNavigator({
    items: navItems,
    activeIndex: previewActivePageIndex,
    title: "Trang & lát",
    ariaLabel: "Điều hướng trang xử lý",
    onSelect: (index) => {
      previewActivePageIndex = index;
      renderPreview();
    },
  });

  const workspace = document.createElement("div");
  workspace.className = "preview-main workbench-canvas-column";
  const surface = document.createElement("section");
  surface.className = "preview-canvas-surface preview-card-active";
  surface.dataset.pageIndex = previewActivePageIndex;
  workspace.appendChild(surface);

  const inspector = document.createElement("aside");
  inspector.className = "context-inspector process-inspector";
  inspector.setAttribute("aria-label", "Thuộc tính trang xử lý");
  const inspectorHeading = document.createElement("div");
  inspectorHeading.className = "context-inspector-heading";
  const inspectorEyebrow = document.createElement("span");
  inspectorEyebrow.className = "ui-eyebrow";
  inspectorEyebrow.textContent = "Trang đang chọn";
  const inspectorTitle = document.createElement("strong");
  inspectorTitle.textContent = pageLabel(pages, previewActivePageIndex);
  inspectorHeading.append(inspectorEyebrow, inspectorTitle);
  inspector.appendChild(inspectorHeading);

  layout.append(navigator.element, workspace, inspector);
  container.appendChild(layout);

  const page = pages[previewActivePageIndex];
  renderPreviewPage(surface, page, previewActivePageIndex, pages, inspector);
  window.setupWorkbenchPanels?.("preview");

  if (typeof setWorkflowCheckpoint === "function") {
    setWorkflowCheckpoint("preview", previewActivePageIndex);
  }
}

function renderPreviewPage(card, page, pageIndex, pages, inspector = null) {
  const header = document.createElement("div");
  header.className = "preview-card-header";

  const labelWrap = document.createElement("div");
  labelWrap.className = "preview-page-label-wrap";
  const label = document.createElement("div");
  label.className = "preview-label";
  label.textContent = pageLabel(pages, pageIndex);
  const status = document.createElement("span");
  status.className = "preview-page-status";
  status.textContent = page.skipped ? "Đã bỏ qua" : "Sẵn sàng xử lý";
  labelWrap.appendChild(label);
  labelWrap.appendChild(status);
  header.appendChild(labelWrap);

  const zoomBar = document.createElement("div");
  zoomBar.className = "zoom-controls";
  const zoomOutBtn = document.createElement("button");
  zoomOutBtn.className = "zoom-btn";
  zoomOutBtn.textContent = "−";
  zoomOutBtn.title = "Thu nhỏ";
  const zoomLevelText = document.createElement("span");
  zoomLevelText.className = "zoom-level";
  const zoomInBtn = document.createElement("button");
  zoomInBtn.className = "zoom-btn";
  zoomInBtn.textContent = "+";
  zoomInBtn.title = "Phóng to";
  const zoomResetBtn = document.createElement("button");
  zoomResetBtn.className = "zoom-btn zoom-reset";
  zoomResetBtn.textContent = "1:1";
  zoomResetBtn.title = "Đặt lại zoom";
  zoomBar.append(zoomOutBtn, zoomLevelText, zoomInBtn, zoomResetBtn);
  header.appendChild(zoomBar);
  card.appendChild(header);

  const viewport = document.createElement("div");
  viewport.className = "preview-viewport";
  const imgWrap = document.createElement("div");
  imgWrap.className = "preview-image-wrap";
  const img = document.createElement("img");
  img.src = page.original;
  img.alt = pageLabel(pages, pageIndex);
  img.draggable = false;

  const overlayContainer = document.createElement("div");
  overlayContainer.className = "excluded-overlay-container";
  const drawLayer = document.createElement("div");
  drawLayer.className = "excluded-draw-layer";
  imgWrap.append(img, overlayContainer, drawLayer);
  viewport.appendChild(imgWrap);
  card.appendChild(viewport);

  previewZoomScale = 1.0;
  const updateZoom = (newScale) => {
    previewZoomScale = Math.max(0.5, Math.min(4.0, newScale));
    zoomLevelText.textContent = Math.round(previewZoomScale * 100) + "%";
    imgWrap.style.transform = `scale(${previewZoomScale})`;
  };
  updateZoom(1);
  zoomInBtn.addEventListener("click", () => updateZoom(previewZoomScale + 0.25));
  zoomOutBtn.addEventListener("click", () => updateZoom(previewZoomScale - 0.25));
  zoomResetBtn.addEventListener("click", () => updateZoom(1));
  viewport.addEventListener("wheel", (e) => {
    if (e.ctrlKey || card.classList.contains("draw-excluded-active")) {
      e.preventDefault();
      updateZoom(previewZoomScale + (e.deltaY < 0 ? 0.15 : -0.15));
    }
  }, { passive: false });

  const renderExcludedBoxes = () => {
    overlayContainer.innerHTML = "";
    if (!img.naturalWidth || !img.clientWidth) return;
    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;
    (page.excluded_regions || []).forEach((region, rIdx) => {
      const boxEl = document.createElement("div");
      boxEl.className = "excluded-region-box";
      boxEl.style.left = region.x1 * scaleX + "px";
      boxEl.style.top = region.y1 * scaleY + "px";
      boxEl.style.width = (region.x2 - region.x1) * scaleX + "px";
      boxEl.style.height = (region.y2 - region.y1) * scaleY + "px";

      const delBtn = document.createElement("button");
      delBtn.className = "excluded-region-del";
      delBtn.textContent = "×";
      delBtn.title = "Xóa vùng cấm này";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        page.excluded_regions.splice(rIdx, 1);
        void saveExcludedRegions(pageIndex, page.excluded_regions);
        renderExcludedBoxes();
      });
      boxEl.appendChild(delBtn);
      overlayContainer.appendChild(boxEl);
    });
  };
  if (img.complete && img.naturalWidth > 0) renderExcludedBoxes();
  else img.onload = renderExcludedBoxes;

  const tools = document.createElement("div");
  tools.className = "preview-tools";
  const drawToggleBtn = document.createElement("button");
  drawToggleBtn.className = "excluded-toggle-btn";
  drawToggleBtn.textContent = "Đánh dấu vùng loại trừ";
  const clearBtn = document.createElement("button");
  clearBtn.className = "excluded-clear-btn";
  clearBtn.textContent = "Xóa vùng loại trừ";
  clearBtn.title = "Xóa toàn bộ vùng loại trừ của lát ảnh này";

  drawToggleBtn.addEventListener("click", () => {
    const active = card.classList.toggle("draw-excluded-active");
    drawToggleBtn.textContent = active ? "Đang đánh dấu · Chọn để kết thúc" : "Đánh dấu vùng loại trừ";
    drawToggleBtn.classList.toggle("active", active);
  });
  clearBtn.addEventListener("click", () => {
    page.excluded_regions = [];
    void saveExcludedRegions(pageIndex, page.excluded_regions);
    renderExcludedBoxes();
  });
  tools.append(drawToggleBtn, clearBtn);
  (inspector || card).appendChild(tools);

  let isDragging = false;
  let startPos = null;
  let tempDrawBox = null;

  const removeDrawListeners = () => {
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
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

  drawLayer.addEventListener("mousedown", (e) => {
    if (!card.classList.contains("draw-excluded-active")) return;
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

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    previewDrawCleanup = stopDrawing;
  });

  const updateTempDrawBox = (x, y) => {
    if (!tempDrawBox || !startPos) return;
    const left = Math.min(startPos.x, x);
    const top = Math.min(startPos.y, y);
    tempDrawBox.style.left = left + "px";
    tempDrawBox.style.top = top + "px";
    tempDrawBox.style.width = Math.abs(x - startPos.x) + "px";
    tempDrawBox.style.height = Math.abs(y - startPos.y) + "px";
  };

  const onMouseMove = (e) => {
    if (!isDragging || !tempDrawBox) return;
    const rect = imgWrap.getBoundingClientRect();
    updateTempDrawBox(
      (e.clientX - rect.left) / previewZoomScale,
      (e.clientY - rect.top) / previewZoomScale
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

    if (w < 5 || h < 5) return;

    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;
    page.excluded_regions.push({
      x1: Math.round(left * scaleX),
      y1: Math.round(top * scaleY),
      x2: Math.round((left + w) * scaleX),
      y2: Math.round((top + h) * scaleY)
    });
    void saveExcludedRegions(pageIndex, page.excluded_regions);
    renderExcludedBoxes();
  };

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

  const footer = document.createElement("div");
  footer.className = "preview-card-footer";
  const skipBtn = document.createElement("button");
  skipBtn.className = "skip-btn";
  skipBtn.textContent = page.skipped ? "Đã bỏ qua · Chọn để khôi phục" : "Bỏ qua lát ảnh";
  skipBtn.addEventListener("click", async () => {
    await toggleSkip(pageIndex, card, skipBtn);
    status.textContent = page.skipped ? "Đã bỏ qua" : "Sẵn sàng xử lý";
    renderPreview();
  });
  footer.appendChild(skipBtn);
  (inspector || card).appendChild(footer);
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
