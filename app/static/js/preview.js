function pageLabel(pages, pageIndex) {

  const page = pages[pageIndex];
  const total = pages.filter((p) => p.source_page === page.source_page).length;
  if (total <= 1) return "Trang " + (page.source_page + 1);
  return "Trang " + (page.source_page + 1) + " - Lát " + (page.slice_index + 1) + "/" + total;
}

let previewActivePageIndex = 0;
let previewZoomScale = 1.0;
let previewDrawCleanup = null;
let previewLastChapterId = null;

function cleanupPreviewDrawListeners() {
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

  cleanupPreviewDrawListeners();

  const pages = currentManifest.pages;
  previewActivePageIndex = Math.max(0, Math.min(previewActivePageIndex, pages.length - 1));
  container.innerHTML = "";
  container.className = "preview-workspace";

  pages.forEach((page) => {
    if (!page.excluded_regions) page.excluded_regions = [];
  });

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";

  const heading = document.createElement("div");
  heading.className = "preview-workspace-heading";
  const title = document.createElement("div");
  title.className = "preview-workspace-title";
  title.textContent = "Xem trước & chọn lát";
  const subtitle = document.createElement("div");
  subtitle.className = "preview-workspace-subtitle";
  subtitle.textContent = "Xem từng lát một để kiểm tra ảnh trước khi xử lý.";
  heading.appendChild(title);
  heading.appendChild(subtitle);

  const processBtn = document.createElement("button");
  processBtn.className = "preview-primary-action";
  processBtn.textContent = "Xử lý các trang đã chọn";
  processBtn.addEventListener("click", processSelectedPages);

  toolbar.appendChild(heading);
  toolbar.appendChild(processBtn);
  container.appendChild(toolbar);

  const navigation = document.createElement("nav");
  navigation.className = "preview-navigation workspace-nav-bar";
  navigation.setAttribute("aria-label", "Điều hướng trang xem trước");

  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.className = "preview-nav-btn workspace-nav-btn";
  prevBtn.textContent = "← Trước";
  prevBtn.setAttribute("aria-label", "Trang trước");
  prevBtn.disabled = previewActivePageIndex === 0;
  prevBtn.addEventListener("click", () => {
    if (previewActivePageIndex <= 0) return;
    previewActivePageIndex -= 1;
    if (typeof setWorkflowCheckpoint === "function") {
      setWorkflowCheckpoint("preview", previewActivePageIndex);
    }
    renderPreview();
  });

  const position = document.createElement("div");
  position.className = "preview-page-position workspace-nav-position";
  position.setAttribute("aria-live", "polite");

  const jumpWrap = document.createElement("label");
  jumpWrap.className = "workspace-nav-jump-wrap";
  jumpWrap.textContent = "Trang ";

  const jumpInput = document.createElement("input");
  jumpInput.type = "number";
  jumpInput.min = "1";
  jumpInput.max = String(pages.length);
  jumpInput.value = String(previewActivePageIndex + 1);
  jumpInput.className = "workspace-nav-jump-input";
  jumpInput.setAttribute("aria-label", "Nhảy tới số trang");

  const doJump = () => {
    const rawVal = jumpInput.value.trim();
    if (!rawVal) {
      jumpInput.value = String(previewActivePageIndex + 1);
      return;
    }
    const val = parseInt(rawVal, 10);
    if (!Number.isFinite(val) || val < 1 || val > pages.length) {
      jumpInput.value = String(previewActivePageIndex + 1);
      return;
    }
    if (val - 1 !== previewActivePageIndex) {
      previewActivePageIndex = val - 1;
      if (typeof setWorkflowCheckpoint === "function") {
        setWorkflowCheckpoint("preview", previewActivePageIndex);
      }
      renderPreview();
    }
  };

  jumpInput.addEventListener("change", doJump);
  jumpInput.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") {
      e.preventDefault();
      doJump();
    }
  });

  jumpWrap.appendChild(jumpInput);

  const totalText = document.createElement("span");
  totalText.textContent = ` / ${pages.length}`;

  const labelSpan = document.createElement("span");
  labelSpan.textContent = ` · ${pageLabel(pages, previewActivePageIndex)}`;

  position.append(jumpWrap, totalText, labelSpan);

  const pageItem = pages[previewActivePageIndex];
  if (pageItem && pageItem.skipped) {
    const badge = document.createElement("span");
    badge.className = "page-status-badge skipped";
    badge.textContent = "Bỏ qua";
    position.appendChild(badge);
  }

  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.className = "preview-nav-btn workspace-nav-btn";
  nextBtn.textContent = "Sau →";
  nextBtn.setAttribute("aria-label", "Trang sau");
  nextBtn.disabled = previewActivePageIndex >= pages.length - 1;
  nextBtn.addEventListener("click", () => {
    if (previewActivePageIndex >= pages.length - 1) return;
    previewActivePageIndex += 1;
    if (typeof setWorkflowCheckpoint === "function") {
      setWorkflowCheckpoint("preview", previewActivePageIndex);
    }
    renderPreview();
  });

  navigation.appendChild(prevBtn);
  navigation.appendChild(position);
  navigation.appendChild(nextBtn);
  container.appendChild(navigation);

  const workspace = document.createElement("div");
  workspace.className = "preview-main";

  const card = document.createElement("section");
  card.className = "preview-card preview-card-active";
  card.dataset.pageIndex = previewActivePageIndex;
  workspace.appendChild(card);
  container.appendChild(workspace);

  const page = pages[previewActivePageIndex];
  renderPreviewPage(card, page, previewActivePageIndex, pages);

  if (typeof setWorkflowCheckpoint === "function") {
    setWorkflowCheckpoint("preview", previewActivePageIndex);
  }

  const strip = document.createElement("div");
  strip.className = "preview-thumbnail-strip";
  strip.setAttribute("aria-label", "Chọn lát xem trước");

  pages.forEach((item, index) => {
    const thumb = document.createElement("button");
    thumb.className = "preview-thumbnail";
    thumb.dataset.pageIndex = index;
    thumb.title = pageLabel(pages, index);
    if (index === previewActivePageIndex) thumb.classList.add("active");
    if (item.skipped) thumb.classList.add("skipped");

    const thumbImage = document.createElement("img");
    thumbImage.src = item.original;
    thumbImage.alt = pageLabel(pages, index);
    thumbImage.loading = "lazy";
    thumb.appendChild(thumbImage);

    const thumbLabel = document.createElement("span");
    thumbLabel.textContent = String(index + 1).padStart(2, "0");
    thumb.appendChild(thumbLabel);

    thumb.addEventListener("click", () => {
      previewActivePageIndex = index;
      if (typeof setWorkflowCheckpoint === "function") {
        setWorkflowCheckpoint("preview", previewActivePageIndex);
      }
      renderPreview();
    });
    strip.appendChild(thumb);
  });

  container.appendChild(strip);
}

function renderPreviewPage(card, page, pageIndex, pages) {
  const header = document.createElement("div");
  header.className = "preview-card-header";

  const labelWrap = document.createElement("div");
  labelWrap.className = "preview-page-label-wrap";
  const label = document.createElement("div");
  label.className = "preview-label";
  label.textContent = pageLabel(pages, pageIndex);
  const status = document.createElement("span");
  status.className = "preview-page-status";
  status.textContent = page.skipped ? "Đã bỏ qua" : "Đang xem";
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
        saveExcludedRegions(pageIndex, page.excluded_regions);
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
  drawToggleBtn.textContent = "Đánh dấu vùng cấm dịch";
  const clearBtn = document.createElement("button");
  clearBtn.className = "excluded-clear-btn";
  clearBtn.textContent = "Xóa vùng cấm";
  clearBtn.title = "Xóa toàn bộ vùng cấm dịch của lát này";

  drawToggleBtn.addEventListener("click", () => {
    const active = card.classList.toggle("draw-excluded-active");
    drawToggleBtn.textContent = active ? "Đang đánh dấu · bấm để tắt" : "Đánh dấu vùng cấm dịch";
    drawToggleBtn.classList.toggle("active", active);
  });
  clearBtn.addEventListener("click", () => {
    page.excluded_regions = [];
    saveExcludedRegions(pageIndex, page.excluded_regions);
    renderExcludedBoxes();
  });
  tools.append(drawToggleBtn, clearBtn);
  card.appendChild(tools);

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
    saveExcludedRegions(pageIndex, page.excluded_regions);
    renderExcludedBoxes();
  };

  previewDrawCleanup = stopDrawing;

  const footer = document.createElement("div");
  footer.className = "preview-card-footer";
  const skipBtn = document.createElement("button");
  skipBtn.className = "skip-btn";
  skipBtn.textContent = page.skipped ? "Đã bỏ qua · bấm để hủy" : "Bỏ qua lát này";
  skipBtn.addEventListener("click", () => {
    toggleSkip(pageIndex, card, skipBtn);
    status.textContent = page.skipped ? "Đã bỏ qua" : "Đang xem";
    document.querySelectorAll(`.preview-thumbnail[data-page-index="${pageIndex}"]`).forEach((el) => {
      el.classList.toggle("skipped", !!page.skipped);
    });
  });
  footer.appendChild(skipBtn);
  card.appendChild(footer);
}
