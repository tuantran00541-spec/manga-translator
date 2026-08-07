// preview.js - Quản lý Giao diện Xem trước, Zoom/Pan & Vùng cấm dịch (Excluded Regions)

function pageLabel(pages, pageIndex) {
  const page = pages[pageIndex];
  const total = pages.filter((p) => p.source_page === page.source_page).length;
  if (total <= 1) return "Trang " + (page.source_page + 1);
  return "Trang " + (page.source_page + 1) + " - Lát " + (page.slice_index + 1) + "/" + total;
}

function renderPreview() {
  const container = document.getElementById("page-view");
  if (!container) return;
  container.innerHTML = "";
  container.className = "";

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";

  const processBtn = document.createElement("button");
  processBtn.textContent = "Xử lý các trang đã chọn (bỏ qua trang đã đánh dấu)";
  processBtn.addEventListener("click", processSelectedPages);
  toolbar.appendChild(processBtn);

  container.appendChild(toolbar);

  currentManifest.pages.forEach((page, pageIndex) => {
    if (!page.excluded_regions) {
      page.excluded_regions = [];
    }

    const card = document.createElement("div");
    card.className = "preview-card";
    card.dataset.pageIndex = pageIndex;
    if (page.skipped) card.classList.add("skipped");

    // Header & Zoom controls
    const cardHeader = document.createElement("div");
    cardHeader.className = "preview-card-header";

    const label = document.createElement("div");
    label.className = "preview-label";
    label.textContent = pageLabel(currentManifest.pages, pageIndex);
    cardHeader.appendChild(label);

    const zoomBar = document.createElement("div");
    zoomBar.className = "zoom-controls";

    let zoomScale = 1.0;

    const zoomOutBtn = document.createElement("button");
    zoomOutBtn.className = "zoom-btn";
    zoomOutBtn.textContent = "-";
    zoomOutBtn.title = "Thu nhỏ";

    const zoomLevelText = document.createElement("span");
    zoomLevelText.className = "zoom-level";
    zoomLevelText.textContent = "100%";

    const zoomInBtn = document.createElement("button");
    zoomInBtn.className = "zoom-btn";
    zoomInBtn.textContent = "+";
    zoomInBtn.title = "Phóng to";

    const zoomResetBtn = document.createElement("button");
    zoomResetBtn.className = "zoom-btn zoom-reset";
    zoomResetBtn.textContent = "1:1";
    zoomResetBtn.title = "Đặt lại zoom";

    zoomBar.appendChild(zoomOutBtn);
    zoomBar.appendChild(zoomLevelText);
    zoomBar.appendChild(zoomInBtn);
    zoomBar.appendChild(zoomResetBtn);
    cardHeader.appendChild(zoomBar);
    card.appendChild(cardHeader);

    // Viewport & Image wrapper
    const viewport = document.createElement("div");
    viewport.className = "preview-viewport";

    const imgWrap = document.createElement("div");
    imgWrap.className = "preview-image-wrap";

    const img = document.createElement("img");
    img.src = page.original;

    const overlayContainer = document.createElement("div");
    overlayContainer.className = "excluded-overlay-container";

    const drawLayer = document.createElement("div");
    drawLayer.className = "excluded-draw-layer";

    imgWrap.appendChild(img);
    imgWrap.appendChild(overlayContainer);
    imgWrap.appendChild(drawLayer);
    viewport.appendChild(imgWrap);
    card.appendChild(viewport);

    // Zoom update logic
    const updateZoom = (newScale) => {
      zoomScale = Math.max(0.5, Math.min(4.0, newScale));
      zoomLevelText.textContent = Math.round(zoomScale * 100) + "%";
      imgWrap.style.transform = `scale(${zoomScale})`;
      imgWrap.style.transformOrigin = "50% 0";
      if (zoomScale > 1.0) {
        viewport.style.overflow = "auto";
      } else {
        viewport.style.overflow = "hidden";
        viewport.scrollTop = 0;
        viewport.scrollLeft = 0;
      }
    };

    zoomInBtn.addEventListener("click", () => updateZoom(zoomScale + 0.25));
    zoomOutBtn.addEventListener("click", () => updateZoom(zoomScale - 0.25));
    zoomResetBtn.addEventListener("click", () => updateZoom(1.0));

    viewport.addEventListener("wheel", (e) => {
      if (e.ctrlKey || card.classList.contains("draw-excluded-active")) {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.15 : -0.15;
        updateZoom(zoomScale + delta);
      }
    }, { passive: false });

    // Excluded regions rendering
    const renderExcludedBoxes = () => {
      overlayContainer.innerHTML = "";
      if (!img.naturalWidth || !img.clientWidth) return;
      const scaleX = img.clientWidth / img.naturalWidth;
      const scaleY = img.clientHeight / img.naturalHeight;

      page.excluded_regions.forEach((box, rIdx) => {
        const rect = document.createElement("div");
        rect.className = "excluded-region-box";
        rect.style.left = (box.x1 * scaleX) + "px";
        rect.style.top = (box.y1 * scaleY) + "px";
        rect.style.width = ((box.x2 - box.x1) * scaleX) + "px";
        rect.style.height = ((box.y2 - box.y1) * scaleY) + "px";

        const delBtn = document.createElement("div");
        delBtn.className = "excluded-region-del";
        delBtn.textContent = "×";
        delBtn.title = "Xóa vùng cấm này";
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteExcludedRegion(pageIndex, rIdx);
        });
        rect.appendChild(delBtn);

        overlayContainer.appendChild(rect);
      });
    };

    if (img.complete && img.naturalWidth > 0) {
      renderExcludedBoxes();
    } else {
      img.onload = renderExcludedBoxes;
    }

    // Interactive excluded region drawing
    enableExcludedDraw(card, imgWrap, img, pageIndex, renderExcludedBoxes);

    // Bottom tool buttons
    const toolsDiv = document.createElement("div");
    toolsDiv.className = "preview-tools";

    const drawBtn = document.createElement("button");
    drawBtn.className = "excluded-toggle-btn";
    drawBtn.textContent = "Đánh dấu vùng cấm dịch";
    drawBtn.addEventListener("click", () => {
      const isActive = card.classList.toggle("draw-excluded-active");
      drawBtn.classList.toggle("active", isActive);
      drawBtn.textContent = isActive ? "Kéo chuột để khoanh vùng cấm..." : "Đánh dấu vùng cấm dịch";
    });
    toolsDiv.appendChild(drawBtn);

    const clearBtn = document.createElement("button");
    clearBtn.className = "excluded-clear-btn";
    clearBtn.textContent = "Xóa vùng cấm";
    clearBtn.addEventListener("click", () => clearAllExcludedRegions(pageIndex));
    toolsDiv.appendChild(clearBtn);

    card.appendChild(toolsDiv);

    // Skip page button
    const skipBtn = document.createElement("button");
    skipBtn.className = "skip-btn";
    skipBtn.textContent = page.skipped ? "Khôi phục trang này" : "Bỏ qua trang này";
    skipBtn.addEventListener("click", () => toggleSkipPage(pageIndex));
    card.appendChild(skipBtn);

    container.appendChild(card);
  });
}

function enableExcludedDraw(card, imgWrap, img, pageIndex, renderCallback) {
  const drawLayer = imgWrap.querySelector(".excluded-draw-layer");
  if (!drawLayer) return;

  let dragging = false;
  let startX = 0, startY = 0;
  let drawBox = null;

  drawLayer.addEventListener("mousedown", (e) => {
    if (!card.classList.contains("draw-excluded-active")) return;
    e.preventDefault();
    dragging = true;
    const rect = imgWrap.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;

    drawBox = document.createElement("div");
    drawBox.className = "excluded-region-box drawing";
    drawBox.style.left = startX + "px";
    drawBox.style.top = startY + "px";
    drawBox.style.width = "0px";
    drawBox.style.height = "0px";
    imgWrap.querySelector(".excluded-overlay-container").appendChild(drawBox);
  });

  drawLayer.addEventListener("mousemove", (e) => {
    if (!dragging || !drawBox) return;
    const rect = imgWrap.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;

    const left = Math.min(startX, curX);
    const top = Math.min(startY, curY);
    const width = Math.abs(curX - startX);
    const height = Math.abs(curY - startY);

    drawBox.style.left = left + "px";
    drawBox.style.top = top + "px";
    drawBox.style.width = width + "px";
    drawBox.style.height = height + "px";
  });

  document.addEventListener("mouseup", async () => {
    if (!dragging) return;
    dragging = false;
    if (!drawBox) return;

    const left = parseFloat(drawBox.style.left) || 0;
    const top = parseFloat(drawBox.style.top) || 0;
    const width = parseFloat(drawBox.style.width) || 0;
    const height = parseFloat(drawBox.style.height) || 0;
    drawBox.remove();
    drawBox = null;

    if (width < 8 || height < 8) return;

    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;

    const x1 = Math.round(left * scaleX);
    const y1 = Math.round(top * scaleY);
    const x2 = Math.round((left + width) * scaleX);
    const y2 = Math.round((top + height) * scaleY);

    await addExcludedRegion(pageIndex, x1, y1, x2, y2);
    renderCallback();
  });
}

async function addExcludedRegion(pageIndex, x1, y1, x2, y2) {
  const page = currentManifest.pages[pageIndex];
  if (!page.excluded_regions) page.excluded_regions = [];
  page.excluded_regions.push({ x1, y1, x2, y2 });
  await saveExcludedRegionsApi(pageIndex, page.excluded_regions);
}

async function deleteExcludedRegion(pageIndex, regionIndex) {
  const page = currentManifest.pages[pageIndex];
  if (!page.excluded_regions) return;
  page.excluded_regions.splice(regionIndex, 1);
  await saveExcludedRegionsApi(pageIndex, page.excluded_regions);
  renderPreview();
}

async function clearAllExcludedRegions(pageIndex) {
  const page = currentManifest.pages[pageIndex];
  page.excluded_regions = [];
  await saveExcludedRegionsApi(pageIndex, []);
  renderPreview();
}

async function toggleSkipPage(pageIndex) {
  const page = currentManifest.pages[pageIndex];
  const newSkipped = !page.skipped;
  page.skipped = newSkipped;
  await setPageSkipApi(pageIndex, newSkipped);
  renderPreview();
}

async function processSelectedPages() {
  const processBtn = document.querySelector("#preview-toolbar button");
  if (processBtn) {
    processBtn.disabled = true;
    processBtn.textContent = "Đang xử lý...";
  }
  try {
    const res = await processChapterApi();
    if (res.ok) {
      showToast("Xử lý thành công! Chuyển sang bước kiểm tra tẩy chữ.", "success");
      currentManifest = res.manifest;
      renderReview();
    } else {
      showToast("Lỗi xử lý: " + (res.error || "Không xác định"), "error");
    }
  } catch (err) {
    showToast("Lỗi kết nối server: " + err.message, "error");
  } finally {
    if (processBtn) {
      processBtn.disabled = false;
      processBtn.textContent = "Xử lý các trang đã chọn (bỏ qua trang đã đánh dấu)";
    }
  }
}
