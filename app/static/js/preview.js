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

      (page.excluded_regions || []).forEach((region, rIdx) => {
        const boxEl = document.createElement("div");
        boxEl.className = "excluded-region-box";
        boxEl.style.left = (region.x1 * scaleX) + "px";
        boxEl.style.top = (region.y1 * scaleY) + "px";
        boxEl.style.width = ((region.x2 - region.x1) * scaleX) + "px";
        boxEl.style.height = ((region.y2 - region.y1) * scaleY) + "px";

        const delBtn = document.createElement("span");
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

    if (img.complete && img.naturalWidth > 0) {
      renderExcludedBoxes();
    } else {
      img.onload = () => renderExcludedBoxes();
    }

    // Excluded region drawing tool
    const tools = document.createElement("div");
    tools.className = "preview-tools";

    const drawToggleBtn = document.createElement("button");
    drawToggleBtn.className = "excluded-toggle-btn";
    drawToggleBtn.textContent = "Đánh dấu vùng cấm dịch";

    const clearBtn = document.createElement("button");
    clearBtn.className = "excluded-clear-btn";
    clearBtn.textContent = "Xóa vùng cấm";
    clearBtn.title = "Xóa toàn bộ vùng cấm dịch của trang này";

    drawToggleBtn.addEventListener("click", () => {
      const active = card.classList.toggle("draw-excluded-active");
      drawToggleBtn.textContent = active ? "Đang đánh dấu (bấm để tắt)" : "Đánh dấu vùng cấm dịch";
      drawToggleBtn.classList.toggle("active", active);
    });

    clearBtn.addEventListener("click", () => {
      page.excluded_regions = [];
      saveExcludedRegions(pageIndex, page.excluded_regions);
      renderExcludedBoxes();
    });

    tools.appendChild(drawToggleBtn);
    tools.appendChild(clearBtn);
    card.appendChild(tools);

    // Mouse drag drawing logic
    let isDragging = false;
    let startPos = null;
    let tempDrawBox = null;

    drawLayer.addEventListener("mousedown", (e) => {
      if (!card.classList.contains("draw-excluded-active")) return;
      e.preventDefault();
      const rect = imgWrap.getBoundingClientRect();
      const clientX = e.clientX;
      const clientY = e.clientY;

      const x = (clientX - rect.left) / zoomScale;
      const y = (clientY - rect.top) / zoomScale;

      isDragging = true;
      startPos = { x, y };

      tempDrawBox = document.createElement("div");
      tempDrawBox.className = "excluded-region-box drawing";
      tempDrawBox.style.left = x + "px";
      tempDrawBox.style.top = y + "px";
      tempDrawBox.style.width = "0px";
      tempDrawBox.style.height = "0px";
      overlayContainer.appendChild(tempDrawBox);
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging || !tempDrawBox) return;
      const rect = imgWrap.getBoundingClientRect();
      const x = (e.clientX - rect.left) / zoomScale;
      const y = (e.clientY - rect.top) / zoomScale;

      const left = Math.min(startPos.x, x);
      const top = Math.min(startPos.y, y);
      const width = Math.abs(x - startPos.x);
      const height = Math.abs(y - startPos.y);

      tempDrawBox.style.left = left + "px";
      tempDrawBox.style.top = top + "px";
      tempDrawBox.style.width = width + "px";
      tempDrawBox.style.height = height + "px";
    });

    window.addEventListener("mouseup", (e) => {
      if (!isDragging || !tempDrawBox) return;
      isDragging = false;

      const left = parseFloat(tempDrawBox.style.left) || 0;
      const top = parseFloat(tempDrawBox.style.top) || 0;
      const w = parseFloat(tempDrawBox.style.width) || 0;
      const h = parseFloat(tempDrawBox.style.height) || 0;
      tempDrawBox.remove();
      tempDrawBox = null;

      if (w >= 5 && h >= 5) {
        const scaleX = img.naturalWidth / img.clientWidth;
        const scaleY = img.naturalHeight / img.clientHeight;

        const x1 = Math.round(left * scaleX);
        const y1 = Math.round(top * scaleY);
        const x2 = Math.round((left + w) * scaleX);
        const y2 = Math.round((top + h) * scaleY);

        if (!page.excluded_regions) page.excluded_regions = [];
        page.excluded_regions.push({ x1, y1, x2, y2 });

        saveExcludedRegions(pageIndex, page.excluded_regions);
        renderExcludedBoxes();
      }
    });

    // Skip button
    const skipBtn = document.createElement("button");
    skipBtn.className = "skip-btn";
    skipBtn.textContent = page.skipped ? "Đã bỏ qua (bấm để hủy)" : "Bỏ qua trang này";
    skipBtn.addEventListener("click", () => toggleSkip(pageIndex, card, skipBtn));
    card.appendChild(skipBtn);

    container.appendChild(card);
  });
}
