(() => {
  const MAX_CANVAS_CHUNK_HEIGHT = 12000;
  const ZOOM_STEPS = [25, 50, 75, 100, 125, 150, 200, 300, 400];
  const MIN_TEXT_BOX_SIZE = 10;
  let activeSourcePage = null;
  let renderToken = 0;
  let zoomPercent = 100;
  let fitWidth = true;
  let variant = "clean";
  const stitchedMaskSnapshots = new Map();
  const autoSyncChecked = new Set();

  function regionHasPaint(canvas, x, y, width, height) {
    const scale = Math.min(1, 512 / Math.max(width, height));
    const probe = document.createElement("canvas");
    probe.width = Math.max(1, Math.ceil(width * scale));
    probe.height = Math.max(1, Math.ceil(height * scale));
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(canvas, x, y, width, height, 0, 0, probe.width, probe.height);
    const pixels = ctx.getImageData(0, 0, probe.width, probe.height).data;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 20) return true;
    return false;
  }

  function cleanSourcePage(page, fallbackIndex) {
    return Number.isInteger(page?.source_page) ? page.source_page : fallbackIndex;
  }

  function cleanSliceIndex(page) {
    return Number.isInteger(page?.slice_index) ? page.slice_index : 0;
  }

  function groupsFromManifest() {
    const groups = new Map();
    (window.currentManifest?.pages || []).forEach((page, canonicalIndex) => {
      if (!page) return;
      const sourcePage = cleanSourcePage(page, canonicalIndex);
      if (!groups.has(sourcePage)) groups.set(sourcePage, []);
      groups.get(sourcePage).push({ page, canonicalIndex });
    });
    for (const items of groups.values()) items.sort((a, b) => cleanSliceIndex(a.page) - cleanSliceIndex(b.page));
    return new Map([...groups.entries()].sort((a, b) => a[0] - b[0]));
  }

  function imageUrl(page, targetVariant = variant) {
    let url;
    let revision;
    if (targetVariant === "original") {
      url = page.original || page.clean;
      revision = Number(page.source_revision || 0);
    } else if (targetVariant === "rendered") {
      url = page._reviewRenderedUrl || (typeof page.rendered === "string" ? page.rendered : null) || page.clean || page.original;
      revision = Number(page.render_revision || page.clean_revision || page.process_revision || page.source_revision || 0);
    } else {
      url = page.skipped ? page.original : (page.clean || page.original);
      revision = Number(page.skipped
        ? (page.source_revision || 0)
        : (page.clean_revision || page.process_revision || page.source_revision || 0));
    }
    if (!url) return null;
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}review_revision=${revision}`;
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = "async";
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Không tải được ảnh: ${url}`));
      img.src = url;
    });
  }

  function coreMetadata(page, imageHeight) {
    const core = page?.stitch_core;
    if (!core || typeof core !== "object") return null;
    const localY1 = Number(core.core_y1);
    const localY2 = Number(core.core_y2);
    const sourceY1 = Number(core.core_source_y1);
    const sourceY2 = Number(core.core_source_y2);
    const sourceHeight = Number(core.source_height);
    if (![localY1, localY2, sourceY1, sourceY2, sourceHeight].every(Number.isFinite)) return null;
    if (localY1 < 0 || localY2 <= localY1 || localY2 > imageHeight) return null;
    if (sourceY1 < 0 || sourceY2 <= sourceY1 || sourceHeight < sourceY2) return null;
    if ((localY2 - localY1) !== (sourceY2 - sourceY1)) return null;
    return { localY1, localY2, sourceY1, sourceY2, sourceHeight };
  }

  function makeChunkCanvases(host, width, height) {
    const chunks = [];
    for (let y = 0; y < height; y += MAX_CANVAS_CHUNK_HEIGHT) {
      const chunkHeight = Math.min(MAX_CANVAS_CHUNK_HEIGHT, height - y);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = chunkHeight;
      canvas.dataset.sourceY = String(y);
      const ctx = canvas.getContext("2d", { alpha: false });
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, chunkHeight);
      host.appendChild(canvas);
      chunks.push({ canvas, ctx, y1: y, y2: y + chunkHeight });
    }
    return chunks;
  }

  function drawIntoChunks(chunks, img, sx, sy, sw, sh, targetY) {
    const targetEnd = targetY + sh;
    for (const chunk of chunks) {
      const iy1 = Math.max(targetY, chunk.y1);
      const iy2 = Math.min(targetEnd, chunk.y2);
      if (iy2 <= iy1) continue;
      const offset = iy1 - targetY;
      const h = iy2 - iy1;
      chunk.ctx.drawImage(img, sx, sy + offset, sw, h, 0, iy1 - chunk.y1, sw, h);
    }
  }

  function captureSnapshot(shell) {
    if (activeSourcePage === null || !shell || variant !== "clean") return;
    const canvas = shell.querySelector("canvas.stitched-brush-canvas");
    if (!canvas || !canvas._reviewDirty || !canvas.width || !canvas.height) {
      stitchedMaskSnapshots.delete(activeSourcePage);
      return;
    }
    try { stitchedMaskSnapshots.set(activeSourcePage, canvas.toDataURL("image/png")); }
    catch (err) { console.warn("Could not preserve stitched mask snapshot:", err); }
  }

  function restoreSnapshot(brushCanvas, sourcePage) {
    const dataUrl = stitchedMaskSnapshots.get(sourcePage);
    if (!dataUrl) return;
    const overlay = new Image();
    overlay.onload = () => {
      if (!brushCanvas.isConnected) return;
      brushCanvas.getContext("2d").drawImage(overlay, 0, 0, brushCanvas.width, brushCanvas.height);
      brushCanvas._reviewDirty = true;
    };
    overlay.src = dataUrl;
  }

  window.hasUnsavedStitchedMarks = () => {
    if (stitchedMaskSnapshots.size > 0) return true;
    const canvas = document.querySelector(".review-stitched-image canvas.stitched-brush-canvas");
    return Boolean(canvas && canvas._reviewDirty);
  };

  function zoomScale(viewport, sourceWidth) {
    if (!sourceWidth) return 1;
    if (fitWidth) {
      const available = Math.max(240, viewport.clientWidth - 48);
      return Math.min(1, available / sourceWidth);
    }
    return zoomPercent / 100;
  }

  function applyZoom(shell) {
    const viewport = shell.querySelector(".review-stitched-viewport");
    const stage = shell.querySelector(".review-stitched-zoom-stage");
    const imageHost = shell.querySelector(".review-stitched-image");
    const zoomLabel = shell.querySelector(".review-zoom-value");
    const width = Number(imageHost?.dataset.sourceWidth || 0);
    const height = Number(imageHost?.dataset.sourceHeight || 0);
    if (!viewport || !stage || !imageHost || !width || !height) return;
    const scale = zoomScale(viewport, width);
    imageHost.style.transform = `scale(${scale})`;
    stage.style.width = `${Math.max(1, Math.round(width * scale))}px`;
    stage.style.height = `${Math.max(1, Math.round(height * scale))}px`;
    if (zoomLabel) zoomLabel.textContent = fitWidth ? `Fit · ${Math.round(scale * 100)}%` : `${zoomPercent}%`;
  }

  function stepZoom(delta) {
    fitWidth = false;
    let index = ZOOM_STEPS.findIndex((value) => value >= zoomPercent);
    if (index < 0) index = ZOOM_STEPS.length - 1;
    if (delta < 0 && ZOOM_STEPS[index] >= zoomPercent) index--;
    if (delta > 0 && ZOOM_STEPS[index] <= zoomPercent) index++;
    index = Math.max(0, Math.min(ZOOM_STEPS.length - 1, index));
    zoomPercent = ZOOM_STEPS[index];
  }

  function descriptorForCanonical(shell, canonicalIndex) {
    return (shell._descriptors || []).find((desc) => Number(desc.item?.canonicalIndex) === Number(canonicalIndex)) || null;
  }

  function ownedTextObjects(shell) {
    const results = [];
    for (const desc of shell._descriptors || []) {
      const pageIndex = desc.item.canonicalIndex;
      const page = window.currentManifest?.pages?.[pageIndex];
      if (!page || page.skipped) continue;
      for (const obj of page.text_objects || []) {
        if (!obj?.region || !obj.id) continue;
        const centerY = (Number(obj.region.y1) + Number(obj.region.y2)) / 2;
        if (centerY < desc.localY1 || centerY >= desc.localY2) continue;
        results.push({ desc, pageIndex, obj });
      }
    }
    return results;
  }

  function sourceRegionFor(desc, region) {
    return {
      x1: Number(region.x1),
      y1: desc.sourceY1 + (Number(region.y1) - desc.localY1),
      x2: Number(region.x2),
      y2: desc.sourceY1 + (Number(region.y2) - desc.localY1),
    };
  }

  function syncReviewOverlay(shell, pageIndex, id) {
    if (!shell?.isConnected) return false;
    const desc = descriptorForCanonical(shell, pageIndex);
    const obj = typeof window.findTextObject === "function" ? window.findTextObject(pageIndex, id) : null;
    const overlay = shell.querySelector(`.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`);
    if (!desc || !obj?.region || !overlay) return false;
    const r = sourceRegionFor(desc, obj.region);
    overlay.style.left = `${r.x1}px`;
    overlay.style.top = `${r.y1}px`;
    overlay.style.width = `${Math.max(MIN_TEXT_BOX_SIZE, r.x2 - r.x1)}px`;
    overlay.style.height = `${Math.max(MIN_TEXT_BOX_SIZE, r.y2 - r.y1)}px`;
    const label = overlay.querySelector(".review-text-preview");
    if (label) label.textContent = obj.translation?.trim() || obj.ocr_text?.trim() || "Text";
    return true;
  }

  function installReviewOverlaySync(shell) {
    if (!window._reviewBaseSyncOverlayForObject) {
      window._reviewBaseSyncOverlayForObject = window.syncOverlayForObject || null;
    }
    window.syncOverlayForObject = (pageIndex, id) => {
      const liveShell = document.querySelector("#page-view.review-mode .review-document-shell");
      if (liveShell && syncReviewOverlay(liveShell, pageIndex, id)) return;
      window._reviewBaseSyncOverlayForObject?.(pageIndex, id);
    };
  }

  function selectReviewTextObject(shell, pageIndex, id) {
    if (!window.editorState) return;
    window.editorState.activePageIndex = Number(pageIndex);
    window.editorState.selectedTextObjectId = id;
    shell.querySelectorAll(".review-text-object-overlay").forEach((el) => {
      el.classList.toggle("selected", Number(el.dataset.pageIndex) === Number(pageIndex) && el.dataset.objectId === String(id));
    });
    if (typeof window.renderEditorPanel === "function") window.renderEditorPanel(Number(pageIndex));
    window.showWorkbenchInspector?.();
  }

  function clearReviewTextSelection(shell) {
    if (window.editorState) window.editorState.selectedTextObjectId = null;
    shell.querySelectorAll(".review-text-object-overlay").forEach((el) => el.classList.remove("selected"));
    const host = document.querySelector(".review-inspector .translation-panel-host");
    if (host) host.innerHTML = '<div class="ui-empty-state">Chọn một vùng chữ trên trang để chỉnh OCR, bản dịch và typography.</div>';
  }

  function overlayMode(event, overlay) {
    const rect = overlay.getBoundingClientRect();
    const edge = Math.max(4, Math.min(10, Math.min(rect.width, rect.height) / 3));
    const left = event.clientX - rect.left < edge;
    const right = rect.right - event.clientX < edge;
    const top = event.clientY - rect.top < edge;
    const bottom = rect.bottom - event.clientY < edge;
    if (top && left) return "nw";
    if (top && right) return "ne";
    if (bottom && left) return "sw";
    if (bottom && right) return "se";
    if (left) return "w";
    if (right) return "e";
    if (top) return "n";
    if (bottom) return "s";
    return "move";
  }

  function cursorForMode(mode) {
    if (mode === "nw" || mode === "se") return "nwse-resize";
    if (mode === "ne" || mode === "sw") return "nesw-resize";
    if (mode === "n" || mode === "s") return "ns-resize";
    if (mode === "e" || mode === "w") return "ew-resize";
    return "move";
  }

  function installOverlayTransform(shell, overlay, desc, pageIndex, obj, signal) {
    let active = null;
    const imageHost = shell.querySelector(".review-stitched-image");

    overlay.addEventListener("pointerdown", (event) => {
      if (variant !== "clean" || window.editorState?.tool !== "select" || event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      selectReviewTextObject(shell, pageIndex, obj.id);
      const rect = imageHost.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const sourceW = Number(imageHost.dataset.sourceWidth || 0);
      const sourceH = Number(imageHost.dataset.sourceHeight || 0);
      active = {
        mode: overlayMode(event, overlay),
        pointerId: event.pointerId,
        startX: (event.clientX - rect.left) * sourceW / rect.width,
        startY: (event.clientY - rect.top) * sourceH / rect.height,
        original: { ...obj.region },
      };
      overlay.classList.add("transforming");
      overlay.setPointerCapture?.(event.pointerId);
    }, { signal });

    const move = (event) => {
      if (!active) {
        if (window.editorState?.tool === "select") overlay.style.cursor = cursorForMode(overlayMode(event, overlay));
        return;
      }
      const rect = imageHost.getBoundingClientRect();
      const sourceW = Number(imageHost.dataset.sourceWidth || 0);
      const sourceH = Number(imageHost.dataset.sourceHeight || 0);
      if (!rect.width || !rect.height || !sourceW || !sourceH) return;
      const x = (event.clientX - rect.left) * sourceW / rect.width;
      const y = (event.clientY - rect.top) * sourceH / rect.height;
      const dx = x - active.startX;
      const dy = y - active.startY;
      const original = active.original;
      const W = desc.img.naturalWidth;
      const minY = desc.localY1;
      const maxY = desc.localY2;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      let { x1, y1, x2, y2 } = original;
      if (active.mode === "move") {
        const w = original.x2 - original.x1;
        const h = original.y2 - original.y1;
        x1 = clamp(original.x1 + dx, 0, Math.max(0, W - w));
        y1 = clamp(original.y1 + dy, minY, Math.max(minY, maxY - h));
        x2 = x1 + w;
        y2 = y1 + h;
      } else {
        if (active.mode.includes("w")) x1 = clamp(original.x1 + dx, 0, original.x2 - MIN_TEXT_BOX_SIZE);
        if (active.mode.includes("e")) x2 = clamp(original.x2 + dx, original.x1 + MIN_TEXT_BOX_SIZE, W);
        if (active.mode.includes("n")) y1 = clamp(original.y1 + dy, minY, original.y2 - MIN_TEXT_BOX_SIZE);
        if (active.mode.includes("s")) y2 = clamp(original.y2 + dy, original.y1 + MIN_TEXT_BOX_SIZE, maxY);
      }
      obj.region = { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
      syncReviewOverlay(shell, pageIndex, obj.id);
    };

    const end = () => {
      if (!active) return;
      const original = active.original;
      active = null;
      overlay.classList.remove("transforming");
      const r = obj.region;
      const changed = r.x1 !== original.x1 || r.y1 !== original.y1 || r.x2 !== original.x2 || r.y2 !== original.y2;
      if (changed) {
        window.scheduleGeomPersist?.(pageIndex, obj.id);
        window.refreshGeometryControls?.(pageIndex, obj.id);
      }
    };

    overlay.addEventListener("pointermove", move, { signal });
    overlay.addEventListener("pointerup", end, { signal });
    overlay.addEventListener("pointercancel", end, { signal });
  }

  function renderTextOverlays(shell, signal) {
    const imageHost = shell.querySelector(".review-stitched-image");
    if (!imageHost) return;
    imageHost.querySelectorAll(".review-text-object-overlay,.review-text-drawing").forEach((node) => node.remove());
    if (variant !== "clean") return;

    for (const { desc, pageIndex, obj } of ownedTextObjects(shell)) {
      const overlay = document.createElement("div");
      overlay.className = "text-object-overlay review-text-object-overlay" + (obj.shape === "ellipse" ? " ellipse" : "");
      overlay.dataset.pageIndex = String(pageIndex);
      overlay.dataset.objectId = String(obj.id);
      const preview = document.createElement("span");
      preview.className = "review-text-preview";
      preview.textContent = obj.translation?.trim() || obj.ocr_text?.trim() || "Text";
      preview.style.cssText = "position:absolute;left:2px;right:2px;top:50%;transform:translateY(-50%);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:2px 4px;background:rgba(18,18,18,.72);color:#fff;font:11px/1.2 system-ui,sans-serif;text-align:center;pointer-events:none;border-radius:2px;";
      overlay.appendChild(preview);
      overlay.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectReviewTextObject(shell, pageIndex, obj.id);
      }, { signal });
      overlay.addEventListener("dblclick", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectReviewTextObject(shell, pageIndex, obj.id);
        document.querySelector(".review-inspector .translation-textarea")?.focus();
      }, { signal });
      imageHost.appendChild(overlay);
      syncReviewOverlay(shell, pageIndex, obj.id);
      installOverlayTransform(shell, overlay, desc, pageIndex, obj, signal);
    }
  }

  async function ensureSourceTextObjects(shell) {
    if (typeof window.ensureAutoTextObjects !== "function") return;
    for (const desc of shell._descriptors || []) {
      const pageIndex = Number(desc.item.canonicalIndex);
      const key = `${window.currentChapterId}:${pageIndex}`;
      if (autoSyncChecked.has(key)) continue;
      autoSyncChecked.add(key);
      try { await window.ensureAutoTextObjects(pageIndex); }
      catch (err) { console.warn("Could not ensure text objects for review page", pageIndex, err); }
    }
  }

  function ownerDescriptorForSourceY(shell, sourceY) {
    return (shell._descriptors || []).find((desc) => sourceY >= desc.sourceY1 && sourceY < desc.sourceY2) || null;
  }

  function installTextDrawing(shell, signal, onChanged) {
    const imageHost = shell.querySelector(".review-stitched-image");
    let drawing = null;

    const sourcePoint = (event) => {
      const rect = imageHost.getBoundingClientRect();
      const W = Number(imageHost.dataset.sourceWidth || 0);
      const H = Number(imageHost.dataset.sourceHeight || 0);
      if (!rect.width || !rect.height || !W || !H) return null;
      return {
        x: Math.max(0, Math.min(W, (event.clientX - rect.left) * W / rect.width)),
        y: Math.max(0, Math.min(H, (event.clientY - rect.top) * H / rect.height)),
      };
    };

    const update = (point) => {
      if (!drawing || !point) return;
      const x1 = Math.min(drawing.start.x, point.x);
      const y1 = Math.min(drawing.start.y, point.y);
      const x2 = Math.max(drawing.start.x, point.x);
      const y2 = Math.max(drawing.start.y, point.y);
      Object.assign(drawing.preview.style, {
        left: `${x1}px`, top: `${y1}px`, width: `${x2 - x1}px`, height: `${y2 - y1}px`,
      });
      drawing.last = point;
    };

    imageHost.addEventListener("pointerdown", (event) => {
      const tool = window.editorState?.tool || "select";
      if (variant !== "clean" || !["rectangle", "ellipse"].includes(tool) || event.button !== 0) return;
      if (event.target.closest(".review-text-object-overlay")) return;
      const point = sourcePoint(event);
      if (!point) return;
      event.preventDefault();
      const preview = document.createElement("div");
      preview.className = "text-object-overlay drawing review-text-drawing" + (tool === "ellipse" ? " ellipse" : "");
      imageHost.appendChild(preview);
      drawing = { tool, start: point, last: point, preview, pointerId: event.pointerId };
      imageHost.setPointerCapture?.(event.pointerId);
      update(point);
    }, { signal });

    imageHost.addEventListener("pointermove", (event) => {
      if (!drawing) return;
      update(sourcePoint(event));
    }, { signal });

    const finish = async () => {
      if (!drawing) return;
      const current = drawing;
      drawing = null;
      current.preview.remove();
      const x1 = Math.round(Math.min(current.start.x, current.last.x));
      const y1 = Math.round(Math.min(current.start.y, current.last.y));
      const x2 = Math.round(Math.max(current.start.x, current.last.x));
      const y2 = Math.round(Math.max(current.start.y, current.last.y));
      if (x2 - x1 < MIN_TEXT_BOX_SIZE || y2 - y1 < MIN_TEXT_BOX_SIZE) return;
      const owner = ownerDescriptorForSourceY(shell, (y1 + y2) / 2);
      if (!owner) return;
      const pageIndex = Number(owner.item.canonicalIndex);
      const clippedSourceY1 = Math.max(owner.sourceY1, y1);
      const clippedSourceY2 = Math.min(owner.sourceY2, y2);
      const localY1 = Math.round(owner.localY1 + (clippedSourceY1 - owner.sourceY1));
      const localY2 = Math.round(owner.localY1 + (clippedSourceY2 - owner.sourceY1));
      const region = {
        x1: Math.max(0, x1),
        y1: Math.max(owner.localY1, localY1),
        x2: Math.min(owner.img.naturalWidth, x2),
        y2: Math.min(owner.localY2, localY2),
      };
      if (region.x2 - region.x1 < MIN_TEXT_BOX_SIZE || region.y2 - region.y1 < MIN_TEXT_BOX_SIZE) return;
      try {
        if (typeof window.createTextObject !== "function") throw new Error("Công cụ tạo vùng chữ chưa sẵn sàng");
        await window.createTextObject(pageIndex, current.tool, region);
        window.editorState.activePageIndex = pageIndex;
        renderTextOverlays(shell, signal);
        onChanged?.();
      } catch (err) {
        window.showToast?.("Không thể tạo vùng chữ: " + err.message, "error");
      }
    };

    window.addEventListener("pointerup", finish, { signal });
    window.addEventListener("pointercancel", finish, { signal });
    imageHost.addEventListener("click", (event) => {
      if ((window.editorState?.tool || "select") !== "select") return;
      if (event.target.closest(".review-text-object-overlay")) return;
      clearReviewTextSelection(shell);
    }, { signal });
  }

  async function renderSourcePage(shell, sourcePage, items, brushHandlers, signal) {
    const token = ++renderToken;
    const stage = shell.querySelector(".review-stitched-zoom-stage");
    const imageHost = shell.querySelector(".review-stitched-image");
    const meta = shell.querySelector(".review-stitched-meta");
    const warning = shell.querySelector(".review-stitched-warning");
    imageHost.replaceChildren();
    stage.style.width = "auto";
    stage.style.height = "auto";
    warning.hidden = true;

    const loading = document.createElement("div");
    loading.className = "ui-state review-stitched-loading";
    loading.textContent = variant === "original"
      ? "Đang dựng ảnh gốc…"
      : (variant === "rendered" ? "Đang dựng bản lettering…" : "Đang dựng ảnh đã inpaint…");
    imageHost.appendChild(loading);

    try {
      const firstUrl = imageUrl(items[0]?.page);
      if (!firstUrl) throw new Error("Trang không có ảnh để hiển thị.");
      const first = await loadImage(firstUrl);
      if (token !== renderToken) return;
      const firstCore = coreMetadata(items[0].page, first.naturalHeight);
      let sourceHeight = firstCore?.sourceHeight || first.naturalHeight;
      const width = first.naturalWidth;
      let metadataValid = Boolean(firstCore) || items.length === 1;
      let expectedSourceY = 0;
      let fallbackY = 0;
      const descriptors = [];

      for (let index = 0; index < items.length; index++) {
        const item = items[index];
        const page = window.currentManifest?.pages?.[item.canonicalIndex] || item.page;
        const url = imageUrl(page);
        if (!url) throw new Error(`Vùng ảnh ${index + 1} không có dữ liệu.`);
        const img = index === 0 ? first : await loadImage(url);
        if (token !== renderToken) return;
        if (img.naturalWidth !== width) throw new Error("Các vùng ảnh trong cùng trang không cùng chiều rộng.");
        const core = coreMetadata(page, img.naturalHeight);
        if (core) {
          sourceHeight = core.sourceHeight;
          if (core.sourceY1 !== expectedSourceY) metadataValid = false;
          expectedSourceY = core.sourceY2;
          descriptors.push({ item: { ...item, page }, img, localY1: core.localY1, localY2: core.localY2, sourceY1: core.sourceY1, sourceY2: core.sourceY2 });
        } else {
          metadataValid = false;
          descriptors.push({ item: { ...item, page }, img, localY1: 0, localY2: img.naturalHeight, sourceY1: fallbackY, sourceY2: fallbackY + img.naturalHeight });
          fallbackY += img.naturalHeight;
        }
      }

      if (!metadataValid) {
        let y = 0;
        for (const desc of descriptors) {
          desc.sourceY1 = y;
          desc.sourceY2 = y + (desc.localY2 - desc.localY1);
          y = desc.sourceY2;
        }
        sourceHeight = y;
        warning.hidden = false;
        warning.textContent = "Metadata ghép trang không liên tục; đang dùng nối tuần tự để kiểm tra hình ảnh.";
      }

      imageHost.replaceChildren();
      imageHost.style.width = `${width}px`;
      imageHost.style.height = `${sourceHeight}px`;
      imageHost.dataset.sourceWidth = String(width);
      imageHost.dataset.sourceHeight = String(sourceHeight);
      const chunks = makeChunkCanvases(imageHost, width, sourceHeight);
      for (const desc of descriptors) {
        drawIntoChunks(chunks, desc.img, 0, desc.localY1, width, desc.localY2 - desc.localY1, desc.sourceY1);
      }
      shell._descriptors = descriptors;

      if (variant === "clean") {
        const brushCanvas = document.createElement("canvas");
        brushCanvas.className = "brush-canvas stitched-brush-canvas";
        brushCanvas.width = width;
        brushCanvas.height = sourceHeight;
        imageHost.appendChild(brushCanvas);
        brushHandlers?.bindCanvas?.(brushCanvas);
        restoreSnapshot(brushCanvas, sourcePage);
        shell._brushCanvas = brushCanvas;
      } else {
        shell._brushCanvas = null;
      }

      const variantLabel = variant === "original" ? "Ảnh gốc" : (variant === "rendered" ? "Bản lettering" : "Sau inpaint");
      meta.textContent = `Trang ${sourcePage + 1} · ${width} × ${sourceHeight}px · ${variantLabel}`;
      applyZoom(shell);
      await ensureSourceTextObjects(shell);
      if (token !== renderToken) return;
      renderTextOverlays(shell, signal);
    } catch (err) {
      if (token !== renderToken) return;
      imageHost.replaceChildren();
      const error = document.createElement("div");
      error.className = "ui-state ui-state-error review-stitched-error";
      error.textContent = `Không dựng được trang: ${err.message}`;
      imageHost.appendChild(error);
    }
  }

  function mountTextInspector(workspace, shell, signal) {
    const inspector = workspace.querySelector(".review-inspector");
    if (!inspector) return null;
    let host = inspector.querySelector(".translation-panel-host");
    if (!host) {
      host = document.createElement("section");
      host.className = "translation-panel-host review-text-inspector-host";
      host.setAttribute("aria-label", "Chỉnh OCR, bản dịch và typography");
      inspector.appendChild(host);
    }
    host.innerHTML = '<div class="ui-empty-state">Chọn một vùng chữ trên trang để chỉnh OCR, bản dịch và typography.</div>';
    host.addEventListener("input", (event) => {
      if (!event.target.classList?.contains("translation-textarea")) return;
      const pageIndex = Number(window.editorState?.activePageIndex || 0);
      const id = window.editorState?.selectedTextObjectId;
      if (!id) return;
      const overlay = shell.querySelector(`.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"] .review-text-preview`);
      if (overlay) overlay.textContent = event.target.value.trim() || window.findTextObject?.(pageIndex, id)?.ocr_text?.trim() || "Text";
    }, { signal });

    let refreshTimer = null;
    const observer = new MutationObserver(() => {
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => renderTextOverlays(shell, signal), 80);
    });
    observer.observe(host, { childList: true, subtree: true });
    signal.addEventListener("abort", () => { observer.disconnect(); clearTimeout(refreshTimer); });
    return host;
  }

  function mountTextToolbar(workspace, shell, signal, rerender) {
    const actions = workspace.querySelector(".review-actions-group");
    if (!actions) return;

    const oldContinue = actions.querySelector(".review-primary-action");
    oldContinue?.remove();

    const tools = document.createElement("div");
    tools.className = "editor-tools review-lettering-tools";
    tools.setAttribute("role", "group");
    tools.setAttribute("aria-label", "Công cụ lettering");
    [
      ["select", "Chọn"],
      ["rectangle", "Text box"],
      ["ellipse", "Bubble text"],
    ].forEach(([tool, label]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ui-btn ui-btn-ghost ui-btn-compact editor-tool-btn";
      btn.dataset.tool = tool;
      btn.textContent = label;
      btn.addEventListener("click", () => {
        window.setEditorTool?.(tool);
        workspace.classList.toggle("review-text-draw-mode", tool !== "select");
        shell.querySelectorAll(".review-text-object-overlay").forEach((overlay) => {
          overlay.style.pointerEvents = tool === "select" ? "auto" : "none";
        });
      }, { signal });
      tools.appendChild(btn);
    });
    actions.prepend(tools);

    const renderBtn = document.createElement("button");
    renderBtn.type = "button";
    renderBtn.className = "ui-btn ui-btn-primary editor-render-btn review-render-text-btn";
    renderBtn.textContent = "Render chữ";
    renderBtn.addEventListener("click", async () => {
      const descriptors = shell._descriptors || [];
      const indices = [...new Set(descriptors.map((d) => Number(d.item.canonicalIndex)))];
      if (!indices.length || typeof window.renderTranslations !== "function") return;
      renderBtn.disabled = true;
      renderBtn.textContent = "Đang render…";
      try {
        let renderedCount = 0;
        for (const pageIndex of indices) {
          await window.renderTranslations(pageIndex);
          const renderedPreview = document.querySelector(".review-inspector .translation-panel-host .render-result img");
          const page = window.currentManifest?.pages?.[pageIndex];
          if (page?.rendered && renderedPreview?.getAttribute("src")) {
            page._reviewRenderedUrl = renderedPreview.getAttribute("src");
            renderedCount += 1;
          }
        }
        if (!renderedCount) {
          window.showToast?.("Không có vùng chữ nào được render trên trang này.", "info");
          return;
        }
        window.showToast?.(`Đã render lettering cho trang ${activeSourcePage + 1}.`, "success");
        variant = "rendered";
        rerender();
      } catch (err) {
        window.showToast?.("Không thể render lettering: " + err.message, "error");
      } finally {
        renderBtn.disabled = false;
        renderBtn.textContent = "Render chữ";
      }
    }, { signal });
    actions.appendChild(renderBtn);

    if (typeof window.buildChapterTranslateControls === "function") {
      const translate = window.buildChapterTranslateControls();
      translate.classList.add("review-translate-controls");
      actions.appendChild(translate);
    }
    if (typeof window.buildChapterExportButton === "function") {
      const exportBtn = window.buildChapterExportButton();
      exportBtn.classList.add("review-export-button");
      actions.appendChild(exportBtn);
    }

    window.setEditorTool?.(window.editorState?.tool || "select");
    installTextDrawing(shell, signal, rerender);
  }

  function mount(workspace) {
    if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchInspectorMounted === "1") return;
    workspace._stitchAbort?.abort();
    window._reviewStitchAbort?.abort();
    const aborter = new AbortController();
    workspace._stitchAbort = aborter;
    window._reviewStitchAbort = aborter;
    const { signal } = aborter;
    const toolbar = workspace.querySelector(".review-sticky-toolbar");
    const layout = workspace.querySelector(".review-workbench-grid");
    const canvasHost = layout?.querySelector(".review-canvas-host");
    const controlsSlot = workspace._stitchedControlsSlot;
    if (!toolbar || !layout || !canvasHost || !controlsSlot) return;
    workspace.dataset.stitchInspectorMounted = "1";
    workspace.classList.add("review-single-document", "review-lettering-workspace");

    const groups = groupsFromManifest();
    const sourcePages = [...groups.keys()];
    if (!sourcePages.length) return;

    const currentCard = workspace.querySelector(".review-card");
    const currentCanonical = Number.parseInt(currentCard?.dataset.pageIndex || "", 10);
    const currentPage = Number.isInteger(currentCanonical) ? window.currentManifest?.pages?.[currentCanonical] : null;
    activeSourcePage = currentPage ? cleanSourcePage(currentPage, currentCanonical) : (activeSourcePage ?? sourcePages[0]);
    if (!sourcePages.includes(activeSourcePage)) activeSourcePage = sourcePages[0];

    canvasHost.replaceChildren();
    const compatibilityCard = document.createElement("div");
    compatibilityCard.className = "review-card review-card-compat";
    compatibilityCard.hidden = true;
    canvasHost.appendChild(compatibilityCard);

    const shell = document.createElement("section");
    shell.className = "review-stitched-shell review-document-shell";

    const docbar = document.createElement("div");
    docbar.className = "review-stitched-toolbar review-document-toolbar";
    const prev = document.createElement("button");
    prev.type = "button"; prev.className = "ui-btn ui-btn-ghost"; prev.title = "Trang trước";
    prev.append(window.createUiIcon("chevron-left"));
    const select = document.createElement("select");
    select.className = "ui-select review-stitched-select";
    select.setAttribute("aria-label", "Chọn trang gốc");
    sourcePages.forEach((sourcePage) => select.add(new Option(`Trang ${sourcePage + 1}`, String(sourcePage))));
    const next = document.createElement("button");
    next.type = "button"; next.className = "ui-btn ui-btn-ghost"; next.title = "Trang sau";
    next.append(window.createUiIcon("chevron-right"));

    const cleanBtn = document.createElement("button");
    cleanBtn.type = "button"; cleanBtn.className = "ui-btn ui-btn-ghost ui-btn-compact"; cleanBtn.textContent = "Clean";
    const renderedBtn = document.createElement("button");
    renderedBtn.type = "button"; renderedBtn.className = "ui-btn ui-btn-ghost ui-btn-compact"; renderedBtn.textContent = "Rendered";
    const originalBtn = document.createElement("button");
    originalBtn.type = "button"; originalBtn.className = "ui-btn ui-btn-ghost ui-btn-compact"; originalBtn.textContent = "Original";

    const zoomOut = document.createElement("button");
    zoomOut.type = "button"; zoomOut.className = "ui-btn ui-btn-ghost ui-btn-compact"; zoomOut.textContent = "−";
    const zoomValue = document.createElement("button");
    zoomValue.type = "button"; zoomValue.className = "ui-btn ui-btn-ghost ui-btn-compact review-zoom-value"; zoomValue.textContent = "Fit";
    const zoomIn = document.createElement("button");
    zoomIn.type = "button"; zoomIn.className = "ui-btn ui-btn-ghost ui-btn-compact"; zoomIn.textContent = "+";
    const oneToOne = document.createElement("button");
    oneToOne.type = "button"; oneToOne.className = "ui-btn ui-btn-ghost ui-btn-compact"; oneToOne.textContent = "100%";

    docbar.append(prev, select, next, cleanBtn, renderedBtn, originalBtn, zoomOut, zoomValue, zoomIn, oneToOne);
    const meta = document.createElement("div"); meta.className = "review-stitched-meta";
    const warning = document.createElement("div"); warning.className = "review-stitched-warning"; warning.hidden = true;
    const viewport = document.createElement("div"); viewport.className = "review-stitched-viewport review-document-viewport";
    const zoomStage = document.createElement("div"); zoomStage.className = "review-stitched-zoom-stage";
    const imageHost = document.createElement("div"); imageHost.className = "review-stitched-image";
    zoomStage.appendChild(imageHost);
    viewport.appendChild(zoomStage);
    shell.append(docbar, meta, warning, viewport);
    canvasHost.appendChild(shell);

    const controls = document.createElement("div");
    controls.className = "review-controls review-stitched-controls";
    const brushBtn = document.createElement("button");
    brushBtn.type = "button"; brushBtn.className = "ui-btn ui-btn-ghost brush-toggle-btn"; brushBtn.textContent = "Brush / Inpaint"; brushBtn.setAttribute("aria-pressed", "false");
    const clearBtn = document.createElement("button");
    clearBtn.type = "button"; clearBtn.className = "ui-btn ui-btn-ghost clear-brush-btn"; clearBtn.textContent = "Xóa mask";
    const submitBtn = document.createElement("button");
    submitBtn.type = "button"; submitBtn.className = "ui-btn ui-btn-primary repaint-btn"; submitBtn.textContent = "Inpaint vùng chọn";
    const brushSizeWrap = document.createElement("label");
    brushSizeWrap.className = "ui-range-field brush-size-control"; brushSizeWrap.hidden = true; brushSizeWrap.textContent = "Brush ";
    const brushSizeValue = document.createElement("output"); brushSizeValue.textContent = "48px";
    const brushSize = document.createElement("input");
    brushSize.type = "range"; brushSize.min = "8"; brushSize.max = "80"; brushSize.step = "1"; brushSize.value = "24"; brushSize.className = "brush-size-slider";
    brushSizeWrap.append(brushSize, brushSizeValue);
    controls.append(brushBtn, clearBtn, submitBtn, brushSizeWrap);
    controlsSlot.replaceChildren(controls);

    installReviewOverlaySync(shell);
    mountTextInspector(workspace, shell, signal);

    let brushOn = false;
    let painting = false;
    let brushRadius = 24;
    let last = { x: 0, y: 0 };
    let boundCanvas = null;

    const syncVariantUI = () => {
      cleanBtn.classList.toggle("ui-btn-primary", variant === "clean");
      renderedBtn.classList.toggle("ui-btn-primary", variant === "rendered");
      originalBtn.classList.toggle("ui-btn-primary", variant === "original");
      const readonly = variant !== "clean";
      brushBtn.disabled = readonly;
      clearBtn.disabled = readonly;
      submitBtn.disabled = readonly;
      workspace.classList.toggle("review-readonly-document", readonly);
      if (readonly) { brushOn = false; brushSizeWrap.hidden = true; imageHost.classList.remove("brush-mode"); }
    };

    const syncBrushUI = () => {
      imageHost.classList.toggle("brush-mode", brushOn && variant === "clean");
      brushBtn.classList.toggle("ui-btn-primary", brushOn && variant === "clean");
      brushBtn.textContent = brushOn ? "Brush đang bật" : "Brush / Inpaint";
      brushBtn.setAttribute("aria-pressed", String(brushOn));
      brushSizeWrap.hidden = !brushOn || variant !== "clean";
      shell.querySelectorAll(".review-text-object-overlay").forEach((overlay) => {
        overlay.style.pointerEvents = brushOn ? "none" : "auto";
        overlay.style.opacity = brushOn ? ".35" : "1";
      });
    };

    const stopPainting = (event) => {
      painting = false;
      if (event?.pointerId !== undefined && boundCanvas?.hasPointerCapture?.(event.pointerId)) {
        try { boundCanvas.releasePointerCapture(event.pointerId); } catch (_) {}
      }
    };

    const brushHandlers = {
      bindCanvas(canvas) {
        boundCanvas = canvas;
        const ctx = canvas.getContext("2d");
        const coords = (event) => {
          const rect = canvas.getBoundingClientRect();
          return {
            x: Math.round((event.clientX - rect.left) * canvas.width / rect.width),
            y: Math.round((event.clientY - rect.top) * canvas.height / rect.height),
          };
        };
        const dot = (x, y) => {
          ctx.fillStyle = "rgba(220,38,38,.7)";
          ctx.beginPath(); ctx.arc(x, y, brushRadius, 0, Math.PI * 2); ctx.fill(); canvas._reviewDirty = true;
        };
        const stroke = (a, b) => {
          ctx.strokeStyle = "rgba(220,38,38,.7)"; ctx.lineWidth = brushRadius * 2; ctx.lineCap = "round"; ctx.lineJoin = "round";
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); canvas._reviewDirty = true;
        };
        canvas.addEventListener("pointerdown", (event) => {
          if (!brushOn || variant !== "clean" || event.button !== 0) return;
          painting = true; last = coords(event); canvas.setPointerCapture?.(event.pointerId); dot(last.x, last.y);
        }, { signal });
        canvas.addEventListener("pointermove", (event) => {
          if (!brushOn || !painting || variant !== "clean") return;
          const nextPos = coords(event); stroke(last, nextPos); last = nextPos;
        }, { signal });
        canvas.addEventListener("pointerup", stopPainting, { signal });
        canvas.addEventListener("pointercancel", stopPainting, { signal });
      },
    };

    const setBusy = (busy, text = "Đang xử lý…") => {
      workspace.classList.toggle("review-busy", busy);
      [brushBtn, clearBtn, submitBtn, brushSize, prev, next, select, cleanBtn, renderedBtn, originalBtn, zoomOut, zoomIn, oneToOne, zoomValue].forEach((el) => { if (el) el.disabled = busy; });
      submitBtn.textContent = busy ? text : "Inpaint vùng chọn";
      workspace._pageNavigator?.setBusy(busy);
      if (!busy) syncVariantUI();
    };

    const updateCompatibility = () => {
      const items = groups.get(activeSourcePage) || [];
      const canonical = items[0]?.canonicalIndex ?? 0;
      compatibilityCard.dataset.pageIndex = String(canonical);
      workspace.dataset.reviewCanonicalIndex = String(canonical);
      window.setWorkflowCheckpoint?.("review", canonical);
    };

    const rerender = () => {
      const items = groups.get(activeSourcePage);
      if (!items) return;
      select.value = String(activeSourcePage);
      const pos = sourcePages.indexOf(activeSourcePage);
      prev.disabled = pos <= 0;
      next.disabled = pos >= sourcePages.length - 1;
      updateCompatibility();
      const title = toolbar.querySelector(".review-toolbar-title");
      if (title) title.textContent = `Trang ${pos + 1} / ${sourcePages.length} · Review + Lettering`;
      syncVariantUI();
      void renderSourcePage(shell, activeSourcePage, items, brushHandlers, signal);
    };

    mountTextToolbar(workspace, shell, signal, rerender);

    const selectSource = (sourcePage) => {
      if (!groups.has(sourcePage) || sourcePage === activeSourcePage) return;
      captureSnapshot(shell);
      clearReviewTextSelection(shell);
      activeSourcePage = sourcePage;
      rerender();
    };

    const navigator = workspace._pageNavigator;
    if (navigator) {
      const sourceItems = sourcePages.map((sourcePage) => {
        const items = groups.get(sourcePage) || [];
        const needsReview = items.some(({ page }) => page.needs_review || page.detection_state === "needs_review" || (page.detection_issues || []).length);
        const textCount = items.reduce((sum, { canonicalIndex }) => sum + (window.currentManifest?.pages?.[canonicalIndex]?.text_objects?.length || 0), 0);
        return {
          key: sourcePage,
          label: `Trang ${sourcePage + 1}`,
          meta: textCount ? `${textCount} vùng chữ` : "Chưa có vùng chữ",
          image: items[0]?.page?.clean || items[0]?.page?.original || "",
          state: needsReview ? "review" : "ready",
          stateLabel: needsReview ? "Cần kiểm tra" : "Sẵn sàng",
        };
      });
      navigator.reconfigure({
        items: sourceItems,
        activeIndex: sourcePages.indexOf(activeSourcePage),
        title: "Trang",
        ariaLabel: "Điều hướng trang gốc",
        onSelect: (index) => { selectSource(sourcePages[index]); },
      });
      navigator.selectByKey = (canonicalIndex) => {
        const page = window.currentManifest?.pages?.[Number(canonicalIndex)];
        if (!page) return Promise.resolve(false);
        const sourcePage = cleanSourcePage(page, Number(canonicalIndex));
        const index = sourcePages.indexOf(sourcePage);
        if (index < 0) return Promise.resolve(false);
        return navigator.select(index);
      };
    }

    if (window._reviewKeyDownHandler) {
      window.removeEventListener("keydown", window._reviewKeyDownHandler);
      window._reviewKeyDownHandler = null;
    }
    const onKeyDown = (event) => {
      const editable = event.target?.matches?.("input,textarea,select") || event.target?.isContentEditable;
      if (editable) return;
      if (event.key === "ArrowLeft" || event.key === "PageUp") {
        const selected = shell.querySelector(".review-text-object-overlay.selected");
        if (selected && ["ArrowLeft"].includes(event.key)) return;
        const pos = sourcePages.indexOf(activeSourcePage); if (pos > 0) { event.preventDefault(); selectSource(sourcePages[pos - 1]); navigator?.setActive(pos - 1); }
      } else if (event.key === "ArrowRight" || event.key === "PageDown") {
        const selected = shell.querySelector(".review-text-object-overlay.selected");
        if (selected && ["ArrowRight"].includes(event.key)) return;
        const pos = sourcePages.indexOf(activeSourcePage); if (pos < sourcePages.length - 1) { event.preventDefault(); selectSource(sourcePages[pos + 1]); navigator?.setActive(pos + 1); }
      } else if ((event.ctrlKey || event.metaKey) && event.key === "0") {
        event.preventDefault(); fitWidth = true; applyZoom(shell);
      } else if ((event.ctrlKey || event.metaKey) && event.key === "1") {
        event.preventDefault(); fitWidth = false; zoomPercent = 100; applyZoom(shell);
      } else if (event.key === "[" && brushOn) {
        event.preventDefault(); brushRadius = Math.max(8, brushRadius - 2); brushSize.value = String(brushRadius); brushSizeValue.textContent = `${brushRadius * 2}px`;
      } else if (event.key === "]" && brushOn) {
        event.preventDefault(); brushRadius = Math.min(80, brushRadius + 2); brushSize.value = String(brushRadius); brushSizeValue.textContent = `${brushRadius * 2}px`;
      } else if ((event.key === "Delete" || event.key === "Backspace") && window.editorState?.selectedTextObjectId) {
        const pageIndex = Number(window.editorState.activePageIndex || 0);
        const id = window.editorState.selectedTextObjectId;
        event.preventDefault();
        window.deleteTextObject?.(pageIndex, id)?.then(() => renderTextOverlays(shell, signal)).catch((err) => window.showToast?.("Không thể xóa vùng chữ: " + err.message, "error"));
      }
    };
    window._reviewKeyDownHandler = onKeyDown;
    window.addEventListener("keydown", onKeyDown);

    let panning = false;
    let panStart = null;
    let spaceDown = false;
    window.addEventListener("keydown", (event) => { if (event.code === "Space" && !event.repeat) spaceDown = true; }, { signal });
    window.addEventListener("keyup", (event) => { if (event.code === "Space") { spaceDown = false; panning = false; panStart = null; viewport.classList.remove("is-panning"); } }, { signal });
    viewport.addEventListener("pointerdown", (event) => {
      if (brushOn || event.button !== 0 || !spaceDown) return;
      panning = true; panStart = { x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop };
      viewport.setPointerCapture?.(event.pointerId); viewport.classList.add("is-panning");
    }, { signal });
    viewport.addEventListener("pointermove", (event) => {
      if (!panning || !panStart) return;
      viewport.scrollLeft = panStart.left - (event.clientX - panStart.x);
      viewport.scrollTop = panStart.top - (event.clientY - panStart.y);
    }, { signal });
    viewport.addEventListener("pointerup", () => { panning = false; panStart = null; viewport.classList.remove("is-panning"); }, { signal });
    viewport.addEventListener("wheel", (event) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault(); stepZoom(event.deltaY < 0 ? 1 : -1); applyZoom(shell);
    }, { passive: false, signal });

    brushBtn.addEventListener("click", () => {
      if (variant !== "clean") return;
      brushOn = !brushOn;
      if (brushOn) window.setEditorTool?.("select");
      stopPainting(); syncBrushUI();
    }, { signal });
    clearBtn.addEventListener("click", () => {
      if (!boundCanvas) return;
      boundCanvas.getContext("2d").clearRect(0, 0, boundCanvas.width, boundCanvas.height);
      boundCanvas._reviewDirty = false; stitchedMaskSnapshots.delete(activeSourcePage);
    }, { signal });
    brushSize.addEventListener("input", () => { brushRadius = Number(brushSize.value); brushSizeValue.textContent = `${brushRadius * 2}px`; }, { signal });

    cleanBtn.addEventListener("click", () => { if (variant === "clean") return; variant = "clean"; rerender(); }, { signal });
    renderedBtn.addEventListener("click", () => { if (variant === "rendered") return; captureSnapshot(shell); variant = "rendered"; rerender(); }, { signal });
    originalBtn.addEventListener("click", () => { if (variant === "original") return; captureSnapshot(shell); variant = "original"; rerender(); }, { signal });
    zoomOut.addEventListener("click", () => { stepZoom(-1); applyZoom(shell); }, { signal });
    zoomIn.addEventListener("click", () => { stepZoom(1); applyZoom(shell); }, { signal });
    zoomValue.addEventListener("click", () => { fitWidth = true; applyZoom(shell); }, { signal });
    oneToOne.addEventListener("click", () => { fitWidth = false; zoomPercent = 100; applyZoom(shell); }, { signal });
    select.addEventListener("change", () => { const source = Number.parseInt(select.value, 10); selectSource(source); navigator?.setActive(sourcePages.indexOf(source)); }, { signal });
    prev.addEventListener("click", () => { const pos = sourcePages.indexOf(activeSourcePage); if (pos > 0) { selectSource(sourcePages[pos - 1]); navigator?.setActive(pos - 1); } }, { signal });
    next.addEventListener("click", () => { const pos = sourcePages.indexOf(activeSourcePage); if (pos < sourcePages.length - 1) { selectSource(sourcePages[pos + 1]); navigator?.setActive(pos + 1); } }, { signal });
    window.addEventListener("resize", () => { if (fitWidth) applyZoom(shell); }, { signal });

    submitBtn.addEventListener("click", async () => {
      const chapterId = window.currentChapterId;
      const brushCanvas = shell.querySelector("canvas.stitched-brush-canvas");
      if (!chapterId || !brushCanvas || !brushCanvas._reviewDirty || !regionHasPaint(brushCanvas, 0, 0, brushCanvas.width, brushCanvas.height)) {
        window.showToast?.("Chưa có vùng nào được đánh dấu.", "error"); return;
      }
      const repaintMode = typeof window.chooseRepaintMode === "function" ? await window.chooseRepaintMode() : "standard";
      if (!repaintMode || chapterId !== window.currentChapterId) return;
      setBusy(true, repaintMode === "lama" ? "LaMa đang xử lý…" : "Đang xử lý…");
      try {
        const descriptors = shell._descriptors || [];
        let affected = 0;
        for (const desc of descriptors) {
          const subH = desc.sourceY2 - desc.sourceY1;
          if (subH <= 0 || !regionHasPaint(brushCanvas, 0, desc.sourceY1, brushCanvas.width, subH)) continue;
          const maskCanvas = document.createElement("canvas");
          maskCanvas.width = brushCanvas.width;
          maskCanvas.height = desc.img.naturalHeight;
          maskCanvas.getContext("2d").drawImage(
            brushCanvas,
            0, desc.sourceY1, brushCanvas.width, subH,
            0, desc.localY1, brushCanvas.width, subH,
          );
          const maskBlob = await window.canvasToBlob(maskCanvas);
          const formData = new FormData();
          formData.append("chapter_id", chapterId);
          formData.append("page_index", desc.item.canonicalIndex);
          formData.append("mode", repaintMode);
          formData.append("mask", maskBlob, "mask.png");
          const response = await fetch("/api/repaint_mask", { method: "POST", body: formData });
          const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({})));
          const data = await parse(response);
          if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
          if (window.currentManifest?.pages?.[desc.item.canonicalIndex] && data.pages?.[desc.item.canonicalIndex]) {
            window.currentManifest.pages[desc.item.canonicalIndex] = data.pages[desc.item.canonicalIndex];
          }
          affected++;
        }
        brushCanvas.getContext("2d").clearRect(0, 0, brushCanvas.width, brushCanvas.height);
        brushCanvas._reviewDirty = false;
        stitchedMaskSnapshots.delete(activeSourcePage);
        window.showToast?.(`Đã xử lý ${affected} vùng ảnh trên trang ${activeSourcePage + 1}.`, affected ? "success" : "info");
        rerender();
      } catch (err) {
        window.showToast?.("Không thể xử lý vùng đánh dấu: " + err.message, "error");
      } finally { setBusy(false); }
    }, { signal });

    syncVariantUI();
    syncBrushUI();
    rerender();
  }

  function scan() {
    document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount);
  }

  window.mountStitchInspector = scan;

  // Legacy manifests may still point at the retired editor stage. Keep their
  // data, but route the old entry point into the unified review + lettering workspace.
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelector('.sidebar-link[data-stage="editor"]')?.remove();
    window.renderEditor = function renderEditorIntoUnifiedReview() {
      const pageIndex = Number(window.editorState?.activePageIndex || 0);
      window.initialReviewCanonicalPageIndex = pageIndex;
      window.renderReview?.();
    };
  });
})();
