(() => {
  const MAX_CANVAS_CHUNK_HEIGHT = 12000;
  const ZOOM_STEPS = [25, 50, 75, 100, 125, 150, 200, 300, 400];
  const MIN_TEXT_BOX_SIZE = 10;
  const TOOL_SHORTCUTS = {
    v: "select",
    r: "rectangle",
    o: "ellipse",
    b: "brush",
    e: "eraser",
    h: "hand",
    z: "zoom",
  };

  let activeSourcePage = null;
  let renderToken = 0;
  let zoomPercent = 100;
  let fitWidth = true;
  let variant = "clean";
  let activeTool = "select";
  let activeChapterKey = null;
  const stitchedMaskSnapshots = new Map();
  const autoSyncChecked = new Set();

  function chapterKey() {
    return String(window.currentChapterId || "");
  }

  function snapshotKey(sourcePage) {
    return `${chapterKey()}:${Number(sourcePage)}`;
  }

  function resetChapterScopedState() {
    const key = chapterKey();
    if (activeChapterKey === key) return;
    activeChapterKey = key;
    activeSourcePage = null;
    variant = "clean";
    activeTool = "select";
    fitWidth = true;
    zoomPercent = 100;
    stitchedMaskSnapshots.clear();
    autoSyncChecked.clear();
  }

  function ensureToolStyles() {
    if (document.getElementById("review-photopea-tool-styles")) return;
    const style = document.createElement("style");
    style.id = "review-photopea-tool-styles";
    style.textContent = `
      body.review-tool-rail-active .app-sidebar > :not(.review-tool-rail) { display: none !important; }
      .review-tool-rail {
        display: flex;
        width: 100%;
        height: 100%;
        min-height: 0;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        padding: 3px 0;
      }
      .review-tool-group {
        display: flex;
        width: 100%;
        flex-direction: column;
        align-items: center;
        gap: 2px;
      }
      .review-tool-group-bottom {
        margin-top: auto;
        padding-top: 5px;
        border-top: 1px solid var(--border-subtle);
      }
      .review-rail-tool {
        position: relative;
        display: grid;
        width: 42px;
        height: 42px;
        place-items: center;
        padding: 0;
        border: 1px solid transparent;
        border-radius: 3px;
        background: transparent;
        color: var(--text-secondary);
        cursor: pointer;
      }
      .review-rail-tool:hover,
      .review-rail-tool:focus-visible {
        outline: 0;
        border-color: var(--border-subtle);
        background: var(--surface-hover);
        color: var(--text-primary);
      }
      .review-rail-tool.active {
        border-color: var(--border-strong);
        background: color-mix(in srgb, var(--accent) 24%, var(--surface-raised));
        color: var(--text-primary);
      }
      .review-rail-tool:disabled { opacity: .45; cursor: default; }
      .review-rail-tool > .ui-icon { width: 21px; height: 21px; }
      .review-tool-tooltip {
        position: absolute;
        z-index: calc(var(--z-toolbar) + 40);
        left: 48px;
        top: 50%;
        display: none;
        width: 230px;
        padding: 7px 9px;
        border: 1px solid var(--border-strong);
        border-radius: 4px;
        background: var(--surface-panel);
        box-shadow: var(--shadow-float);
        color: var(--text-secondary);
        text-align: left;
        transform: translateY(-50%);
        pointer-events: none;
      }
      .review-tool-tooltip strong {
        display: block;
        margin-bottom: 2px;
        color: var(--text-primary);
        font-size: 11px;
        line-height: 1.25;
      }
      .review-tool-tooltip span {
        display: block;
        font-size: 10px;
        line-height: 1.35;
      }
      .review-rail-tool:hover .review-tool-tooltip,
      .review-rail-tool:focus-visible .review-tool-tooltip { display: block; }

      .review-single-document .review-stitched-image > .review-image-chunk {
        position: relative;
        z-index: 1;
        display: block;
        width: 100%;
        height: auto;
      }
      .review-single-document .review-stitched-image > .stitched-brush-chunk {
        position: absolute;
        z-index: 5;
        left: 0;
        display: block;
        width: 100%;
        height: auto;
        pointer-events: none;
      }
      .review-text-object-overlay {
        z-index: 12;
        overflow: visible !important;
        border: 1.5px solid rgba(55, 155, 255, .95);
        background: rgba(55, 155, 255, .045);
      }
      .review-text-object-overlay.ellipse { border-radius: 999px; }
      .review-text-object-overlay.selected {
        border-color: #57a8ff;
        box-shadow: 0 0 0 1px rgba(87, 168, 255, .45);
      }
      .review-text-object-overlay.tool-muted {
        opacity: .42;
        pointer-events: none !important;
      }
      .review-inline-ocr {
        position: absolute;
        bottom: calc(100% + 4px);
        min-height: 22px;
        overflow: hidden;
        padding: 4px 6px;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 3px;
        background: rgba(25,25,25,.94);
        box-shadow: 0 2px 8px rgba(0,0,0,.32);
        color: #f2f2f2;
        font: 11px/1.25 system-ui, sans-serif;
        text-overflow: ellipsis;
        white-space: nowrap;
        user-select: text;
        pointer-events: auto;
      }
      .review-inline-translation {
        position: absolute;
        top: calc(100% + 4px);
        min-height: 28px;
        max-height: 156px;
        overflow: hidden;
        resize: none;
        padding: 5px 7px;
        border: 1px solid rgba(87,168,255,.75);
        border-radius: 3px;
        outline: 0;
        background: rgba(250,250,250,.97);
        box-shadow: 0 2px 8px rgba(0,0,0,.28);
        color: #171717;
        font: 12px/18px system-ui, sans-serif;
        white-space: pre-wrap;
        pointer-events: auto;
      }
      .review-inline-translation:focus {
        border-color: #57a8ff;
        box-shadow: 0 0 0 1px rgba(87,168,255,.38), 0 2px 8px rgba(0,0,0,.28);
      }
      .review-text-drawing {
        position: absolute;
        z-index: 20;
        border: 1.5px dashed #57a8ff;
        background: rgba(87,168,255,.08);
        pointer-events: none;
      }
      .review-text-drawing.ellipse { border-radius: 999px; }

      .review-document-viewport[data-active-tool="hand"] { cursor: grab; }
      .review-document-viewport[data-active-tool="hand"].is-panning { cursor: grabbing; }
      .review-document-viewport[data-active-tool="zoom"] { cursor: zoom-in; }
      .review-stitched-image[data-active-tool="rectangle"],
      .review-stitched-image[data-active-tool="ellipse"],
      .review-stitched-image[data-active-tool="brush"],
      .review-stitched-image[data-active-tool="eraser"] { cursor: crosshair; }

      @media (max-width: 720px) {
        .review-rail-tool { width: 38px; height: 38px; }
        .review-tool-tooltip { display: none !important; }
      }
    `;
    document.head.appendChild(style);
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
      groups.get(sourcePage).push({ canonicalIndex });
    });
    for (const items of groups.values()) {
      items.sort((a, b) => {
        const pageA = window.currentManifest?.pages?.[a.canonicalIndex];
        const pageB = window.currentManifest?.pages?.[b.canonicalIndex];
        return cleanSliceIndex(pageA) - cleanSliceIndex(pageB);
      });
    }
    return new Map([...groups.entries()].sort((a, b) => a[0] - b[0]));
  }

  function livePage(item) {
    return window.currentManifest?.pages?.[Number(item?.canonicalIndex)] || null;
  }

  function imageUrl(page, targetVariant = variant) {
    if (!page) return null;
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

  function makeImageChunks(host, width, height) {
    const chunks = [];
    for (let y = 0; y < height; y += MAX_CANVAS_CHUNK_HEIGHT) {
      const chunkHeight = Math.min(MAX_CANVAS_CHUNK_HEIGHT, height - y);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = chunkHeight;
      canvas.dataset.sourceY = String(y);
      canvas.className = "review-image-chunk";
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

  function makeBrushChunks(host, width, height) {
    const chunks = [];
    for (let y = 0; y < height; y += MAX_CANVAS_CHUNK_HEIGHT) {
      const chunkHeight = Math.min(MAX_CANVAS_CHUNK_HEIGHT, height - y);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = chunkHeight;
      canvas.dataset.sourceY = String(y);
      canvas.className = "stitched-brush-chunk";
      canvas.style.top = `${y}px`;
      host.appendChild(canvas);
      chunks.push({
        canvas,
        ctx: canvas.getContext("2d", { willReadFrequently: true }),
        y1: y,
        y2: y + chunkHeight,
        dirty: false,
      });
    }
    return chunks;
  }

  function chunkHasPaint(chunk, sourceY1 = chunk.y1, sourceY2 = chunk.y2) {
    if (!chunk?.dirty) return false;
    const iy1 = Math.max(chunk.y1, sourceY1);
    const iy2 = Math.min(chunk.y2, sourceY2);
    if (iy2 <= iy1) return false;
    const localY1 = iy1 - chunk.y1;
    const h = iy2 - iy1;
    const width = chunk.canvas.width;
    const scale = Math.min(1, 512 / Math.max(width, h));
    const probe = document.createElement("canvas");
    probe.width = Math.max(1, Math.ceil(width * scale));
    probe.height = Math.max(1, Math.ceil(h * scale));
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(chunk.canvas, 0, localY1, width, h, 0, 0, probe.width, probe.height);
    const pixels = ctx.getImageData(0, 0, probe.width, probe.height).data;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 20) return true;
    }
    return false;
  }

  function brushRegionHasPaint(chunks, sourceY1, sourceY2) {
    return (chunks || []).some((chunk) => chunkHasPaint(chunk, sourceY1, sourceY2));
  }

  function clearBrushChunks(chunks) {
    (chunks || []).forEach((chunk) => {
      chunk.ctx.clearRect(0, 0, chunk.canvas.width, chunk.canvas.height);
      chunk.dirty = false;
    });
  }

  function captureBrushSnapshot(shell) {
    if (activeSourcePage === null || !shell || variant !== "clean") return;
    const chunks = shell._brushChunks || [];
    const dirty = chunks.filter((chunk) => chunk.dirty && chunkHasPaint(chunk));
    const key = snapshotKey(activeSourcePage);
    if (!dirty.length) {
      stitchedMaskSnapshots.delete(key);
      return;
    }
    const payload = dirty.map((chunk) => ({
      y1: chunk.y1,
      dataUrl: chunk.canvas.toDataURL("image/png"),
    }));
    stitchedMaskSnapshots.set(key, payload);
  }

  function restoreBrushSnapshot(shell, sourcePage) {
    const payload = stitchedMaskSnapshots.get(snapshotKey(sourcePage));
    if (!Array.isArray(payload) || !payload.length) return;
    const byY = new Map((shell._brushChunks || []).map((chunk) => [chunk.y1, chunk]));
    payload.forEach((item) => {
      const chunk = byY.get(Number(item.y1));
      if (!chunk) return;
      const img = new Image();
      img.onload = () => {
        if (!chunk.canvas.isConnected) return;
        chunk.ctx.drawImage(img, 0, 0);
        chunk.dirty = true;
      };
      img.src = item.dataUrl;
    });
  }

  window.hasUnsavedStitchedMarks = () => {
    const key = activeSourcePage === null ? null : snapshotKey(activeSourcePage);
    if (key && stitchedMaskSnapshots.has(key)) return true;
    const shell = document.querySelector("#page-view.review-mode .review-document-shell");
    return Boolean(shell?._brushChunks?.some((chunk) => chunk.dirty && chunkHasPaint(chunk)));
  };

  function zoomScale(viewport, sourceWidth) {
    if (!sourceWidth) return 1;
    if (fitWidth) {
      const available = Math.max(240, viewport.clientWidth - 48);
      return Math.min(1, available / sourceWidth);
    }
    return zoomPercent / 100;
  }

  function applyZoom(shell, focus = null) {
    const viewport = shell.querySelector(".review-stitched-viewport");
    const stage = shell.querySelector(".review-stitched-zoom-stage");
    const imageHost = shell.querySelector(".review-stitched-image");
    const zoomLabel = shell.querySelector(".review-zoom-value");
    const width = Number(imageHost?.dataset.sourceWidth || 0);
    const height = Number(imageHost?.dataset.sourceHeight || 0);
    if (!viewport || !stage || !imageHost || !width || !height) return;
    const previousScale = Number(imageHost.dataset.zoomScale || 1);
    const scale = zoomScale(viewport, width);
    let sourceFocus = null;
    if (focus && previousScale > 0) {
      sourceFocus = {
        x: (viewport.scrollLeft + focus.x) / previousScale,
        y: (viewport.scrollTop + focus.y) / previousScale,
      };
    }
    imageHost.style.transform = `scale(${scale})`;
    imageHost.dataset.zoomScale = String(scale);
    stage.style.width = `${Math.max(1, Math.round(width * scale))}px`;
    stage.style.height = `${Math.max(1, Math.round(height * scale))}px`;
    if (sourceFocus) {
      viewport.scrollLeft = Math.max(0, sourceFocus.x * scale - focus.x);
      viewport.scrollTop = Math.max(0, sourceFocus.y * scale - focus.y);
    }
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
      const pageIndex = Number(desc.item.canonicalIndex);
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

  function autoGrowTranslation(textarea) {
    if (!textarea) return;
    textarea.style.height = "0px";
    const lineHeight = 18;
    const minHeight = lineHeight + 10;
    const maxHeight = lineHeight * 8 + 12;
    textarea.style.height = `${Math.max(minHeight, Math.min(maxHeight, textarea.scrollHeight))}px`;
  }

  function syncReviewOverlay(shell, pageIndex, id) {
    if (!shell?.isConnected) return false;
    const desc = descriptorForCanonical(shell, pageIndex);
    const obj = typeof window.findTextObject === "function" ? window.findTextObject(pageIndex, id) : null;
    const overlay = shell.querySelector(`.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`);
    if (!desc || !obj?.region || !overlay) return false;
    const r = sourceRegionFor(desc, obj.region);
    const width = Math.max(MIN_TEXT_BOX_SIZE, r.x2 - r.x1);
    const height = Math.max(MIN_TEXT_BOX_SIZE, r.y2 - r.y1);
    overlay.style.left = `${r.x1}px`;
    overlay.style.top = `${r.y1}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    const sourceWidth = Number(shell.querySelector(".review-stitched-image")?.dataset.sourceWidth || 0);
    const inlineWidth = Math.max(150, Math.min(360, width));
    const alignRight = sourceWidth > 0 && r.x1 + inlineWidth > sourceWidth;
    const ocr = overlay.querySelector(".review-inline-ocr");
    const translation = overlay.querySelector(".review-inline-translation");
    if (ocr) {
      ocr.textContent = obj.ocr_text?.trim() || "Đang OCR…";
      ocr.style.width = `${inlineWidth}px`;
      ocr.style.left = alignRight ? "auto" : "0";
      ocr.style.right = alignRight ? "0" : "auto";
    }
    if (translation) {
      if (translation !== document.activeElement) translation.value = obj.translation || "";
      translation.style.width = `${inlineWidth}px`;
      translation.style.left = alignRight ? "auto" : "0";
      translation.style.right = alignRight ? "0" : "auto";
      autoGrowTranslation(translation);
    }
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
      if (variant !== "clean" || activeTool !== "select" || event.button !== 0) return;
      if (event.target.closest(".review-inline-ocr,.review-inline-translation")) return;
      event.preventDefault();
      event.stopPropagation();
      selectReviewTextObject(shell, pageIndex, obj.id);
      const rect = imageHost.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const sourceW = Number(imageHost.dataset.sourceWidth || 0);
      const sourceH = Number(imageHost.dataset.sourceHeight || 0);
      active = {
        mode: overlayMode(event, overlay),
        startX: (event.clientX - rect.left) * sourceW / rect.width,
        startY: (event.clientY - rect.top) * sourceH / rect.height,
        original: { ...obj.region },
      };
      overlay.classList.add("transforming");
      overlay.setPointerCapture?.(event.pointerId);
    }, { signal });

    const move = (event) => {
      if (!active) {
        if (activeTool === "select") overlay.style.cursor = cursorForMode(overlayMode(event, overlay));
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

  function createInlineEditor(shell, desc, pageIndex, obj, signal) {
    const overlay = document.createElement("div");
    overlay.className = "text-object-overlay review-text-object-overlay" + (obj.shape === "ellipse" ? " ellipse" : "");
    overlay.dataset.pageIndex = String(pageIndex);
    overlay.dataset.objectId = String(obj.id);

    const ocr = document.createElement("div");
    ocr.className = "review-inline-ocr";
    ocr.title = "OCR nguồn";
    ocr.textContent = obj.ocr_text?.trim() || "Đang OCR…";
    ocr.addEventListener("pointerdown", (event) => event.stopPropagation(), { signal });
    ocr.addEventListener("click", (event) => {
      event.stopPropagation();
      selectReviewTextObject(shell, pageIndex, obj.id);
    }, { signal });

    const translation = document.createElement("textarea");
    translation.className = "review-inline-translation";
    translation.rows = 1;
    translation.placeholder = "Bản dịch…";
    translation.value = obj.translation || "";
    translation.spellcheck = false;
    translation.addEventListener("pointerdown", (event) => event.stopPropagation(), { signal });
    translation.addEventListener("click", (event) => {
      event.stopPropagation();
      selectReviewTextObject(shell, pageIndex, obj.id);
    }, { signal });
    translation.addEventListener("input", () => {
      const live = window.findTextObject?.(pageIndex, obj.id);
      if (!live) return;
      live.translation = translation.value;
      autoGrowTranslation(translation);
      window.scheduleTextObjectPersist?.(pageIndex, obj.id);
      const panelText = document.querySelector(`.review-inspector .translation-textarea[data-text-object-id="${CSS.escape(String(obj.id))}"]`);
      if (panelText && panelText !== document.activeElement) panelText.value = translation.value;
    }, { signal });

    overlay.append(ocr, translation);
    overlay.addEventListener("click", (event) => {
      if (event.target.closest(".review-inline-ocr,.review-inline-translation")) return;
      event.preventDefault();
      event.stopPropagation();
      selectReviewTextObject(shell, pageIndex, obj.id);
    }, { signal });
    shell.querySelector(".review-stitched-image")?.appendChild(overlay);
    syncReviewOverlay(shell, pageIndex, obj.id);
    installOverlayTransform(shell, overlay, desc, pageIndex, obj, signal);
    return overlay;
  }

  function renderTextOverlays(shell, signal) {
    const imageHost = shell.querySelector(".review-stitched-image");
    if (!imageHost) return;
    imageHost.querySelectorAll(".review-text-object-overlay,.review-text-drawing").forEach((node) => node.remove());
    if (variant !== "clean") return;
    for (const { desc, pageIndex, obj } of ownedTextObjects(shell)) {
      createInlineEditor(shell, desc, pageIndex, obj, signal);
    }
    syncToolPresentation(shell);
  }

  async function waitForOcr(shell, pageIndex, id, signal) {
    const started = Date.now();
    let last = "";
    while (!signal.aborted && Date.now() - started < 8000) {
      const obj = window.findTextObject?.(pageIndex, id);
      const text = obj?.ocr_text?.trim() || "";
      if (text && text !== last) {
        last = text;
        syncReviewOverlay(shell, pageIndex, id);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    syncReviewOverlay(shell, pageIndex, id);
  }

  async function ensureSourceTextObjects(shell) {
    if (typeof window.ensureAutoTextObjects !== "function") return;
    for (const desc of shell._descriptors || []) {
      const pageIndex = Number(desc.item.canonicalIndex);
      const key = `${chapterKey()}:${pageIndex}`;
      if (autoSyncChecked.has(key)) continue;
      autoSyncChecked.add(key);
      try {
        await window.ensureAutoTextObjects(pageIndex);
      } catch (err) {
        console.warn("Could not ensure text objects for review page", pageIndex, err);
      }
    }
  }

  function ownerDescriptorForSourceY(shell, sourceY) {
    return (shell._descriptors || []).find((desc) => sourceY >= desc.sourceY1 && sourceY < desc.sourceY2) || null;
  }

  function sourcePoint(imageHost, event) {
    const rect = imageHost.getBoundingClientRect();
    const W = Number(imageHost.dataset.sourceWidth || 0);
    const H = Number(imageHost.dataset.sourceHeight || 0);
    if (!rect.width || !rect.height || !W || !H) return null;
    return {
      x: Math.max(0, Math.min(W, (event.clientX - rect.left) * W / rect.width)),
      y: Math.max(0, Math.min(H, (event.clientY - rect.top) * H / rect.height)),
    };
  }

  function installTextDrawing(shell, signal) {
    const imageHost = shell.querySelector(".review-stitched-image");
    let drawing = null;

    const update = (point) => {
      if (!drawing || !point) return;
      const x1 = Math.min(drawing.start.x, point.x);
      const y1 = Math.min(drawing.start.y, point.y);
      const x2 = Math.max(drawing.start.x, point.x);
      const y2 = Math.max(drawing.start.y, point.y);
      Object.assign(drawing.preview.style, {
        left: `${x1}px`,
        top: `${y1}px`,
        width: `${x2 - x1}px`,
        height: `${y2 - y1}px`,
      });
      drawing.last = point;
    };

    imageHost.addEventListener("pointerdown", (event) => {
      if (variant !== "clean" || !["rectangle", "ellipse"].includes(activeTool) || event.button !== 0) return;
      if (event.target.closest(".review-text-object-overlay")) return;
      const point = sourcePoint(imageHost, event);
      if (!point) return;
      event.preventDefault();
      const preview = document.createElement("div");
      preview.className = "text-object-overlay drawing review-text-drawing" + (activeTool === "ellipse" ? " ellipse" : "");
      imageHost.appendChild(preview);
      drawing = { tool: activeTool, start: point, last: point, preview };
      imageHost.setPointerCapture?.(event.pointerId);
      update(point);
    }, { signal });

    imageHost.addEventListener("pointermove", (event) => {
      if (!drawing) return;
      update(sourcePoint(imageHost, event));
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
        const id = window.editorState?.selectedTextObjectId;
        window.editorState.activePageIndex = pageIndex;
        renderTextOverlays(shell, signal);
        if (id) {
          selectReviewTextObject(shell, pageIndex, id);
          void waitForOcr(shell, pageIndex, id, signal);
        }
        setActiveTool(shell, "select");
      } catch (err) {
        window.showToast?.("Không thể tạo vùng chữ: " + err.message, "error");
      }
    };

    window.addEventListener("pointerup", finish, { signal });
    window.addEventListener("pointercancel", finish, { signal });

    imageHost.addEventListener("click", (event) => {
      if (activeTool !== "select") return;
      if (event.target.closest(".review-text-object-overlay")) return;
      clearReviewTextSelection(shell);
    }, { signal });
  }

  function renderSourcePage(shell, sourcePage, items, signal) {
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

    return (async () => {
      const firstPage = livePage(items[0]);
      const firstUrl = imageUrl(firstPage);
      if (!firstUrl) throw new Error("Trang không có ảnh để hiển thị.");
      const first = await loadImage(firstUrl);
      if (token !== renderToken) return;
      const firstCore = coreMetadata(firstPage, first.naturalHeight);
      let sourceHeight = firstCore?.sourceHeight || first.naturalHeight;
      const width = first.naturalWidth;
      let metadataValid = Boolean(firstCore) || items.length === 1;
      let expectedSourceY = 0;
      let fallbackY = 0;
      const descriptors = [];

      for (let index = 0; index < items.length; index++) {
        const item = items[index];
        const page = livePage(item);
        if (!page) throw new Error(`Vùng ảnh ${index + 1} không còn trong manifest.`);
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
          descriptors.push({ item: { canonicalIndex: item.canonicalIndex }, img, localY1: core.localY1, localY2: core.localY2, sourceY1: core.sourceY1, sourceY2: core.sourceY2 });
        } else {
          metadataValid = false;
          descriptors.push({ item: { canonicalIndex: item.canonicalIndex }, img, localY1: 0, localY2: img.naturalHeight, sourceY1: fallbackY, sourceY2: fallbackY + img.naturalHeight });
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
      const imageChunks = makeImageChunks(imageHost, width, sourceHeight);
      for (const desc of descriptors) {
        drawIntoChunks(imageChunks, desc.img, 0, desc.localY1, width, desc.localY2 - desc.localY1, desc.sourceY1);
      }
      shell._descriptors = descriptors;

      if (variant === "clean") {
        shell._brushChunks = makeBrushChunks(imageHost, width, sourceHeight);
        restoreBrushSnapshot(shell, sourcePage);
      } else {
        shell._brushChunks = [];
      }

      const variantLabel = variant === "original" ? "Ảnh gốc" : (variant === "rendered" ? "Bản lettering" : "Sau inpaint");
      meta.textContent = `Trang ${sourcePage + 1} · ${width} × ${sourceHeight}px · ${variantLabel}`;
      applyZoom(shell);
      await ensureSourceTextObjects(shell);
      if (token !== renderToken) return;
      renderTextOverlays(shell, signal);
      syncToolPresentation(shell);
    })().catch((err) => {
      if (token !== renderToken) return;
      imageHost.replaceChildren();
      shell._brushChunks = [];
      const error = document.createElement("div");
      error.className = "ui-state ui-state-error review-stitched-error";
      error.textContent = `Không dựng được trang: ${err.message}`;
      imageHost.appendChild(error);
    });
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
      const target = event.target;
      const pageIndex = Number(window.editorState?.activePageIndex || 0);
      const id = window.editorState?.selectedTextObjectId;
      if (!id) return;
      const obj = window.findTextObject?.(pageIndex, id);
      if (!obj) return;
      if (target.classList?.contains("translation-textarea")) {
        const inline = shell.querySelector(`.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"] .review-inline-translation`);
        if (inline && inline !== document.activeElement) {
          inline.value = target.value;
          autoGrowTranslation(inline);
        }
      } else if (target.classList?.contains("ocr-textarea")) {
        const bar = shell.querySelector(`.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"] .review-inline-ocr`);
        if (bar) bar.textContent = target.value.trim() || "OCR nguồn";
      }
    }, { signal });
    return host;
  }

  function setToolButtonState(tool) {
    document.querySelectorAll(".review-rail-tool").forEach((button) => {
      const active = button.dataset.tool === tool;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function syncToolPresentation(shell) {
    const imageHost = shell?.querySelector(".review-stitched-image");
    const viewport = shell?.querySelector(".review-document-viewport");
    if (!imageHost || !viewport) return;
    imageHost.dataset.activeTool = activeTool;
    viewport.dataset.activeTool = activeTool;
    const paintMode = activeTool === "brush" || activeTool === "eraser";
    const textMode = activeTool === "rectangle" || activeTool === "ellipse";
    (shell._brushChunks || []).forEach((chunk) => {
      chunk.canvas.style.pointerEvents = paintMode && variant === "clean" ? "auto" : "none";
    });
    shell.querySelectorAll(".review-text-object-overlay").forEach((overlay) => {
      overlay.style.pointerEvents = activeTool === "select" && variant === "clean" ? "auto" : "none";
      overlay.classList.toggle("tool-muted", variant === "clean" && activeTool !== "select");
    });
    imageHost.classList.toggle("text-draw-mode", textMode && variant === "clean");
    imageHost.classList.toggle("brush-mode", paintMode && variant === "clean");
    viewport.classList.toggle("hand-mode", activeTool === "hand");
    viewport.classList.toggle("zoom-mode", activeTool === "zoom");
    setToolButtonState(activeTool);
  }

  function setActiveTool(shell, tool) {
    const allowed = new Set(["select", "rectangle", "ellipse", "brush", "eraser", "hand", "zoom"]);
    if (!allowed.has(tool)) tool = "select";
    if (variant !== "clean" && ["rectangle", "ellipse", "brush", "eraser"].includes(tool)) {
      variant = "clean";
      const items = groupsFromManifest().get(activeSourcePage);
      if (items) void renderSourcePage(shell, activeSourcePage, items, shell._toolSignal);
    }
    activeTool = tool;
    if (window.editorState) window.editorState.tool = ["rectangle", "ellipse"].includes(tool) ? tool : "select";
    window.setEditorTool?.(window.editorState?.tool || "select");
    syncToolPresentation(shell);
  }

  function createToolButton(tool, icon, title, description, shell, signal, handler = null) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "review-rail-tool";
    button.dataset.tool = tool;
    button.setAttribute("aria-label", `${title}: ${description}`);
    button.setAttribute("aria-pressed", "false");
    button.append(window.createUiIcon(icon));
    const tip = document.createElement("span");
    tip.className = "review-tool-tooltip";
    const strong = document.createElement("strong");
    strong.textContent = title;
    const desc = document.createElement("span");
    desc.textContent = description;
    tip.append(strong, desc);
    button.appendChild(tip);
    button.addEventListener("click", () => {
      if (handler) handler();
      else setActiveTool(shell, tool);
    }, { signal });
    return button;
  }

  function mountToolRail(workspace, shell, signal) {
    const sidebar = document.getElementById("app-sidebar");
    if (!sidebar) return;
    sidebar.querySelector(".review-tool-rail")?.remove();
    document.body.classList.add("review-tool-rail-active");

    const rail = document.createElement("div");
    rail.className = "review-tool-rail";
    rail.setAttribute("role", "toolbar");
    rail.setAttribute("aria-label", "Công cụ chỉnh sửa ảnh và lettering");

    const primary = document.createElement("div");
    primary.className = "review-tool-group";
    [
      ["select", "cursor", "Chọn", "Chọn, kéo và thay đổi kích thước vùng chữ."],
      ["rectangle", "rect-select", "Vùng chữ nhật", "Kéo vùng chữ nhật quanh bong bóng để tạo vùng OCR."],
      ["ellipse", "ellipse-select", "Vùng elip", "Kéo vùng elip quanh bong bóng tròn để tạo vùng OCR."],
      ["brush", "brush", "Cọ Inpaint", "Tô trực tiếp vùng cần xóa rồi chạy Inpaint."],
      ["eraser", "eraser", "Tẩy mask", "Xóa phần mask inpaint đã tô nhầm."],
      ["hand", "hand", "Bàn tay", "Kéo trang tự do theo mọi hướng."],
      ["zoom", "zoom", "Thu phóng", "Nhấp để phóng to, Alt + nhấp để thu nhỏ."],
    ].forEach(([tool, icon, title, description]) => {
      primary.appendChild(createToolButton(tool, icon, title, description, shell, signal));
    });

    const secondary = document.createElement("div");
    secondary.className = "review-tool-group review-tool-group-bottom";
    secondary.appendChild(createToolButton(
      "settings",
      "settings",
      "Cài đặt",
      "Mở cài đặt ứng dụng và dịch vụ AI.",
      shell,
      signal,
      () => window.openAppSettings?.(),
    ));

    rail.append(primary, secondary);
    sidebar.appendChild(rail);

    signal.addEventListener("abort", () => {
      rail.remove();
      document.body.classList.remove("review-tool-rail-active");
    }, { once: true });
    setToolButtonState(activeTool);
  }

  function mountActionToolbar(workspace, shell, signal, rerender) {
    const actions = workspace.querySelector(".review-actions-group");
    if (!actions) return;
    actions.querySelector(".review-primary-action")?.remove();

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
          const page = window.currentManifest?.pages?.[pageIndex];
          if (page?.rendered) {
            page._reviewRenderedUrl = page.rendered;
            renderedCount++;
          }
        }
        if (!renderedCount) {
          window.showToast?.("Không có vùng chữ nào được render trên trang này.", "info");
          return;
        }
        variant = "rendered";
        window.showToast?.(`Đã render lettering cho trang ${activeSourcePage + 1}.`, "success");
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
  }

  function paintPoint(shell, x, y, radius, erase) {
    const chunk = (shell._brushChunks || []).find((item) => y >= item.y1 && y < item.y2);
    if (!chunk) return;
    const ctx = chunk.ctx;
    ctx.save();
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    ctx.fillStyle = "rgba(220,38,38,.72)";
    ctx.beginPath();
    ctx.arc(x, y - chunk.y1, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    chunk.dirty = true;
  }

  function paintStroke(shell, a, b, radius, erase) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const distance = Math.hypot(dx, dy);
    const step = Math.max(1, radius * 0.45);
    const count = Math.max(1, Math.ceil(distance / step));
    for (let i = 0; i <= count; i++) {
      const t = i / count;
      paintPoint(shell, a.x + dx * t, a.y + dy * t, radius, erase);
    }
  }

  function drawBrushChunksIntoMask(maskCtx, chunks, desc, width) {
    for (const chunk of chunks || []) {
      const iy1 = Math.max(chunk.y1, desc.sourceY1);
      const iy2 = Math.min(chunk.y2, desc.sourceY2);
      if (iy2 <= iy1) continue;
      const h = iy2 - iy1;
      maskCtx.drawImage(
        chunk.canvas,
        0,
        iy1 - chunk.y1,
        width,
        h,
        0,
        desc.localY1 + (iy1 - desc.sourceY1),
        width,
        h,
      );
    }
  }

  function canvasToBlob(canvas) {
    if (typeof window.canvasToBlob === "function") return window.canvasToBlob(canvas);
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Không thể mã hóa mask")), "image/png");
    });
  }

  function mount(workspace) {
    if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchInspectorMounted === "1") return;
    resetChapterScopedState();
    ensureToolStyles();
    workspace._stitchAbort?.abort();
    window._reviewStitchAbort?.abort();
    const aborter = new AbortController();
    workspace._stitchAbort = aborter;
    window._reviewStitchAbort = aborter;
    const { signal } = aborter;

    if (window._reviewKeyDownHandler) {
      window.removeEventListener("keydown", window._reviewKeyDownHandler);
      window._reviewKeyDownHandler = null;
    }

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
    shell._toolSignal = signal;

    const docbar = document.createElement("div");
    docbar.className = "review-stitched-toolbar review-document-toolbar";

    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "ui-btn ui-btn-ghost ui-btn-compact";
    prev.title = "Trang trước";
    prev.setAttribute("aria-label", "Trang trước");
    prev.append(window.createUiIcon("chevron-left"));

    const select = document.createElement("select");
    select.className = "ui-select review-stitched-select";
    select.setAttribute("aria-label", "Chọn trang gốc");
    sourcePages.forEach((sourcePage) => select.add(new Option(`Trang ${sourcePage + 1}`, String(sourcePage))));

    const next = document.createElement("button");
    next.type = "button";
    next.className = "ui-btn ui-btn-ghost ui-btn-compact";
    next.title = "Trang sau";
    next.setAttribute("aria-label", "Trang sau");
    next.append(window.createUiIcon("chevron-right"));

    const cleanBtn = document.createElement("button");
    cleanBtn.type = "button";
    cleanBtn.className = "ui-btn ui-btn-ghost ui-btn-compact";
    cleanBtn.textContent = "Clean";

    const renderedBtn = document.createElement("button");
    renderedBtn.type = "button";
    renderedBtn.className = "ui-btn ui-btn-ghost ui-btn-compact";
    renderedBtn.textContent = "Rendered";

    const originalBtn = document.createElement("button");
    originalBtn.type = "button";
    originalBtn.className = "ui-btn ui-btn-ghost ui-btn-compact";
    originalBtn.textContent = "Original";

    const zoomOut = document.createElement("button");
    zoomOut.type = "button";
    zoomOut.className = "ui-btn ui-btn-ghost ui-btn-compact";
    zoomOut.setAttribute("aria-label", "Thu nhỏ");
    zoomOut.append(window.createUiIcon("minus"));

    const zoomValue = document.createElement("button");
    zoomValue.type = "button";
    zoomValue.className = "ui-btn ui-btn-ghost ui-btn-compact review-zoom-value";
    zoomValue.textContent = "Fit";

    const zoomIn = document.createElement("button");
    zoomIn.type = "button";
    zoomIn.className = "ui-btn ui-btn-ghost ui-btn-compact";
    zoomIn.setAttribute("aria-label", "Phóng to");
    zoomIn.append(window.createUiIcon("plus"));

    const oneToOne = document.createElement("button");
    oneToOne.type = "button";
    oneToOne.className = "ui-btn ui-btn-ghost ui-btn-compact";
    oneToOne.textContent = "100%";

    docbar.append(prev, select, next, cleanBtn, renderedBtn, originalBtn, zoomOut, zoomValue, zoomIn, oneToOne);

    const meta = document.createElement("div");
    meta.className = "review-stitched-meta";

    const warning = document.createElement("div");
    warning.className = "review-stitched-warning";
    warning.hidden = true;

    const viewport = document.createElement("div");
    viewport.className = "review-stitched-viewport review-document-viewport";

    const zoomStage = document.createElement("div");
    zoomStage.className = "review-stitched-zoom-stage";

    const imageHost = document.createElement("div");
    imageHost.className = "review-stitched-image";

    zoomStage.appendChild(imageHost);
    viewport.appendChild(zoomStage);
    shell.append(docbar, meta, warning, viewport);
    canvasHost.appendChild(shell);

    const controls = document.createElement("div");
    controls.className = "review-controls review-stitched-controls";

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "ui-btn ui-btn-ghost clear-brush-btn";
    clearBtn.textContent = "Xóa toàn bộ mask";

    const submitBtn = document.createElement("button");
    submitBtn.type = "button";
    submitBtn.className = "ui-btn ui-btn-primary repaint-btn";
    submitBtn.textContent = "Inpaint vùng chọn";

    const brushSizeWrap = document.createElement("label");
    brushSizeWrap.className = "ui-range-field brush-size-control";
    brushSizeWrap.textContent = "Cỡ cọ ";

    const brushSizeValue = document.createElement("output");
    brushSizeValue.textContent = "48px";

    const brushSize = document.createElement("input");
    brushSize.type = "range";
    brushSize.min = "4";
    brushSize.max = "100";
    brushSize.step = "1";
    brushSize.value = "24";
    brushSize.className = "brush-size-slider";

    brushSizeWrap.append(brushSize, brushSizeValue);
    controls.append(submitBtn, clearBtn, brushSizeWrap);
    controlsSlot.replaceChildren(controls);

    installReviewOverlaySync(shell);
    mountTextInspector(workspace, shell, signal);

    let painting = false;
    let brushRadius = 24;
    let lastPaint = null;
    let panning = false;
    let panStart = null;
    let spaceDown = false;

    const syncVariantUI = () => {
      cleanBtn.classList.toggle("ui-btn-primary", variant === "clean");
      renderedBtn.classList.toggle("ui-btn-primary", variant === "rendered");
      originalBtn.classList.toggle("ui-btn-primary", variant === "original");
      const readonly = variant !== "clean";
      submitBtn.disabled = readonly;
      clearBtn.disabled = readonly;
      brushSize.disabled = readonly;
      workspace.classList.toggle("review-readonly-document", readonly);
      syncToolPresentation(shell);
    };

    const setBusy = (busy, text = "Đang xử lý…") => {
      workspace.classList.toggle("review-busy", busy);
      [submitBtn, clearBtn, brushSize, prev, next, select, cleanBtn, renderedBtn, originalBtn, zoomOut, zoomIn, oneToOne, zoomValue].forEach((el) => {
        if (el) el.disabled = busy;
      });
      submitBtn.textContent = busy ? text : "Inpaint vùng chọn";
      workspace._pageNavigator?.setBusy(busy);
      document.querySelectorAll(".review-rail-tool").forEach((button) => { button.disabled = busy; });
      if (!busy) syncVariantUI();
    };

    const updateCompatibility = () => {
      const items = groups.get(activeSourcePage) || [];
      const canonical = Number(items[0]?.canonicalIndex ?? 0);
      compatibilityCard.dataset.pageIndex = String(canonical);
      workspace.dataset.reviewCanonicalIndex = String(canonical);
      window.setWorkflowCheckpoint?.("review", canonical);
    };

    const rerender = () => {
      const items = groupsFromManifest().get(activeSourcePage);
      if (!items) return;
      captureBrushSnapshot(shell);
      select.value = String(activeSourcePage);
      const pos = sourcePages.indexOf(activeSourcePage);
      prev.disabled = pos <= 0;
      next.disabled = pos >= sourcePages.length - 1;
      updateCompatibility();
      const title = toolbar.querySelector(".review-toolbar-title");
      if (title) title.textContent = `Trang ${pos + 1} / ${sourcePages.length} · Xử lý & Biên tập`;
      syncVariantUI();
      void renderSourcePage(shell, activeSourcePage, items, signal);
    };

    const selectSource = (sourcePage) => {
      if (!groups.has(sourcePage) || sourcePage === activeSourcePage) return;
      captureBrushSnapshot(shell);
      clearReviewTextSelection(shell);
      activeSourcePage = sourcePage;
      rerender();
    };

    mountToolRail(workspace, shell, signal);
    mountActionToolbar(workspace, shell, signal, rerender);
    installTextDrawing(shell, signal);

    const navigator = workspace._pageNavigator;
    if (navigator) {
      const sourceItems = sourcePages.map((sourcePage) => {
        const items = groups.get(sourcePage) || [];
        const pages = items.map(livePage).filter(Boolean);
        const needsReview = pages.some((page) => page.needs_review || page.detection_state === "needs_review" || (page.detection_issues || []).length);
        const textCount = pages.reduce((sum, page) => sum + (page.text_objects?.length || 0), 0);
        return {
          key: sourcePage,
          label: `Trang ${sourcePage + 1}`,
          meta: textCount ? `${textCount} vùng chữ` : "Chưa có vùng chữ",
          image: pages[0]?.clean || pages[0]?.original || "",
          state: needsReview ? "review" : "ready",
          stateLabel: needsReview ? "Cần kiểm tra" : "Sẵn sàng",
        };
      });
      navigator.reconfigure({
        items: sourceItems,
        activeIndex: sourcePages.indexOf(activeSourcePage),
        title: "Trang",
        ariaLabel: "Điều hướng trang gốc",
        onSelect: (index) => selectSource(sourcePages[index]),
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

    const imagePoint = (event) => sourcePoint(imageHost, event);

    imageHost.addEventListener("pointerdown", (event) => {
      if (variant !== "clean" || !["brush", "eraser"].includes(activeTool) || event.button !== 0) return;
      const point = imagePoint(event);
      if (!point) return;
      event.preventDefault();
      painting = true;
      lastPaint = point;
      imageHost.setPointerCapture?.(event.pointerId);
      paintPoint(shell, point.x, point.y, brushRadius, activeTool === "eraser");
    }, { signal });

    imageHost.addEventListener("pointermove", (event) => {
      if (!painting || !lastPaint || !["brush", "eraser"].includes(activeTool)) return;
      const nextPoint = imagePoint(event);
      if (!nextPoint) return;
      paintStroke(shell, lastPaint, nextPoint, brushRadius, activeTool === "eraser");
      lastPaint = nextPoint;
    }, { signal });

    const stopPainting = () => {
      painting = false;
      lastPaint = null;
    };
    imageHost.addEventListener("pointerup", stopPainting, { signal });
    imageHost.addEventListener("pointercancel", stopPainting, { signal });

    window.addEventListener("keydown", (event) => {
      const tag = event.target?.tagName?.toLowerCase();
      const editable = event.target?.isContentEditable || tag === "input" || tag === "textarea" || tag === "select";
      if (editable) return;

      if (event.code === "Space" && !event.repeat) {
        spaceDown = true;
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key === "0") {
        event.preventDefault();
        fitWidth = true;
        applyZoom(shell);
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key === "1") {
        event.preventDefault();
        fitWidth = false;
        zoomPercent = 100;
        applyZoom(shell);
        return;
      }
      if (event.key === "[" && ["brush", "eraser"].includes(activeTool)) {
        event.preventDefault();
        brushRadius = Math.max(4, brushRadius - 2);
        brushSize.value = String(brushRadius);
        brushSizeValue.textContent = `${brushRadius * 2}px`;
        return;
      }
      if (event.key === "]" && ["brush", "eraser"].includes(activeTool)) {
        event.preventDefault();
        brushRadius = Math.min(100, brushRadius + 2);
        brushSize.value = String(brushRadius);
        brushSizeValue.textContent = `${brushRadius * 2}px`;
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && window.editorState?.selectedTextObjectId) {
        const pageIndex = Number(window.editorState.activePageIndex || 0);
        const id = window.editorState.selectedTextObjectId;
        event.preventDefault();
        window.deleteTextObject?.(pageIndex, id)
          ?.then(() => renderTextOverlays(shell, signal))
          .catch((err) => window.showToast?.("Không thể xóa vùng chữ: " + err.message, "error"));
        return;
      }
      if (event.key === "Escape") {
        setActiveTool(shell, "select");
        return;
      }
      if (!event.ctrlKey && !event.metaKey && !event.altKey) {
        const shortcut = TOOL_SHORTCUTS[event.key.toLowerCase()];
        if (shortcut) {
          event.preventDefault();
          setActiveTool(shell, shortcut);
        }
      }
    }, { signal });

    window.addEventListener("keyup", (event) => {
      if (event.code === "Space") {
        spaceDown = false;
        panning = false;
        panStart = null;
        viewport.classList.remove("is-panning");
      }
    }, { signal });

    viewport.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !(spaceDown || activeTool === "hand")) return;
      if (event.target.closest(".review-inline-translation,.review-inline-ocr")) return;
      panning = true;
      panStart = { x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop };
      viewport.setPointerCapture?.(event.pointerId);
      viewport.classList.add("is-panning");
      event.preventDefault();
    }, { signal });

    viewport.addEventListener("pointermove", (event) => {
      if (!panning || !panStart) return;
      viewport.scrollLeft = panStart.left - (event.clientX - panStart.x);
      viewport.scrollTop = panStart.top - (event.clientY - panStart.y);
    }, { signal });

    viewport.addEventListener("pointerup", () => {
      panning = false;
      panStart = null;
      viewport.classList.remove("is-panning");
    }, { signal });

    viewport.addEventListener("click", (event) => {
      if (activeTool !== "zoom" || event.target.closest("button,input,textarea,select")) return;
      const rect = viewport.getBoundingClientRect();
      const focus = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      stepZoom(event.altKey ? -1 : 1);
      applyZoom(shell, focus);
    }, { signal });

    viewport.addEventListener("wheel", (event) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const focus = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      stepZoom(event.deltaY < 0 ? 1 : -1);
      applyZoom(shell, focus);
    }, { passive: false, signal });

    brushSize.addEventListener("input", () => {
      brushRadius = Number(brushSize.value);
      brushSizeValue.textContent = `${brushRadius * 2}px`;
    }, { signal });

    clearBtn.addEventListener("click", () => {
      clearBrushChunks(shell._brushChunks);
      stitchedMaskSnapshots.delete(snapshotKey(activeSourcePage));
    }, { signal });

    cleanBtn.addEventListener("click", () => {
      if (variant === "clean") return;
      variant = "clean";
      rerender();
    }, { signal });
    renderedBtn.addEventListener("click", () => {
      if (variant === "rendered") return;
      captureBrushSnapshot(shell);
      variant = "rendered";
      rerender();
    }, { signal });
    originalBtn.addEventListener("click", () => {
      if (variant === "original") return;
      captureBrushSnapshot(shell);
      variant = "original";
      rerender();
    }, { signal });

    zoomOut.addEventListener("click", () => {
      stepZoom(-1);
      applyZoom(shell);
    }, { signal });
    zoomIn.addEventListener("click", () => {
      stepZoom(1);
      applyZoom(shell);
    }, { signal });
    zoomValue.addEventListener("click", () => {
      fitWidth = true;
      applyZoom(shell);
    }, { signal });
    oneToOne.addEventListener("click", () => {
      fitWidth = false;
      zoomPercent = 100;
      applyZoom(shell);
    }, { signal });

    select.addEventListener("change", () => {
      const source = Number.parseInt(select.value, 10);
      selectSource(source);
      navigator?.setActive(sourcePages.indexOf(source));
    }, { signal });
    prev.addEventListener("click", () => {
      const pos = sourcePages.indexOf(activeSourcePage);
      if (pos > 0) {
        selectSource(sourcePages[pos - 1]);
        navigator?.setActive(pos - 1);
      }
    }, { signal });
    next.addEventListener("click", () => {
      const pos = sourcePages.indexOf(activeSourcePage);
      if (pos < sourcePages.length - 1) {
        selectSource(sourcePages[pos + 1]);
        navigator?.setActive(pos + 1);
      }
    }, { signal });

    window.addEventListener("resize", () => {
      if (fitWidth) applyZoom(shell);
    }, { signal });

    submitBtn.addEventListener("click", async () => {
      const chapterId = window.currentChapterId;
      const chunks = shell._brushChunks || [];
      if (!chapterId || !chunks.some((chunk) => chunk.dirty && chunkHasPaint(chunk))) {
        window.showToast?.("Chưa có vùng nào được đánh dấu.", "error");
        return;
      }
      const repaintMode = typeof window.chooseRepaintMode === "function" ? await window.chooseRepaintMode() : "standard";
      if (!repaintMode || chapterId !== window.currentChapterId) return;
      setBusy(true, repaintMode === "lama" ? "LaMa đang xử lý…" : "Đang xử lý…");
      try {
        let affected = 0;
        for (const desc of shell._descriptors || []) {
          const subH = desc.sourceY2 - desc.sourceY1;
          if (subH <= 0 || !brushRegionHasPaint(chunks, desc.sourceY1, desc.sourceY2)) continue;
          const maskCanvas = document.createElement("canvas");
          maskCanvas.width = desc.img.naturalWidth;
          maskCanvas.height = desc.img.naturalHeight;
          drawBrushChunksIntoMask(maskCanvas.getContext("2d"), chunks, desc, maskCanvas.width);
          const maskBlob = await canvasToBlob(maskCanvas);
          const formData = new FormData();
          formData.append("chapter_id", chapterId);
          formData.append("page_index", desc.item.canonicalIndex);
          formData.append("mode", repaintMode);
          formData.append("mask", maskBlob, "mask.png");
          const response = await fetch("/api/repaint_mask", { method: "POST", body: formData });
          const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({})));
          const data = await parse(response);
          if (!response.ok) {
            throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
          }
          if (window.currentManifest?.pages?.[desc.item.canonicalIndex] && data.pages?.[desc.item.canonicalIndex]) {
            window.currentManifest.pages[desc.item.canonicalIndex] = data.pages[desc.item.canonicalIndex];
          }
          affected++;
        }
        clearBrushChunks(chunks);
        stitchedMaskSnapshots.delete(snapshotKey(activeSourcePage));
        window.showToast?.(`Đã xử lý ${affected} vùng ảnh trên trang ${activeSourcePage + 1}.`, affected ? "success" : "info");
        rerender();
      } catch (err) {
        window.showToast?.("Không thể xử lý vùng đánh dấu: " + err.message, "error");
      } finally {
        setBusy(false);
      }
    }, { signal });

    signal.addEventListener("abort", () => captureBrushSnapshot(shell), { once: true });

    syncVariantUI();
    setActiveTool(shell, activeTool);
    rerender();
  }

  function scan() {
    document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount);
  }

  window.mountStitchInspector = scan;
})();