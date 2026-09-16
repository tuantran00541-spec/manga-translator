(() => {
  const MAX_CANVAS_CHUNK_HEIGHT = 12000;
  const ZOOM_STEPS = [25, 50, 75, 100, 125, 150, 200, 300, 400];
  let activeSourcePage = null;
  let renderToken = 0;
  let zoomPercent = 100;
  let fitWidth = true;
  let variant = "clean";
  const stitchedMaskSnapshots = new Map();

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

  async function renderSourcePage(shell, sourcePage, items, brushHandlers) {
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
    loading.textContent = variant === "original" ? "Đang dựng ảnh gốc…" : "Đang dựng ảnh đã inpaint…";
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
        const url = imageUrl(item.page);
        if (!url) throw new Error(`Vùng ảnh ${index + 1} không có dữ liệu.`);
        const img = index === 0 ? first : await loadImage(url);
        if (token !== renderToken) return;
        if (img.naturalWidth !== width) throw new Error("Các vùng ảnh trong cùng trang không cùng chiều rộng.");
        const core = coreMetadata(item.page, img.naturalHeight);
        if (core) {
          sourceHeight = core.sourceHeight;
          if (core.sourceY1 !== expectedSourceY) metadataValid = false;
          expectedSourceY = core.sourceY2;
          descriptors.push({ item, img, localY1: core.localY1, localY2: core.localY2, sourceY1: core.sourceY1, sourceY2: core.sourceY2 });
        } else {
          metadataValid = false;
          descriptors.push({ item, img, localY1: 0, localY2: img.naturalHeight, sourceY1: fallbackY, sourceY2: fallbackY + img.naturalHeight });
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

      meta.textContent = `Trang ${sourcePage + 1} · ${width} × ${sourceHeight}px · ${variant === "original" ? "Ảnh gốc" : "Sau inpaint"}`;
      applyZoom(shell);
    } catch (err) {
      if (token !== renderToken) return;
      imageHost.replaceChildren();
      const error = document.createElement("div");
      error.className = "ui-state ui-state-error review-stitched-error";
      error.textContent = `Không dựng được trang: ${err.message}`;
      imageHost.appendChild(error);
    }
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
    workspace.classList.add("review-single-document");

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

    docbar.append(prev, select, next, cleanBtn, originalBtn, zoomOut, zoomValue, zoomIn, oneToOne);
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
    brushBtn.type = "button"; brushBtn.className = "ui-btn ui-btn-ghost brush-toggle-btn"; brushBtn.textContent = "Brush"; brushBtn.setAttribute("aria-pressed", "false");
    const clearBtn = document.createElement("button");
    clearBtn.type = "button"; clearBtn.className = "ui-btn ui-btn-ghost clear-brush-btn"; clearBtn.textContent = "Clear mask";
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

    let brushOn = false;
    let painting = false;
    let brushRadius = 24;
    let last = { x: 0, y: 0 };
    let boundCanvas = null;

    const syncVariantUI = () => {
      cleanBtn.classList.toggle("ui-btn-primary", variant === "clean");
      originalBtn.classList.toggle("ui-btn-primary", variant === "original");
      const readonly = variant !== "clean";
      brushBtn.disabled = readonly;
      clearBtn.disabled = readonly;
      submitBtn.disabled = readonly;
      if (readonly) { brushOn = false; brushSizeWrap.hidden = true; imageHost.classList.remove("brush-mode"); }
    };

    const syncBrushUI = () => {
      imageHost.classList.toggle("brush-mode", brushOn && variant === "clean");
      brushBtn.classList.toggle("ui-btn-primary", brushOn && variant === "clean");
      brushBtn.textContent = brushOn ? "Brush đang bật" : "Brush";
      brushBtn.setAttribute("aria-pressed", String(brushOn));
      brushSizeWrap.hidden = !brushOn || variant !== "clean";
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
      [brushBtn, clearBtn, submitBtn, brushSize, prev, next, select, cleanBtn, originalBtn, zoomOut, zoomIn, oneToOne, zoomValue].forEach((el) => { if (el) el.disabled = busy; });
      submitBtn.textContent = busy ? text : "Inpaint vùng chọn";
      workspace._pageNavigator?.setBusy(busy);
      const continueBtn = workspace.querySelector(".review-primary-action");
      if (continueBtn) continueBtn.disabled = busy;
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
      if (title) title.textContent = `Trang ${pos + 1} / ${sourcePages.length}`;
      syncVariantUI();
      void renderSourcePage(shell, activeSourcePage, items, brushHandlers);
    };

    const selectSource = (sourcePage) => {
      if (!groups.has(sourcePage) || sourcePage === activeSourcePage) return;
      captureSnapshot(shell);
      activeSourcePage = sourcePage;
      rerender();
    };

    const navigator = workspace._pageNavigator;
    if (navigator) {
      const sourceItems = sourcePages.map((sourcePage) => {
        const items = groups.get(sourcePage) || [];
        const needsReview = items.some(({ page }) => page.needs_review || page.detection_state === "needs_review" || (page.detection_issues || []).length);
        return {
          key: sourcePage,
          label: `Trang ${sourcePage + 1}`,
          meta: `${items.length} vùng xử lý`,
          image: items[0]?.page?.clean || items[0]?.page?.original || "",
          state: needsReview ? "review" : "ready",
          stateLabel: needsReview ? "Cần kiểm tra" : "Đã xác minh",
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
        const pos = sourcePages.indexOf(activeSourcePage); if (pos > 0) { event.preventDefault(); selectSource(sourcePages[pos - 1]); navigator?.setActive(pos - 1); }
      } else if (event.key === "ArrowRight" || event.key === "PageDown") {
        const pos = sourcePages.indexOf(activeSourcePage); if (pos < sourcePages.length - 1) { event.preventDefault(); selectSource(sourcePages[pos + 1]); navigator?.setActive(pos + 1); }
      } else if ((event.ctrlKey || event.metaKey) && event.key === "0") {
        event.preventDefault(); fitWidth = true; applyZoom(shell);
      } else if ((event.ctrlKey || event.metaKey) && event.key === "1") {
        event.preventDefault(); fitWidth = false; zoomPercent = 100; applyZoom(shell);
      } else if (event.key === "[" && brushOn) {
        event.preventDefault(); brushRadius = Math.max(8, brushRadius - 2); brushSize.value = String(brushRadius); brushSizeValue.textContent = `${brushRadius * 2}px`;
      } else if (event.key === "]" && brushOn) {
        event.preventDefault(); brushRadius = Math.min(80, brushRadius + 2); brushSize.value = String(brushRadius); brushSizeValue.textContent = `${brushRadius * 2}px`;
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

    brushBtn.addEventListener("click", () => { if (variant !== "clean") return; brushOn = !brushOn; stopPainting(); syncBrushUI(); }, { signal });
    clearBtn.addEventListener("click", () => {
      if (!boundCanvas) return;
      boundCanvas.getContext("2d").clearRect(0, 0, boundCanvas.width, boundCanvas.height);
      boundCanvas._reviewDirty = false; stitchedMaskSnapshots.delete(activeSourcePage);
    }, { signal });
    brushSize.addEventListener("input", () => { brushRadius = Number(brushSize.value); brushSizeValue.textContent = `${brushRadius * 2}px`; }, { signal });

    cleanBtn.addEventListener("click", () => { if (variant === "clean") return; variant = "clean"; rerender(); }, { signal });
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
})();