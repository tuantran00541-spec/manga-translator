(() => {
  const MODE_KEY = "mt_review_display_mode";
  const DEFAULT_MODE = "slices";
  const MAX_CANVAS_CHUNK_HEIGHT = 12000;
  let mode = localStorage.getItem(MODE_KEY) || DEFAULT_MODE;
  let activeSourcePage = null;
  let renderToken = 0;
  const stitchedMaskSnapshots = new Map();

  function regionHasPaint(canvas, x, y, width, height) {
    const scale = Math.min(1, 512 / Math.max(width, height));
    const probe = document.createElement("canvas");
    probe.width = Math.max(1, Math.ceil(width * scale));
    probe.height = Math.max(1, Math.ceil(height * scale));
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(canvas, x, y, width, height, 0, 0, probe.width, probe.height);
    const pixels = ctx.getImageData(0, 0, probe.width, probe.height).data;
    for (let index = 3; index < pixels.length; index += 4) {
      if (pixels[index] > 20) return true;
    }
    return false;
  }

  function captureStitchedSnapshot(shell) {
    if (activeSourcePage === null || !shell) return;
    const canvas = shell.querySelector("canvas.stitched-brush-canvas");
    if (!canvas || !canvas._reviewDirty || !canvas.width || !canvas.height) {
      stitchedMaskSnapshots.delete(activeSourcePage);
      return;
    }
    try {
      stitchedMaskSnapshots.set(activeSourcePage, canvas.toDataURL("image/png"));
    } catch (err) {
      console.warn("Could not preserve stitched mask snapshot:", err);
    }
  }

  function restoreStitchedSnapshot(brushCanvas, sourcePage) {
    if (sourcePage === null || !stitchedMaskSnapshots.has(sourcePage)) return;
    const dataUrl = stitchedMaskSnapshots.get(sourcePage);
    if (!dataUrl) return;
    const overlay = new Image();
    overlay.onload = () => {
      if (!brushCanvas.isConnected) return;
      const ctx = brushCanvas.getContext("2d");
      ctx.drawImage(overlay, 0, 0, brushCanvas.width, brushCanvas.height);
      brushCanvas._reviewDirty = true;
    };
    overlay.src = dataUrl;
  }

  window.hasUnsavedStitchedMarks = () => {
    if (stitchedMaskSnapshots.size > 0) return true;
    const canvas = document.querySelector(".review-stitched-image canvas.stitched-brush-canvas");
    return Boolean(canvas && canvas._reviewDirty);
  };

  function cleanSourcePage(page, fallbackIndex) {
    return Number.isInteger(page?.source_page) ? page.source_page : fallbackIndex;
  }

  function cleanSliceIndex(page) {
    return Number.isInteger(page?.slice_index) ? page.slice_index : 0;
  }

  function groupsFromManifest() {
    const groups = new Map();
    const pages = window.currentManifest?.pages || [];
    pages.forEach((page, canonicalIndex) => {
      if (!page) return;
      const sourcePage = cleanSourcePage(page, canonicalIndex);
      if (!groups.has(sourcePage)) groups.set(sourcePage, []);
      groups.get(sourcePage).push({ page, canonicalIndex });
    });
    for (const items of groups.values()) {
      items.sort((a, b) => cleanSliceIndex(a.page) - cleanSliceIndex(b.page));
    }
    return new Map([...groups.entries()].sort((a, b) => a[0] - b[0]));
  }

  function imageUrl(page) {
    // A skipped slice still owns part of the source page.  Showing only clean
    // slices makes a mixed source page look truncated and no longer match the
    // raw image.  Use the original pixels for skipped slices and the latest
    // clean artifact everywhere else.
    const url = page.skipped ? page.original : (page.clean || page.original);
    if (!url) return null;
    const revision = Number(
      page.skipped
        ? (page.source_revision || 0)
        : (page.clean_revision || page.process_revision || page.source_revision || 0),
    );
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}review_revision=${revision}&t=${Date.now()}`;
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
      chunk.ctx.drawImage(
        img,
        sx,
        sy + offset,
        sw,
        h,
        0,
        iy1 - chunk.y1,
        sw,
        h,
      );
    }
  }

  async function renderSourcePage(shell, sourcePage, items, brushHandlers) {
    const token = ++renderToken;
    const imageHost = shell.querySelector(".review-stitched-image");
    const meta = shell.querySelector(".review-stitched-meta");
    const warning = shell.querySelector(".review-stitched-warning");
    imageHost.replaceChildren();
    warning.hidden = true;

    const loading = document.createElement("div");
    loading.className = "ui-state review-stitched-loading";
    loading.textContent = "Đang ghép ảnh theo ownership của từng lát…";
    imageHost.appendChild(loading);

    try {
      const firstUrl = imageUrl(items[0]?.page);
      if (!firstUrl) throw new Error("Trang không có ảnh đã xử lý hoặc ảnh gốc.");
      const first = await loadImage(firstUrl);
      if (token !== renderToken) return;

      const firstCore = coreMetadata(items[0].page, first.naturalHeight);
      let sourceHeight = firstCore?.sourceHeight || first.naturalHeight;
      let width = first.naturalWidth;
      let metadataValid = Boolean(firstCore) || items.length === 1;
      let expectedSourceY = 0;

      const descriptors = [];
      let fallbackY = 0;
      for (let index = 0; index < items.length; index += 1) {
        const item = items[index];
        const url = imageUrl(item.page);
        if (!url) throw new Error(`Lát ${index + 1} không có ảnh.`);
        const img = index === 0 ? first : await loadImage(url);
        if (token !== renderToken) return;
        if (img.naturalWidth !== width) {
          throw new Error(`Lát ${index + 1} có chiều rộng ${img.naturalWidth}px, khác ${width}px.`);
        }
        const core = coreMetadata(item.page, img.naturalHeight);
        if (core) {
          sourceHeight = core.sourceHeight;
          if (core.sourceY1 !== expectedSourceY) metadataValid = false;
          expectedSourceY = core.sourceY2;
          descriptors.push({
            item,
            img,
            localY1: core.localY1,
            localY2: core.localY2,
            sourceY1: core.sourceY1,
            sourceY2: core.sourceY2,
          });
        } else {
          metadataValid = false;
          descriptors.push({
            item,
            img,
            localY1: 0,
            localY2: img.naturalHeight,
            sourceY1: fallbackY,
            sourceY2: fallbackY + img.naturalHeight,
          });
          fallbackY += img.naturalHeight;
        }
      }

      if (!metadataValid) {
        let y = 0;
        for (const descriptor of descriptors) {
          descriptor.sourceY1 = y;
          descriptor.sourceY2 = y + (descriptor.localY2 - descriptor.localY1);
          y = descriptor.sourceY2;
        }
        sourceHeight = y;
        warning.hidden = false;
        warning.textContent = "Metadata stitch_core không phủ liên tục; ảnh dưới đây dùng fallback nối tuần tự để vẫn nhìn được lỗi. Không coi fallback này là output chuẩn.";
      } else if (expectedSourceY && expectedSourceY !== sourceHeight) {
        warning.hidden = false;
        warning.textContent = `Ownership kết thúc ở y=${expectedSourceY}, nhưng source_height=${sourceHeight}. Có khả năng stitch metadata đang thiếu/gãy.`;
      }

      imageHost.replaceChildren();
      const chunks = makeChunkCanvases(imageHost, width, sourceHeight);
      for (const descriptor of descriptors) {
        drawIntoChunks(
          chunks,
          descriptor.img,
          0,
          descriptor.localY1,
          width,
          descriptor.localY2 - descriptor.localY1,
          descriptor.sourceY1,
        );
      }

      shell._descriptors = descriptors;

      const brushCanvas = document.createElement("canvas");
      brushCanvas.className = "brush-canvas stitched-brush-canvas";
      brushCanvas.width = width;
      brushCanvas.height = sourceHeight;
      imageHost.appendChild(brushCanvas);
      shell._brushCanvas = brushCanvas;

      if (brushHandlers && typeof brushHandlers.bindCanvas === "function") {
        brushHandlers.bindCanvas(brushCanvas);
      }

      restoreStitchedSnapshot(brushCanvas, sourcePage);

      const skippedCount = items.filter((item) => item.page?.skipped).length;
      const processedCount = items.length - skippedCount;
      const sliceSummary = skippedCount
        ? `${processedCount} lát đã xử lý + ${skippedCount} lát giữ nguyên`
        : `${items.length} lát đã xử lý`;
      meta.textContent = `Trang gốc ${sourcePage + 1} · ${sliceSummary} · ${width} × ${sourceHeight}px · ghép theo stitch_core`;
    } catch (err) {
      if (token !== renderToken) return;
      imageHost.replaceChildren();
      const error = document.createElement("div");
      error.className = "ui-state ui-state-error review-stitched-error";
      error.textContent = `Không dựng được ảnh ghép: ${err.message}`;
      imageHost.appendChild(error);
    }
  }

  function mount(workspace) {
    if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchInspectorMounted === "1") return;
    workspace._stitchAbort?.abort();
    window._reviewStitchAbort?.abort();
    const stitchAbort = new AbortController();
    workspace._stitchAbort = stitchAbort;
    window._reviewStitchAbort = stitchAbort;
    const signal = stitchAbort.signal;
    const host = workspace.closest("#page-view.review-mode");
    const toolbar = workspace.querySelector(".review-sticky-toolbar");
    const actions = toolbar?.querySelector(".review-actions-group");
    const layout = workspace.querySelector(".review-workbench-grid");
    const canvasHost = layout?.querySelector(".review-canvas-host");
    const stitchedControlsSlot = workspace._stitchedControlsSlot;
    if (!host || !toolbar || !actions || !layout || !canvasHost) return;
    workspace.dataset.stitchInspectorMounted = "1";

    const groups = groupsFromManifest();
    if (!groups.size) return;
    const sourcePages = [...groups.keys()];
    if (!sourcePages.includes(activeSourcePage)) {
      const activeCard = workspace.querySelector(".review-card");
      const canonical = Number.parseInt(activeCard?.dataset.pageIndex || "", 10);
      const activePage = Number.isFinite(canonical) ? window.currentManifest?.pages?.[canonical] : null;
      activeSourcePage = activePage ? cleanSourcePage(activePage, canonical) : sourcePages[0];
    }

    // Top view switcher
    const switcher = document.createElement("div");
    switcher.className = "review-view-switch";
    switcher.setAttribute("role", "group");
    switcher.setAttribute("aria-label", "Kiểu hiển thị ảnh kiểm tra");
    const slicesBtn = document.createElement("button");
    slicesBtn.type = "button";
    slicesBtn.className = "ui-btn ui-btn-ghost ui-btn-compact";
    slicesBtn.dataset.reviewMode = "slices";
    slicesBtn.textContent = "Từng lát";
    const stitchedBtn = document.createElement("button");
    stitchedBtn.type = "button";
    stitchedBtn.className = "ui-btn ui-btn-ghost ui-btn-compact";
    stitchedBtn.dataset.reviewMode = "stitched";
    stitchedBtn.textContent = "Ghép như ảnh gốc";
    switcher.append(slicesBtn, stitchedBtn);
    actions.prepend(switcher);

    // Stitched Canvas Shell
    const shell = document.createElement("section");
    shell.className = "review-stitched-shell";
    const stitchedToolbar = document.createElement("div");
    stitchedToolbar.className = "review-stitched-toolbar";
    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "ui-btn ui-btn-ghost";
    prev.append(window.createUiIcon("chevron-left"), document.createTextNode("Trang trước"));
    const select = document.createElement("select");
    select.className = "ui-select review-stitched-select";
    select.setAttribute("aria-label", "Chọn trang gốc đã ghép");
    sourcePages.forEach((sourcePage) => {
      const option = document.createElement("option");
      option.value = String(sourcePage);
      option.textContent = `Trang gốc ${sourcePage + 1} · ${groups.get(sourcePage).length} lát`;
      select.appendChild(option);
    });
    select.value = String(activeSourcePage);
    const next = document.createElement("button");
    next.type = "button";
    next.className = "ui-btn ui-btn-ghost";
    next.append(document.createTextNode("Trang sau"), window.createUiIcon("chevron-right"));
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.className = "ui-btn ui-btn-ghost";
    refresh.textContent = "Làm mới ảnh ghép";
    stitchedToolbar.append(prev, select, next, refresh);

    const note = document.createElement("div");
    note.className = "review-stitched-note";
    note.textContent = "Chế độ xem ghép toàn trang: Bạn có thể dùng cọ đánh dấu và xử lý lỗi trực tiếp trên ảnh ghép hoàn chỉnh hoặc chuyển sang “Từng lát” để chỉnh sửa chi tiết từng lát cắt.";
    const meta = document.createElement("div");
    meta.className = "review-stitched-meta";
    const warning = document.createElement("div");
    warning.className = "review-stitched-warning";
    warning.hidden = true;
    const viewport = document.createElement("div");
    viewport.className = "review-stitched-viewport";
    const imageHost = document.createElement("div");
    imageHost.className = "review-stitched-image";
    viewport.appendChild(imageHost);
    shell.append(stitchedToolbar, note, meta, warning, viewport);
    canvasHost.appendChild(shell);

    // Stitched Inspector Controls inside stitchedControlsSlot
    const controlsContainer = document.createElement("div");
    controlsContainer.className = "review-controls review-stitched-controls";

    const brushBtn = document.createElement("button");
    brushBtn.type = "button";
    brushBtn.className = "ui-btn ui-btn-ghost brush-toggle-btn";
    brushBtn.textContent = "Đánh dấu vùng lỗi";
    brushBtn.setAttribute("aria-pressed", "false");

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "ui-btn ui-btn-ghost clear-brush-btn";
    clearBtn.textContent = "Xóa nét đánh dấu";

    const submitBtn = document.createElement("button");
    submitBtn.type = "button";
    submitBtn.className = "ui-btn ui-btn-primary repaint-btn";
    submitBtn.textContent = "Xử lý vùng đánh dấu";

    const brushSizeWrap = document.createElement("label");
    brushSizeWrap.className = "ui-range-field brush-size-control";
    brushSizeWrap.hidden = true;
    brushSizeWrap.textContent = "Kích thước cọ ";
    const brushSizeValue = document.createElement("output");
    brushSizeValue.className = "brush-size-value";
    brushSizeValue.textContent = "48px";
    const brushSize = document.createElement("input");
    brushSize.type = "range";
    brushSize.min = "8";
    brushSize.max = "80";
    brushSize.step = "1";
    brushSize.value = "24";
    brushSize.className = "brush-size-slider";
    brushSize.title = "Điều chỉnh kích thước cọ";
    brushSizeWrap.append(brushSize, brushSizeValue);

    controlsContainer.append(brushBtn, clearBtn, submitBtn, brushSizeWrap);
    if (stitchedControlsSlot) {
      stitchedControlsSlot.replaceChildren(controlsContainer);
    }

    // Brush State & Event Handling
    let brushOn = false;
    let painting = false;
    let brushRadius = 24;
    let lastStrokePos = { x: 0, y: 0 };
    let currentBoundCanvas = null;

    brushSize.value = String(brushRadius);
    brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;

    const adjustBrushRadius = (delta) => {
      const nextRadius = Math.max(8, Math.min(80, brushRadius + delta));
      if (nextRadius !== brushRadius) {
        brushRadius = nextRadius;
        brushSize.value = String(brushRadius);
        brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;
      }
    };

    brushSize.addEventListener("input", () => {
      brushRadius = Number(brushSize.value);
      brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;
    });

    const onStitchKeyDown = (e) => {
      if (e.target.matches && e.target.matches("input, textarea, select")) return;
      if (e.key === "[") {
        e.preventDefault();
        adjustBrushRadius(-2);
      } else if (e.key === "]") {
        e.preventDefault();
        adjustBrushRadius(2);
      }
    };
    document.addEventListener("keydown", onStitchKeyDown, { signal });

    const syncBrushUI = () => {
      imageHost.classList.toggle("brush-mode", brushOn);
      brushBtn.classList.toggle("ui-btn-primary", brushOn);
      brushBtn.classList.toggle("ui-btn-ghost", !brushOn);
      brushBtn.textContent = brushOn ? "Đang đánh dấu · Chọn để kết thúc" : "Đánh dấu vùng lỗi";
      brushBtn.setAttribute("aria-pressed", String(brushOn));
      brushSizeWrap.hidden = !brushOn;
    };

    const stopPainting = (e) => {
      if (!painting) return;
      painting = false;
      if (e && e.pointerId !== undefined && currentBoundCanvas?.releasePointerCapture) {
        try {
          if (currentBoundCanvas.hasPointerCapture && currentBoundCanvas.hasPointerCapture(e.pointerId)) {
            currentBoundCanvas.releasePointerCapture(e.pointerId);
          }
        } catch (_) {}
      }
    };

    brushBtn.addEventListener("click", () => {
      brushOn = !brushOn;
      if (!brushOn) stopPainting();
      syncBrushUI();
    });

    clearBtn.addEventListener("click", () => {
      stopPainting();
      if (currentBoundCanvas) {
        const ctx = currentBoundCanvas.getContext("2d");
        ctx.clearRect(0, 0, currentBoundCanvas.width, currentBoundCanvas.height);
        currentBoundCanvas._reviewDirty = false;
      }
      stitchedMaskSnapshots.delete(activeSourcePage);
      showToast("Đã xóa nét đánh dấu trên ảnh ghép.", "info");
    });

    function getCanvasCoords(e, canvas) {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return { x: 0, y: 0 };
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return {
        x: Math.round((e.clientX - rect.left) * scaleX),
        y: Math.round((e.clientY - rect.top) * scaleY),
      };
    }

    function paintDot(ctx, canvas, x, y) {
      ctx.fillStyle = "rgba(220, 38, 38, 0.7)";
      ctx.beginPath();
      ctx.arc(x, y, brushRadius, 0, Math.PI * 2);
      ctx.fill();
      canvas._reviewDirty = true;
    }

    function paintStrokeSegment(ctx, canvas, x0, y0, x1, y1) {
      ctx.strokeStyle = "rgba(220, 38, 38, 0.7)";
      ctx.lineWidth = brushRadius * 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      canvas._reviewDirty = true;
    }

    const brushHandlers = {
      bindCanvas(canvas) {
        currentBoundCanvas = canvas;
        const ctx = canvas.getContext("2d");

        const handleStart = (e) => {
          if (!brushOn) return;
          if (e.button !== undefined && e.button !== 0) return;
          painting = true;
          const { x, y } = getCanvasCoords(e, canvas);
          lastStrokePos = { x, y };
          if (e.pointerId !== undefined && canvas.setPointerCapture) {
            try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
          }
          paintDot(ctx, canvas, x, y);
        };

        const handleMove = (e) => {
          if (!brushOn || !painting) return;
          if (e.buttons !== undefined && (e.buttons & 1) !== 1) {
            stopPainting(e);
            return;
          }
          const { x, y } = getCanvasCoords(e, canvas);
          paintStrokeSegment(ctx, canvas, lastStrokePos.x, lastStrokePos.y, x, y);
          lastStrokePos = { x, y };
        };

        const handleEnd = (e) => {
          stopPainting(e);
        };

        if (window.PointerEvent) {
          canvas.addEventListener("pointerdown", handleStart);
          canvas.addEventListener("pointermove", handleMove);
          canvas.addEventListener("pointerup", handleEnd);
          canvas.addEventListener("pointercancel", handleEnd);
        } else {
          canvas.addEventListener("mousedown", handleStart);
          canvas.addEventListener("mousemove", handleMove);
          canvas.addEventListener("mouseup", handleEnd);
        }

        canvas.addEventListener("wheel", (e) => {
          if (!brushOn) return;
          if (e.ctrlKey || e.altKey) {
            e.preventDefault();
            brushRadius = Math.round(Math.max(8, Math.min(80, brushRadius + (e.deltaY < 0 ? 2 : -2))));
            brushSize.value = String(brushRadius);
            brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;
            showToast(`Kích thước cọ: ${Math.round(brushRadius * 2)}px`, "info");
          }
        }, { passive: false });
      },
    };

    window.addEventListener("mouseup", stopPainting, { signal });
    window.addEventListener("pointerup", stopPainting, { signal });
    window.addEventListener("blur", () => stopPainting(), { signal });

    const setStitchedBusy = (busy, label = "Đang xử lý…") => {
      workspace.classList.toggle("review-busy", busy);
      brushBtn.disabled = busy;
      clearBtn.disabled = busy;
      submitBtn.disabled = busy;
      submitBtn.textContent = busy ? label : "Xử lý vùng đánh dấu";
      brushSize.disabled = busy;
      const pos = sourcePages.indexOf(activeSourcePage);
      prev.disabled = busy || pos <= 0;
      next.disabled = busy || pos >= sourcePages.length - 1;
      select.disabled = busy;
      refresh.disabled = busy;
      if (workspace._pageNavigator) workspace._pageNavigator.setBusy(busy);
      const continueBtn = workspace.querySelector(".review-primary-action");
      if (continueBtn) continueBtn.disabled = busy;
    };

    submitBtn.addEventListener("click", async () => {
      const chapterId = window.currentChapterId;
      if (!chapterId) return;

      const brushCanvas = shell.querySelector("canvas.stitched-brush-canvas");
      if (!brushCanvas || !brushCanvas._reviewDirty) {
        showToast("Chưa có vùng nào được đánh dấu để xử lý trên ảnh ghép.", "error");
        return;
      }

      const w = brushCanvas.width;
      const h = brushCanvas.height;
      const hasPaint = regionHasPaint(brushCanvas, 0, 0, w, h);
      if (!hasPaint) {
        brushCanvas._reviewDirty = false;
        showToast("Chưa có vùng nào được đánh dấu để xử lý.", "error");
        return;
      }

      const repaintMode = typeof window.chooseRepaintMode === "function"
        ? await window.chooseRepaintMode()
        : "standard";
      if (!repaintMode || chapterId !== window.currentChapterId) return;

      setStitchedBusy(true, repaintMode === "lama" ? "LaMa đang xử lý…" : "Đang xử lý…");

      try {
        const descriptors = shell._descriptors || [];
        if (!descriptors.length) throw new Error("Không tìm thấy thông tin lát ảnh của trang ghép.");

        let affectedCount = 0;
        for (const desc of descriptors) {
          const subH = desc.sourceY2 - desc.sourceY1;
          if (subH <= 0) continue;

          const sliceSubCanvas = document.createElement("canvas");
          sliceSubCanvas.width = w;
          sliceSubCanvas.height = subH;
          const subCtx = sliceSubCanvas.getContext("2d");
          subCtx.drawImage(
            brushCanvas,
            0, desc.sourceY1, w, subH,
            0, 0, w, subH
          );

          const slicePainted = regionHasPaint(brushCanvas, 0, desc.sourceY1, w, subH);
          if (!slicePainted) continue;

          const sliceMaskCanvas = document.createElement("canvas");
          sliceMaskCanvas.width = w;
          sliceMaskCanvas.height = desc.img.naturalHeight;
          const sliceMaskCtx = sliceMaskCanvas.getContext("2d");
          sliceMaskCtx.drawImage(
            sliceSubCanvas,
            0, 0, w, subH,
            0, desc.localY1, w, subH
          );

          const toBlobFn = typeof window.canvasToBlob === "function"
            ? window.canvasToBlob
            : (c) => new Promise((res, rej) => {
                try {
                  const d = c.toDataURL("image/png").split(",")[1];
                  const b = atob(d);
                  const a = new Uint8Array(b.length);
                  for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i);
                  res(new Blob([a], { type: "image/png" }));
                } catch (e) { rej(e); }
              });
          const maskBlob = await toBlobFn(sliceMaskCanvas);

          const formData = new FormData();
          formData.append("chapter_id", chapterId);
          formData.append("page_index", desc.item.canonicalIndex);
          formData.append("mode", repaintMode);
          formData.append("mask", maskBlob, "mask.png");

          const resp = await fetch("/api/repaint_mask", { method: "POST", body: formData });
          const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => (await r.json().catch(() => ({})));
          const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d.detail || `Server trả về ${s}`;
          const manifest = await parse(resp);
          if (!resp.ok) throw new Error(getErr(resp.status, manifest));

          if (window.currentManifest?.pages) {
            window.currentManifest.pages[desc.item.canonicalIndex] = manifest.pages[desc.item.canonicalIndex];
          }
          affectedCount++;
        }

        if (affectedCount > 0) {
          const ctx = brushCanvas.getContext("2d");
          ctx.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
          brushCanvas._reviewDirty = false;
          stitchedMaskSnapshots.delete(activeSourcePage);
          showToast(
            repaintMode === "lama"
              ? `Đã tái inpaint bằng LaMa cho ${affectedCount} lát ảnh trên trang ghép.`
              : `Đã xử lý vùng đánh dấu cho ${affectedCount} lát ảnh trên trang ghép.`,
            "success"
          );
          rerender();
        } else {
          showToast("Không tìm thấy lát ảnh tương ứng với vùng đã đánh dấu.", "info");
        }
      } catch (err) {
        showToast("Không thể xử lý vùng đánh dấu trên ảnh ghép: " + err.message, "error");
      } finally {
        setStitchedBusy(false);
      }
    });

    const rerender = () => {
      const items = groups.get(activeSourcePage);
      if (!items) return;
      select.value = String(activeSourcePage);
      const pos = sourcePages.indexOf(activeSourcePage);
      prev.disabled = pos <= 0;
      next.disabled = pos < 0 || pos >= sourcePages.length - 1;
      const titleEl = toolbar.querySelector(".review-toolbar-title");
      if (titleEl && mode === "stitched") {
        titleEl.textContent = `Trang gốc ${activeSourcePage + 1} / ${sourcePages.length} · ${items.length} lát (Ghép như ảnh gốc)`;
      }
      void renderSourcePage(shell, activeSourcePage, items, brushHandlers);
    };

    function applyMode(targetHost, targetShell, targetSwitcher) {
      const stitched = mode === "stitched";
      targetHost.classList.toggle("review-show-stitched", stitched);
      targetHost.classList.toggle("review-show-slices", !stitched);
      workspace.classList.toggle("review-show-stitched", stitched);
      workspace.classList.toggle("review-show-slices", !stitched);
      targetSwitcher.querySelectorAll("button[data-review-mode]").forEach((button) => {
        const active = button.dataset.reviewMode === mode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
      targetShell.hidden = !stitched;
      const titleEl = toolbar.querySelector(".review-toolbar-title");
      if (titleEl) {
        if (stitched) {
          const sliceCount = groups.get(activeSourcePage)?.length || 0;
          titleEl.textContent = `Trang gốc ${activeSourcePage + 1} / ${sourcePages.length} · ${sliceCount} lát (Ghép như ảnh gốc)`;
        } else {
          const pageIndices = (window.currentManifest?.pages || []).filter((p) => !p.skipped);
          titleEl.textContent = `${pageIndices.length} lát đã xử lý (Xem từng lát)`;
        }
      }
      window.syncWorkbenchPanels?.();
    }

    const setMode = (nextMode) => {
      mode = nextMode === "stitched" ? "stitched" : "slices";
      localStorage.setItem(MODE_KEY, mode);
      applyMode(host, shell, switcher);
      if (mode === "stitched") {
        rerender();
      } else {
        const activeCard = workspace.querySelector(".review-card");
        if (activeCard) {
          const img = activeCard.querySelector("img");
          const canonical = parseInt(activeCard.dataset.pageIndex, 10);
          if (img && Number.isFinite(canonical) && window.currentManifest?.pages?.[canonical]) {
            const page = window.currentManifest.pages[canonical];
            const cleanSrc = (page.clean || page.original) + "?t=" + Date.now();
            if (img.src !== cleanSrc) img.src = cleanSrc;
          }
        }
      }
    };
    window.setReviewDisplayMode = setMode;

    window.onReviewSliceSelect = (canonicalIndex) => {
      if (mode === "stitched") {
        const page = window.currentManifest?.pages?.[canonicalIndex];
        if (page) {
          const srcPage = cleanSourcePage(page, canonicalIndex);
          if (srcPage !== activeSourcePage && sourcePages.includes(srcPage)) {
            captureStitchedSnapshot(shell);
            activeSourcePage = srcPage;
            rerender();
          }
        }
      }
    };

    stitchedBtn.addEventListener("click", () => setMode("stitched"));
    slicesBtn.addEventListener("click", () => setMode("slices"));

    select.addEventListener("change", () => {
      captureStitchedSnapshot(shell);
      activeSourcePage = Number.parseInt(select.value, 10);
      rerender();
    });

    prev.addEventListener("click", () => {
      const pos = sourcePages.indexOf(activeSourcePage);
      if (pos > 0) {
        captureStitchedSnapshot(shell);
        activeSourcePage = sourcePages[pos - 1];
        rerender();
      }
    });

    next.addEventListener("click", () => {
      const pos = sourcePages.indexOf(activeSourcePage);
      if (pos >= 0 && pos < sourcePages.length - 1) {
        captureStitchedSnapshot(shell);
        activeSourcePage = sourcePages[pos + 1];
        rerender();
      }
    });

    refresh.addEventListener("click", rerender);

    applyMode(host, shell, switcher);
    if (mode === "stitched") rerender();
  }

  function scan() {
    document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount);
  }

  window.mountStitchInspector = scan;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan, { once: true });
  else scan();
})();
