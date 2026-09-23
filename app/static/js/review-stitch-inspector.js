(() => {
  "use strict";

  const BRUSH_CHUNK_H = 4096;
  const ZOOM_STEPS = [25, 50, 75, 100, 125, 150, 200, 300, 400];
  const MIN_BOX = 10;
  const SHORTCUTS = { v: "select", r: "rectangle", o: "ellipse", b: "brush", e: "eraser", h: "hand", z: "zoom" };

  let variant = "clean";
  let tool = "select";
  let zoom = 100;
  let fitWidth = true;
  let chapter = null;
  let renderToken = 0;
  const snapshots = new Map();
  const autoSynced = new Set();

  const chapterKey = () => String(window.currentChapterId || "");
  const snapshotKey = () => `${chapterKey()}:strip`;
  const livePage = (item) => window.currentManifest?.pages?.[Number(item?.canonicalIndex)] || null;

  function resetChapterState() {
    const key = chapterKey();
    if (chapter === key) return;
    chapter = key;
    variant = "clean";
    tool = "select";
    zoom = 100;
    fitWidth = true;
    snapshots.clear();
    autoSynced.clear();
  }

  function orderedSlices() {
    return (window.currentManifest?.pages || [])
      .map((page, canonicalIndex) => ({ page, canonicalIndex }))
      .filter(({ page }) => Boolean(page))
      .map(({ canonicalIndex }) => ({ canonicalIndex }));
  }

  function ensureStyles() {
    if (document.getElementById("review-photopea-tool-styles")) return;
    const style = document.createElement("style");
    style.id = "review-photopea-tool-styles";
    style.textContent = `
      .review-tool-rail{position:absolute;z-index:calc(var(--z-toolbar) + 4);top:52px;left:8px;display:flex;width:44px;max-height:calc(100% - 68px);flex-direction:column;align-items:center;padding:3px;border:1px solid var(--border-strong);background:var(--surface-panel);overflow-y:auto}
      .review-tool-group{display:flex;width:100%;flex-direction:column;align-items:center;gap:2px}
      .review-rail-tool{position:relative;display:grid;width:36px;height:36px;flex:0 0 36px;place-items:center;padding:0;border:1px solid transparent;border-radius:3px;background:transparent;color:var(--text-primary);cursor:pointer}
      .review-rail-tool:hover,.review-rail-tool:focus-visible,.review-rail-tool.active{outline:0;border-color:var(--border-strong);background:var(--surface-panel);color:var(--text-primary)}
      .review-rail-tool:disabled{opacity:.45;cursor:default}.review-rail-tool>.ui-icon{width:19px;height:19px}
      .review-tool-tooltip{position:absolute;z-index:calc(var(--z-toolbar) + 40);left:42px;top:50%;display:none;width:220px;padding:7px 9px;border:1px solid var(--border-strong);border-radius:3px;background:var(--surface-panel);color:var(--text-primary);text-align:left;transform:translateY(-50%);pointer-events:none}
      .review-tool-tooltip strong{display:block;margin-bottom:2px;color:var(--text-primary);font-size:11px;line-height:1.25}.review-tool-tooltip span{display:block;font-size:10px;line-height:1.35}
      .review-rail-tool:hover .review-tool-tooltip,.review-rail-tool:focus-visible .review-tool-tooltip{display:block}
      .review-single-document .review-stitched-image>.review-image-chunk{position:relative;z-index:1;display:block;width:100%;height:auto}
      .review-single-document .review-stitched-image{position:relative}
      .review-strip-slice{position:absolute;z-index:1;left:0;width:100%;overflow:hidden;pointer-events:none}
      .review-strip-slice>img{position:absolute;left:0;display:block;width:100%;max-width:none;height:auto;pointer-events:none;user-select:none}
      .review-single-document .review-stitched-image>.stitched-brush-chunk{position:absolute;z-index:5;left:0;display:block;width:100%;height:auto;pointer-events:none}
      .review-text-object-overlay{z-index:12;overflow:visible!important;border:1.5px solid var(--text-primary);background:transparent}
      .review-text-object-overlay.ellipse{border-radius:999px}.review-text-object-overlay.selected{border-width:2px}
      .review-text-object-overlay.tool-muted{opacity:.42;pointer-events:none!important}
      .review-text-drawing{position:absolute;z-index:20;border:1.5px dashed var(--text-primary);background:transparent;pointer-events:none}.review-text-drawing.ellipse{border-radius:999px}
      .review-document-viewport[data-active-tool="hand"]{cursor:grab}.review-document-viewport[data-active-tool="hand"].is-panning{cursor:grabbing}.review-document-viewport[data-active-tool="zoom"]{cursor:zoom-in}
      .review-stitched-image[data-active-tool="rectangle"],.review-stitched-image[data-active-tool="ellipse"],.review-stitched-image[data-active-tool="brush"],.review-stitched-image[data-active-tool="eraser"]{cursor:crosshair}
      @media(max-width:720px){.review-tool-rail{top:48px;left:4px;width:40px;max-height:calc(100% - 58px)}.review-rail-tool{width:32px;height:32px;flex-basis:32px}.review-tool-tooltip{display:none!important}}
    `;
    document.head.appendChild(style);
  }

  function imageUrl(page, mode = variant) {
    if (!page) return null;
    let url;
    let rev;
    if (mode === "original") {
      url = page.original || page.clean;
      rev = Number(page.source_revision || 0);
    } else if (mode === "rendered") {
      url = page._reviewRenderedUrl || (typeof page.rendered === "string" ? page.rendered : null) || page.clean || page.original;
      rev = Number(page.render_revision || page.clean_revision || page.process_revision || page.source_revision || 0);
    } else {
      url = page.skipped ? page.original : (page.clean || page.original);
      rev = Number(page.skipped ? page.source_revision || 0 : page.clean_revision || page.process_revision || page.source_revision || 0);
    }
    if (!url) return null;
    return `${url}${url.includes("?") ? "&" : "?"}review_revision=${rev}`;
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

  function coreMeta(page, imageHeight) {
    const core = page?.stitch_core;
    if (!core || typeof core !== "object") return null;
    const localY1 = Number(core.core_y1), localY2 = Number(core.core_y2);
    const sourceY1 = Number(core.core_source_y1), sourceY2 = Number(core.core_source_y2), sourceHeight = Number(core.source_height);
    if (![localY1, localY2, sourceY1, sourceY2, sourceHeight].every(Number.isFinite)) return null;
    if (localY1 < 0 || localY2 <= localY1 || localY2 > imageHeight) return null;
    if (sourceY1 < 0 || sourceY2 <= sourceY1 || sourceHeight < sourceY2) return null;
    if (localY2 - localY1 !== sourceY2 - sourceY1) return null;
    return { localY1, localY2, sourceY1, sourceY2, sourceHeight };
  }

  function makeBrushChunks(host, width, height) {
    const chunks = [];
    for (let y = 0; y < height; y += BRUSH_CHUNK_H) {
      chunks.push({
        host,
        width,
        y1: y,
        y2: Math.min(height, y + BRUSH_CHUNK_H),
        canvas: null,
        ctx: null,
        dirty: false,
      });
    }
    return chunks;
  }

  function ensureBrushCanvas(chunk) {
    if (chunk.canvas && chunk.ctx) return chunk;
    const canvas = document.createElement("canvas");
    canvas.className = "stitched-brush-chunk";
    canvas.width = chunk.width;
    canvas.height = chunk.y2 - chunk.y1;
    canvas.dataset.sourceY = String(chunk.y1);
    canvas.style.top = `${chunk.y1}px`;
    chunk.host.appendChild(canvas);
    chunk.canvas = canvas;
    chunk.ctx = canvas.getContext("2d", { willReadFrequently: true });
    return chunk;
  }

  function chunkHasPaint(chunk, y1 = chunk.y1, y2 = chunk.y2) {
    if (!chunk?.dirty || !chunk.canvas) return false;
    const iy1 = Math.max(chunk.y1, y1), iy2 = Math.min(chunk.y2, y2);
    if (iy2 <= iy1) return false;
    const h = iy2 - iy1, width = chunk.canvas.width, scale = Math.min(1, 512 / Math.max(width, h));
    const probe = document.createElement("canvas");
    probe.width = Math.max(1, Math.ceil(width * scale));
    probe.height = Math.max(1, Math.ceil(h * scale));
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(chunk.canvas, 0, iy1 - chunk.y1, width, h, 0, 0, probe.width, probe.height);
    const data = ctx.getImageData(0, 0, probe.width, probe.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 20) return true;
    return false;
  }

  function captureSnapshot(shell) {
    if (!shell || variant !== "clean") return;
    const dirty = (shell._brushChunks || []).filter((chunk) => chunk.dirty && chunkHasPaint(chunk));
    const key = snapshotKey();
    if (!dirty.length) return void snapshots.delete(key);
    snapshots.set(key, dirty.map((chunk) => ({ y1: chunk.y1, dataUrl: chunk.canvas.toDataURL("image/png") })));
  }

  function restoreSnapshot(shell) {
    const saved = snapshots.get(snapshotKey());
    if (!Array.isArray(saved)) return;
    const byY = new Map((shell._brushChunks || []).map((chunk) => [chunk.y1, chunk]));
    for (const item of saved) {
      const chunk = byY.get(Number(item.y1));
      if (!chunk) continue;
      ensureBrushCanvas(chunk);
      const img = new Image();
      img.onload = () => {
        if (!chunk.canvas?.isConnected) return;
        chunk.ctx.drawImage(img, 0, 0);
        chunk.dirty = true;
      };
      img.src = item.dataUrl;
    }
  }

  window.hasUnsavedStitchedMarks = () => {
    const prefix = `${chapterKey()}:`;
    if ([...snapshots.keys()].some((key) => key.startsWith(prefix))) return true;
    const shell = document.querySelector("#page-view.review-mode .review-document-shell");
    return Boolean(shell?._brushChunks?.some((chunk) => chunk.dirty && chunkHasPaint(chunk)));
  };

  function zoomScale(viewport, width) {
    if (!width) return 1;
    if (fitWidth) return Math.min(1, Math.max(240, viewport.clientWidth - 48) / width);
    return zoom / 100;
  }

  function applyZoom(shell, focus = null) {
    const viewport = shell.querySelector(".review-document-viewport");
    const stage = shell.querySelector(".review-stitched-zoom-stage");
    const image = shell.querySelector(".review-stitched-image");
    const label = shell.querySelector(".review-zoom-value");
    const width = Number(image?.dataset.sourceWidth || 0), height = Number(image?.dataset.sourceHeight || 0);
    if (!viewport || !stage || !image || !width || !height) return;
    const prev = Number(image.dataset.zoomScale || 1), scale = zoomScale(viewport, width);
    let src = null;
    if (focus && prev > 0) src = { x: (viewport.scrollLeft + focus.x) / prev, y: (viewport.scrollTop + focus.y) / prev };
    image.style.transform = `scale(${scale})`;
    image.dataset.zoomScale = String(scale);
    stage.style.width = `${Math.max(1, Math.round(width * scale))}px`;
    stage.style.height = `${Math.max(1, Math.round(height * scale))}px`;
    stage.style.marginTop = `${Math.max(24, Math.round((viewport.clientHeight - height * scale) / 2))}px`;
    stage.style.marginBottom = "24px";
    if (src) {
      viewport.scrollLeft = Math.max(0, src.x * scale - focus.x);
      viewport.scrollTop = Math.max(0, src.y * scale - focus.y);
    }
    if (label) label.textContent = fitWidth ? `Fit · ${Math.round(scale * 100)}%` : `${zoom}%`;
  }

  function stepZoom(delta) {
    fitWidth = false;
    let i = ZOOM_STEPS.findIndex((n) => n >= zoom);
    if (i < 0) i = ZOOM_STEPS.length - 1;
    if (delta < 0 && ZOOM_STEPS[i] >= zoom) i--;
    if (delta > 0 && ZOOM_STEPS[i] <= zoom) i++;
    zoom = ZOOM_STEPS[Math.max(0, Math.min(ZOOM_STEPS.length - 1, i))];
  }

  function descriptorFor(shell, pageIndex) {
    return (shell._descriptors || []).find((d) => Number(d.item.canonicalIndex) === Number(pageIndex)) || null;
  }

  function ownedObjects(shell) {
    const out = [];
    for (const desc of shell._descriptors || []) {
      const pageIndex = Number(desc.item.canonicalIndex), page = window.currentManifest?.pages?.[pageIndex];
      if (!page || page.skipped) continue;
      for (const obj of page.text_objects || []) {
        if (!obj?.id || !obj.region) continue;
        const center = (Number(obj.region.y1) + Number(obj.region.y2)) / 2;
        if (center >= desc.localY1 && center < desc.localY2) out.push({ desc, pageIndex, obj });
      }
    }
    return out;
  }

  const sourceRegion = (desc, r) => ({
    x1: Number(r.x1), y1: desc.sourceY1 + (Number(r.y1) - desc.localY1),
    x2: Number(r.x2), y2: desc.sourceY1 + (Number(r.y2) - desc.localY1),
  });

  function syncOverlay(shell, pageIndex, id) {
    const desc = descriptorFor(shell, pageIndex);
    const obj = window.findTextObject?.(pageIndex, id);
    const overlay = shell.querySelector(
      `.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`,
    );
    if (!desc || !obj?.region || !overlay) return false;
    const r = sourceRegion(desc, obj.region);
    Object.assign(overlay.style, {
      left: `${r.x1}px`,
      top: `${r.y1}px`,
      width: `${Math.max(MIN_BOX, r.x2 - r.x1)}px`,
      height: `${Math.max(MIN_BOX, r.y2 - r.y1)}px`,
    });
    return true;
  }

  function installOverlaySync() {
    if (!window._reviewBaseSyncOverlayForObject) window._reviewBaseSyncOverlayForObject = window.syncOverlayForObject || null;
    window.syncOverlayForObject = (pageIndex, id) => {
      const shell = document.querySelector("#page-view.review-mode .review-document-shell");
      if (shell && syncOverlay(shell, pageIndex, id)) return;
      window._reviewBaseSyncOverlayForObject?.(pageIndex, id);
    };
  }

  function selectObject(shell, pageIndex, id) {
    if (!window.editorState) return;
    window.editorState.activePageIndex = Number(pageIndex);
    window.editorState.selectedTextObjectId = id;
    shell.querySelectorAll(".review-text-object-overlay").forEach((el) => el.classList.toggle("selected", Number(el.dataset.pageIndex) === Number(pageIndex) && el.dataset.objectId === String(id)));
    const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
    if (floating) {
      floating.hidden = false;
      floating.classList.add("is-open");
    }
    window.renderEditorPanel?.(Number(pageIndex));
  }

  function clearSelection(shell) {
    if (window.editorState) window.editorState.selectedTextObjectId = null;
    shell.querySelectorAll(".review-text-object-overlay").forEach((el) => el.classList.remove("selected"));
    const floating = shell._floatingInspector || shell.querySelector(".review-floating-inspector");
    if (floating) {
      floating.classList.remove("is-open");
      floating.hidden = true;
    }
    const panel = shell.querySelector(".translation-panel-host");
    if (panel) panel.innerHTML = '<div class="ui-empty-state">Chọn một vùng chữ trên trang để chỉnh sửa.</div>';
  }

  function overlayMode(event, overlay) {
    const rect = overlay.getBoundingClientRect(), edge = Math.max(4, Math.min(10, Math.min(rect.width, rect.height) / 3));
    const l = event.clientX - rect.left < edge, r = rect.right - event.clientX < edge, t = event.clientY - rect.top < edge, b = rect.bottom - event.clientY < edge;
    if (t && l) return "nw"; if (t && r) return "ne"; if (b && l) return "sw"; if (b && r) return "se";
    if (l) return "w"; if (r) return "e"; if (t) return "n"; if (b) return "s"; return "move";
  }
  function cursor(mode) {
    if (mode === "nw" || mode === "se") return "nwse-resize";
    if (mode === "ne" || mode === "sw") return "nesw-resize";
    if (mode === "n" || mode === "s") return "ns-resize";
    if (mode === "e" || mode === "w") return "ew-resize";
    return "move";
  }

  function installTransform(shell, overlay, desc, pageIndex, obj, signal) {
    let drag = null;
    const image = shell.querySelector(".review-stitched-image");
    overlay.addEventListener("pointerdown", (event) => {
      if (variant !== "clean" || tool !== "select" || event.button !== 0) return;
      event.preventDefault(); event.stopPropagation(); selectObject(shell, pageIndex, obj.id);
      const rect = image.getBoundingClientRect(), W = Number(image.dataset.sourceWidth || 0), H = Number(image.dataset.sourceHeight || 0);
      if (!rect.width || !rect.height || !W || !H) return;
      drag = { mode: overlayMode(event, overlay), x: (event.clientX - rect.left) * W / rect.width, y: (event.clientY - rect.top) * H / rect.height, original: { ...obj.region } };
      overlay.classList.add("transforming"); overlay.setPointerCapture?.(event.pointerId);
    }, { signal });
    overlay.addEventListener("pointermove", (event) => {
      if (!drag) { if (tool === "select") overlay.style.cursor = cursor(overlayMode(event, overlay)); return; }
      const rect = image.getBoundingClientRect(), W0 = Number(image.dataset.sourceWidth || 0), H0 = Number(image.dataset.sourceHeight || 0);
      if (!rect.width || !rect.height || !W0 || !H0) return;
      const x = (event.clientX - rect.left) * W0 / rect.width, y = (event.clientY - rect.top) * H0 / rect.height, dx = x - drag.x, dy = y - drag.y;
      const o = drag.original, W = desc.img.naturalWidth, minY = 0, maxY = desc.img.naturalHeight, clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      let { x1, y1, x2, y2 } = o;
      if (drag.mode === "move") {
        const w = o.x2 - o.x1, h = o.y2 - o.y1;
        x1 = clamp(o.x1 + dx, 0, Math.max(0, W - w)); y1 = clamp(o.y1 + dy, minY, Math.max(minY, maxY - h)); x2 = x1 + w; y2 = y1 + h;
      } else {
        if (drag.mode.includes("w")) x1 = clamp(o.x1 + dx, 0, o.x2 - MIN_BOX);
        if (drag.mode.includes("e")) x2 = clamp(o.x2 + dx, o.x1 + MIN_BOX, W);
        if (drag.mode.includes("n")) y1 = clamp(o.y1 + dy, minY, o.y2 - MIN_BOX);
        if (drag.mode.includes("s")) y2 = clamp(o.y2 + dy, o.y1 + MIN_BOX, maxY);
      }
      obj.region = { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
      syncOverlay(shell, pageIndex, obj.id);
    }, { signal });
    const end = () => {
      if (!drag) return;
      const o = drag.original; drag = null; overlay.classList.remove("transforming");
      const r = obj.region;
      if (r.x1 !== o.x1 || r.y1 !== o.y1 || r.x2 !== o.x2 || r.y2 !== o.y2) {
        window.scheduleGeomPersist?.(pageIndex, obj.id); window.refreshGeometryControls?.(pageIndex, obj.id);
      }
    };
    overlay.addEventListener("pointerup", end, { signal }); overlay.addEventListener("pointercancel", end, { signal });
  }

  function createInlineEditor(shell, desc, pageIndex, obj, signal) {
    const overlay = document.createElement("div");
    overlay.className = `text-object-overlay review-text-object-overlay${obj.shape === "ellipse" ? " ellipse" : ""}`;
    overlay.dataset.pageIndex = String(pageIndex);
    overlay.dataset.objectId = String(obj.id);
    overlay.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      selectObject(shell, pageIndex, obj.id);
    }, { signal });
    shell.querySelector(".review-stitched-image")?.appendChild(overlay);
    syncOverlay(shell, pageIndex, obj.id);
    installTransform(shell, overlay, desc, pageIndex, obj, signal);
  }

  function renderOverlays(shell, signal) {
    const image = shell.querySelector(".review-stitched-image"); if (!image) return;
    image.querySelectorAll(".review-text-object-overlay,.review-text-drawing").forEach((n) => n.remove());
    if (variant !== "clean") return;
    for (const { desc, pageIndex, obj } of ownedObjects(shell)) createInlineEditor(shell, desc, pageIndex, obj, signal);
    syncTool(shell);
  }

  function restoreWorkspaceState(shell) {
    if (shell._restoreApplied) return;
    const workspace = shell.closest(".review-workspace-shell");
    const state = workspace?._reviewRestoreState;
    if (!state || state.chapterId !== chapterKey()) return;
    shell._restoreApplied = true;
    workspace._reviewRestoreState = null;

    const viewport = shell.querySelector(".review-document-viewport");
    if (viewport) {
      viewport.scrollLeft = Math.max(0, Number(state.scrollLeft || 0));
      viewport.scrollTop = Math.max(0, Number(state.scrollTop || 0));
    }

    const pageIndex = Number(state.activePageIndex);
    const id = state.selectedTextObjectId;
    if (variant === "clean" && Number.isInteger(pageIndex) && id) {
      const overlay = shell.querySelector(
        `.review-text-object-overlay[data-page-index="${pageIndex}"][data-object-id="${CSS.escape(String(id))}"]`,
      );
      if (overlay) selectObject(shell, pageIndex, id);
    }
  }

  async function waitForOcr(shell, pageIndex, id, signal) {
    const start = Date.now();
    while (!signal.aborted && Date.now() - start < 8000) {
      if (window.findTextObject?.(pageIndex, id)?.ocr_text?.trim()) break;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    syncOverlay(shell, pageIndex, id);
    if (window.editorState?.selectedTextObjectId === id) {
      window.renderEditorPanel?.(Number(pageIndex));
    }
  }

  async function ensureObjects(shell) {
    if (typeof window.ensureAutoTextObjects !== "function") return;
    for (const desc of shell._descriptors || []) {
      const pageIndex = Number(desc.item.canonicalIndex), key = `${chapterKey()}:${pageIndex}`;
      if (autoSynced.has(key)) continue;
      autoSynced.add(key);
      try { await window.ensureAutoTextObjects(pageIndex); } catch (err) { console.warn("Could not ensure text objects", pageIndex, err); }
    }
  }

  function sourceCoverage(desc) {
    return { y1: desc.sourceY1 - desc.localY1, y2: desc.sourceY1 + (desc.img.naturalHeight - desc.localY1) };
  }
  function ownerForRegion(shell, y1, y2) {
    const center = (y1 + y2) / 2;
    const complete = (shell._descriptors || []).filter((d) => { const c = sourceCoverage(d); return y1 >= c.y1 && y2 <= c.y2; });
    if (!complete.length) return null;
    return complete.find((d) => center >= d.sourceY1 && center < d.sourceY2) || complete.sort((a, b) => {
      const ca = sourceCoverage(a), cb = sourceCoverage(b);
      return Math.min(y1 - cb.y1, cb.y2 - y2) - Math.min(y1 - ca.y1, ca.y2 - y2);
    })[0];
  }

  function sourcePoint(image, event) {
    const rect = image.getBoundingClientRect(), W = Number(image.dataset.sourceWidth || 0), H = Number(image.dataset.sourceHeight || 0);
    if (!rect.width || !rect.height || !W || !H) return null;
    return { x: Math.max(0, Math.min(W, (event.clientX - rect.left) * W / rect.width)), y: Math.max(0, Math.min(H, (event.clientY - rect.top) * H / rect.height)) };
  }

  function installTextDrawing(shell, signal) {
    const image = shell.querySelector(".review-stitched-image"); let drawing = null;
    const update = (p) => {
      if (!drawing || !p) return;
      const x1 = Math.min(drawing.start.x, p.x), y1 = Math.min(drawing.start.y, p.y), x2 = Math.max(drawing.start.x, p.x), y2 = Math.max(drawing.start.y, p.y);
      Object.assign(drawing.preview.style, { left: `${x1}px`, top: `${y1}px`, width: `${x2 - x1}px`, height: `${y2 - y1}px` }); drawing.last = p;
    };
    image.addEventListener("pointerdown", (e) => {
      if (variant !== "clean" || !["rectangle", "ellipse"].includes(tool) || e.button !== 0 || e.target.closest(".review-text-object-overlay")) return;
      const p = sourcePoint(image, e); if (!p) return; e.preventDefault();
      const preview = document.createElement("div"); preview.className = `text-object-overlay drawing review-text-drawing${tool === "ellipse" ? " ellipse" : ""}`; image.appendChild(preview);
      drawing = { tool, start: p, last: p, preview }; image.setPointerCapture?.(e.pointerId); update(p);
    }, { signal });
    image.addEventListener("pointermove", (e) => { if (drawing) update(sourcePoint(image, e)); }, { signal });
    const finish = async () => {
      if (!drawing) return;
      const d = drawing; drawing = null; d.preview.remove();
      const x1 = Math.round(Math.min(d.start.x, d.last.x)), y1 = Math.round(Math.min(d.start.y, d.last.y)), x2 = Math.round(Math.max(d.start.x, d.last.x)), y2 = Math.round(Math.max(d.start.y, d.last.y));
      if (x2 - x1 < MIN_BOX || y2 - y1 < MIN_BOX) return;
      const owner = ownerForRegion(shell, y1, y2);
      if (!owner) return window.showToast?.("Vùng chữ vượt quá phần overlap an toàn giữa hai lát. Hãy thu vùng lại một chút.", "warning");
      const pageIndex = Number(owner.item.canonicalIndex), localY1 = Math.round(owner.localY1 + y1 - owner.sourceY1), localY2 = Math.round(owner.localY1 + y2 - owner.sourceY1);
      const region = { x1: Math.max(0, x1), y1: Math.max(0, localY1), x2: Math.min(owner.img.naturalWidth, x2), y2: Math.min(owner.img.naturalHeight, localY2) };
      if (region.x2 - region.x1 < MIN_BOX || region.y2 - region.y1 < MIN_BOX) return;
      try {
        if (typeof window.createTextObject !== "function") throw new Error("Công cụ tạo vùng chữ chưa sẵn sàng");
        await window.createTextObject(pageIndex, d.tool, region);
        const id = window.editorState?.selectedTextObjectId; window.editorState.activePageIndex = pageIndex; renderOverlays(shell, signal);
        if (id) { selectObject(shell, pageIndex, id); void waitForOcr(shell, pageIndex, id, signal); }
        setTool(shell, "select");
      } catch (err) { window.showToast?.("Không thể tạo vùng chữ: " + err.message, "error"); }
    };
    window.addEventListener("pointerup", finish, { signal }); window.addEventListener("pointercancel", finish, { signal });
    image.addEventListener("click", (e) => { if (tool === "select" && !e.target.closest(".review-text-object-overlay")) clearSelection(shell); }, { signal });
  }

  async function renderStrip(shell, items, signal) {
    const token = ++renderToken;
    const stage = shell.querySelector(".review-stitched-zoom-stage");
    const image = shell.querySelector(".review-stitched-image");
    const meta = shell.querySelector(".review-stitched-meta");
    const warning = shell.querySelector(".review-stitched-warning");
    image.replaceChildren();
    stage.style.width = "auto";
    stage.style.height = "auto";
    warning.hidden = true;

    const loading = document.createElement("div");
    loading.className = "ui-state review-stitched-loading";
    loading.textContent = variant === "original"
      ? "Đang dựng strip ảnh gốc…"
      : variant === "rendered"
        ? "Đang dựng strip lettering…"
        : "Đang dựng strip sau inpaint…";
    image.appendChild(loading);

    try {
      const descriptors = [];
      let stripWidth = 0;
      let stripHeight = 0;
      let fallbackSlices = 0;

      let firstPreloaded = null;
      for (const item of items) {
        const live = livePage(item);
        if (!live) continue;
        const url = imageUrl(live);
        if (!url) throw new Error(`Lát ${item.canonicalIndex + 1} không có dữ liệu ảnh.`);

        let width = Number(live.width || 0);
        let height = Number(live.height || 0);
        const rawCore = live.stitch_core;
        if (!(height > 0) && rawCore && typeof rawCore === "object") {
          const sourceY1 = Number(rawCore.source_y1);
          const sourceY2 = Number(rawCore.source_y2);
          if (Number.isFinite(sourceY1) && Number.isFinite(sourceY2) && sourceY2 > sourceY1) {
            height = sourceY2 - sourceY1;
          }
        }

        let preloaded = null;
        if (!(stripWidth > 0) && !(width > 0)) {
          preloaded = await loadImage(url);
          if (token !== renderToken) return;
          width = preloaded.naturalWidth;
          if (!(height > 0)) height = preloaded.naturalHeight;
          firstPreloaded = preloaded;
        }
        if (!(stripWidth > 0)) stripWidth = width;
        if (!(width > 0)) width = stripWidth;

        if (!(height > 0)) {
          preloaded = preloaded || await loadImage(url);
          if (token !== renderToken) return;
          height = preloaded.naturalHeight;
          width = preloaded.naturalWidth || width;
        }
        if (!(width > 0)) throw new Error(`Lát ${item.canonicalIndex + 1} không có chiều rộng hợp lệ.`);
        stripWidth = Math.max(stripWidth, width);

        const core = coreMeta(live, height);
        const localY1 = core ? core.localY1 : 0;
        const localY2 = core ? core.localY2 : height;
        if (!core) fallbackSlices += 1;
        const ownedHeight = localY2 - localY1;
        descriptors.push({
          item: { canonicalIndex: item.canonicalIndex },
          img: { naturalWidth: width, naturalHeight: height },
          localY1,
          localY2,
          sourceY1: stripHeight,
          sourceY2: stripHeight + ownedHeight,
          url,
          preloaded,
        });
        stripHeight += ownedHeight;
      }

      if (!descriptors.length || !stripWidth || !stripHeight) {
        throw new Error("Chapter không có lát ảnh hợp lệ để hiển thị.");
      }

      image.replaceChildren();
      Object.assign(image.style, {
        width: `${stripWidth}px`,
        height: `${stripHeight}px`,
      });
      image.dataset.sourceWidth = String(stripWidth);
      image.dataset.sourceHeight = String(stripHeight);
      image.dataset.stripSlices = String(descriptors.length);

      for (const desc of descriptors) {
        const slice = document.createElement("div");
        slice.className = "review-strip-slice";
        slice.dataset.pageIndex = String(desc.item.canonicalIndex);
        Object.assign(slice.style, {
          top: `${desc.sourceY1}px`,
          width: `${desc.img.naturalWidth}px`,
          height: `${desc.sourceY2 - desc.sourceY1}px`,
        });

        const img = desc.preloaded || new Image();
        img.className = "review-strip-slice-image";
        img.alt = "";
        img.decoding = "async";
        img.loading = desc.sourceY1 === 0 ? "eager" : "lazy";
        Object.assign(img.style, {
          top: `-${desc.localY1}px`,
        });
        if (!desc.preloaded) {
          img.addEventListener("error", () => {
            slice.classList.add("review-strip-slice-error");
            slice.textContent = `Không tải được lát ${desc.item.canonicalIndex + 1}`;
          }, { once: true });
          img.src = desc.url;
        }
        slice.appendChild(img);
        image.appendChild(slice);

        delete desc.preloaded;
        delete desc.url;
      }

      shell._descriptors = descriptors;
      shell._brushChunks = variant === "clean"
        ? makeBrushChunks(image, stripWidth, stripHeight)
        : [];
      if (variant === "clean") restoreSnapshot(shell);

      meta.textContent = `1 strip · ${descriptors.length} lát · ${stripWidth} × ${stripHeight}px · ${
        variant === "original"
          ? "Ảnh gốc"
          : variant === "rendered"
            ? "Bản lettering"
            : "Sau inpaint"
      }`;
      if (fallbackSlices) {
        warning.hidden = false;
        warning.textContent = `${fallbackSlices} lát thiếu ownership metadata; đang nối toàn bộ lát theo thứ tự.`;
      }

      applyZoom(shell);
      await ensureObjects(shell);
      if (token !== renderToken) return;
      renderOverlays(shell, signal);
      syncTool(shell);
      restoreWorkspaceState(shell);
    } catch (err) {
      if (token !== renderToken) return;
      image.replaceChildren();
      shell._brushChunks = [];
      const error = document.createElement("div");
      error.className = "ui-state ui-state-error review-stitched-error";
      error.textContent = `Không dựng được strip: ${err.message}`;
      image.appendChild(error);
    }
  }

  function mountTextInspector(workspace, shell, signal) {
    const floating = document.createElement("aside");
    floating.className = "review-floating-inspector";
    floating.hidden = true;
    floating.setAttribute("aria-label", "Thuộc tính vùng chữ");

    const header = document.createElement("div");
    header.className = "review-floating-inspector-header";
    const title = document.createElement("strong");
    title.textContent = "Text";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "ui-icon-btn ui-btn-ghost ui-btn-compact";
    close.setAttribute("aria-label", "Đóng thuộc tính");
    close.append(window.createUiIcon("close"));
    header.append(title, close);

    const host = document.createElement("section");
    host.className = "translation-panel-host review-text-inspector-host";
    host.setAttribute("aria-label", "Chỉnh OCR, bản dịch và typography");
    host.innerHTML = '<div class="ui-empty-state">Chọn một vùng chữ trên trang để chỉnh sửa.</div>';

    floating.append(header, host);
    shell.appendChild(floating);
    shell._floatingInspector = floating;

    close.addEventListener("click", () => clearSelection(shell), { signal });


  }

  function syncToolButtons() {
    document.querySelectorAll(".review-rail-tool[data-tool]").forEach((b) => { const active = b.dataset.tool === tool; b.classList.toggle("active", active); b.setAttribute("aria-pressed", String(active)); });
  }
  function syncTool(shell) {
    const image = shell?.querySelector(".review-stitched-image"), viewport = shell?.querySelector(".review-document-viewport"); if (!image || !viewport) return;
    image.dataset.activeTool = tool; viewport.dataset.activeTool = tool;
    const paint = ["brush", "eraser"].includes(tool), text = ["rectangle", "ellipse"].includes(tool);
    shell.querySelectorAll(".review-text-object-overlay").forEach((el) => { el.style.pointerEvents = tool === "select" && variant === "clean" ? "auto" : "none"; el.classList.toggle("tool-muted", variant === "clean" && tool !== "select"); });
    image.classList.toggle("text-draw-mode", text && variant === "clean"); image.classList.toggle("brush-mode", paint && variant === "clean"); syncToolButtons();
  }
  function setTool(shell, next) {
    if (!["select", "rectangle", "ellipse", "brush", "eraser", "hand", "zoom"].includes(next)) next = "select";
    tool = next;
    if (variant !== "clean" && ["rectangle", "ellipse", "brush", "eraser"].includes(tool)) { variant = "clean"; shell._syncVariantUI?.(); shell._rerender?.(); }
    if (window.editorState) window.editorState.tool = ["rectangle", "ellipse"].includes(tool) ? tool : "select";
    window.setEditorTool?.(window.editorState?.tool || "select"); syncTool(shell);
  }

  function toolButton(shell, signal, name, icon, title, description, action = null) {
    const b = document.createElement("button"); b.type = "button"; b.className = "review-rail-tool"; b.dataset.tool = name; b.setAttribute("aria-label", `${title}: ${description}`); b.setAttribute("aria-pressed", "false"); b.append(window.createUiIcon(icon));
    const tip = document.createElement("span"); tip.className = "review-tool-tooltip"; const strong = document.createElement("strong"); strong.textContent = title; const text = document.createElement("span"); text.textContent = description; tip.append(strong, text); b.appendChild(tip);
    b.addEventListener("click", () => action ? action() : setTool(shell, name), { signal }); return b;
  }
  function mountToolRail(shell, signal) {
    document.body.classList.remove("review-tool-rail-active");
    shell.querySelector(".review-tool-rail")?.remove();
    const rail = document.createElement("div");
    rail.className = "review-tool-rail";
    rail.setAttribute("role", "toolbar");
    rail.setAttribute("aria-label", "Công cụ chỉnh sửa ảnh và lettering");

    const main = document.createElement("div");
    main.className = "review-tool-group";
    [
      ["select","cursor","Chọn","Chọn, kéo và thay đổi kích thước vùng chữ."],
      ["rectangle","rect-select","Vùng chữ nhật","Kéo quanh bong bóng để tạo vùng OCR."],
      ["ellipse","ellipse-select","Vùng elip","Kéo quanh bong bóng tròn để tạo vùng OCR."],
      ["brush","brush","Cọ Inpaint","Tô vùng cần xóa rồi chạy Inpaint."],
      ["eraser","eraser","Tẩy mask","Xóa phần mask inpaint đã tô nhầm."],
      ["hand","hand","Bàn tay","Kéo trang tự do theo mọi hướng."],
      ["zoom","zoom","Thu phóng","Nhấp để phóng to, Alt + nhấp để thu nhỏ."],
    ].forEach((x) => main.appendChild(toolButton(shell, signal, ...x)));

    rail.appendChild(main);
    shell.appendChild(rail);
    signal.addEventListener("abort", () => rail.remove(), { once: true });
    syncToolButtons();
  }

  function mountActions(shell, signal, rerender) {
    const actions = shell.querySelector(".review-docbar-actions");
    if (!actions) return;
    actions.replaceChildren();

    const render = document.createElement("button");
    render.type = "button";
    render.className = "ui-btn ui-btn-ghost ui-btn-compact review-render-text-btn";
    render.textContent = "Render";
    render.title = "Render chữ lên trang";

    render.addEventListener("click", async () => {
      const indices = [...new Set((shell._descriptors || []).map((d) => Number(d.item.canonicalIndex)))];
      if (!indices.length || typeof window.renderTranslations !== "function") return;
      render.disabled = true;
      render.textContent = "Rendering…";
      try {
        let count = 0;
        for (const pageIndex of indices) {
          await window.renderTranslations(pageIndex);
          const page = window.currentManifest?.pages?.[pageIndex];
          if (page?.rendered) {
            page._reviewRenderedUrl = page.rendered;
            count++;
          }
        }
        if (!count) return window.showToast?.("Không có vùng chữ nào được render trên trang này.", "info");
        variant = "rendered";
        window.showToast?.(`Đã render lettering cho ${count} lát.`, "success");
        rerender();
      } catch (err) {
        window.showToast?.("Không thể render lettering: " + err.message, "error");
      } finally {
        render.disabled = false;
        render.textContent = "Render";
      }
    }, { signal });

    actions.appendChild(render);
    if (typeof window.buildChapterTranslateControls === "function") {
      actions.appendChild(window.buildChapterTranslateControls());
    }
    if (typeof window.buildChapterExportButton === "function") {
      actions.appendChild(window.buildChapterExportButton());
    }
  }

  function paintPoint(shell, x, y, radius, erase) {
    for (const chunk of shell._brushChunks || []) {
      if (y + radius < chunk.y1 || y - radius >= chunk.y2) continue;
      ensureBrushCanvas(chunk);
      const ctx = chunk.ctx; ctx.save(); ctx.globalCompositeOperation = erase ? "destination-out" : "source-over"; ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--text-primary").trim() || "#000000"; ctx.beginPath(); ctx.arc(x, y - chunk.y1, radius, 0, Math.PI * 2); ctx.fill(); ctx.restore(); chunk.dirty = true;
    }
  }
  function paintStroke(shell, a, b, radius, erase) {
    const dx = b.x - a.x, dy = b.y - a.y, count = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(1, radius * .45)));
    for (let i = 0; i <= count; i++) { const t = i / count; paintPoint(shell, a.x + dx * t, a.y + dy * t, radius, erase); }
  }
  function drawBrushMask(ctx, chunks, desc, width) {
    for (const chunk of chunks || []) {
      if (!chunk.dirty || !chunk.canvas) continue;
      const y1 = Math.max(chunk.y1, desc.sourceY1), y2 = Math.min(chunk.y2, desc.sourceY2);
      if (y2 <= y1) continue;
      const h = y2 - y1;
      ctx.drawImage(
        chunk.canvas,
        0,
        y1 - chunk.y1,
        width,
        h,
        0,
        desc.localY1 + y1 - desc.sourceY1,
        width,
        h,
      );
    }
  }
  function canvasBlob(canvas) {
    if (typeof window.canvasToBlob === "function") return window.canvasToBlob(canvas);
    return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Không thể mã hóa mask")), "image/png"));
  }

  function mount(workspace) {
    if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchInspectorMounted === "1") return;
    resetChapterState();
    ensureStyles();
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
    shell.className = "review-stitched-shell review-document-shell";

    const docbar = document.createElement("div");
    docbar.className = "review-stitched-toolbar review-document-toolbar review-document-toolbar-compact";

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
    const center = document.createElement("div");
    center.className = "review-docbar-group review-docbar-center";
    const right = document.createElement("div");
    right.className = "review-docbar-group review-docbar-right";

    const stripMeta = document.createElement("span");
    stripMeta.className = "review-docbar-mini-meta review-strip-meta";
    stripMeta.textContent = `1 strip · ${items.length} lát`;

    const clean = button(null, "Clean");
    const rendered = button(null, "Rendered");
    const original = button(null, "Original");

    const zoomOut = button("minus", "Thu nhỏ");
    const zoomValue = button(null, "Fit");
    zoomValue.classList.add("review-zoom-value");
    const zoomIn = button("plus", "Phóng to");
    const one = button(null, "100%");

    const submit = document.createElement("button");
    submit.type = "button";
    submit.className = "ui-btn ui-btn-primary ui-btn-compact repaint-btn";
    submit.textContent = "Inpaint";

    const clearMask = document.createElement("button");
    clearMask.type = "button";
    clearMask.className = "ui-btn ui-btn-ghost ui-btn-compact clear-brush-btn";
    clearMask.textContent = "Clear";

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

    const actionsHost = document.createElement("div");
    actionsHost.className = "review-docbar-actions";

    left.append(stripMeta);
    center.append(clean, rendered, original, zoomOut, zoomValue, zoomIn, one);
    right.append(submit, clearMask, sizeWrap, actionsHost);
    docbar.append(left, center, right);

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

    shell.append(docbar, meta, warning, viewport);
    canvasHost.appendChild(shell);

    installOverlaySync();
    mountTextInspector(workspace, shell, signal);
    mountToolRail(shell, signal);
    installTextDrawing(shell, signal);
    mountActions(shell, signal, () => shell._rerender?.());
    window.mountChapterOCR?.();

    let painting = false, radius = 24, last = null, panning = false, pan = null, space = false;

    const syncVariant = () => { clean.classList.toggle("ui-btn-primary", variant === "clean"); rendered.classList.toggle("ui-btn-primary", variant === "rendered"); original.classList.toggle("ui-btn-primary", variant === "original"); const readonly = variant !== "clean"; submit.disabled = readonly; clearMask.disabled = readonly; size.disabled = readonly; workspace.classList.toggle("review-readonly-document", readonly); syncTool(shell); };
    shell._syncVariantUI = syncVariant;
    const busy = (on, text = "Đang xử lý…") => { workspace.classList.toggle("review-busy", on); [submit, clearMask, size, clean, rendered, original, zoomOut, zoomIn, one, zoomValue].forEach((el) => el.disabled = on); submit.textContent = on ? text : "Inpaint"; document.querySelectorAll(".review-rail-tool").forEach((b) => b.disabled = on); if (!on) syncVariant(); };
    const updateCompat = () => {
      const canonical = Number(items[0]?.canonicalIndex ?? 0);
      compatibility.dataset.pageIndex = String(canonical);
      workspace.dataset.reviewCanonicalIndex = String(canonical);
      window.setWorkflowCheckpoint?.("review", canonical);
    };
    const rerender = () => {
      captureSnapshot(shell);
      updateCompat();
      syncVariant();
      void renderStrip(shell, items, signal);
    };
    shell._rerender = rerender;
    shell._showRendered = () => { variant = "rendered"; rerender(); };
    signal.addEventListener("abort", () => captureSnapshot(shell), { once: true });

    image.addEventListener("pointerdown", (e) => { if (variant !== "clean" || !["brush", "eraser"].includes(tool) || e.button !== 0) return; const p = sourcePoint(image, e); if (!p) return; e.preventDefault(); painting = true; last = p; image.setPointerCapture?.(e.pointerId); paintPoint(shell, p.x, p.y, radius, tool === "eraser"); }, { signal });
    image.addEventListener("pointermove", (e) => { if (!painting || !last || !["brush", "eraser"].includes(tool)) return; const p = sourcePoint(image, e); if (!p) return; paintStroke(shell, last, p, radius, tool === "eraser"); last = p; }, { signal });
    const stopPaint = () => { painting = false; last = null; }; image.addEventListener("pointerup", stopPaint, { signal }); image.addEventListener("pointercancel", stopPaint, { signal });

    window.addEventListener("keydown", (e) => {
      const tag = e.target?.tagName?.toLowerCase(), editable = e.target?.isContentEditable || tag === "input" || tag === "textarea" || tag === "select"; if (editable) return;
      if (e.code === "Space" && !e.repeat) { space = true; return; }
      if ((e.ctrlKey || e.metaKey) && e.key === "0") { e.preventDefault(); fitWidth = true; return applyZoom(shell); }
      if ((e.ctrlKey || e.metaKey) && e.key === "1") { e.preventDefault(); fitWidth = false; zoom = 100; return applyZoom(shell); }
      if (e.key === "[" && ["brush","eraser"].includes(tool)) { e.preventDefault(); radius = Math.max(4, radius - 2); size.value = String(radius); sizeOut.textContent = `${radius * 2}px`; return; }
      if (e.key === "]" && ["brush","eraser"].includes(tool)) { e.preventDefault(); radius = Math.min(100, radius + 2); size.value = String(radius); sizeOut.textContent = `${radius * 2}px`; return; }
      if ((e.key === "Delete" || e.key === "Backspace") && window.editorState?.selectedTextObjectId) { const pageIndex = Number(window.editorState.activePageIndex || 0), id = window.editorState.selectedTextObjectId; e.preventDefault(); window.deleteTextObject?.(pageIndex, id)?.then(() => renderOverlays(shell, signal)).catch((err) => window.showToast?.("Không thể xóa vùng chữ: " + err.message, "error")); return; }
      if (e.key === "Escape") return setTool(shell, "select");
      if (!e.ctrlKey && !e.metaKey && !e.altKey && SHORTCUTS[e.key.toLowerCase()]) { e.preventDefault(); setTool(shell, SHORTCUTS[e.key.toLowerCase()]); }
    }, { signal });
    window.addEventListener("keyup", (e) => { if (e.code === "Space") { space = false; panning = false; pan = null; viewport.classList.remove("is-panning"); } }, { signal });
    viewport.addEventListener("pointerdown", (e) => { if (e.button !== 0 || !(space || tool === "hand")) return; panning = true; pan = { x: e.clientX, y: e.clientY, left: viewport.scrollLeft, top: viewport.scrollTop }; viewport.setPointerCapture?.(e.pointerId); viewport.classList.add("is-panning"); e.preventDefault(); }, { signal });
    viewport.addEventListener("pointermove", (e) => { if (!panning || !pan) return; viewport.scrollLeft = pan.left - (e.clientX - pan.x); viewport.scrollTop = pan.top - (e.clientY - pan.y); }, { signal });
    viewport.addEventListener("pointerup", () => { panning = false; pan = null; viewport.classList.remove("is-panning"); }, { signal });
    viewport.addEventListener("click", (e) => { if (tool !== "zoom" || e.target.closest("button,input,textarea,select")) return; const r = viewport.getBoundingClientRect(); stepZoom(e.altKey ? -1 : 1); applyZoom(shell, { x: e.clientX - r.left, y: e.clientY - r.top }); }, { signal });
    viewport.addEventListener("wheel", (e) => { if (!(e.ctrlKey || e.metaKey)) return; e.preventDefault(); const r = viewport.getBoundingClientRect(); stepZoom(e.deltaY < 0 ? 1 : -1); applyZoom(shell, { x: e.clientX - r.left, y: e.clientY - r.top }); }, { passive: false, signal });

    size.addEventListener("input", () => { radius = Number(size.value); sizeOut.textContent = `${radius * 2}px`; }, { signal });
    clearMask.addEventListener("click", () => { for (const c of shell._brushChunks || []) { if (c.canvas && c.ctx) c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height); c.dirty = false; } snapshots.delete(snapshotKey()); }, { signal });
    clean.addEventListener("click", () => { if (variant !== "clean") { variant = "clean"; rerender(); } }, { signal }); rendered.addEventListener("click", () => { if (variant !== "rendered") { captureSnapshot(shell); variant = "rendered"; rerender(); } }, { signal }); original.addEventListener("click", () => { if (variant !== "original") { captureSnapshot(shell); variant = "original"; rerender(); } }, { signal });
    zoomOut.addEventListener("click", () => { stepZoom(-1); applyZoom(shell); }, { signal }); zoomIn.addEventListener("click", () => { stepZoom(1); applyZoom(shell); }, { signal }); zoomValue.addEventListener("click", () => { fitWidth = true; applyZoom(shell); }, { signal }); one.addEventListener("click", () => { fitWidth = false; zoom = 100; applyZoom(shell); }, { signal });
    window.addEventListener("resize", () => { if (fitWidth) applyZoom(shell); }, { signal });

    submit.addEventListener("click", async () => {
      const chapterId = window.currentChapterId, chunks = shell._brushChunks || []; if (!chapterId || !chunks.some((c) => c.dirty && chunkHasPaint(c))) return window.showToast?.("Chưa có vùng nào được đánh dấu.", "error");
      const mode = typeof window.chooseRepaintMode === "function" ? await window.chooseRepaintMode() : "standard"; if (!mode || chapterId !== window.currentChapterId) return; busy(true, mode === "lama" ? "LaMa đang xử lý…" : "Đang xử lý…");
      try {
        let affected = 0;
        for (const desc of shell._descriptors || []) {
          if (!(shell._brushChunks || []).some((c) => chunkHasPaint(c, desc.sourceY1, desc.sourceY2))) continue;
          const mask = document.createElement("canvas"); mask.width = desc.img.naturalWidth; mask.height = desc.img.naturalHeight; drawBrushMask(mask.getContext("2d"), chunks, desc, mask.width);
          const form = new FormData(); form.append("chapter_id", chapterId); form.append("page_index", desc.item.canonicalIndex); form.append("mode", mode); form.append("mask", await canvasBlob(mask), "mask.png");
          const response = await fetch("/api/repaint_mask", { method: "POST", body: form }), parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({}))), data = await parse(response); if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
          if (window.currentManifest?.pages?.[desc.item.canonicalIndex] && data.pages?.[desc.item.canonicalIndex]) window.currentManifest.pages[desc.item.canonicalIndex] = data.pages[desc.item.canonicalIndex]; affected++;
        }
        for (const c of chunks) { if (c.canvas && c.ctx) c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height); c.dirty = false; } snapshots.delete(snapshotKey()); window.showToast?.(`Đã xử lý ${affected} lát trên strip.`, affected ? "success" : "info"); rerender();
      } catch (err) { window.showToast?.("Không thể xử lý vùng đánh dấu: " + err.message, "error"); } finally { busy(false); }
    }, { signal });

    syncVariant(); setTool(shell, tool); rerender();
  }

  function scan() { document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount); }
  window.mountStitchInspector = scan;
})();
