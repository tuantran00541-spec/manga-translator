(() => {
  "use strict";

  const nativeFetch = window.fetch.bind(window);
  const writes = new Map();
  const ops = new Map();
  const reviewDraftPages = new Set();
  let opSeq = 0;
  let chapterSeq = 0;
  let activeChapterSeq = 0;
  let switchBusy = false;
  let queuedSwitch = null;

  const toast = (message, type = "info") => window.showToast?.(message, type);
  const pathOf = (input) => {
    try { return new URL(typeof input === "string" ? input : input?.url, location.href).pathname; }
    catch (_) { return ""; }
  };
  const emitOps = () => {
    const detail = [...ops.values()].map((x) => ({ ...x }));
    try { window.dispatchEvent(new CustomEvent("frontend:operationchange", { detail })); } catch (_) {}
    syncReviewBusy();
  };
  const operationState = {
    begin(name, scope) {
      const token = ++opSeq;
      ops.set(token, { token, name, scope, chapterId: window.currentChapterId || null, startedAt: Date.now() });
      emitOps();
      return token;
    },
    end(token) { if (ops.delete(token)) emitOps(); },
    isBusy(query = null) {
      const list = [...ops.values()];
      if (!query) return list.length > 0;
      if (typeof query === "string") return list.some((x) => x.scope === query || x.name === query);
      return list.some((x) => (!query.scope || x.scope === query.scope) && (!query.chapterId || x.chapterId === query.chapterId));
    },
    snapshot: () => [...ops.values()].map((x) => ({ ...x })),
  };
  window.frontendOperationState = operationState;

  const classify = (path) => {
    if (!path || path === "/api/visual_qc/settings" || /\/key$/.test(path)) return null;
    if (/text_object\/update(_bulk)?$/.test(path)) return ["save-object", "save"];
    if (/visual_qc|quality|repaint|repair|inpaint_region/i.test(path)) return ["quality-check", "review"];
    if (/ocr/i.test(path)) return ["ocr", document.body?.dataset?.appStage || "app"];
    if (/translate/i.test(path)) return ["translate", "editor"];
    if (/render\/chapter|export/i.test(path)) return ["export", "editor"];
    if (/process_pages/i.test(path)) return ["process", "preview"];
    return null;
  };
  const updateMeta = (init) => {
    if (typeof init?.body !== "string") return null;
    try {
      const x = JSON.parse(init.body);
      if (x.chapter_id == null || x.page_index == null || x.id == null) return null;
      return { chapterId: String(x.chapter_id), pageIndex: Number(x.page_index), id: String(x.id) };
    } catch (_) { return null; }
  };

  window.fetch = function coordinatedFetch(input, init) {
    const path = pathOf(input);
    const kind = classify(path);
    const opToken = kind ? operationState.begin(kind[0], kind[1]) : null;
    if (!/text_object\/update$/.test(path)) {
      const request = nativeFetch(input, init);
      if (opToken !== null) request.finally(() => operationState.end(opToken)).catch(() => {});
      return request;
    }

    const meta = updateMeta(init);
    if (!meta) {
      const request = nativeFetch(input, init);
      if (opToken !== null) request.finally(() => operationState.end(opToken)).catch(() => {});
      return request;
    }
    const key = `${meta.chapterId}:${meta.pageIndex}:${meta.id}`;
    const state = writes.get(key) || {
      ...meta, revision: 0, pending: 0, tail: Promise.resolve(), lastSuccess: 0, lastFailure: 0, error: null,
    };
    writes.set(key, state);
    const rev = ++state.revision;
    state.pending += 1;
    const network = state.tail.catch(() => {}).then(() => nativeFetch(input, init));
    const observed = network.then((response) => {
      if (response.ok) {
        state.lastSuccess = Math.max(state.lastSuccess, rev);
        if (state.lastSuccess >= state.lastFailure) state.error = null;
      } else {
        state.lastFailure = Math.max(state.lastFailure, rev);
        state.error = new Error(`Lưu thay đổi thất bại (HTTP ${response.status})`);
      }
      return response;
    }, (error) => {
      state.lastFailure = Math.max(state.lastFailure, rev);
      state.error = error instanceof Error ? error : new Error(String(error));
      throw error;
    });
    state.tail = observed.catch(() => {});
    observed.finally(() => {
      state.pending = Math.max(0, state.pending - 1);
      if (opToken !== null) operationState.end(opToken);
    }).catch(() => {});
    return network;
  };

  const legacyFlush = window.flushAllPendingPersists?.bind(window) || (async () => {});
  const relevantWrites = (pageIndex) => {
    const chapterId = window.currentChapterId == null ? null : String(window.currentChapterId);
    return [...writes.values()].filter((x) => (!chapterId || x.chapterId === chapterId) && (pageIndex === undefined || x.pageIndex === Number(pageIndex)));
  };
  async function waitWrites(pageIndex) {
    const list = relevantWrites(pageIndex);
    await Promise.all(list.map((x) => x.tail));
    const failed = list.find((x) => x.lastFailure > x.lastSuccess);
    if (failed) throw failed.error || new Error("Không lưu được thay đổi gần nhất.");
  }
  async function awaitSaved(pageIndex) {
    for (let pass = 0; pass < 12; pass += 1) {
      await legacyFlush(pageIndex);
      await waitWrites(pageIndex);
      await Promise.resolve();
      const pending = relevantWrites(pageIndex).some((x) => x.pending > 0);
      const geom = window.hasPendingGeom?.() || window.isGeomSaving?.();
      if (!pending && !geom) {
        await legacyFlush(pageIndex);
        await waitWrites(pageIndex);
        if (!relevantWrites(pageIndex).some((x) => x.pending > 0) && !window.hasPendingGeom?.() && !window.isGeomSaving?.()) return true;
      }
    }
    throw new Error("Các thay đổi vẫn đang được lưu.");
  }
  window.awaitSaved = awaitSaved;
  window.flushAllPendingPersists = awaitSaved;

  const snapshotReviewDraft = () => {
    const card = document.querySelector(".review-canvas-host .review-card");
    if (!card) return;
    const index = Number(card.dataset.pageIndex);
    if (!Number.isFinite(index)) return;
    if (card.querySelector("canvas.brush-canvas")?._reviewDirty) reviewDraftPages.add(index);
    else reviewDraftPages.delete(index);
  };
  const hasReviewDraft = () => {
    snapshotReviewDraft();
    if (reviewDraftPages.size) return true;
    return Boolean(window.hasUnsavedStitchedMarks?.());
  };
  window.frontendHasPendingReviewDrafts = hasReviewDraft;

  async function guardExit(targetStage) {
    const stage = document.body?.dataset?.appStage || "landing";
    if (stage === "review" && targetStage !== "review") {
      if (hasReviewDraft()) {
        toast("Còn vùng đánh dấu chưa được xử lý. Hãy xử lý hoặc xóa trước khi rời bước Kiểm tra.", "error");
        return false;
      }
      if (document.querySelector(".review-workspace-shell.review-busy")) {
        toast("Đang xử lý kiểm tra chất lượng. Hãy hoàn tất thao tác hiện tại trước khi rời bước Kiểm tra.", "warning");
        return false;
      }
    }
    if (stage === "editor" && targetStage !== "editor") {
      try { await awaitSaved(); }
      catch (error) {
        toast("Không thể rời trình biên tập vì thay đổi chưa lưu: " + (error?.message || String(error)), "error");
        return false;
      }
    }
    return true;
  }
  window.guardFrontendNavigation = guardExit;

  const legacyNavigate = window.navigateAppStage?.bind(window);
  if (legacyNavigate) window.navigateAppStage = async (stage) => (await guardExit(stage)) ? legacyNavigate(stage) : false;
  document.addEventListener("click", (event) => {
    const step = event.target?.closest?.(".workbench-stage-link[data-stage]");
    const home = event.target?.closest?.("#app-home");
    if (!step && !home) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void window.navigateAppStage?.(step ? step.dataset.stage : "landing");
  }, true);

  const legacySwitch = window.switchEditorPage?.bind(window);
  async function safeSwitch(index, delegate = legacySwitch) {
    if (switchBusy) { queuedSwitch = index; return false; }
    switchBusy = true;
    let target = index;
    try {
      while (target !== null && target !== undefined) {
        queuedSwitch = null;
        try { await awaitSaved(window.editorState?.activePageIndex); }
        catch (error) {
          toast("Chưa thể chuyển trang vì thay đổi chưa lưu: " + (error?.message || String(error)), "error");
          return false;
        }
        if (delegate) await delegate(target);
        polishEditor();
        target = queuedSwitch;
        if (target === window.editorState?.activePageIndex) target = null;
      }
      return true;
    } finally { switchBusy = false; }
  }
  if (legacySwitch) window.switchEditorPage = (index) => safeSwitch(index, legacySwitch);

  const legacyNavigator = window.createPageNavigator?.bind(window);
  if (legacyNavigator) window.createPageNavigator = (options = {}) => {
    const next = { ...options };
    if (next.title === "Trang kiểm tra") {
      next.items = (next.items || []).map((x) => x.stateLabel === "Đã xác minh" ? { ...x, stateLabel: "Chưa duyệt" } : x);
      if (next.onSelect) {
        const select = next.onSelect;
        next.onSelect = (index) => { snapshotReviewDraft(); return select(index); };
      }
    } else if (next.title === "Trang biên tập" && next.onSelect) {
      const select = next.onSelect;
      next.onSelect = (index) => safeSwitch(index, select);
    } else if (next.title === "Trang & lát" && next.onSelect) {
      const select = next.onSelect;
      next.onSelect = (index) => { const result = select(index); queueMicrotask(attachPointerBridges); return result; };
    }
    const nav = legacyNavigator(next);
    queueMicrotask(() => { normalizeReviewLabels(); polishEditor(); attachPointerBridges(); });
    return nav;
  };

  function normalizeReviewLabels() {
    document.querySelectorAll(".page-navigator *").forEach((node) => {
      if (!node.childElementCount && node.textContent?.trim() === "Đã xác minh") node.textContent = "Chưa duyệt";
    });
  }
  function syncReviewBusy() {
    const workspace = document.querySelector(".review-workspace-shell");
    if (!workspace) return;
    const explicit = operationState.isBusy({ scope: "review", chapterId: window.currentChapterId || undefined });
    const internal = Boolean(workspace.querySelector(".review-card")?._reviewBusy);
    const busy = explicit || internal;
    workspace.classList.toggle("review-busy", busy);
    workspace.querySelectorAll(".brush-toggle-btn,.clear-brush-btn,.repaint-btn,.reset-manual-btn,.ai-qc-btn,.review-primary-action").forEach((x) => { x.disabled = busy; });
  }
  function chapterName() {
    const m = window.currentManifest || {};
    const direct = m.title || m.chapter_name || m.name || m.source_title;
    if (typeof direct === "string" && direct.trim()) return direct.trim();
    if (typeof m.source_url === "string") {
      try {
        const tail = new URL(m.source_url).pathname.split("/").filter(Boolean).pop();
        if (tail) return decodeURIComponent(tail).replace(/[-_]+/g, " ");
      } catch (_) {}
    }
    return window.currentChapterId ? String(window.currentChapterId) : "Chưa mở chương";
  }
  function polishEditor() {
    const preview = document.querySelector(".editor-render-btn");
    if (preview) {
      preview.classList.remove("ui-btn-primary"); preview.classList.add("ui-btn-ghost");
      if (!/đang/i.test(preview.textContent || "")) preview.textContent = "Xem trước trang";
      preview.title = "Kết xuất và xem trước trang đang mở";
    }
    document.querySelector(".chapter-export-run")?.classList.add("ui-btn-primary");
    const context = document.getElementById("app-context");
    if (context && window.currentChapterId) { context.textContent = chapterName(); context.title = `Chương ${window.currentChapterId}`; }
  }

  const restoreLanguage = (manifest) => {
    const value = manifest?.source_lang || manifest?.language || manifest?.config?.source_lang || manifest?.processing?.source_lang;
    const select = document.getElementById("lang-select");
    if (value && select && [...select.options].some((x) => x.value === String(value))) select.value = String(value);
  };
  const persistChapter = () => {
    try {
      sessionStorage.setItem("mt_active_chapter", window.currentChapterId);
      if (location.hash !== `#${window.currentChapterId}`) history.replaceState(null, "", `#${window.currentChapterId}`);
    } catch (_) {}
  };
  const renderManifest = (manifest, forcedStage = null) => {
    const pages = manifest?.pages || [];
    let wf = forcedStage ? { stage: forcedStage, page_index: 0 } : manifest?.workflow;
    if (!wf?.stage) wf = pages.some((p) => p.rendered) ? { stage: "editor", page_index: 0 } : pages.some((p) => p.clean) ? { stage: "review", page_index: 0 } : { stage: "preview", page_index: 0 };
    const index = Math.max(0, Math.min(parseInt(wf.page_index, 10) || 0, Math.max(0, pages.length - 1)));
    if (wf.stage === "preview") { window.initialPreviewCanonicalPageIndex = index; window.renderPreview?.(); }
    else if (wf.stage === "review") { window.initialReviewCanonicalPageIndex = index; window.renderReview?.(); }
    else { if (window.editorState) window.editorState.activePageIndex = index; window.renderEditor?.(); }
    polishEditor();
  };
  const setOpenBusy = (busy, upload = false) => {
    document.querySelectorAll(".recent-resume-btn").forEach((x) => { x.disabled = busy; });
    const load = document.getElementById("load-btn"); if (load) load.disabled = busy;
    const drop = document.getElementById("upload-dropzone");
    if (drop) { drop.classList.toggle("uploading", busy && upload); drop.setAttribute("aria-busy", String(busy && upload)); }
  };
  async function beginOpen(kind) {
    if (!(await guardExit("landing"))) return null;
    const token = ++chapterSeq;
    activeChapterSeq = token;
    reviewDraftPages.clear();
    setOpenBusy(true, kind === "upload");
    return token;
  }
  const currentOpen = (token) => token === activeChapterSeq && token === chapterSeq;
  const finishOpen = (token) => { if (currentOpen(token)) setOpenBusy(false); };
  async function getManifest(token, input, init) {
    const response = await window.fetch(input, init);
    const data = await (window.parseApiResponse ? window.parseApiResponse(response) : response.json());
    if (!response.ok) throw new Error(window.getErrorMessage ? window.getErrorMessage(response.status, data) : data?.detail || `HTTP ${response.status}`);
    return currentOpen(token) ? data : null;
  }
  function activate(data, forcedStage = null) {
    window.currentManifest = data; window.currentChapterId = data.chapter_id;
    restoreLanguage(data); persistChapter(); renderManifest(data, forcedStage);
  }
  window.resumeChapter = async (chapterId) => {
    if (!chapterId) return;
    const token = await beginOpen("resume"); if (!token) return;
    try { const data = await getManifest(token, `/api/chapter/${encodeURIComponent(chapterId)}`); if (data) activate(data); }
    catch (error) { if (currentOpen(token)) toast("Không thể tiếp tục chương: " + error.message, "error"); }
    finally { finishOpen(token); }
  };
  window.loadChapter = async () => {
    const url = document.getElementById("chapter-url")?.value?.trim(); if (!url) return;
    const token = await beginOpen("url"); if (!token) return;
    const button = document.getElementById("load-btn"); const label = button?.textContent || "Tải chương";
    if (button) button.textContent = "Đang tải chương…";
    try {
      const data = await getManifest(token, "/api/chapter", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url, workers: window.getWorkersSetting?.() || 2 }) });
      if (data) activate(data, "preview");
    } catch (error) { if (currentOpen(token)) toast("Không tải được chương: " + error.message, "error"); }
    finally { if (currentOpen(token) && button) button.textContent = label; finishOpen(token); }
  };
  window.uploadChapter = async (files) => {
    if (!Array.isArray(files) || !files.length) return;
    const token = await beginOpen("upload"); if (!token) return;
    const hint = document.getElementById("upload-hint"); const label = hint?.textContent || "";
    if (hint) hint.textContent = `Đang tải ${files.length} tệp…`;
    try {
      const form = new FormData(); files.forEach((f) => form.append("files", f)); form.append("workers", String(window.getWorkersSetting?.() || 2));
      const data = await getManifest(token, "/api/chapter/upload", { method: "POST", body: form });
      if (data) activate(data, "preview");
    } catch (error) { if (currentOpen(token)) toast("Không thể tải tệp lên: " + error.message, "error"); }
    finally { if (currentOpen(token) && hint) hint.textContent = label; finishOpen(token); }
  };
  window.frontendChapterActivation = { get generation() { return chapterSeq; } };

  function attachEditorPointer() {
    const wrap = document.querySelector(".translation-canvas-host .page-image-wrap");
    const img = wrap?.querySelector("img");
    if (!wrap || !img || wrap.dataset.pointerBridge) return;
    wrap.dataset.pointerBridge = "1";
    let drag = null; let temp = null;
    const syncTouch = () => { wrap.style.touchAction = wrap.classList.contains("draw-mode") ? "none" : "auto"; };
    syncTouch(); new MutationObserver(syncTouch).observe(wrap, { attributes: true, attributeFilter: ["class"] });
    const point = (e) => {
      const r = img.getBoundingClientRect(); if (!img.naturalWidth || !r.width || !r.height) return null;
      return { x: Math.max(0, Math.min(img.naturalWidth, (e.clientX - r.left) * img.naturalWidth / r.width)), y: Math.max(0, Math.min(img.naturalHeight, (e.clientY - r.top) * img.naturalHeight / r.height)) };
    };
    const draw = (end) => {
      if (!drag || !temp || !end) return;
      const sx = img.clientWidth / img.naturalWidth, sy = img.clientHeight / img.naturalHeight;
      temp.style.left = `${img.offsetLeft + Math.min(drag.start.x, end.x) * sx}px`; temp.style.top = `${img.offsetTop + Math.min(drag.start.y, end.y) * sy}px`;
      temp.style.width = `${Math.abs(end.x - drag.start.x) * sx}px`; temp.style.height = `${Math.abs(end.y - drag.start.y) * sy}px`;
    };
    const reset = () => { drag = null; temp?.remove(); temp = null; };
    wrap.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "mouse" || window.editorState?.tool === "select" || e.target.closest(".text-object-overlay")) return;
      const start = point(e); if (!start) return; e.preventDefault(); wrap.setPointerCapture?.(e.pointerId);
      drag = { id: e.pointerId, start, end: start, shape: window.editorState.tool }; temp = document.createElement("div");
      temp.className = "text-object-overlay drawing" + (drag.shape === "ellipse" ? " ellipse" : ""); wrap.appendChild(temp); draw(start);
    });
    wrap.addEventListener("pointermove", (e) => { if (!drag || e.pointerId !== drag.id) return; e.preventDefault(); drag.end = point(e) || drag.end; draw(drag.end); });
    wrap.addEventListener("pointercancel", (e) => { if (drag?.id === e.pointerId) reset(); });
    wrap.addEventListener("pointerup", (e) => {
      if (!drag || e.pointerId !== drag.id) return; e.preventDefault(); const x = drag; x.end = point(e) || x.end; reset();
      const region = { x1: Math.round(Math.min(x.start.x, x.end.x)), y1: Math.round(Math.min(x.start.y, x.end.y)), x2: Math.round(Math.max(x.start.x, x.end.x)), y2: Math.round(Math.max(x.start.y, x.end.y)) };
      if (region.x2 - region.x1 < 10 || region.y2 - region.y1 < 10) return;
      if (typeof window.createTextObject === "function") Promise.resolve(window.createTextObject(Number(window.editorState?.activePageIndex || 0), x.shape, region)).catch((err) => toast("Không thể tạo vùng chữ: " + err.message, "error"));
    });
  }
  function attachPreviewPointer() {
    const card = document.querySelector(".preview-card-active"), layer = card?.querySelector(".excluded-draw-layer"), wrap = card?.querySelector(".preview-image-wrap"), img = wrap?.querySelector("img"), overlay = card?.querySelector(".excluded-overlay-container");
    if (!card || !layer || !wrap || !img || !overlay || layer.dataset.pointerBridge) return;
    layer.dataset.pointerBridge = "1"; let drag = null; let temp = null;
    const syncTouch = () => { layer.style.touchAction = card.classList.contains("draw-excluded-active") ? "none" : "auto"; };
    syncTouch(); new MutationObserver(syncTouch).observe(card, { attributes: true, attributeFilter: ["class"] });
    const point = (e) => { const r = wrap.getBoundingClientRect(); if (!r.width || !r.height) return null; return { x: (e.clientX - r.left) * wrap.clientWidth / r.width, y: (e.clientY - r.top) * wrap.clientHeight / r.height }; };
    const draw = (end) => { if (!drag || !temp) return; temp.style.left = `${Math.min(drag.start.x, end.x)}px`; temp.style.top = `${Math.min(drag.start.y, end.y)}px`; temp.style.width = `${Math.abs(end.x - drag.start.x)}px`; temp.style.height = `${Math.abs(end.y - drag.start.y)}px`; };
    const reset = () => { drag = null; temp?.remove(); temp = null; };
    layer.addEventListener("pointerdown", (e) => { if (e.pointerType === "mouse" || !card.classList.contains("draw-excluded-active")) return; const start = point(e); if (!start) return; e.preventDefault(); layer.setPointerCapture?.(e.pointerId); drag = { id: e.pointerId, start, end: start }; temp = document.createElement("div"); temp.className = "excluded-region-box drawing"; overlay.appendChild(temp); draw(start); });
    layer.addEventListener("pointermove", (e) => { if (!drag || e.pointerId !== drag.id) return; e.preventDefault(); drag.end = point(e) || drag.end; draw(drag.end); });
    layer.addEventListener("pointercancel", (e) => { if (drag?.id === e.pointerId) reset(); });
    layer.addEventListener("pointerup", (e) => {
      if (!drag || e.pointerId !== drag.id) return; e.preventDefault(); const x = drag; x.end = point(e) || x.end; reset();
      if (Math.abs(x.end.x - x.start.x) < 5 || Math.abs(x.end.y - x.start.y) < 5 || !img.clientWidth) return;
      const index = Number(card.dataset.pageIndex || 0), page = window.currentManifest?.pages?.[index]; if (!page) return;
      page.excluded_regions ||= []; const sx = img.naturalWidth / img.clientWidth, sy = img.naturalHeight / img.clientHeight;
      page.excluded_regions.push({ x1: Math.round(Math.min(x.start.x, x.end.x) * sx), y1: Math.round(Math.min(x.start.y, x.end.y) * sy), x2: Math.round(Math.max(x.start.x, x.end.x) * sx), y2: Math.round(Math.max(x.start.y, x.end.y) * sy) });
      Promise.resolve(window.saveExcludedRegions?.(index, page.excluded_regions)).catch((err) => toast("Không lưu được vùng loại trừ: " + err.message, "error")); window.renderPreview?.();
    });
  }
  const attachPointerBridges = () => { attachEditorPointer(); attachPreviewPointer(); };

  const wrapRenderer = (name, after) => {
    const original = window[name]?.bind(window); if (!original) return;
    window[name] = (...args) => { const result = original(...args); after(); return result; };
  };
  wrapRenderer("renderPreview", () => { attachPreviewPointer(); polishEditor(); });
  wrapRenderer("renderReview", () => { normalizeReviewLabels(); syncReviewBusy(); polishEditor(); });
  wrapRenderer("renderEditor", () => { polishEditor(); attachEditorPointer(); });

  function mountAISettings() {
    const host = document.getElementById("ai-settings-host");
    if (!host || host.querySelector(".gemini-qc-config")) return;
    if (typeof window.createAIProviderSettings === "function") {
      window.createAIProviderSettings();
      return;
    }
    const root = document.createElement("div"); root.className = "gemini-qc-config frontend-ai-settings";
    const status = document.createElement("span"); status.className = "gemini-qc-status"; status.textContent = "Kiểm tra AI: Đang kiểm tra cấu hình…";
    const input = document.createElement("input"); input.type = "password"; input.className = "gemini-key-input"; input.placeholder = "Gemini API key"; input.autocomplete = "off";
    const save = document.createElement("button"); save.type = "button"; save.className = "ui-btn ui-btn-primary gemini-key-save-btn"; save.textContent = "Lưu khóa API";
    const clear = document.createElement("button"); clear.type = "button"; clear.className = "ui-btn ui-btn-ghost gemini-key-clear-btn"; clear.textContent = "Xóa khóa API";
    root.append(status, input, save, clear);
    window.setupGeminiQCSettings?.(status, input, save, clear);
    const deep = document.createElement("p"); deep.className = "gemini-qc-privacy-note"; deep.textContent = "DeepSeek Vision: cấu hình và trạng thái được tải khi mở phần Kiểm tra; ảnh chỉ được gửi khi bạn chủ động chạy kiểm tra.";
    root.appendChild(deep); host.replaceChildren(root);
    window.fetch("/api/visual_qc/settings").then(async (r) => {
      const data = await (window.parseApiResponse ? window.parseApiResponse(r) : r.json());
      const ds = data?.providers?.deepseek; if (ds) deep.textContent = `DeepSeek Vision: ${ds.configured ? "Sẵn sàng" : "Chưa cấu hình"}${ds.model ? ` · ${ds.model}` : ""}.`;
    }).catch(() => {});
  }
  window.ensureAISettingsMounted = mountAISettings;

  const add = document.addEventListener.bind(document), remove = document.removeEventListener.bind(document), keyMap = new WeakMap();
  document.addEventListener = function (type, listener, options) {
    const capture = options === true || Boolean(options?.capture);
    if (type === "keydown" && capture && typeof listener === "function") {
      const wrapped = function (event) { if (event.key === "Tab" && !document.body.classList.contains("settings-open")) return; return listener.call(this, event); };
      keyMap.set(listener, wrapped); return add(type, wrapped, options);
    }
    return add(type, listener, options);
  };
  document.removeEventListener = (type, listener, options) => remove(type, keyMap.get(listener) || listener, options);
  add("DOMContentLoaded", () => queueMicrotask(() => {
    document.addEventListener = add; document.removeEventListener = remove;
    const focus = document.getElementById("toggle-focus-mode"); if (focus) focus.title = "Chế độ tập trung";
    mountAISettings(); polishEditor();
    const view = document.getElementById("page-view");
    if (view) {
      const repair = () => { attachPointerBridges(); normalizeReviewLabels(); polishEditor(); syncReviewBusy(); };
      let repairQueued = false;
      const scheduleRepair = () => {
        if (repairQueued) return;
        repairQueued = true;
        const run = () => { repairQueued = false; repair(); };
        if (typeof window.requestAnimationFrame === "function") window.requestAnimationFrame(run);
        else setTimeout(run, 0);
      };
      // Observe DOM rebuilds only.  repair() changes classes itself, therefore
      // observing attributes here creates an observer -> repair -> observer loop.
      new MutationObserver(scheduleRepair).observe(view, { childList: true, subtree: true });
      scheduleRepair();
    }
  }));
  add("click", (event) => { if (event.target?.closest?.("#settings-toggle")) mountAISettings(); }, true);

  window.frontendCoordinatorDebug = { writes, reviewDraftPages, waitWrites, classify, pathOf };
})();
