async function parseApiResponse(resp) {
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    try {
      data = { detail: await resp.text() };
    } catch (_) {
      data = {};
    }
  }
  return data || {};
}
window.parseApiResponse = parseApiResponse;

function getErrorMessage(status, data) {
  if (data && data.detail) {
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((e) => (typeof e === "string" ? e : e.msg || JSON.stringify(e)))
        .join("; ");
    }
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail.trim();
    }
  }
  const statusMessages = {
    400: "Yêu cầu không hợp lệ (400)",
    404: "Không tìm thấy dữ liệu (404)",
    422: "Dữ liệu gửi lên không đúng định dạng (422)",
    500: "Lỗi nội bộ máy chủ (500)",
    502: "Máy chủ cổng không phản hồi (502)",
    504: "Hết thời gian chờ phản hồi từ máy chủ (504)",
  };
  return statusMessages[status] || `Máy chủ trả về lỗi HTTP ${status}`;
}
window.getErrorMessage = getErrorMessage;

function parsePageNumber(value, totalPages) {
  if (typeof value !== "string" && typeof value !== "number") return null;
  const str = String(value).trim();
  if (!/^[1-9]\d*$/.test(str)) return null;
  const num = Number(str);
  if (!Number.isSafeInteger(num) || num < 1 || num > totalPages) return null;
  return num - 1;
}
window.parsePageNumber = parsePageNumber;

let _lastCheckpointStage = null;
let _lastCheckpointPage = null;
let _lastCheckpointChapter = null;
let _checkpointSeq = 0;
let _chapterNavigationSeq = 0;

async function setWorkflowCheckpoint(stage, pageIndex) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const canonicalIndex = Math.max(0, parseInt(pageIndex, 10) || 0);
  if (
    _lastCheckpointChapter === chapterId
    && _lastCheckpointStage === stage
    && _lastCheckpointPage === canonicalIndex
  ) {
    return;
  }
  _lastCheckpointChapter = chapterId;
  _lastCheckpointStage = stage;
  _lastCheckpointPage = canonicalIndex;
  const seq = ++_checkpointSeq;

  try {
    const resp = await fetch("/api/workflow_checkpoint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: chapterId,
        stage,
        page_index: canonicalIndex,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (
      seq === _checkpointSeq
      && chapterId === currentChapterId
      && currentManifest
    ) {
      currentManifest.workflow = { stage, page_index: canonicalIndex };
    }
  } catch (err) {
    if (
      seq === _checkpointSeq
      && _lastCheckpointChapter === chapterId
      && _lastCheckpointStage === stage
      && _lastCheckpointPage === canonicalIndex
    ) {
      _lastCheckpointChapter = null;
      _lastCheckpointStage = null;
      _lastCheckpointPage = null;
    }
    console.error("Workflow checkpoint save failed:", err);
  }
}
window.setWorkflowCheckpoint = setWorkflowCheckpoint;

function appendText(parent, tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

async function loadRecentChapters() {
  const container = document.getElementById("recent-chapters");
  if (!container) return;
  const panel = container.closest(".recent-panel");
  const setPanelVisible = (visible) => {
    if (panel) panel.hidden = !visible;
  };

  try {
    const resp = await fetch("/api/chapters");
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      showToast(
        "Không tải được danh sách chương: " + getErrorMessage(resp.status, data),
        "error",
      );
      container.replaceChildren();
      setPanelVisible(false);
      return;
    }

    const chapters = Array.isArray(data) ? data : [];
    if (chapters.length === 0) {
      container.replaceChildren();
      setPanelVisible(false);
      return;
    }

    setPanelVisible(true);
    container.replaceChildren();
    appendText(container, "div", "recent-title", "Chương đang xử lý");
    const list = document.createElement("div");
    list.className = "recent-list";
    const stageLabels = {
      preview: "Xử lý ảnh",
      review: "Kiểm tra chất lượng",
      editor: "Biên tập bản dịch",
    };

    chapters.forEach((ch) => {
      const card = document.createElement("div");
      card.className = "recent-card";
      const info = document.createElement("div");
      info.className = "recent-info";

      appendText(info, "strong", "", String(ch?.chapter_id || "(không rõ chương)"));
      info.appendChild(document.createElement("br"));
      appendText(
        info,
        "span",
        "recent-url",
        String(ch?.source_url || "(không có liên kết nguồn)"),
      );
      info.appendChild(document.createElement("br"));

      const meta = document.createElement("span");
      meta.className = "recent-meta";
      meta.appendChild(
        document.createTextNode(`${Number(ch?.total_pages) || 0} trang · `),
      );
      const rawStage = String(ch?.workflow?.stage || "");
      appendText(
        meta,
        "span",
        "recent-stage-badge",
        stageLabels[rawStage] || rawStage || "Đang xử lý",
      );
      info.appendChild(meta);
      card.appendChild(info);

      const btn = document.createElement("button");
      btn.className = "recent-resume-btn";
      btn.type = "button";
      btn.textContent = "Tiếp tục xử lý";
      btn.addEventListener("click", () => resumeChapter(String(ch?.chapter_id || "")));
      card.appendChild(btn);
      list.appendChild(card);
    });

    container.appendChild(list);
  } catch (err) {
    showToast("Không tải được danh sách chương: " + err.message, "error");
    container.replaceChildren();
    setPanelVisible(false);
  }
}

async function refreshChapterManifest(chapterId) {
  if (!chapterId) return null;
  const resp = await fetch(`/api/chapter/${encodeURIComponent(chapterId)}`);
  const data = await parseApiResponse(resp);
  if (!resp.ok) {
    throw new Error(getErrorMessage(resp.status, data));
  }
  if (chapterId !== currentChapterId) return null;
  currentManifest = data;
  return data;
}
window.refreshChapterManifest = refreshChapterManifest;

async function resumeChapter(chapterId) {
  if (!chapterId) return;
  const navigationSeq = ++_chapterNavigationSeq;
  try {
    const resp = await fetch(`/api/chapter/${encodeURIComponent(chapterId)}`);
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (navigationSeq !== _chapterNavigationSeq) return;
    currentManifest = data;
    currentChapterId = currentManifest.chapter_id;
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      if (window.location.hash !== `#${currentChapterId}`) {
        window.history.replaceState(null, "", `#${currentChapterId}`);
      }
    } catch (_) {}

    const recentEl = document.getElementById("recent-chapters");
    if (recentEl) recentEl.innerHTML = "";

    const pages = currentManifest.pages || [];
    let workflow = currentManifest.workflow;
    if (!workflow || !workflow.stage) {
      if (pages.some((p) => p.rendered)) {
        workflow = { stage: "editor", page_index: 0 };
      } else if (pages.some((p) => p.clean)) {
        workflow = { stage: "review", page_index: 0 };
      } else {
        workflow = { stage: "preview", page_index: 0 };
      }
    }

    const stage = workflow.stage;
    const rawIndex = parseInt(workflow.page_index, 10) || 0;
    const pageIndex = Math.max(0, Math.min(rawIndex, Math.max(0, pages.length - 1)));

    if (stage === "preview") {
      window.initialPreviewCanonicalPageIndex = pageIndex;
      renderPreview();
    } else if (stage === "review") {
      window.initialReviewCanonicalPageIndex = pageIndex;
      renderReview();
    } else {
      if (window.editorState) {
        window.editorState.activePageIndex = pageIndex;
      }
      renderEditor();
    }
  } catch (err) {
    if (navigationSeq === _chapterNavigationSeq) {
      showToast("Không thể tiếp tục chương: " + err.message, "error");
    }
  }
}

async function loadChapter() {
  const urlEl = document.getElementById("chapter-url");
  if (!urlEl) return;
  const url = urlEl.value.trim();
  if (!url) return;
  const navigationSeq = ++_chapterNavigationSeq;

  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) {
    loadBtn.disabled = true;
    loadBtn.textContent = "Đang tải chương…";
  }

  try {
    const resp = await fetch("/api/chapter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, workers: getWorkersSetting() }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (navigationSeq !== _chapterNavigationSeq) return;
    currentManifest = data;
    currentChapterId = currentManifest.chapter_id;
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      window.history.replaceState(null, "", `#${currentChapterId}`);
    } catch (_) {}
    window.previewActivePageIndex = 0;
    renderPreview();
  } catch (err) {
    if (navigationSeq === _chapterNavigationSeq) {
      showToast("Không tải được chương: " + err.message, "error");
    }
  } finally {
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.textContent = "Tải chương";
    }
  }
}

async function toggleSkip(pageIndex, card, btn) {
  const chapterId = currentChapterId;
  const page = currentManifest.pages[pageIndex];
  const newSkipped = !page.skipped;

  try {
    if (typeof window.flushExcludedRegionSaves === "function") {
      await window.flushExcludedRegionSaves(chapterId, pageIndex);
    }
    if (chapterId !== currentChapterId) return;
    const resp = await fetch("/api/skip_pages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: chapterId,
        page_indices: [pageIndex],
        skipped: newSkipped,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (chapterId !== currentChapterId) return;
    currentManifest = data;
    window.previewActivePageIndex = pageIndex;
    renderPreview();
  } catch (err) {
    showToast("Không đổi được trạng thái bỏ qua: " + err.message, "error");
  }
}

function getWorkersSetting() {
  const el = document.getElementById("workers-select");
  const n = el ? parseInt(el.value, 10) : 2;
  if (!Number.isFinite(n) || n < 1) return 2;
  return Math.min(8, n);
}

const PROCESS_BATCH_SIZE = 16;

async function processSelectedPages() {
  const pages = currentManifest?.pages || [];
  const indices = pages
    .map((page, index) => (page?.skipped ? null : index))
    .filter((index) => index !== null);

  if (indices.length === 0) {
    showToast("Không có trang nào để xử lý.", "error");
    return;
  }

  const chapterId = currentChapterId;
  const total = indices.length;
  const btn = document.querySelector("#preview-toolbar .preview-primary-action")
    || document.querySelector("#preview-toolbar button");
  if (btn) btn.disabled = true;

  try {
    if (btn) btn.textContent = "Đang lưu vùng loại trừ…";
    if (typeof window.flushExcludedRegionSaves === "function") {
      await window.flushExcludedRegionSaves(chapterId);
    }
  } catch (err) {
    showToast(
      "Không thể bắt đầu xử lý vì vùng loại trừ chưa được lưu: " + err.message,
      "error",
    );
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Bắt đầu xử lý";
    }
    return;
  }
  if (chapterId !== currentChapterId) return;

  let completed = 0;
  try {
    for (let start = 0; start < total; start += PROCESS_BATCH_SIZE) {
      const batch = indices.slice(start, start + PROCESS_BATCH_SIZE);
      if (btn) btn.textContent = `Đang xử lý ${completed}/${total}…`;

      const resp = await fetch("/api/process_pages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          page_indices: batch,
          workers: getWorkersSetting(),
        }),
      });
      const data = await parseApiResponse(resp);
      if (!resp.ok) {
        throw new Error(getErrorMessage(resp.status, data));
      }
      if (chapterId !== currentChapterId) return;

      currentManifest = data;
      completed += batch.length;
      if (btn) btn.textContent = `Đã xử lý ${completed}/${total}…`;
    }

    renderReview();
  } catch (err) {
    let resynced = false;
    if (chapterId === currentChapterId) {
      try {
        await refreshChapterManifest(chapterId);
        resynced = true;
      } catch (syncErr) {
        console.error("Could not resync chapter after partial process failure:", syncErr);
      }
    }

    const prefix = completed > 0
      ? `Đã xử lý ít nhất ${completed}/${total} trang. Phần tiếp theo thất bại: `
      : "Xử lý trang thất bại: ";
    showToast(prefix + err.message, "error");

    if (resynced) {
      window.previewActivePageIndex = Math.min(
        Number(window.previewActivePageIndex) || 0,
        Math.max(0, (currentManifest?.pages?.length || 1) - 1),
      );
      renderPreview();
      return;
    }
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Tiếp tục xử lý";
    }
  }
}

async function fetchOcr(pageIndex, boxIndex, originalEl) {
  const page = currentManifest.pages[pageIndex];
  const langEl = document.getElementById("lang-select");
  const lang = langEl ? langEl.value : "ja";
  const box = page && page.boxes ? page.boxes[boxIndex] : null;

  if (box && box.ocr_text && box.ocr_lang === lang) {
    originalEl.textContent = box.ocr_text;
    return;
  }
  try {
    const resp = await fetch("/api/ocr_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: currentChapterId,
        page_index: pageIndex,
        box_index: boxIndex,
        lang,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (box) {
      box.ocr_text = data.text || "";
      box.ocr_lang = lang;
    }
    originalEl.textContent = data.text || "(không nhận dạng được)";
  } catch (err) {
    originalEl.textContent = "(Lỗi OCR: " + err.message + ")";
    showToast("Không thể hoàn tất OCR: " + err.message, "error");
  }
}

function scheduleSaveDraft() {
  if (_saveDraftTimer) clearTimeout(_saveDraftTimer);
  _saveDraftTimer = setTimeout(saveDraftNow, 800);
}

async function saveDraftNow() {
  if (!currentChapterId) return;
  const textareas = document.querySelectorAll("textarea[data-page-index]");
  const drafts = currentManifest ? (currentManifest.drafts || (currentManifest.drafts = {})) : {};
  textareas.forEach((ta) => {
    const key = `${ta.dataset.pageIndex}_${ta.dataset.boxIndex}`;
    drafts[key] = {
      text: ta.value,
      color: ta.dataset.color || "auto",
      font: ta.dataset.font || "default",
      fontSize: ta.dataset.fontSize || "auto",
      bold: ta.dataset.bold === "true",
      strokeWidth: ta.dataset.strokeWidth || "auto",
      strokeColor: ta.dataset.strokeColor || "auto",
      bgColor: ta.dataset.bgColor || "transparent",
      cornerRadius: ta.dataset.cornerRadius || "0",
    };
  });
  try {
    const resp = await fetch("/api/save_draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, drafts }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      console.error("Save draft failed:", getErrorMessage(resp.status, data));
    }
  } catch (e) {
    console.error("Save draft network error:", e);
  }
}
window.saveDraftNow = saveDraftNow;
window.scheduleSaveDraft = scheduleSaveDraft;

async function renderTranslations(pageIndex) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  try {
    if (typeof window.flushAllPendingPersists === "function") {
      await window.flushAllPendingPersists(pageIndex);
    } else if (typeof window.flushTextObjectPersist === "function") {
      await window.flushTextObjectPersist(pageIndex);
    }
  } catch (err) {
    showToast("Không thể chèn chữ vì lưu dữ liệu không thành công.", "error");
    return;
  }
  if (chapterId !== currentChapterId) return;
  const page = currentManifest && currentManifest.pages ? currentManifest.pages[pageIndex] : null;
  if (!page) return;

  const translations = {};
  const colors = {};
  const fonts = {};
  const font_sizes = {};
  const bolds = {};
  const stroke_widths = {};
  const stroke_colors = {};
  const bg_colors = {};
  const corner_radii = {};
  const horizontal_aligns = {};
  const vertical_aligns = {};
  (page.text_objects || []).forEach((obj) => {
    if (!obj || !obj.translation || !obj.translation.trim()) return;
    translations[obj.id] = obj.translation.trim();
    const s = obj.style || {};
    colors[obj.id] = s.color || "auto";
    fonts[obj.id] = s.font || "default";
    font_sizes[obj.id] = s.fontSize || "auto";
    bolds[obj.id] = s.bold === true;
    stroke_widths[obj.id] = s.strokeWidth || "auto";
    stroke_colors[obj.id] = s.strokeColor || "auto";
    bg_colors[obj.id] = s.bgColor || "transparent";
    corner_radii[obj.id] = parseInt(s.cornerRadius || "0", 10);
    horizontal_aligns[obj.id] = ["left", "center", "right"].includes(s.horizontalAlign)
      ? s.horizontalAlign
      : "center";
    vertical_aligns[obj.id] = ["top", "middle", "bottom"].includes(s.verticalAlign)
      ? s.verticalAlign
      : "middle";
  });

  const btn = document.querySelector(".editor-render-btn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Đang kết xuất…";
  }

  try {
    const resp = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: chapterId,
        page_index: pageIndex,
        translations,
        colors,
        fonts,
        font_sizes,
        bolds,
        stroke_widths,
        stroke_colors,
        bg_colors,
        corner_radii,
        horizontal_aligns,
        vertical_aligns,
      }),
    });

    const data = await parseApiResponse(resp);
    if (chapterId !== currentChapterId) return;
    if (!resp.ok) {
      showToast("Kết xuất ảnh thất bại: " + getErrorMessage(resp.status, data), "error");
      return;
    }

    if (currentManifest && currentManifest.pages && currentManifest.pages[pageIndex]) {
      currentManifest.pages[pageIndex].rendered = true;
    }
    showRenderResult(pageIndex, data.output);
    if (data.warning) {
      showToast(data.warning, "info");
    }
  } catch (err) {
    if (chapterId === currentChapterId) {
      showToast("Kết xuất ảnh thất bại: " + err.message, "error");
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Kết xuất bản dịch";
    }
  }
}

async function loadFonts() {
  try {
    const resp = await fetch("/api/fonts");
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    availableFonts = Array.isArray(data) ? data : [{ id: "default", name: "Mặc định (Comic)" }];
  } catch (e) {
    availableFonts = [{ id: "default", name: "Mặc định (Comic)" }];
    showToast("Không thể tải danh sách phông chữ, dùng phông mặc định.", "info");
  }
}

const _excludedRegionSaveStates = new Map();

function _cloneExcludedRegions(excludedRegions) {
  if (!Array.isArray(excludedRegions)) return [];
  return excludedRegions.map((region) => ({
    x1: Number(region.x1),
    y1: Number(region.y1),
    x2: Number(region.x2),
    y2: Number(region.y2),
  }));
}

function _excludedRegionSaveKey(chapterId, pageIndex) {
  return `${chapterId}:${pageIndex}`;
}

async function _drainExcludedRegionSaves(state) {
  while (state.persistedVersion < state.version) {
    const requestVersion = state.version;
    const regions = _cloneExcludedRegions(state.regions);
    const resp = await fetch("/api/save_excluded_regions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: state.chapterId,
        page_index: state.pageIndex,
        excluded_regions: regions,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    state.persistedVersion = requestVersion;

    if (
      requestVersion === state.version
      && state.chapterId === currentChapterId
      && currentManifest?.pages?.[state.pageIndex]
    ) {
      const serverRegions = data?.pages?.[state.pageIndex]?.excluded_regions;
      currentManifest.pages[state.pageIndex].excluded_regions =
        _cloneExcludedRegions(Array.isArray(serverRegions) ? serverRegions : regions);
    }
  }
}

function _ensureExcludedRegionSave(state) {
  if (state.promise) return state.promise;
  const key = _excludedRegionSaveKey(state.chapterId, state.pageIndex);
  state.promise = _drainExcludedRegionSaves(state).finally(() => {
    state.promise = null;
    if (state.persistedVersion >= state.version) {
      _excludedRegionSaveStates.delete(key);
    }
  });
  return state.promise;
}

function saveExcludedRegions(pageIndex, excludedRegions) {
  const chapterId = currentChapterId;
  const key = _excludedRegionSaveKey(chapterId, pageIndex);
  let state = _excludedRegionSaveStates.get(key);
  if (!state) {
    state = {
      chapterId,
      pageIndex,
      regions: [],
      version: 0,
      persistedVersion: 0,
      notifiedVersion: 0,
      promise: null,
    };
    _excludedRegionSaveStates.set(key, state);
  }
  state.regions = _cloneExcludedRegions(excludedRegions);
  state.version += 1;
  const requestedVersion = state.version;
  return _ensureExcludedRegionSave(state).catch((err) => {
    if (state.notifiedVersion < requestedVersion) {
      state.notifiedVersion = state.version;
      showToast("Không lưu được vùng loại trừ: " + err.message, "error");
    }
  });
}

async function flushExcludedRegionSaves(chapterId = currentChapterId, pageIndex) {
  const jobs = [];
  for (const state of _excludedRegionSaveStates.values()) {
    if (state.chapterId !== chapterId) continue;
    if (pageIndex !== undefined && state.pageIndex !== pageIndex) continue;
    if (state.persistedVersion < state.version || state.promise) {
      jobs.push(_ensureExcludedRegionSave(state));
    }
  }
  await Promise.all(jobs);
}
window.flushExcludedRegionSaves = flushExcludedRegionSaves;

async function resetManualMask(pageIndex, img, canvas, ctx, resetBtn) {
  const chapterId = currentChapterId;
  const card = resetBtn?.closest(".review-card") || null;
  if (card) {
    card._reviewBusy = true;
    if (typeof card._syncReviewBusy === "function") card._syncReviewBusy();
  }
  if (canvas && typeof canvas._stopBrush === "function") canvas._stopBrush();
  if (resetBtn) {
    resetBtn.disabled = true;
    resetBtn.textContent = "Đang xóa…";
  }
  try {
    const resp = await fetch("/api/reset_manual_mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: chapterId, page_index: pageIndex }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (chapterId !== currentChapterId || !currentManifest?.pages?.[pageIndex]) {
      return;
    }
    currentManifest.pages[pageIndex] = data.pages[pageIndex];
    if (img) img.src = data.pages[pageIndex].clean + "?t=" + Date.now();
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
  } catch (err) {
    showToast("Không xóa được vùng chỉnh sửa thủ công: " + err.message, "error");
  } finally {
    if (card) {
      card._reviewBusy = false;
      if (typeof card._syncReviewBusy === "function") card._syncReviewBusy();
    }
    if (resetBtn) {
      resetBtn.disabled = false;
      resetBtn.textContent = "Xóa vùng chỉnh sửa";
    }
  }
}
