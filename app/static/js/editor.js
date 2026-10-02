const editorState = {
  activePageIndex: 0,
  selectedTextObjectId: null,
  tool: "select",
  lastChapterId: null,
};
window.editorState = editorState;

const DEFAULT_TEXT_OBJECT_STYLE = {
  color: "auto",
  font: "default",
  fontSize: "auto",
  bold: false,
  strokeWidth: "auto",
  strokeColor: "auto",
  bgColor: "transparent",
  cornerRadius: "0",
  horizontalAlign: "center",
  verticalAlign: "middle",
};
window.DEFAULT_TEXT_OBJECT_STYLE = DEFAULT_TEXT_OBJECT_STYLE;

function findTextObject(pageIndex, id) {
  const page = currentManifest && currentManifest.pages ? currentManifest.pages[pageIndex] : null;
  if (!page) return null;
  const list = page.text_objects || [];
  return list.find((o) => o && o.id === id) || null;
}
window.findTextObject = findTextObject;

async function apiTextObject(action, payload) {
  const resp = await fetch(`/api/text_object/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const parse = typeof window.parseApiResponse === "function"
    ? window.parseApiResponse
    : async (r) => (await r.json().catch(() => ({})));
  const getErr = typeof window.getErrorMessage === "function"
    ? window.getErrorMessage
    : (s, d) => (d && d.detail) || `lỗi ${s}`;
  const data = await parse(resp);
  if (!resp.ok) throw new Error(getErr(resp.status, data));
  return data;
}

function collectPanelState(pageIndex, id) {
  const panelHost = document.querySelector(".translation-panel-host");
  if (!panelHost) return null;
  const state = { ocr_text: null, translation: null, style: null };
  panelHost.querySelectorAll(`textarea[data-text-object-id="${id}"]`).forEach((ta) => {
    if (ta.classList.contains("ocr-textarea")) state.ocr_text = ta.value;
    if (ta.classList.contains("translation-textarea")) state.translation = ta.value;
  });
  const panel = panelHost.querySelector(`.text-editor-panel[data-page-index="${pageIndex}"]`);
  if (panel && panel.dataset.objectId === String(id) && panel.dataset.font !== undefined) {
    state.style = {
      color: panel.dataset.color,
      font: panel.dataset.font,
      fontSize: panel.dataset.fontSize,
      bold: panel.dataset.bold === "true",
      strokeWidth: panel.dataset.strokeWidth,
      strokeColor: panel.dataset.strokeColor,
      bgColor: panel.dataset.bgColor,
      cornerRadius: panel.dataset.cornerRadius,
      horizontalAlign: panel.dataset.horizontalAlign || "center",
      verticalAlign: panel.dataset.verticalAlign || "middle",
    };
  }
  return state;
}

function _styleEqual(a, b) {
  return JSON.stringify(a || null) === JSON.stringify(b || null);
}

function _applyStateDiff(obj, current, snapshot) {
  if (current.ocr_text !== null && (snapshot === null || current.ocr_text !== snapshot.ocr_text)) {
    obj.ocr_text = current.ocr_text;
  }
  if (current.translation !== null && (snapshot === null || current.translation !== snapshot.translation)) {
    obj.translation = current.translation;
  }
  if (
    current.style
    && (snapshot === null || !snapshot.style || !_styleEqual(current.style, snapshot.style))
  ) {
    obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE, obj.style || {}, current.style);
  }
}

function applyManifestResponse(manifest, pageIndex, opts = {}) {
  if (!manifest || (manifest.chapter_id && manifest.chapter_id !== currentChapterId)) {
    return false;
  }
  const snapshot = opts.snapshot || null;
  const targetId = opts.id || editorState.selectedTextObjectId;
  const currentState = targetId ? collectPanelState(pageIndex, targetId) : null;
  if (typeof window.capturePendingGeom === "function") window.capturePendingGeom();
  currentManifest = manifest;
  if (currentState) {
    const obj = findTextObject(pageIndex, targetId);
    if (obj) _applyStateDiff(obj, currentState, snapshot);
  }
  _textDirty.forEach((entry) => {
    if (entry.pageIndex === pageIndex && entry.id === targetId) return;
    const obj = findTextObject(entry.pageIndex, entry.id);
    if (!obj) return;
    if (entry.ocr_text != null) obj.ocr_text = entry.ocr_text;
    if (entry.translation != null) obj.translation = entry.translation;
    if (entry.style) obj.style = Object.assign({}, DEFAULT_TEXT_OBJECT_STYLE, obj.style || {}, entry.style);
  });
  if (typeof window.reapplyPendingGeom === "function") window.reapplyPendingGeom();
  renderEditorPanel(pageIndex);
  return true;
}

async function createTextObject(pageIndex, shape, region) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  if (chapterId !== currentChapterId) return;
  const manifest = await apiTextObject("create", {
    chapter_id: chapterId,
    page_index: pageIndex,
    shape,
    region,
  });
  if (chapterId !== currentChapterId) return;
  const page = manifest.pages[pageIndex];
  const objs = page.text_objects || [];
  const obj = objs[objs.length - 1];
  if (!obj) throw new Error("Không tạo được vùng chữ");
  editorState.selectedTextObjectId = obj.id;
  applyManifestResponse(manifest, pageIndex, { id: obj.id });
  window.editorHistory?.recordCreate(pageIndex, obj.id);
  associateTextObjectOcr(pageIndex, obj.id).catch((err) => {
    showToast("Không thể nhóm OCR tự động. Bạn vẫn có thể nhập nội dung thủ công: " + err.message, "info");
  });
}

async function deleteTextObject(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  if (chapterId !== currentChapterId) return;
  const removed = findTextObject(pageIndex, id);
  const removedCopy = removed ? JSON.parse(JSON.stringify(removed)) : null;
  const manifest = await apiTextObject("delete", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id,
  });
  if (chapterId !== currentChapterId) return;
  if (editorState.selectedTextObjectId === id) editorState.selectedTextObjectId = null;
  if (typeof window.removePendingPersist === "function") window.removePendingPersist(pageIndex, id);
  applyManifestResponse(manifest, pageIndex, { id });
  window.editorHistory?.recordDelete(pageIndex, removedCopy);
}
window.deleteTextObject = deleteTextObject;

// Re-creates a deleted text object from a saved snapshot and returns its new id.
async function restoreTextObject(pageIndex, snap) {
  const chapterId = currentChapterId;
  if (!chapterId || !snap?.region) return null;
  await window.flushAllPendingPersists?.();
  if (chapterId !== currentChapterId) return null;
  const manifest = await apiTextObject("create", {
    chapter_id: chapterId,
    page_index: pageIndex,
    shape: snap.shape || "rectangle",
    region: snap.region,
  });
  const objs = manifest.pages[pageIndex].text_objects || [];
  const created = objs[objs.length - 1];
  if (!created) return null;
  const updated = await apiTextObject("update", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id: created.id,
    ocr_text: snap.ocr_text,
    translation: snap.translation,
    style: snap.style,
    font_selection_mode: snap.font_selection_mode,
    font_match: snap.font_match,
    font_ai_id: snap.font_ai_id,
  });
  if (chapterId !== currentChapterId) return null;
  editorState.selectedTextObjectId = created.id;
  applyManifestResponse(updated, pageIndex, { id: created.id });
  return created.id;
}
window.restoreTextObject = restoreTextObject;

async function associateTextObjectOcr(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const lang = await window.resolveSourceLang();
  const snapshot = collectPanelState(pageIndex, id);
  const manifest = await apiTextObject("ocr", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id,
    lang,
  });
  if (chapterId !== currentChapterId) return;
  if (!findTextObject(pageIndex, id)) return;
  applyManifestResponse(manifest, pageIndex, { snapshot, id });
}
window.associateTextObjectOcr = associateTextObjectOcr;

function setEditorTool(tool) {
  editorState.tool = tool;
  document.querySelectorAll(".editor-tool-btn").forEach((btn) => {
    const active = btn.dataset.tool === tool;
    btn.classList.toggle("active", active);
    btn.classList.toggle("ui-btn-primary", active);
    btn.classList.toggle("ui-btn-ghost", !active);
    if (btn.dataset.tool) btn.setAttribute("aria-pressed", String(active));
  });
}
window.setEditorTool = setEditorTool;

function setSelectedTextObject(pageIndex, id) {
  editorState.selectedTextObjectId = id;
  document.querySelectorAll(".text-object-overlay").forEach((el) => {
    el.classList.toggle("selected", el.dataset.objectId === id);
  });
  renderEditorPanel(pageIndex);
}
window.setSelectedTextObject = setSelectedTextObject;

function clearSelectedTextObject() {
  editorState.selectedTextObjectId = null;
  document.querySelectorAll(".text-object-overlay").forEach((el) => el.classList.remove("selected"));
  renderEditorPanel(editorState.activePageIndex);
}
window.clearSelectedTextObject = clearSelectedTextObject;

function switchEditorPage(newIndex) {
  const pages = currentManifest ? currentManifest.pages : null;
  if (!pages || newIndex < 0 || newIndex >= pages.length) return;
  const target = Math.max(0, Math.min(Number(newIndex) || 0, pages.length - 1));
  if (target === editorState.activePageIndex) return true;

  void flushAllPendingPersists().catch((err) => {
    console.warn("Không thể lưu bản nháp trước khi chuyển trang:", err);
  });
  editorState.activePageIndex = target;
  editorState.selectedTextObjectId = null;
  renderEditor();
  return true;
}

window.switchEditorPage = switchEditorPage;

const _pendingAutoSync = new Map();

async function ensureAutoTextObjects(pageIndex) {
  const chapterId = window.currentChapterId;
  if (!chapterId) return null;
  const key = `${chapterId}:${pageIndex}`;
  if (_pendingAutoSync.has(key)) return _pendingAutoSync.get(key);

  const job = (async () => {
    const response = await fetch("/api/text_objects/ensure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: chapterId, page_indices: [pageIndex] }),
    });
    const parse = typeof window.parseApiResponse === "function"
      ? window.parseApiResponse
      : async (r) => r.json().catch(() => ({}));
    const data = await parse(response);
    if (!response.ok) {
      const getError = typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, payload) => payload?.detail || `HTTP ${status}`;
      throw new Error(getError(response.status, data));
    }
    if (chapterId !== window.currentChapterId) return null;
    window.currentManifest = data;
    return data;
  })();

  _pendingAutoSync.set(key, job);
  try {
    return await job;
  } finally {
    _pendingAutoSync.delete(key);
  }
}
window.ensureAutoTextObjects = ensureAutoTextObjects;

function buildChapterExportButton() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ui-btn ui-btn-primary chapter-export-run";
  button.textContent = "Xuất chương (.zip)";

  const renderChapter = async (chapterId, force) => {
    const response = await fetch(`/api/render/chapter?chapter_id=${encodeURIComponent(chapterId)}${force ? "&force=true" : ""}`, {
      method: "POST",
    });
    const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
    return { response, data: await parse(response) };
  };

  button.addEventListener("click", async () => {
    const chapterId = window.currentChapterId;
    if (!chapterId) return;
    button.disabled = true;
    button.textContent = "Đang kết xuất chương…";
    try {
      if (typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists();
      }
      if (chapterId !== window.currentChapterId) return;
      let { response, data } = await renderChapter(chapterId, false);
      // Unfinished text asks before exporting; the untranslated regions stay clean in the file.
      if (response.status === 409 && data?.detail?.code === "editorial_preflight") {
        if (!window.confirm(`${data.detail.message}\nVẫn xuất chương? Những vùng đó sẽ để trống chữ.`)) return;
        ({ response, data } = await renderChapter(chapterId, true));
      }
      if (!response.ok) {
        const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d?.detail || `HTTP ${s}`;
        throw new Error(getErr(response.status, data));
      }
      if (chapterId !== window.currentChapterId) return;
      window.currentManifest = data;

      const href = data.chapter_render?.download_url || `/api/export/${encodeURIComponent(chapterId)}.zip`;
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = `manga-translator-${chapterId}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      if (typeof window.showToast === "function") {
        window.showToast(`Đã kết xuất ${data.chapter_render?.rendered || 0} trang.`, "info");
      }
    } catch (err) {
      if (chapterId === window.currentChapterId && typeof window.showToast === "function") {
        window.showToast("Xuất chương thất bại: " + err.message, "error");
      }
    } finally {
      button.disabled = false;
      button.textContent = "Xuất chương (.zip)";
    }
  });

  return button;
}

window.buildChapterExportButton = buildChapterExportButton;

// The editor lives inside the review workspace; opening it opens review at the same page.
function renderEditor() {
  const pages = window.currentManifest?.pages || [];
  const rawIndex = Number(editorState.activePageIndex ?? window.currentManifest?.workflow?.page_index ?? 0);
  const pageIndex = Math.max(0, Math.min(Number.isFinite(rawIndex) ? rawIndex : 0, Math.max(0, pages.length - 1)));
  window.initialReviewCanonicalPageIndex = pageIndex;
  window.setWorkflowCheckpoint?.("review", pageIndex);
  return window.renderReview?.();
}
window.renderEditor = renderEditor;
