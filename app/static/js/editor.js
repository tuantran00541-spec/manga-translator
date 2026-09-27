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
  const skipOverlays = opts.skipOverlays === true;
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
  const wrapper = document.querySelector(".translation-canvas-host .page-block-wrapper");
  const page = currentManifest ? currentManifest.pages[pageIndex] : null;
  if (!skipOverlays && wrapper && page) renderTextObjectOverlays(pageIndex, page);
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
  const manifest = await apiTextObject("delete", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id,
  });
  if (chapterId !== currentChapterId) return;
  if (editorState.selectedTextObjectId === id) editorState.selectedTextObjectId = null;
  if (typeof window.removePendingPersist === "function") window.removePendingPersist(pageIndex, id);
  applyManifestResponse(manifest, pageIndex, { id });
}
window.deleteTextObject = deleteTextObject;

const DUPLICATE_OFFSET = 24;

async function duplicateTextObject(pageIndex, id) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const obj = findTextObject(pageIndex, id);
  if (!obj) throw new Error("Không tìm thấy vùng chữ");
  const source = {
    shape: obj.shape,
    region: Object.assign({}, obj.region),
    ocrText: obj.ocr_text || "",
    translation: obj.translation || "",
    style: JSON.parse(JSON.stringify(obj.style || DEFAULT_TEXT_OBJECT_STYLE)),
  };
  const { w: W, h: H } = getPageImageSize(pageIndex);
  await Promise.all([
    typeof window.flushTextObjectPersist === "function" ? window.flushTextObjectPersist() : Promise.resolve(),
    typeof window.flushGeomPersist === "function" ? window.flushGeomPersist() : Promise.resolve(),
  ]);
  if (chapterId !== currentChapterId) return;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const bw = source.region.x2 - source.region.x1;
  const bh = source.region.y2 - source.region.y1;
  const x1 = clamp(source.region.x1 + DUPLICATE_OFFSET, 0, Math.max(0, W - bw));
  const y1 = clamp(source.region.y1 + DUPLICATE_OFFSET, 0, Math.max(0, H - bh));

  const manifest = await apiTextObject("create", {
    chapter_id: chapterId,
    page_index: pageIndex,
    shape: source.shape,
    region: { x1, y1, x2: x1 + bw, y2: y1 + bh },
  });
  const objs = manifest.pages[pageIndex].text_objects || [];
  const created = objs[objs.length - 1];
  if (!created) throw new Error("Không thể nhân đôi vùng chữ");

  const updated = await apiTextObject("update", {
    chapter_id: chapterId,
    page_index: pageIndex,
    id: created.id,
    ocr_text: source.ocrText,
    translation: source.translation,
    style: source.style,
  });
  if (chapterId !== currentChapterId) return;
  editorState.selectedTextObjectId = created.id;
  applyManifestResponse(updated, pageIndex, { id: created.id });
}
window.duplicateTextObject = duplicateTextObject;

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
  applyManifestResponse(manifest, pageIndex, { skipOverlays: true, snapshot, id });
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
  const imgWrap = document.querySelector(".translation-canvas-host .page-image-wrap");
  if (imgWrap) imgWrap.classList.toggle("draw-mode", tool !== "select");
}
window.setEditorTool = setEditorTool;

function setSelectedTextObject(pageIndex, id) {
  editorState.selectedTextObjectId = id;
  document.querySelectorAll(".text-object-overlay").forEach((el) => {
    el.classList.toggle("selected", el.dataset.objectId === id);
  });
  renderEditorPanel(pageIndex);
  if (id) window.showWorkbenchInspector?.();
}
window.setSelectedTextObject = setSelectedTextObject;

function clearSelectedTextObject() {
  editorState.selectedTextObjectId = null;
  document.querySelectorAll(".text-object-overlay").forEach((el) => el.classList.remove("selected"));
  renderEditorPanel(editorState.activePageIndex);
}
window.clearSelectedTextObject = clearSelectedTextObject;

function editorImageMetrics(img) {
  if (!img || !img.naturalWidth || !img.naturalHeight || !img.clientWidth || !img.clientHeight) {
    return null;
  }
  return {
    offsetX: img.offsetLeft,
    offsetY: img.offsetTop,
    width: img.clientWidth,
    height: img.clientHeight,
    sx: img.clientWidth / img.naturalWidth,
    sy: img.clientHeight / img.naturalHeight,
  };
}
window.editorImageMetrics = editorImageMetrics;

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

function showRenderResult(pageIndex, outputPath, renderRevision = null) {
  const panelHost = document.querySelector(".translation-panel-host");
  if (!panelHost) return;
  let resultBox = panelHost.querySelector(".render-result");
  if (!resultBox) {
    resultBox = document.createElement("div");
    resultBox.className = "render-result";
    panelHost.appendChild(resultBox);
  }
  const cacheBust = renderRevision === null || renderRevision === undefined
    ? "?t=" + Date.now()
    : `?r=${encodeURIComponent(renderRevision)}`;
  resultBox.innerHTML = "";

  const label = document.createElement("div");
  label.className = "render-result-label";
  label.textContent = "Kết quả kết xuất";
  resultBox.appendChild(label);

  const img = document.createElement("img");
  img.src = outputPath + cacheBust;
  resultBox.appendChild(img);

  const link = document.createElement("a");
  link.href = window.currentChapterId ? `/api/download/${encodeURIComponent(window.currentChapterId)}/${pageIndex}` : outputPath + cacheBust;
  link.download = `page_${pageIndex + 1}_rendered.png`;
  link.className = "download-link";
  link.textContent = "Tải ảnh đã kết xuất";
  resultBox.appendChild(link);
}

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

function buildChapterTranslateControls() {
  const controls = document.createElement("details");
  controls.className = "ui-disclosure chapter-translate-controls command-disclosure";
  const summary = document.createElement("summary");
  summary.className = "ui-btn ui-btn-ghost";
  summary.textContent = "Dịch tự động";
  const options = document.createElement("div");
  options.className = "ui-disclosure-panel command-disclosure-panel chapter-translate-options";

  const note = document.createElement("p");
  note.className = "ui-note";
  note.textContent = "Mỗi lát gửi ảnh gốc + ảnh sau inpaint đến model vision. Chỉ nội dung chữ được thay đổi; bố cục và màu đã lưu được dùng để render tự động.";

  const provider = document.createElement("select");
  provider.className = "chapter-translate-provider";
  [["deepseek", "DeepSeek"], ["gemini", "Google Gemini"], ["openai", "OpenAI"], ["openrouter", "OpenRouter"], ["experiential", "Experiential Labs"]]
    .forEach(([value, label]) => provider.add(new Option(label, value)));
  const storedProvider = localStorage.getItem("manga_translation_provider");
  provider.value = [...provider.options].some((o) => o.value === storedProvider)
    ? storedProvider : "deepseek";

  const model = document.createElement("input");
  model.className = "ui-input chapter-translate-model";
  model.placeholder = "Model vision mặc định của dịch vụ";
  model.setAttribute("aria-label", "Model vision");
  const target = document.createElement("select");
  target.className = "chapter-translate-target";
  target.setAttribute("aria-label", "Ngôn ngữ bản dịch");
  [["vi", "Tiếng Việt"], ["en", "English"]].forEach(([value, label]) => target.add(new Option(label, value)));

  const budget = document.createElement("input");
  budget.type = "number";
  budget.min = "0.001";
  budget.max = "0.25";
  budget.step = "0.001";
  budget.value = "0.25";
  budget.className = "chapter-translate-budget";
  budget.title = "Giới hạn ước tính cho toàn chương; giá ảnh tùy API";
  budget.setAttribute("aria-label", "Ngân sách ước tính dịch toàn chương bằng USD");

  const forceLabel = document.createElement("label");
  forceLabel.className = "ui-field";
  const force = document.createElement("input");
  force.type = "checkbox";
  force.className = "chapter-translate-force";
  forceLabel.append(force, document.createTextNode(" Dịch lại vùng đã có bản dịch"));

  const field = (title, input) => {
    const label = document.createElement("label");
    label.className = "ui-field";
    label.append(document.createTextNode(title), input);
    return label;
  };
  const budgetLabel = field("Giới hạn DeepSeek (USD, ước tính)", budget);
  const syncModel = () => {
    model.value = localStorage.getItem("manga_translation_vision_model_" + provider.value) || "";
    localStorage.setItem("manga_translation_provider", provider.value);
    budgetLabel.hidden = provider.value !== "deepseek";
    budget.disabled = provider.value !== "deepseek";
  };
  provider.addEventListener("change", syncModel);
  provider.addEventListener("ai-providers-updated", syncModel);
  model.addEventListener("change", () => {
    localStorage.setItem("manga_translation_vision_model_" + provider.value, model.value.trim());
  });

  const run = document.createElement("button");
  run.type = "button";
  run.className = "ui-btn ui-btn-primary chapter-translate-run";
  run.textContent = "Dịch và render";

  run.addEventListener("click", async () => {
    const chapterId = window.currentChapterId;
    if (!chapterId || run.disabled) return;
    run.disabled = true;
    const originalSummary = "Dịch tự động";
    let done = 0;
    let translatedCount = 0;
    let unreadable = 0;
    let staleCount = 0;
    let actualCost = 0;
    let costKnown = provider.value === "deepseek";
    const renderErrors = [];
    const budgetTotal = Number(budget.value || 0.25);
    try {
      if (typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists();
      } else if (typeof window.flushTextObjectPersist === "function") {
        await window.flushTextObjectPersist();
      }
      if (chapterId !== window.currentChapterId) return;
      const indices = (window.currentManifest?.pages || [])
        .map((page, index) => ({ page, index }))
        .filter(({ page }) => page && !page.skipped && !page.process_required &&
          page.original && page.clean &&
          (page.text_objects?.some((obj) => obj && !obj.source_missing &&
            (force.checked || !String(obj.translation || "").trim())) ||
           page.boxes?.some((box) => box && !box.removed && box.ocr_eligible !== false)))
        .map(({ index }) => index);
      if (!indices.length) {
        window.showToast?.("Không có vùng chữ nào cần dịch trên các lát đã inpaint.", "info");
        return;
      }
      const sourceLang = await window.resolveSourceLang();
      if (chapterId !== window.currentChapterId) return;
      for (const pageIndex of indices) {
        if (chapterId !== window.currentChapterId) return;
        if (provider.value === "deepseek" && actualCost >= budgetTotal) {
          window.showToast?.("Đã chạm giới hạn chi phí ước tính. Các lát đã dịch được lưu.", "info");
          break;
        }
        run.textContent = "Đang dịch " + (done + 1) + "/" + indices.length;
        summary.textContent = run.textContent;
        const response = await fetch("/api/translate/page/vision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter_id: chapterId,
            page_index: pageIndex,
            source_lang: sourceLang,
            target_lang: target.value,
            provider: provider.value,
            model: model.value.trim() || null,
            budget_usd: provider.value === "deepseek"
              ? Math.max(0.001, budgetTotal - actualCost) : 0.25,
            force: force.checked,
          }),
        });
        const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({})));
        const data = await parse(response);
        if (!response.ok) {
          const getErr = window.getErrorMessage || ((s, d) => d?.detail || "HTTP " + s);
          throw new Error("Lát " + (pageIndex + 1) + ": " + getErr(response.status, data));
        }
        if (chapterId !== window.currentChapterId) return;
        window.currentManifest = data;
        (data.pages || []).forEach((page, index) => {
          if (page?.rendered) {
            page._reviewRenderedUrl = "/api/image/" + encodeURIComponent(chapterId)
              + "/" + index + "/rendered";
          }
        });
        const info = data.translation_run || {};
        translatedCount += Number(info.translated || 0);
        unreadable += Number(info.unreadable || 0);
        staleCount += Number(info.stale || 0);
        if (info.estimated_cost_usd == null) costKnown = false;
        else actualCost += Number(info.estimated_cost_usd);
        if (info.render_error) renderErrors.push("Lát " + (pageIndex + 1) + ": " + info.render_error);
        done++;
        if (info.budget_exceeded) break;
      }
      if (chapterId !== window.currentChapterId) return;
      const shell = document.querySelector("#page-view.review-mode .review-document-shell");
      if (shell && typeof shell._showRendered === "function") shell._showRendered();
      const price = costKnown ? " · ~$" + actualCost.toFixed(4) : " · phí ảnh theo provider";
      const warning = renderErrors.length ? " · " + renderErrors.length + " lát chưa render" : "";
      window.showToast?.(
        "Đã dịch " + translatedCount + " vùng / " + done + " lát" +
        (unreadable ? " · không đọc được " + unreadable : "") +
        (staleCount ? " · thay đổi khi dịch " + staleCount : "") + price + warning,
        renderErrors.length ? "error" : "success"
      );
    } catch (err) {
      window.showToast?.(
        "Dịch dừng sau " + done + " lát đã lưu: " + err.message, "error"
      );
    } finally {
      run.disabled = false;
      run.textContent = "Dịch và render";
      summary.textContent = originalSummary;
    }
  });

  options.append(
    note, field("Dịch vụ AI", provider), field("Model vision", model),
    field("Dịch sang", target), budgetLabel, forceLabel, run,
  );
  syncModel();
  controls.append(summary, options);
  window.syncAIProviderSelects?.();
  return controls;
}

function buildChapterExportButton() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ui-btn ui-btn-primary chapter-export-run";
  button.textContent = "Xuất chương (.zip)";

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
      const response = await fetch(`/api/render/chapter?chapter_id=${encodeURIComponent(chapterId)}`, {
        method: "POST",
      });
      const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => r.json().catch(() => ({}));
      const data = await parse(response);
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

window.buildChapterTranslateControls = buildChapterTranslateControls;
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
