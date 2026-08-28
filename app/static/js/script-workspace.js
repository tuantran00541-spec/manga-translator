(() => {
  const state = {
    chapterId: null,
    activeObjectId: null,
    activePageIndex: 0,
    filter: "todo",
    saveTimers: new Map(),
    saveChains: new Map(),
    generation: 0,
  };

  function apiHelpers() {
    return {
      parse: typeof window.parseApiResponse === "function"
        ? window.parseApiResponse
        : async (response) => response.json().catch(() => ({})),
      error: typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, data) => data?.detail || `HTTP ${status}`,
    };
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const helpers = apiHelpers();
    const data = await helpers.parse(response);
    if (!response.ok) throw new Error(helpers.error(response.status, data));
    return data;
  }

  function normalizeStatus(obj) {
    return ["draft", "reviewed", "skip"].includes(obj?.script_status) ? obj.script_status : "draft";
  }

  function objectEntries() {
    const entries = [];
    (window.currentManifest?.pages || []).forEach((page, pageIndex) => {
      if (page?.skipped) return;
      (page?.text_objects || []).forEach((obj) => {
        if (!obj?.id) return;
        entries.push({ page, pageIndex, obj });
      });
    });
    entries.sort((a, b) => {
      if (a.pageIndex !== b.pageIndex) return a.pageIndex - b.pageIndex;
      const ar = a.obj.region || {};
      const br = b.obj.region || {};
      return (Number(ar.y1) || 0) - (Number(br.y1) || 0) || (Number(ar.x1) || 0) - (Number(br.x1) || 0);
    });
    return entries;
  }

  function entryNeedsWork(entry) {
    const status = normalizeStatus(entry.obj);
    if (status === "skip") return false;
    return status !== "reviewed" || !String(entry.obj.translation || "").trim() || !String(entry.obj.ocr_text || "").trim();
  }

  function visibleEntries(entries) {
    if (state.filter === "all") return entries;
    if (state.filter === "reviewed") return entries.filter((entry) => normalizeStatus(entry.obj) === "reviewed");
    if (state.filter === "skip") return entries.filter((entry) => normalizeStatus(entry.obj) === "skip");
    return entries.filter(entryNeedsWork);
  }

  function summary(entries) {
    const reviewed = entries.filter((entry) => normalizeStatus(entry.obj) === "reviewed").length;
    const skipped = entries.filter((entry) => normalizeStatus(entry.obj) === "skip").length;
    const missingTranslation = entries.filter((entry) => normalizeStatus(entry.obj) !== "skip" && !String(entry.obj.translation || "").trim()).length;
    const missingSource = entries.filter((entry) => normalizeStatus(entry.obj) !== "skip" && !String(entry.obj.ocr_text || "").trim()).length;
    return { total: entries.length, reviewed, skipped, missingTranslation, missingSource };
  }

  function updateSummary(root, entries) {
    const info = summary(entries);
    const el = root.querySelector(".script-summary");
    if (el) {
      el.textContent = `${info.reviewed}/${info.total} đã soát · ${info.missingTranslation} thiếu bản dịch · ${info.missingSource} thiếu OCR/source · ${info.skipped} bỏ qua`;
    }
    const next = root.querySelector(".script-open-typeset");
    if (next) next.disabled = info.missingTranslation > 0 || info.missingSource > 0 || (info.reviewed + info.skipped) < info.total;
  }

  function pageLabel(pageIndex) {
    const pages = window.currentManifest?.pages || [];
    return typeof window.pageLabel === "function" ? window.pageLabel(pages, pageIndex) : `Trang ${pageIndex + 1}`;
  }

  function queueSave(entry, payload, delay = 450) {
    const key = `${entry.pageIndex}:${entry.obj.id}`;
    window.clearTimeout(state.saveTimers.get(key));
    state.saveTimers.set(key, window.setTimeout(() => {
      state.saveTimers.delete(key);
      const previous = state.saveChains.get(key) || Promise.resolve();
      const next = previous.catch(() => {}).then(async () => {
        const body = Object.assign({
          chapter_id: window.currentChapterId,
          page_index: entry.pageIndex,
          id: entry.obj.id,
        }, payload());
        await requestJson("/api/text_object/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      });
      state.saveChains.set(key, next);
      next.catch((err) => {
        if (typeof window.showToast === "function") window.showToast("Không lưu được Script: " + err.message, "error");
      }).finally(() => {
        if (state.saveChains.get(key) === next) state.saveChains.delete(key);
      });
    }, delay));
  }

  async function saveStatus(entry, status, row) {
    const source = row.querySelector(".script-source")?.value || "";
    const translation = row.querySelector(".script-translation")?.value || "";
    entry.obj.ocr_text = source;
    entry.obj.translation = translation;
    const key = `${entry.pageIndex}:${entry.obj.id}`;
    window.clearTimeout(state.saveTimers.get(key));
    state.saveTimers.delete(key);
    const previous = state.saveChains.get(key) || Promise.resolve();
    const next = previous.catch(() => {}).then(async () => {
      await requestJson("/api/text_object/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: window.currentChapterId,
          page_index: entry.pageIndex,
          id: entry.obj.id,
          ocr_text: source,
          translation,
        }),
      });
      return requestJson("/api/script/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: window.currentChapterId,
          page_index: entry.pageIndex,
          object_id: entry.obj.id,
          status,
        }),
      });
    });
    state.saveChains.set(key, next);
    const result = await next;
    entry.obj.script_status = result.status;
    if (result.script_review_fingerprint) entry.obj.script_review_fingerprint = result.script_review_fingerprint;
    else delete entry.obj.script_review_fingerprint;
    if (state.saveChains.get(key) === next) state.saveChains.delete(key);
  }

  function updateRowState(row, entry) {
    const status = normalizeStatus(entry.obj);
    row.dataset.status = status;
    row.classList.toggle("script-row-reviewed", status === "reviewed");
    row.classList.toggle("script-row-skip", status === "skip");
    const badge = row.querySelector(".script-status-badge");
    if (badge) badge.textContent = status === "reviewed" ? "Đã soát" : status === "skip" ? "Bỏ qua" : "Cần soát";
    row.querySelectorAll(".script-status-action").forEach((button) => {
      button.classList.toggle("active", button.dataset.status === status);
    });
  }

  function renderPreview(preview, entry) {
    if (!preview || !entry) return;
    state.activeObjectId = entry.obj.id;
    state.activePageIndex = entry.pageIndex;
    preview.innerHTML = "";

    const heading = document.createElement("div");
    heading.className = "script-preview-heading";
    const title = document.createElement("strong");
    title.textContent = pageLabel(entry.pageIndex);
    const meta = document.createElement("span");
    meta.textContent = `Vùng ${entry.obj.id}`;
    heading.append(title, meta);

    const wrap = document.createElement("div");
    wrap.className = "script-preview-image-wrap";
    const img = document.createElement("img");
    img.src = entry.page.clean || entry.page.original;
    img.alt = pageLabel(entry.pageIndex);
    const marker = document.createElement("div");
    marker.className = "script-preview-marker";
    wrap.append(img, marker);

    const position = () => {
      const region = entry.obj.region || {};
      const width = Number(entry.page.width) || img.naturalWidth;
      const height = Number(entry.page.height) || img.naturalHeight;
      if (!width || !height) return;
      marker.style.left = `${Math.max(0, Number(region.x1) || 0) / width * 100}%`;
      marker.style.top = `${Math.max(0, Number(region.y1) || 0) / height * 100}%`;
      marker.style.width = `${Math.max(0, (Number(region.x2) || 0) - (Number(region.x1) || 0)) / width * 100}%`;
      marker.style.height = `${Math.max(0, (Number(region.y2) || 0) - (Number(region.y1) || 0)) / height * 100}%`;
    };
    if (img.complete) position();
    else img.addEventListener("load", position, { once: true });

    preview.append(heading, wrap);
    if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("script", entry.pageIndex);
  }

  function focusNeighbor(root, row, direction) {
    const rows = [...root.querySelectorAll(".script-row")];
    const index = rows.indexOf(row);
    if (index < 0) return;
    const target = rows[index + direction];
    target?.querySelector(".script-translation")?.focus();
    target?.scrollIntoView({ block: "center" });
  }

  function buildRow(root, preview, entry) {
    const row = document.createElement("article");
    row.className = "script-row";
    row.dataset.objectId = entry.obj.id;
    row.dataset.pageIndex = String(entry.pageIndex);

    const head = document.createElement("div");
    head.className = "script-row-head";
    const location = document.createElement("button");
    location.type = "button";
    location.className = "script-row-location";
    location.textContent = pageLabel(entry.pageIndex);
    location.addEventListener("click", () => renderPreview(preview, entry));
    const badge = document.createElement("span");
    badge.className = "script-status-badge";
    head.append(location, badge);

    const fields = document.createElement("div");
    fields.className = "script-row-fields";
    const sourceWrap = document.createElement("label");
    sourceWrap.className = "script-field";
    const sourceLabel = document.createElement("span");
    sourceLabel.textContent = "Source / OCR";
    const source = document.createElement("textarea");
    source.className = "script-source";
    source.rows = 3;
    source.value = entry.obj.ocr_text || "";
    sourceWrap.append(sourceLabel, source);

    const translationWrap = document.createElement("label");
    translationWrap.className = "script-field";
    const translationLabel = document.createElement("span");
    translationLabel.textContent = "Bản dịch";
    const translation = document.createElement("textarea");
    translation.className = "script-translation";
    translation.rows = 3;
    translation.value = entry.obj.translation || "";
    translationWrap.append(translationLabel, translation);
    fields.append(sourceWrap, translationWrap);

    const actions = document.createElement("div");
    actions.className = "script-row-actions";
    [["reviewed", "Đã soát"], ["draft", "Cần soát"], ["skip", "Bỏ qua"]].forEach(([status, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ui-btn ui-btn-ghost script-status-action";
      button.dataset.status = status;
      button.textContent = label;
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await saveStatus(entry, status, row);
          updateRowState(row, entry);
          updateSummary(root, objectEntries());
          if (status === "reviewed") focusNeighbor(root, row, 1);
        } catch (err) {
          if (typeof window.showToast === "function") window.showToast("Không cập nhật được trạng thái Script: " + err.message, "error");
        } finally {
          button.disabled = false;
        }
      });
      actions.appendChild(button);
    });

    const onInput = () => {
      entry.obj.ocr_text = source.value;
      entry.obj.translation = translation.value;
      entry.obj.script_status = "draft";
      delete entry.obj.script_review_fingerprint;
      updateRowState(row, entry);
      updateSummary(root, objectEntries());
      queueSave(entry, () => ({ ocr_text: source.value, translation: translation.value }));
    };
    source.addEventListener("input", onInput);
    translation.addEventListener("input", onInput);
    [source, translation].forEach((field) => {
      field.addEventListener("focus", () => {
        root.querySelectorAll(".script-row-active").forEach((item) => item.classList.remove("script-row-active"));
        row.classList.add("script-row-active");
        renderPreview(preview, entry);
      });
      field.addEventListener("keydown", async (event) => {
        if (event.ctrlKey && event.key === "Enter") {
          event.preventDefault();
          try {
            await saveStatus(entry, "reviewed", row);
            updateRowState(row, entry);
            updateSummary(root, objectEntries());
            focusNeighbor(root, row, 1);
          } catch (err) {
            if (typeof window.showToast === "function") window.showToast("Không duyệt được dòng Script: " + err.message, "error");
          }
        } else if (event.altKey && event.key === "ArrowDown") {
          event.preventDefault();
          focusNeighbor(root, row, 1);
        } else if (event.altKey && event.key === "ArrowUp") {
          event.preventDefault();
          focusNeighbor(root, row, -1);
        }
      });
    });

    row.append(head, fields, actions);
    updateRowState(row, entry);
    return row;
  }

  function renderRows(root, list, preview, entries) {
    list.replaceChildren();
    const visible = visibleEntries(entries);
    if (!visible.length) {
      const empty = document.createElement("div");
      empty.className = "script-empty";
      empty.textContent = entries.length ? "Không có dòng nào khớp bộ lọc hiện tại." : "Chưa có vùng chữ. Có thể quay lại bước Review/OCR hoặc tạo vùng thủ công trong Typeset.";
      list.appendChild(empty);
      return;
    }
    visible.forEach((entry) => list.appendChild(buildRow(root, preview, entry)));
    const preferred = visible.find((entry) => entry.obj.id === state.activeObjectId)
      || visible.find((entry) => entry.pageIndex === state.activePageIndex)
      || visible[0];
    if (preferred) renderPreview(preview, preferred);
  }

  async function prepareScript(generation) {
    const data = await requestJson("/api/text_objects/ensure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: window.currentChapterId, page_indices: null }),
    });
    if (generation !== state.generation || data.chapter_id !== window.currentChapterId) return false;
    window.currentManifest = data;

    // Final-QC's deterministic report is also the source of truth for stale
    // proofreading approvals. If OCR/translation changed after review, surface
    // the row as draft again without teaching every producer about Script state.
    const report = await requestJson(`/api/final_qc/${encodeURIComponent(window.currentChapterId)}`);
    if (generation !== state.generation) return false;
    const staleIds = new Set();
    (report.pages || []).forEach((page) => {
      (page.issues || []).forEach((issue) => {
        if (issue?.code === "script_unreviewed" && issue.object_id) staleIds.add(String(issue.object_id));
      });
    });
    objectEntries().forEach((entry) => {
      if (staleIds.has(String(entry.obj.id)) && entry.obj.script_status === "reviewed") {
        entry.obj.script_status = "draft";
        delete entry.obj.script_review_fingerprint;
      }
    });
    return true;
  }

  function renderWorkspaceBody(container) {
    const entries = objectEntries();
    const shell = document.createElement("section");
    shell.className = "script-workspace";

    const toolbar = document.createElement("header");
    toolbar.className = "script-toolbar";
    const title = document.createElement("div");
    title.className = "script-toolbar-title";
    title.innerHTML = '<span class="ui-eyebrow">Script & Proof</span><strong>Dịch và soát toàn chương</strong><span class="script-summary"></span>';

    const actions = document.createElement("div");
    actions.className = "script-toolbar-actions";
    const filter = document.createElement("select");
    filter.className = "script-filter";
    filter.setAttribute("aria-label", "Lọc trạng thái Script");
    [["todo", "Cần xử lý"], ["all", "Tất cả"], ["reviewed", "Đã soát"], ["skip", "Bỏ qua"]].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      filter.appendChild(option);
    });
    filter.value = state.filter;

    const openTypeset = document.createElement("button");
    openTypeset.type = "button";
    openTypeset.className = "ui-btn ui-btn-primary script-open-typeset";
    openTypeset.textContent = "Mở Typeset";
    openTypeset.addEventListener("click", async () => {
      await Promise.all([...state.saveChains.values()].map((job) => job.catch(() => {})));
      if (window.editorState) window.editorState.activePageIndex = state.activePageIndex || 0;
      if (typeof window.renderEditor === "function") window.renderEditor();
    });
    actions.append(filter, openTypeset);
    toolbar.append(title, actions);

    const grid = document.createElement("div");
    grid.className = "script-workspace-grid";
    const list = document.createElement("main");
    list.className = "script-list";
    const preview = document.createElement("aside");
    preview.className = "script-preview context-inspector";
    grid.append(list, preview);
    shell.append(toolbar, grid);
    container.replaceChildren(shell);

    filter.addEventListener("change", () => {
      state.filter = filter.value;
      renderRows(shell, list, preview, objectEntries());
    });
    updateSummary(shell, entries);
    renderRows(shell, list, preview, entries);
  }

  function renderScript() {
    const container = document.getElementById("page-view");
    if (!container || !window.currentChapterId) return;
    if (state.chapterId !== window.currentChapterId) {
      state.chapterId = window.currentChapterId;
      state.activeObjectId = null;
      state.activePageIndex = Number(window.initialScriptCanonicalPageIndex || 0);
    } else if (window.initialScriptCanonicalPageIndex !== undefined && window.initialScriptCanonicalPageIndex !== null) {
      state.activePageIndex = Number(window.initialScriptCanonicalPageIndex || 0);
    }
    window.initialScriptCanonicalPageIndex = null;
    const generation = ++state.generation;
    container.className = "script-mode";
    container.innerHTML = '<div class="script-loading"><strong>Đang chuẩn bị Script…</strong><span>Đồng bộ vùng nhận diện thành text object mà không ghi đè chỉnh sửa thủ công.</span></div>';
    if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("script", state.activePageIndex);

    prepareScript(generation)
      .then((ok) => {
        if (ok && document.body.dataset.appStage === "script") renderWorkspaceBody(container);
      })
      .catch((err) => {
        if (generation !== state.generation) return;
        container.innerHTML = "";
        const error = document.createElement("div");
        error.className = "script-loading script-error";
        error.textContent = "Không chuẩn bị được Script: " + err.message;
        container.appendChild(error);
      });
  }

  window.renderScript = renderScript;
  window.openScriptWorkspace = (pageIndex = 0, objectId = null) => {
    state.activePageIndex = Math.max(0, Number(pageIndex) || 0);
    state.activeObjectId = objectId || null;
    window.initialScriptCanonicalPageIndex = state.activePageIndex;
    renderScript();
  };
})();
