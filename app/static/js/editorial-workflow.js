(() => {
  "use strict";

  const state = {
    chapterId: null,
    scriptPageIndex: 0,
    finalPageIndex: 0,
    filter: "todo",
    generation: 0,
    busy: false,
    preflight: null,
    staleObjectIds: new Set(),
    saveTimers: new Map(),
    pendingSaves: new Map(),
    saveChains: new Map(),
  };
  window.editorialWorkflowState = state;

  function helpers() {
    return {
      parse: typeof window.parseApiResponse === "function"
        ? window.parseApiResponse
        : async (response) => response.json().catch(() => ({})),
      error: typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, data) => data?.detail || `HTTP ${status}`,
    };
  }

  async function requestJson(url, options = undefined) {
    const response = await fetch(url, options);
    const h = helpers();
    const data = await h.parse(response);
    if (!response.ok) {
      const detail = data?.detail;
      if (detail && typeof detail === "object" && detail.message) {
        throw new Error(detail.message);
      }
      throw new Error(h.error(response.status, data));
    }
    return data;
  }

  function pageLabel(pageIndex) {
    const pages = window.currentManifest?.pages || [];
    return typeof window.pageLabel === "function"
      ? window.pageLabel(pages, pageIndex)
      : `Trang ${pageIndex + 1}`;
  }

  function storyEntries() {
    const entries = [];
    (window.currentManifest?.pages || []).forEach((page, pageIndex) => {
      if (!page || page.skipped) return;
      (page.text_objects || []).forEach((obj) => {
        if (!obj?.id || obj.source_missing) return;
        entries.push({ page, pageIndex, obj });
      });
    });
    entries.sort((a, b) => {
      if (a.pageIndex !== b.pageIndex) return a.pageIndex - b.pageIndex;
      const ar = a.obj.region || {};
      const br = b.obj.region || {};
      return (Number(ar.y1) || 0) - (Number(br.y1) || 0)
        || (Number(ar.x1) || 0) - (Number(br.x1) || 0);
    });
    return entries;
  }

  function objectIsReviewed(obj) {
    return Boolean(obj?.script_reviewed) && !state.staleObjectIds.has(String(obj.id));
  }

  function entryNeedsWork(entry) {
    return !String(entry.obj.ocr_text || "").trim()
      || !String(entry.obj.translation || "").trim()
      || !objectIsReviewed(entry.obj);
  }

  function visibleEntries(entries) {
    if (state.filter === "all") return entries;
    if (state.filter === "reviewed") return entries.filter((entry) => objectIsReviewed(entry.obj));
    return entries.filter(entryNeedsWork);
  }

  function refreshStaleObjectIds(preflight) {
    const stale = new Set();
    (preflight?.blockers || []).forEach((blocker) => {
      if (
        ["script_unreviewed", "script_review_stale"].includes(blocker?.kind)
        && blocker?.object_id
      ) {
        stale.add(String(blocker.object_id));
      }
    });
    state.staleObjectIds = stale;
  }

  async function loadPreflight() {
    if (!window.currentChapterId) return null;
    const report = await requestJson(
      `/api/export/${encodeURIComponent(window.currentChapterId)}/preflight`,
    );
    state.preflight = report;
    refreshStaleObjectIds(report);
    return report;
  }

  function saveKey(entry) {
    return `${entry.pageIndex}:${entry.obj.id}`;
  }

  async function persistPending(key) {
    const pending = state.pendingSaves.get(key);
    if (!pending) return;
    state.pendingSaves.delete(key);
    const previous = state.saveChains.get(key) || Promise.resolve();
    const next = previous.catch(() => {}).then(async () => {
      await requestJson("/api/text_object/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: window.currentChapterId,
          page_index: pending.pageIndex,
          id: pending.objectId,
          ocr_text: pending.ocrText,
          translation: pending.translation,
        }),
      });
    });
    state.saveChains.set(key, next);
    try {
      await next;
    } finally {
      if (state.saveChains.get(key) === next) state.saveChains.delete(key);
    }
  }

  function queueScriptSave(entry, source, translation) {
    const key = saveKey(entry);
    const oldTimer = state.saveTimers.get(key);
    if (oldTimer) window.clearTimeout(oldTimer);
    state.pendingSaves.set(key, {
      pageIndex: entry.pageIndex,
      objectId: String(entry.obj.id),
      ocrText: source,
      translation,
    });
    state.saveTimers.set(key, window.setTimeout(() => {
      state.saveTimers.delete(key);
      persistPending(key).catch((error) => {
        window.showToast?.("Không lưu được Script: " + error.message, "error");
      });
    }, 450));
  }

  async function flushScriptPendingSaves() {
    for (const timer of state.saveTimers.values()) window.clearTimeout(timer);
    state.saveTimers.clear();
    for (const key of [...state.pendingSaves.keys()]) {
      await persistPending(key);
    }
    await Promise.all([...state.saveChains.values()].map((job) => job.catch(() => {})));
  }
  window.flushScriptPendingSaves = flushScriptPendingSaves;

  async function setScriptReviewed(entry, reviewed) {
    await flushScriptPendingSaves();
    const manifest = await requestJson("/api/review/script", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: window.currentChapterId,
        page_index: entry.pageIndex,
        object_id: entry.obj.id,
        reviewed,
      }),
    });
    if (manifest?.chapter_id === window.currentChapterId) {
      window.currentManifest = manifest;
    }
    await loadPreflight();
  }

  function renderScriptPreview(host, entry) {
    host.replaceChildren();
    if (!entry) return;
    state.scriptPageIndex = entry.pageIndex;

    const head = document.createElement("div");
    head.className = "editorial-preview-heading";
    const title = document.createElement("strong");
    title.textContent = pageLabel(entry.pageIndex);
    const meta = document.createElement("span");
    meta.textContent = String(entry.obj.id);
    head.append(title, meta);

    const wrap = document.createElement("div");
    wrap.className = "editorial-preview-image";
    const image = document.createElement("img");
    image.src = entry.page.clean || entry.page.original || "";
    image.alt = pageLabel(entry.pageIndex);
    const marker = document.createElement("div");
    marker.className = "editorial-preview-marker";
    wrap.append(image, marker);

    const position = () => {
      const region = entry.obj.region || {};
      const width = Number(entry.page.width) || image.naturalWidth;
      const height = Number(entry.page.height) || image.naturalHeight;
      if (!width || !height) return;
      marker.style.left = `${Math.max(0, Number(region.x1) || 0) / width * 100}%`;
      marker.style.top = `${Math.max(0, Number(region.y1) || 0) / height * 100}%`;
      marker.style.width = `${Math.max(0, (Number(region.x2) || 0) - (Number(region.x1) || 0)) / width * 100}%`;
      marker.style.height = `${Math.max(0, (Number(region.y2) || 0) - (Number(region.y1) || 0)) / height * 100}%`;
    };
    if (image.complete) position();
    else image.addEventListener("load", position, { once: true });

    host.append(head, wrap);
    window.setWorkflowCheckpoint?.("script", entry.pageIndex);
  }

  function scriptSummary(entries) {
    const reviewed = entries.filter((entry) => objectIsReviewed(entry.obj)).length;
    const missingSource = entries.filter((entry) => !String(entry.obj.ocr_text || "").trim()).length;
    const missingTranslation = entries.filter((entry) => !String(entry.obj.translation || "").trim()).length;
    return { total: entries.length, reviewed, missingSource, missingTranslation };
  }

  function updateScriptSummary(root, entries) {
    const summary = scriptSummary(entries);
    const label = root.querySelector(".editorial-script-summary");
    if (label) {
      label.textContent = `${summary.reviewed}/${summary.total} đã soát · ${summary.missingTranslation} thiếu dịch · ${summary.missingSource} thiếu OCR`;
    }
    const openTypeset = root.querySelector(".editorial-open-typeset");
    if (openTypeset) {
      openTypeset.disabled = (
        summary.total === 0
        || summary.reviewed !== summary.total
        || summary.missingSource > 0
        || summary.missingTranslation > 0
      );
    }
  }

  function buildScriptRow(root, preview, entry) {
    const row = document.createElement("article");
    row.className = "editorial-script-row";
    row.dataset.objectId = String(entry.obj.id);
    row.dataset.pageIndex = String(entry.pageIndex);

    const header = document.createElement("div");
    header.className = "editorial-script-row-head";
    const location = document.createElement("button");
    location.type = "button";
    location.className = "editorial-script-location";
    location.textContent = pageLabel(entry.pageIndex);
    location.addEventListener("click", () => renderScriptPreview(preview, entry));
    const badge = document.createElement("span");
    badge.className = "editorial-script-status";
    const syncStatus = () => {
      const reviewed = objectIsReviewed(entry.obj);
      badge.textContent = reviewed ? "Đã soát" : "Cần soát";
      row.classList.toggle("is-reviewed", reviewed);
    };
    header.append(location, badge);

    const fields = document.createElement("div");
    fields.className = "editorial-script-fields";
    const sourceLabel = document.createElement("label");
    sourceLabel.innerHTML = "<span>Source / OCR</span>";
    const source = document.createElement("textarea");
    source.rows = 3;
    source.value = entry.obj.ocr_text || "";
    source.className = "editorial-script-source";
    sourceLabel.appendChild(source);

    const translationLabel = document.createElement("label");
    translationLabel.innerHTML = "<span>Bản dịch</span>";
    const translation = document.createElement("textarea");
    translation.rows = 3;
    translation.value = entry.obj.translation || "";
    translation.className = "editorial-script-translation";
    translationLabel.appendChild(translation);
    fields.append(sourceLabel, translationLabel);

    const actions = document.createElement("div");
    actions.className = "editorial-script-actions";
    const review = document.createElement("button");
    review.type = "button";
    review.className = "ui-btn ui-btn-primary";
    review.textContent = "Đã soát";
    const reopen = document.createElement("button");
    reopen.type = "button";
    reopen.className = "ui-btn ui-btn-ghost";
    reopen.textContent = "Cần soát";

    const updateLocal = () => {
      entry.obj.ocr_text = source.value;
      entry.obj.translation = translation.value;
      entry.obj.script_reviewed = false;
      state.staleObjectIds.add(String(entry.obj.id));
      syncStatus();
      updateScriptSummary(root, storyEntries());
      queueScriptSave(entry, source.value, translation.value);
    };
    source.addEventListener("input", updateLocal);
    translation.addEventListener("input", updateLocal);

    const applyReview = async (reviewed) => {
      review.disabled = true;
      reopen.disabled = true;
      try {
        await setScriptReviewed(entry, reviewed);
        renderScript();
      } catch (error) {
        window.showToast?.("Không cập nhật được trạng thái Script: " + error.message, "error");
      } finally {
        review.disabled = false;
        reopen.disabled = false;
      }
    };
    review.addEventListener("click", () => applyReview(true));
    reopen.addEventListener("click", () => applyReview(false));
    translation.addEventListener("keydown", (event) => {
      if (event.ctrlKey && event.key === "Enter") {
        event.preventDefault();
        applyReview(true);
      }
    });
    [source, translation].forEach((field) => {
      field.addEventListener("focus", () => renderScriptPreview(preview, entry));
    });

    actions.append(review, reopen);
    row.append(header, fields, actions);
    syncStatus();
    return row;
  }

  function renderScriptBody() {
    const container = document.getElementById("page-view");
    if (!container || document.body.dataset.appStage !== "script") return;
    const entries = storyEntries();

    const shell = document.createElement("section");
    shell.className = "editorial-script-workspace";
    const toolbar = document.createElement("header");
    toolbar.className = "editorial-workflow-toolbar";
    const heading = document.createElement("div");
    heading.className = "editorial-workflow-title";
    heading.innerHTML = '<span class="ui-eyebrow">Script & Proof</span><strong>Soát OCR và bản dịch toàn chương</strong><span class="editorial-script-summary"></span>';

    const controls = document.createElement("div");
    controls.className = "editorial-workflow-actions";
    const filter = document.createElement("select");
    filter.setAttribute("aria-label", "Lọc trạng thái Script");
    [["todo", "Cần xử lý"], ["all", "Tất cả"], ["reviewed", "Đã soát"]].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      filter.appendChild(option);
    });
    filter.value = state.filter;
    const typeset = document.createElement("button");
    typeset.type = "button";
    typeset.className = "ui-btn ui-btn-primary editorial-open-typeset";
    typeset.textContent = "Mở Typeset";
    typeset.addEventListener("click", async () => {
      await flushScriptPendingSaves();
      window.initialReviewCanonicalPageIndex = state.scriptPageIndex;
      window.renderReview?.();
    });
    controls.append(filter, typeset);
    toolbar.append(heading, controls);

    const grid = document.createElement("div");
    grid.className = "editorial-script-grid";
    const list = document.createElement("main");
    list.className = "editorial-script-list";
    const preview = document.createElement("aside");
    preview.className = "context-inspector editorial-script-preview";
    grid.append(list, preview);
    shell.append(toolbar, grid);
    container.replaceChildren(shell);

    const drawRows = () => {
      list.replaceChildren();
      const visible = visibleEntries(storyEntries());
      if (!visible.length) {
        const empty = document.createElement("div");
        empty.className = "editorial-empty";
        empty.textContent = entries.length
          ? "Không có dòng nào khớp bộ lọc."
          : "Chưa có text object. Quay lại Review để OCR/curate vùng chữ trước.";
        list.appendChild(empty);
        preview.replaceChildren();
        return;
      }
      visible.forEach((entry) => list.appendChild(buildScriptRow(shell, preview, entry)));
      const preferred = visible.find((entry) => entry.pageIndex === state.scriptPageIndex) || visible[0];
      renderScriptPreview(preview, preferred);
    };

    filter.addEventListener("change", () => {
      state.filter = filter.value;
      drawRows();
    });
    updateScriptSummary(shell, entries);
    drawRows();
  }

  async function prepareScript(generation) {
    await loadPreflight();
    return generation === state.generation && state.chapterId === window.currentChapterId;
  }

  function renderScript() {
    const container = document.getElementById("page-view");
    if (!container || !window.currentChapterId) return;
    window.setAppStage?.("script");
    if (state.chapterId !== window.currentChapterId) {
      state.chapterId = window.currentChapterId;
      state.scriptPageIndex = Number(window.initialScriptCanonicalPageIndex || 0);
    } else if (window.initialScriptCanonicalPageIndex != null) {
      state.scriptPageIndex = Number(window.initialScriptCanonicalPageIndex || 0);
    }
    window.initialScriptCanonicalPageIndex = null;
    const generation = ++state.generation;
    container.className = "editorial-script-mode";
    container.innerHTML = '<div class="editorial-loading"><strong>Đang kiểm tra Script…</strong><span>Review fingerprint sẽ tự stale khi OCR hoặc bản dịch thay đổi.</span></div>';
    window.setWorkflowCheckpoint?.("script", state.scriptPageIndex);
    prepareScript(generation)
      .then((ok) => {
        if (ok && document.body.dataset.appStage === "script") renderScriptBody();
      })
      .catch((error) => {
        if (generation !== state.generation) return;
        container.replaceChildren();
        const failure = document.createElement("div");
        failure.className = "editorial-loading editorial-error";
        failure.textContent = "Không chuẩn bị được Script: " + error.message;
        container.appendChild(failure);
      });
  }

  function blockersForPage(pageIndex) {
    return (state.preflight?.blockers || []).filter(
      (blocker) => Number(blocker?.page_index) === Number(pageIndex),
    );
  }

  function issueLabel(kind) {
    const labels = {
      unaccounted_story_candidate: "Vùng story chưa được account",
      promote_missing_object: "Promote nhưng thiếu text object",
      implicit_drop_without_disposition: "Story bị drop không có disposition",
      unresolved_cleanup_review: "Cleaning chưa được xác nhận",
      post_inpaint_text_residue: "Còn sót chữ sau inpaint",
      story_object_missing_ocr: "Thiếu OCR/source",
      untranslated_story_object: "Thiếu bản dịch",
      script_unreviewed: "Bản dịch chưa soát",
      script_review_stale: "Review bản dịch đã stale",
      final_review_stale: "Final approval chưa khớp render hiện tại",
    };
    return labels[kind] || kind || "Vấn đề cần kiểm tra";
  }

  function pageApproved(page) {
    const renderRevision = Number(page?.render_revision || 0);
    return Boolean(
      !page?.skipped
      && renderRevision > 0
      && Number(page?.final_review_approved_render_revision || 0) === renderRevision
    );
  }

  function nonApprovalBlockers(pageIndex) {
    return blockersForPage(pageIndex).filter((item) => item?.kind !== "final_review_stale");
  }

  function finalReady() {
    const pages = window.currentManifest?.pages || [];
    const active = pages.filter((page) => page && !page.skipped);
    return Boolean(
      state.preflight?.ok
      && active.every((page) => pageApproved(page))
    );
  }

  function navigateBlocker(blocker) {
    const pageIndex = Math.max(0, Number(blocker?.page_index) || 0);
    if (["story_object_missing_ocr", "untranslated_story_object", "script_unreviewed", "script_review_stale"].includes(blocker?.kind)) {
      state.scriptPageIndex = pageIndex;
      window.initialScriptCanonicalPageIndex = pageIndex;
      renderScript();
      return;
    }
    window.initialReviewCanonicalPageIndex = pageIndex;
    window.renderReview?.();
  }

  async function setFinalApproval(pageIndex, approved) {
    if (state.busy) return;
    state.busy = true;
    try {
      const manifest = await requestJson("/api/review/final", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: window.currentChapterId,
          page_index: pageIndex,
          approved,
        }),
      });
      if (manifest?.chapter_id === window.currentChapterId) window.currentManifest = manifest;
      await loadPreflight();
      renderFinalQCBody();
    } finally {
      state.busy = false;
    }
  }

  async function renderCurrentChapter() {
    if (!window.currentChapterId || state.busy) return;
    state.busy = true;
    try {
      await flushScriptPendingSaves();
      const data = await requestJson(
        `/api/render/chapter?chapter_id=${encodeURIComponent(window.currentChapterId)}`,
        { method: "POST" },
      );
      if (data?.chapter_id === window.currentChapterId) window.currentManifest = data;
      await loadPreflight();
      renderFinalQCBody();
    } finally {
      state.busy = false;
    }
  }

  async function downloadExport(button) {
    if (!finalReady() || state.busy) return;
    state.busy = true;
    button.disabled = true;
    try {
      const response = await fetch(
        `/api/export/${encodeURIComponent(window.currentChapterId)}.zip`,
      );
      if (!response.ok) {
        const h = helpers();
        const data = await h.parse(response);
        throw new Error(h.error(response.status, data));
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `manga-translator-${window.currentChapterId}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      window.showToast?.("Xuất chapter thất bại: " + error.message, "error");
      await loadPreflight().catch(() => {});
      renderFinalQCBody();
    } finally {
      state.busy = false;
    }
  }

  function buildFinalInspector(host, pageIndex, page) {
    host.replaceChildren();
    const heading = document.createElement("div");
    heading.className = "context-inspector-heading";
    const eyebrow = document.createElement("span");
    eyebrow.className = "ui-eyebrow";
    eyebrow.textContent = "Final QC";
    const headingTitle = document.createElement("strong");
    headingTitle.textContent = pageLabel(pageIndex);
    heading.append(eyebrow, headingTitle);
    host.appendChild(heading);

    if (page?.skipped) {
      const note = document.createElement("p");
      note.textContent = "Trang đã bỏ qua.";
      host.appendChild(note);
      return;
    }

    const blockers = nonApprovalBlockers(pageIndex);
    const approved = pageApproved(page);
    const status = document.createElement("div");
    status.className = `editorial-final-status ${approved ? "is-approved" : blockers.length ? "is-blocked" : "is-ready"}`;
    status.textContent = approved
      ? "Đã duyệt ở render revision hiện tại"
      : blockers.length
        ? `${blockers.length} blocker cần xử lý`
        : page?.rendered
          ? "Sẵn sàng để duyệt"
          : "Chưa có render hiện hành";
    host.appendChild(status);

    const list = document.createElement("div");
    list.className = "editorial-final-issues";
    blockers.forEach((blocker) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "editorial-final-issue";
      const strong = document.createElement("strong");
      strong.textContent = issueLabel(blocker.kind);
      const detail = document.createElement("span");
      detail.textContent = blocker.reason || "";
      button.append(strong, detail);
      button.addEventListener("click", () => navigateBlocker(blocker));
      list.appendChild(button);
    });
    host.appendChild(list);

    const actions = document.createElement("div");
    actions.className = "editorial-workflow-actions editorial-final-page-actions";
    if (!blockers.length && page?.rendered) {
      const approve = document.createElement("button");
      approve.type = "button";
      approve.className = approved ? "ui-btn ui-btn-ghost" : "ui-btn ui-btn-primary";
      approve.textContent = approved ? "Bỏ duyệt trang" : "Duyệt trang này";
      approve.addEventListener("click", () => {
        approve.disabled = true;
        setFinalApproval(pageIndex, !approved).catch((error) => {
          approve.disabled = false;
          window.showToast?.("Không cập nhật được Final QC: " + error.message, "error");
        });
      });
      actions.appendChild(approve);
    }
    host.appendChild(actions);
  }

  function renderFinalQCBody() {
    const container = document.getElementById("page-view");
    if (!container || document.body.dataset.appStage !== "final_qc") return;
    const pages = window.currentManifest?.pages || [];
    state.finalPageIndex = Math.max(
      0,
      Math.min(state.finalPageIndex, Math.max(0, pages.length - 1)),
    );

    const shell = document.createElement("section");
    shell.className = "editorial-final-workspace";
    const toolbar = document.createElement("header");
    toolbar.className = "editorial-workflow-toolbar";
    const title = document.createElement("div");
    title.className = "editorial-workflow-title";
    const eyebrow = document.createElement("span");
    eyebrow.className = "ui-eyebrow";
    eyebrow.textContent = "Final QC";
    const titleText = document.createElement("strong");
    titleText.textContent = "Duyệt render trước khi xuất";
    const summary = document.createElement("span");
    summary.textContent = `${state.preflight?.blocker_count || 0} blocker · ${pages.filter((page) => pageApproved(page)).length}/${pages.filter((page) => !page?.skipped).length} trang đã duyệt`;
    title.append(eyebrow, titleText, summary);
    const controls = document.createElement("div");
    controls.className = "editorial-workflow-actions";
    const rerender = document.createElement("button");
    rerender.type = "button";
    rerender.className = "ui-btn ui-btn-ghost";
    rerender.textContent = "Kết xuất phần thay đổi";
    rerender.addEventListener("click", () => renderCurrentChapter().catch((error) => {
      window.showToast?.("Không kết xuất lại được: " + error.message, "error");
    }));
    const exportButton = document.createElement("button");
    exportButton.type = "button";
    exportButton.className = "ui-btn ui-btn-primary";
    exportButton.textContent = "Xuất chapter (.zip)";
    exportButton.disabled = !finalReady();
    exportButton.addEventListener("click", () => downloadExport(exportButton));
    controls.append(rerender, exportButton);
    toolbar.append(title, controls);

    const navItems = pages.map((page, index) => {
      const blockers = nonApprovalBlockers(index);
      const approved = pageApproved(page);
      return {
        key: index,
        label: pageLabel(index),
        image: page.rendered || page.clean || page.original || "",
        state: page.skipped ? "skipped" : approved ? "rendered" : blockers.length ? "review" : "ready",
        stateLabel: page.skipped ? "Bỏ qua" : approved ? "Đã duyệt" : blockers.length ? `${blockers.length} lỗi` : "Chờ duyệt",
      };
    });
    const navigator = window.createPageNavigator?.({
      items: navItems,
      activeIndex: state.finalPageIndex,
      title: "Trang Final QC",
      ariaLabel: "Điều hướng Final QC",
      onSelect: (index) => {
        state.finalPageIndex = index;
        window.setWorkflowCheckpoint?.("final_qc", index);
        renderFinalQCBody();
      },
    });

    const canvas = document.createElement("main");
    canvas.className = "editorial-final-canvas workbench-canvas-column";
    const page = pages[state.finalPageIndex];
    const wrap = document.createElement("div");
    wrap.className = "editorial-final-image";
    const image = document.createElement("img");
    image.src = page?.rendered || page?.clean || page?.original || "";
    image.alt = pageLabel(state.finalPageIndex);
    wrap.appendChild(image);
    canvas.appendChild(wrap);

    const inspector = document.createElement("aside");
    inspector.className = "context-inspector editorial-final-inspector";
    buildFinalInspector(inspector, state.finalPageIndex, page);

    const grid = document.createElement("div");
    grid.className = "workbench-stage-grid editorial-final-grid";
    if (navigator?.element) grid.appendChild(navigator.element);
    grid.append(canvas, inspector);
    shell.append(toolbar, grid);
    container.replaceChildren(shell);
  }

  async function prepareFinal(generation) {
    await loadPreflight();
    return generation === state.generation && state.chapterId === window.currentChapterId;
  }

  function renderFinalQC() {
    const container = document.getElementById("page-view");
    if (!container || !window.currentChapterId) return;
    window.setAppStage?.("final_qc");
    if (state.chapterId !== window.currentChapterId) {
      state.chapterId = window.currentChapterId;
      state.finalPageIndex = Number(window.initialFinalQCCanonicalPageIndex || 0);
    } else if (window.initialFinalQCCanonicalPageIndex != null) {
      state.finalPageIndex = Number(window.initialFinalQCCanonicalPageIndex || 0);
    }
    window.initialFinalQCCanonicalPageIndex = null;
    const generation = ++state.generation;
    container.className = "editorial-final-mode";
    container.innerHTML = '<div class="editorial-loading"><strong>Đang kiểm tra Final QC…</strong><span>Approval tự stale khi render revision thay đổi.</span></div>';
    window.setWorkflowCheckpoint?.("final_qc", state.finalPageIndex);
    prepareFinal(generation)
      .then((ok) => {
        if (ok && document.body.dataset.appStage === "final_qc") renderFinalQCBody();
      })
      .catch((error) => {
        if (generation !== state.generation) return;
        container.replaceChildren();
        const failure = document.createElement("div");
        failure.className = "editorial-loading editorial-error";
        failure.textContent = "Không tải được Final QC: " + error.message;
        container.appendChild(failure);
      });
  }

  window.renderScript = renderScript;
  window.openScriptWorkspace = (pageIndex = 0) => {
    state.scriptPageIndex = Math.max(0, Number(pageIndex) || 0);
    window.initialScriptCanonicalPageIndex = state.scriptPageIndex;
    renderScript();
  };
  window.renderFinalQC = renderFinalQC;
  window.openFinalQC = (pageIndex = 0) => {
    state.finalPageIndex = Math.max(0, Number(pageIndex) || 0);
    window.initialFinalQCCanonicalPageIndex = state.finalPageIndex;
    renderFinalQC();
  };
})();
