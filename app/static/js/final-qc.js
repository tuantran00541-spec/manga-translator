(() => {
  const state = {
    chapterId: null,
    report: null,
    activePageIndex: 0,
    generation: 0,
    busy: false,
  };

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

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const h = helpers();
    const data = await h.parse(response);
    if (!response.ok) {
      const detail = data?.detail;
      if (detail && typeof detail === "object" && detail.message) throw new Error(detail.message);
      throw new Error(h.error(response.status, data));
    }
    return data;
  }

  function pageLabel(pageIndex) {
    const pages = window.currentManifest?.pages || [];
    return typeof window.pageLabel === "function" ? window.pageLabel(pages, pageIndex) : `Trang ${pageIndex + 1}`;
  }

  function issueLabel(issue) {
    const labels = {
      cleanup_review: "Cần xác nhận cleaning",
      source_missing: "Vùng chữ mất liên kết nguồn",
      missing_source_text: "Thiếu OCR/source",
      missing_translation: "Thiếu bản dịch",
      script_unreviewed: "Bản dịch chưa soát",
      invalid_geometry: "Khung chữ không hợp lệ",
      text_overflow: "Text tràn khung",
      render_stale: "Bản render chưa hiện hành",
      invalid_page: "Trang không hợp lệ",
    };
    return labels[issue?.code] || "Vấn đề cần kiểm tra";
  }

  function pageReport(pageIndex) {
    return state.report?.pages?.find((item) => Number(item.page_index) === Number(pageIndex)) || null;
  }

  function reportSummaryText(report) {
    const summary = report?.summary || {};
    return `${summary.pages_approved || 0}/${summary.pages_required || 0} trang đã duyệt · ${summary.blocking_issues || 0} lỗi chặn xuất bản`;
  }

  function navigateIssue(issue, pageIndex) {
    const objectId = issue?.object_id || null;
    if (issue?.code === "cleanup_review") {
      window.initialReviewCanonicalPageIndex = pageIndex;
      if (typeof window.renderReview === "function") window.renderReview();
      return;
    }
    if (["missing_source_text", "missing_translation", "script_unreviewed"].includes(issue?.code)) {
      if (typeof window.openScriptWorkspace === "function") window.openScriptWorkspace(pageIndex, objectId);
      return;
    }
    if (["source_missing", "invalid_geometry", "text_overflow"].includes(issue?.code)) {
      if (window.editorState) {
        window.editorState.activePageIndex = pageIndex;
        window.editorState.selectedTextObjectId = objectId;
      }
      if (typeof window.renderEditor === "function") window.renderEditor();
      return;
    }
    if (issue?.code === "render_stale") {
      renderCurrentChapter().catch((err) => {
        if (typeof window.showToast === "function") window.showToast("Không kết xuất lại được chapter: " + err.message, "error");
      });
    }
  }

  async function renderCurrentChapter() {
    if (!window.currentChapterId || state.busy) return;
    state.busy = true;
    try {
      if (typeof window.flushAllPendingPersists === "function") await window.flushAllPendingPersists();
      const data = await requestJson(`/api/render/chapter?chapter_id=${encodeURIComponent(window.currentChapterId)}`, { method: "POST" });
      if (data?.chapter_id === window.currentChapterId) window.currentManifest = data;
      state.report = data.final_qc || await requestJson(`/api/final_qc/${encodeURIComponent(window.currentChapterId)}`);
      renderBody();
    } finally {
      state.busy = false;
    }
  }

  async function setApproval(pageIndex, approved) {
    if (state.busy) return;
    state.busy = true;
    try {
      state.report = await requestJson("/api/final_qc/page", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: window.currentChapterId,
          page_index: pageIndex,
          approved,
        }),
      });
      renderBody();
    } finally {
      state.busy = false;
    }
  }

  async function downloadExport(button) {
    if (!state.report?.ready_for_export || state.busy) return;
    state.busy = true;
    button.disabled = true;
    const oldText = button.textContent;
    button.textContent = "Đang đóng gói…";
    try {
      const response = await fetch(`/api/export/${encodeURIComponent(window.currentChapterId)}.zip`);
      if (!response.ok) {
        const h = helpers();
        const data = await h.parse(response);
        const detail = data?.detail;
        throw new Error(detail?.message || h.error(response.status, data));
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
      if (typeof window.showToast === "function") window.showToast("Đã xuất chapter sau Final QC.", "info");
    } catch (err) {
      if (typeof window.showToast === "function") window.showToast("Xuất chapter thất bại: " + err.message, "error");
      state.report = await requestJson(`/api/final_qc/${encodeURIComponent(window.currentChapterId)}`).catch(() => state.report);
      renderBody();
    } finally {
      state.busy = false;
      button.disabled = !state.report?.ready_for_export;
      button.textContent = oldText;
    }
  }

  function buildInspector(inspector, pageIndex, item) {
    inspector.replaceChildren();
    const heading = document.createElement("div");
    heading.className = "context-inspector-heading";
    const eyebrow = document.createElement("span");
    eyebrow.className = "ui-eyebrow";
    eyebrow.textContent = "Final QC";
    const title = document.createElement("strong");
    title.textContent = pageLabel(pageIndex);
    heading.append(eyebrow, title);
    inspector.appendChild(heading);

    if (item?.skipped) {
      const note = document.createElement("p");
      note.className = "final-qc-note";
      note.textContent = "Trang đã được bỏ qua và không cần duyệt Final QC.";
      inspector.appendChild(note);
      return;
    }

    const issues = item?.issues || [];
    const status = document.createElement("div");
    status.className = `final-qc-page-status ${item?.approved ? "approved" : issues.length ? "blocked" : "ready"}`;
    status.textContent = item?.approved ? "✓ Đã duyệt ở render revision hiện tại" : issues.length ? `${issues.length} vấn đề cần xử lý` : "Sẵn sàng để người biên tập duyệt";
    inspector.appendChild(status);

    const list = document.createElement("div");
    list.className = "final-qc-issues";
    issues.forEach((issue) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "final-qc-issue";
      const label = document.createElement("strong");
      label.textContent = issueLabel(issue);
      const message = document.createElement("span");
      message.textContent = issue.message || "";
      button.append(label, message);
      button.addEventListener("click", () => navigateIssue(issue, pageIndex));
      list.appendChild(button);
    });
    inspector.appendChild(list);

    const actions = document.createElement("div");
    actions.className = "final-qc-page-actions";
    if (!issues.length) {
      const approve = document.createElement("button");
      approve.type = "button";
      approve.className = item?.approved ? "ui-btn ui-btn-ghost" : "ui-btn ui-btn-primary";
      approve.textContent = item?.approved ? "Bỏ duyệt trang" : "Duyệt trang này";
      approve.addEventListener("click", () => {
        approve.disabled = true;
        setApproval(pageIndex, !item?.approved).catch((err) => {
          approve.disabled = false;
          if (typeof window.showToast === "function") window.showToast("Không cập nhật được Final QC: " + err.message, "error");
        });
      });
      actions.appendChild(approve);
    }
    const rerender = document.createElement("button");
    rerender.type = "button";
    rerender.className = "ui-btn ui-btn-ghost";
    rerender.textContent = "Kết xuất phần thay đổi";
    rerender.addEventListener("click", () => {
      rerender.disabled = true;
      renderCurrentChapter().catch((err) => {
        rerender.disabled = false;
        if (typeof window.showToast === "function") window.showToast("Không kết xuất lại được: " + err.message, "error");
      });
    });
    actions.appendChild(rerender);
    inspector.appendChild(actions);
  }

  function renderBody() {
    const container = document.getElementById("page-view");
    if (!container || document.body.dataset.appStage !== "final_qc") return;
    const report = state.report;
    const pages = window.currentManifest?.pages || [];
    state.activePageIndex = Math.max(0, Math.min(state.activePageIndex, Math.max(0, pages.length - 1)));

    const shell = document.createElement("section");
    shell.className = "final-qc-workspace";
    const toolbar = document.createElement("header");
    toolbar.className = "final-qc-toolbar";
    const title = document.createElement("div");
    title.className = "final-qc-toolbar-title";
    title.innerHTML = '<span class="ui-eyebrow">Final QC</span><strong>Duyệt bản kết xuất trước khi xuất</strong>';
    const summary = document.createElement("span");
    summary.className = "final-qc-summary";
    summary.textContent = reportSummaryText(report);
    title.appendChild(summary);

    const actions = document.createElement("div");
    actions.className = "final-qc-toolbar-actions";
    const renderBtn = document.createElement("button");
    renderBtn.type = "button";
    renderBtn.className = "ui-btn ui-btn-ghost";
    renderBtn.textContent = "Kết xuất phần thay đổi";
    renderBtn.addEventListener("click", () => renderCurrentChapter().catch((err) => {
      if (typeof window.showToast === "function") window.showToast("Không kết xuất lại được: " + err.message, "error");
    }));
    const exportBtn = document.createElement("button");
    exportBtn.type = "button";
    exportBtn.className = "ui-btn ui-btn-primary final-qc-export";
    exportBtn.textContent = "Xuất chapter (.zip)";
    exportBtn.disabled = !report?.ready_for_export;
    exportBtn.title = report?.ready_for_export ? "Xuất chapter đã qua Final QC" : "Cần xử lý hết lỗi và duyệt mọi trang trước khi xuất";
    exportBtn.addEventListener("click", () => downloadExport(exportBtn));
    actions.append(renderBtn, exportBtn);
    toolbar.append(title, actions);

    const navItems = pages.map((page, index) => {
      const item = pageReport(index);
      const issues = item?.issues?.length || 0;
      return {
        key: index,
        label: pageLabel(index),
        image: page.rendered ? `/api/image/${encodeURIComponent(window.currentChapterId)}/${index}/rendered` : (page.clean || page.original),
        state: page.skipped ? "skipped" : item?.approved ? "rendered" : issues ? "review" : "ready",
        stateLabel: page.skipped ? "Bỏ qua" : item?.approved ? "Đã duyệt" : issues ? `${issues} lỗi` : "Chờ duyệt",
      };
    });
    const navigator = window.createPageNavigator({
      items: navItems,
      activeIndex: state.activePageIndex,
      title: "Trang Final QC",
      ariaLabel: "Điều hướng Final QC",
      onSelect: (index) => {
        state.activePageIndex = index;
        if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("final_qc", index);
        renderBody();
      },
    });

    const canvas = document.createElement("main");
    canvas.className = "final-qc-canvas workbench-canvas-column";
    const page = pages[state.activePageIndex];
    const imageWrap = document.createElement("div");
    imageWrap.className = "final-qc-image-wrap";
    const image = document.createElement("img");
    image.src = page?.rendered
      ? `/api/image/${encodeURIComponent(window.currentChapterId)}/${state.activePageIndex}/rendered?t=${Date.now()}`
      : (page?.clean || page?.original || "");
    image.alt = pageLabel(state.activePageIndex);
    imageWrap.appendChild(image);
    canvas.appendChild(imageWrap);

    const inspector = document.createElement("aside");
    inspector.className = "context-inspector final-qc-inspector";
    buildInspector(inspector, state.activePageIndex, pageReport(state.activePageIndex));

    const grid = document.createElement("div");
    grid.className = "workbench-stage-grid final-qc-grid";
    grid.append(navigator.element, canvas, inspector);
    shell.append(toolbar, grid);
    container.replaceChildren(shell);
  }

  async function loadReport(generation) {
    const report = await requestJson(`/api/final_qc/${encodeURIComponent(window.currentChapterId)}`);
    if (generation !== state.generation || report.chapter_id !== window.currentChapterId) return false;
    state.report = report;
    return true;
  }

  function renderFinalQC() {
    const container = document.getElementById("page-view");
    if (!container || !window.currentChapterId) return;
    if (state.chapterId !== window.currentChapterId) {
      state.chapterId = window.currentChapterId;
      state.activePageIndex = Number(window.initialFinalQCCanonicalPageIndex || 0);
    } else if (window.initialFinalQCCanonicalPageIndex !== undefined && window.initialFinalQCCanonicalPageIndex !== null) {
      state.activePageIndex = Number(window.initialFinalQCCanonicalPageIndex || 0);
    }
    window.initialFinalQCCanonicalPageIndex = null;
    const generation = ++state.generation;
    container.className = "final-qc-mode";
    container.innerHTML = '<div class="script-loading"><strong>Đang kiểm tra trạng thái bản kết xuất…</strong><span>Final QC dùng render identity hiện tại; approval cũ tự hết hiệu lực khi nội dung thay đổi.</span></div>';
    if (typeof window.setWorkflowCheckpoint === "function") window.setWorkflowCheckpoint("final_qc", state.activePageIndex);
    loadReport(generation)
      .then((ok) => { if (ok && document.body.dataset.appStage === "final_qc") renderBody(); })
      .catch((err) => {
        if (generation !== state.generation) return;
        container.innerHTML = "";
        const error = document.createElement("div");
        error.className = "script-loading script-error";
        error.textContent = "Không tải được Final QC: " + err.message;
        container.appendChild(error);
      });
  }

  window.renderFinalQC = renderFinalQC;
  window.openFinalQC = (pageIndex = 0) => {
    state.activePageIndex = Math.max(0, Number(pageIndex) || 0);
    window.initialFinalQCCanonicalPageIndex = state.activePageIndex;
    renderFinalQC();
  };
})();
