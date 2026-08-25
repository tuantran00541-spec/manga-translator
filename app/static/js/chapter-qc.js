(() => {
  const POLL_INTERVAL_MS = 900;
  const state = {
    snapshot: null,
    pollTimer: null,
    generation: 0,
  };

  function apiHelpers() {
    const parse = typeof window.parseApiResponse === "function"
      ? window.parseApiResponse
      : async (response) => response.json().catch(() => ({}));
    const getError = typeof window.getErrorMessage === "function"
      ? window.getErrorMessage
      : (status, data) => data?.detail || `Máy chủ trả về ${status}`;
    return { parse, getError };
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const { parse, getError } = apiHelpers();
    const data = await parse(response);
    if (!response.ok) throw new Error(getError(response.status, data));
    return data;
  }

  function isRunning(snapshot = state.snapshot) {
    return snapshot?.status === "pending" || snapshot?.status === "running";
  }

  function visiblePageIndices() {
    return (window.currentManifest?.pages || [])
      .map((page, index) => ({ page, index }))
      .filter(({ page }) => !page?.skipped)
      .map(({ index }) => index);
  }

  function issueLabel(issue) {
    const labels = {
      residual_text: "Còn sót chữ",
      partial_text: "Xóa chữ chưa hết",
      partial_erase: "Xóa chưa hoàn chỉnh",
      smear: "Vệt nhòe",
      inpaint_artifact: "Lỗi inpaint",
      over_erased_art: "Mất chi tiết ảnh",
      suspicious_fill: "Vùng nền bất thường",
      unknown: "Vùng cần kiểm tra",
    };
    return labels[issue?.issue_type] || "Vùng cần kiểm tra";
  }

  function clearHighlight(workspace) {
    workspace?.querySelectorAll(".review-qc-highlight").forEach((node) => node.remove());
  }

  function drawHighlight(workspace, pageIndex, issue, attempt = 0) {
    if (!workspace?.isConnected || attempt > 8) return;
    const card = workspace.querySelector(`.review-card[data-page-index="${pageIndex}"]`);
    const wrap = card?.querySelector(".review-image-wrap");
    const img = wrap?.querySelector("img");
    const bbox = Array.isArray(issue?.bbox) ? issue.bbox.map(Number) : null;
    const page = window.currentManifest?.pages?.[pageIndex];
    const width = Number(page?.width) || img?.naturalWidth || 0;
    const height = Number(page?.height) || img?.naturalHeight || 0;
    if (!wrap || !bbox || bbox.length !== 4 || !width || !height) {
      window.setTimeout(() => drawHighlight(workspace, pageIndex, issue, attempt + 1), 80);
      return;
    }

    const [x1, y1, x2, y2] = bbox;
    if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return;
    clearHighlight(workspace);
    const marker = document.createElement("div");
    marker.className = "review-qc-highlight";
    marker.setAttribute("aria-label", issueLabel(issue));
    marker.title = `${issueLabel(issue)}${issue?.reason ? `: ${issue.reason}` : ""}`;
    marker.style.left = `${Math.max(0, Math.min(100, x1 / width * 100))}%`;
    marker.style.top = `${Math.max(0, Math.min(100, y1 / height * 100))}%`;
    marker.style.width = `${Math.max(0.5, Math.min(100, (x2 - x1) / width * 100))}%`;
    marker.style.height = `${Math.max(0.5, Math.min(100, (y2 - y1) / height * 100))}%`;
    wrap.appendChild(marker);
    marker.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function jumpToResult(workspace, result, issue) {
    const pageIndex = Number(result?.page_index);
    if (!Number.isInteger(pageIndex) || pageIndex < 0) return;
    const pages = visiblePageIndices();
    const visibleIndex = pages.indexOf(pageIndex);
    if (visibleIndex < 0) return;
    const jump = workspace.querySelector(".workspace-nav-jump-input");
    if (!jump) return;
    jump.value = String(visibleIndex + 1);
    jump.dispatchEvent(new Event("change", { bubbles: true }));
    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);
  }

  function flaggedResults(snapshot) {
    return (Array.isArray(snapshot?.results) ? snapshot.results : []).filter((result) =>
      result?.status === "flagged" || result?.status === "ambiguous" ||
      (Array.isArray(result?.issues) && result.issues.length > 0)
    );
  }

  function setLocked(workspace, locked) {
    if (!workspace) return;
    workspace.classList.toggle("review-chapter-qc-running", locked);
    workspace.querySelectorAll(
      ".brush-toggle-btn,.clear-brush-btn,.repaint-btn,.reset-manual-btn,.ai-qc-btn,.brush-size-slider,.review-primary-action"
    ).forEach((control) => {
      if (locked) {
        if (!control.dataset.chapterQcWasDisabled) {
          control.dataset.chapterQcWasDisabled = control.disabled ? "1" : "0";
        }
        control.disabled = true;
      } else if (control.dataset.chapterQcWasDisabled) {
        control.disabled = control.dataset.chapterQcWasDisabled === "1";
        delete control.dataset.chapterQcWasDisabled;
      }
    });
  }

  function summaryText(snapshot) {
    if (!snapshot) return "Chưa chạy kiểm tra toàn chương.";
    const completed = Number(snapshot.completed_regions) || 0;
    const total = Number(snapshot.total_regions) || 0;
    if (isRunning(snapshot)) return `Đang kiểm tra ${completed}/${total} vùng…`;
    if (snapshot.status === "cancelled") return `Đã hủy sau ${completed}/${total} vùng.`;
    return `Hoàn tất · ${snapshot.passed || 0} đạt · ${snapshot.flagged || 0} cần xem · ${snapshot.failed || 0} lỗi.`;
  }

  function renderPanel(workspace) {
    if (!workspace?.isConnected) return;
    const panel = workspace.querySelector(".chapter-qc-panel");
    if (!panel) return;
    const snapshot = state.snapshot;
    const running = isRunning(snapshot);
    setLocked(workspace, running);

    const status = panel.querySelector(".chapter-qc-summary");
    const progress = panel.querySelector("progress");
    const runBtn = workspace.querySelector(".chapter-qc-run");
    const cancelBtn = panel.querySelector(".chapter-qc-cancel");
    const retryBtn = panel.querySelector(".chapter-qc-retry");
    const resultsHost = panel.querySelector(".chapter-qc-results");

    if (status) status.textContent = summaryText(snapshot);
    if (progress) {
      progress.max = Math.max(1, Number(snapshot?.total_regions) || 1);
      progress.value = Math.min(progress.max, Number(snapshot?.completed_regions) || 0);
      progress.hidden = !snapshot;
    }
    if (runBtn) {
      runBtn.disabled = running;
      runBtn.textContent = snapshot && !running ? "Kiểm tra lại toàn chương" : "Kiểm tra toàn chương bằng AI";
    }
    if (cancelBtn) cancelBtn.hidden = !running;
    if (retryBtn) retryBtn.hidden = running || !(Number(snapshot?.failed) > 0);

    if (!resultsHost) return;
    resultsHost.replaceChildren();
    const results = flaggedResults(snapshot);
    if (!snapshot || running) return;
    if (!results.length) {
      const empty = document.createElement("p");
      empty.className = "chapter-qc-empty";
      empty.textContent = snapshot.status === "completed" && !snapshot.failed
        ? "AI không phát hiện vùng cần kiểm tra thêm."
        : "Không có kết quả đánh dấu để hiển thị.";
      resultsHost.appendChild(empty);
      return;
    }

    const heading = document.createElement("strong");
    heading.className = "chapter-qc-results-title";
    heading.textContent = `${results.length} kết quả cần xem`;
    resultsHost.appendChild(heading);
    results.forEach((result) => {
      const issues = Array.isArray(result.issues) && result.issues.length ? result.issues : [null];
      issues.forEach((issue) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "chapter-qc-result";
        const pageNumber = Number(result.page_index) + 1;
        const confidence = issue && Number.isFinite(Number(issue.confidence))
          ? ` · ${Math.round(Number(issue.confidence) * 100)}%`
          : "";
        button.innerHTML = `<span>Trang ${pageNumber}</span><strong>${issueLabel(issue)}${confidence}</strong>`;
        button.addEventListener("click", () => jumpToResult(workspace, result, issue));
        resultsHost.appendChild(button);
      });
    });
  }

  function renderAll() {
    document.querySelectorAll(".review-workspace-shell[data-chapter-qc-bound='1']").forEach(renderPanel);
  }

  function schedulePoll(jobId, generation) {
    window.clearTimeout(state.pollTimer);
    if (!jobId || generation !== state.generation) return;
    state.pollTimer = window.setTimeout(async () => {
      try {
        state.snapshot = await requestJson(`/api/visual_qc/chapter/${encodeURIComponent(jobId)}`);
        renderAll();
        if (isRunning() && generation === state.generation) schedulePoll(jobId, generation);
      } catch (err) {
        if (generation !== state.generation) return;
        if (typeof window.showToast === "function") {
          window.showToast("Không thể cập nhật tiến độ kiểm tra toàn chương: " + err.message, "error");
        }
      }
    }, POLL_INTERVAL_MS);
  }

  async function startChapterQC(workspace) {
    if (!window.currentChapterId || isRunning()) return;
    const generation = ++state.generation;
    try {
      state.snapshot = await requestJson("/api/visual_qc/chapter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chapter_id: window.currentChapterId, concurrency: 2 }),
      });
      renderAll();
      schedulePoll(state.snapshot.job_id, generation);
    } catch (err) {
      if (generation !== state.generation) return;
      if (typeof window.showToast === "function") {
        window.showToast("Không thể bắt đầu kiểm tra toàn chương: " + err.message, "error");
      }
      renderPanel(workspace);
    }
  }

  async function cancelChapterQC() {
    if (!state.snapshot?.job_id || !isRunning()) return;
    try {
      state.snapshot = await requestJson(
        `/api/visual_qc/chapter/${encodeURIComponent(state.snapshot.job_id)}/cancel`,
        { method: "POST" }
      );
      state.generation += 1;
      window.clearTimeout(state.pollTimer);
      renderAll();
    } catch (err) {
      if (typeof window.showToast === "function") window.showToast("Không thể hủy kiểm tra: " + err.message, "error");
    }
  }

  async function retryChapterQC() {
    if (!state.snapshot?.job_id || isRunning()) return;
    const generation = ++state.generation;
    try {
      state.snapshot = await requestJson(
        `/api/visual_qc/chapter/${encodeURIComponent(state.snapshot.job_id)}/retry`,
        { method: "POST" }
      );
      renderAll();
      schedulePoll(state.snapshot.job_id, generation);
    } catch (err) {
      if (typeof window.showToast === "function") window.showToast("Không thể thử lại kiểm tra: " + err.message, "error");
    }
  }

  function bindWorkspace(workspace) {
    if (!workspace || workspace.dataset.chapterQcBound === "1") return;
    workspace.dataset.chapterQcBound = "1";
    const actions = workspace.querySelector(".review-actions-group");
    const toolbar = workspace.querySelector(".review-sticky-toolbar");
    const nav = workspace.querySelector(".review-page-nav");
    if (!actions || !toolbar || !nav) return;

    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.className = "ui-btn ui-btn-ghost chapter-qc-run";
    runBtn.textContent = "Kiểm tra toàn chương bằng AI";
    runBtn.addEventListener("click", () => startChapterQC(workspace));
    actions.insertBefore(runBtn, actions.firstChild);

    const panel = document.createElement("section");
    panel.className = "chapter-qc-panel";
    panel.setAttribute("aria-live", "polite");
    panel.innerHTML = `
      <div class="chapter-qc-head">
        <div><span class="ui-eyebrow">AI Visual QC</span><strong>Kiểm tra toàn chương</strong></div>
        <div class="chapter-qc-job-actions">
          <button type="button" class="ui-btn ui-btn-ghost chapter-qc-retry" hidden>Thử lại phần lỗi</button>
          <button type="button" class="ui-btn ui-btn-ghost chapter-qc-cancel" hidden>Hủy kiểm tra</button>
        </div>
      </div>
      <p class="chapter-qc-summary">Chưa chạy kiểm tra toàn chương.</p>
      <progress class="chapter-qc-progress" max="1" value="0" hidden></progress>
      <div class="chapter-qc-results"></div>
    `;
    panel.querySelector(".chapter-qc-cancel")?.addEventListener("click", cancelChapterQC);
    panel.querySelector(".chapter-qc-retry")?.addEventListener("click", retryChapterQC);
    toolbar.insertAdjacentElement("afterend", panel);

    const observer = new MutationObserver(() => {
      if (isRunning()) setLocked(workspace, true);
    });
    observer.observe(workspace, { childList: true, subtree: true });
    renderPanel(workspace);
  }

  function scan() {
    document.querySelectorAll(".review-workspace-shell").forEach(bindWorkspace);
  }

  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", scan);
  scan();
})();
