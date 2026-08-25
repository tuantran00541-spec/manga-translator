(() => {
  const POLL_INTERVAL_MS = 900;
  const state = { snapshot: null, pollTimer: null, generation: 0 };

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

  function syncChapterState() {
    const chapterId = window.currentChapterId || null;
    const snapshotChapter = state.snapshot?.chapter_id || null;
    if (!snapshotChapter || snapshotChapter === chapterId) return;
    state.generation += 1;
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    state.snapshot = null;
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

  function providerLabel(provider) {
    return provider === "deepseek" ? "DeepSeek" : "Gemini";
  }

  function clearHighlight(workspace) {
    workspace?.querySelectorAll(".review-qc-highlight").forEach((node) => node.remove());
  }

  function highlightGeometry(pageIndex, issue, img) {
    const bbox = Array.isArray(issue?.bbox) ? issue.bbox.map(Number) : null;
    const page = window.currentManifest?.pages?.[pageIndex];
    const width = Number(page?.width) || img?.naturalWidth || 0;
    const height = Number(page?.height) || img?.naturalHeight || 0;
    if (bbox?.length !== 4 || !width || !height) return null;
    const [x1, y1, x2, y2] = bbox;
    if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return null;
    return { x1, y1, x2, y2, width, height };
  }

  function applyHighlightStyle(marker, geometry) {
    const { x1, y1, x2, y2, width, height } = geometry;
    marker.style.left = `${Math.max(0, Math.min(100, x1 / width * 100))}%`;
    marker.style.top = `${Math.max(0, Math.min(100, y1 / height * 100))}%`;
    marker.style.width = `${Math.max(0.5, Math.min(100, (x2 - x1) / width * 100))}%`;
    marker.style.height = `${Math.max(0.5, Math.min(100, (y2 - y1) / height * 100))}%`;
  }

  function drawHighlight(workspace, pageIndex, issue, attempt = 0) {
    if (!workspace?.isConnected || attempt > 8) return;
    const card = workspace.querySelector(`.review-card[data-page-index="${pageIndex}"]`);
    const wrap = card?.querySelector(".review-image-wrap");
    const img = wrap?.querySelector("img");
    const geometry = highlightGeometry(pageIndex, issue, img);
    if (!wrap || !geometry) {
      window.setTimeout(() => drawHighlight(workspace, pageIndex, issue, attempt + 1), 80);
      return;
    }
    clearHighlight(workspace);
    const marker = document.createElement("div");
    const reason = issue?.reason ? ": " + issue.reason : "";
    marker.className = "review-qc-highlight";
    marker.setAttribute("aria-label", issueLabel(issue));
    marker.title = issueLabel(issue) + reason;
    applyHighlightStyle(marker, geometry);
    wrap.appendChild(marker);
    marker.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function jumpToResult(workspace, result, issue) {
    const pageIndex = Number(result?.page_index);
    if (!Number.isInteger(pageIndex) || pageIndex < 0) return;
    const visibleIndex = visiblePageIndices().indexOf(pageIndex);
    if (visibleIndex < 0) return;
    const jump = workspace.querySelector(".workspace-nav-jump-input");
    if (!jump) return;
    jump.value = String(visibleIndex + 1);
    jump.dispatchEvent(new Event("change", { bubbles: true }));
    window.setTimeout(() => drawHighlight(workspace, pageIndex, issue), 40);
  }

  function flaggedResults(snapshot) {
    return (Array.isArray(snapshot?.results) ? snapshot.results : []).filter((result) =>
      result?.status === "flagged" || result?.status === "ambiguous" || (result?.issues?.length || 0) > 0
    );
  }

  function setLocked(workspace, locked) {
    if (!workspace) return;
    workspace.classList.toggle("review-chapter-qc-running", locked);
    workspace.querySelectorAll(
      ".brush-toggle-btn,.clear-brush-btn,.repaint-btn,.reset-manual-btn,.ai-qc-btn,.brush-size-slider,.review-primary-action"
    ).forEach((control) => updateLockedControl(control, locked));
  }

  function updateLockedControl(control, locked) {
    if (locked) {
      control.dataset.chapterQcWasDisabled ??= control.disabled ? "1" : "0";
      control.disabled = true;
      return;
    }
    const previous = control.dataset.chapterQcWasDisabled;
    if (previous === undefined) return;
    control.disabled = previous === "1";
    delete control.dataset.chapterQcWasDisabled;
  }

  function summaryText(snapshot) {
    if (!snapshot) return "Chưa chạy kiểm tra toàn chương.";
    const completed = Number(snapshot.completed_regions) || 0;
    const total = Number(snapshot.total_regions) || 0;
    const provider = providerLabel(snapshot.provider);
    if (isRunning(snapshot) && snapshot.cancel_requested) return `${provider} · Đang hủy · ${completed}/${total} vùng đã xử lý…`;
    if (isRunning(snapshot)) return `${provider} · Đang kiểm tra ${completed}/${total} vùng…`;
    if (snapshot.status === "cancelled") return `${provider} · Đã hủy sau ${completed}/${total} vùng.`;
    return `${provider} · Hoàn tất · ${snapshot.passed || 0} đạt · ${snapshot.flagged || 0} cần xem · ${snapshot.failed || 0} lỗi.`;
  }

  function usageText(snapshot) {
    if (snapshot?.provider !== "deepseek" || !snapshot.usage) return "";
    const spent = Number(snapshot.usage.estimated_cost_usd) || 0;
    const budget = Number(snapshot.usage.budget_usd) || 0;
    const requests = Number(snapshot.usage.requests) || 0;
    return `Chi phí ước tính: $${spent.toFixed(4)} / $${budget.toFixed(3)} · ${requests} request`;
  }

  function updateProgress(panel, snapshot) {
    const status = panel.querySelector(".chapter-qc-summary");
    const progress = panel.querySelector("progress");
    const usage = panel.querySelector(".chapter-qc-usage");
    if (status) status.textContent = summaryText(snapshot);
    if (usage) {
      usage.textContent = usageText(snapshot);
      usage.hidden = !usage.textContent;
    }
    if (!progress) return;
    progress.max = Math.max(1, Number(snapshot?.total_regions) || 1);
    progress.value = Math.min(progress.max, Number(snapshot?.completed_regions) || 0);
    progress.hidden = !snapshot;
  }

  function syncProviderControls(panel, snapshot, running) {
    const select = panel.querySelector(".chapter-qc-provider");
    const budget = panel.querySelector(".chapter-qc-budget-input");
    const budgetWrap = panel.querySelector(".chapter-qc-budget");
    if (!select || !budget || !budgetWrap) return;
    if (running && snapshot?.provider) select.value = snapshot.provider;
    select.disabled = running;
    const deepseek = select.value === "deepseek";
    budgetWrap.hidden = !deepseek;
    budget.disabled = running || !deepseek;
  }

  function updateButtons(workspace, panel, snapshot, running) {
    const runBtn = workspace.querySelector(".chapter-qc-run");
    const cancelBtn = panel.querySelector(".chapter-qc-cancel");
    const retryBtn = panel.querySelector(".chapter-qc-retry");
    if (runBtn) {
      runBtn.disabled = running || workspace.classList.contains("review-busy");
      runBtn.textContent = snapshot && !running ? "Kiểm tra lại toàn chương" : "Kiểm tra toàn chương bằng AI";
    }
    if (cancelBtn) cancelBtn.hidden = !running || Boolean(snapshot?.cancel_requested);
    if (retryBtn) retryBtn.hidden = running || Number(snapshot?.failed) <= 0;
    syncProviderControls(panel, snapshot, running);
  }

  function makeResultButton(workspace, result, issue) {
    const button = document.createElement("button");
    const pageNumber = Number(result.page_index) + 1;
    const confidenceNumber = Number(issue?.confidence);
    const confidence = Number.isFinite(confidenceNumber) ? ` · ${Math.round(confidenceNumber * 100)}%` : "";
    const page = document.createElement("span");
    const label = document.createElement("strong");
    button.type = "button";
    button.className = "chapter-qc-result";
    page.textContent = `Trang ${pageNumber}`;
    label.textContent = issueLabel(issue) + confidence;
    button.append(page, label);
    button.addEventListener("click", () => jumpToResult(workspace, result, issue));
    return button;
  }

  function renderResultList(workspace, host, snapshot) {
    host.replaceChildren();
    if (!snapshot || isRunning(snapshot)) return;
    const results = flaggedResults(snapshot);
    if (results.length === 0) {
      const empty = document.createElement("p");
      empty.className = "chapter-qc-empty";
      empty.textContent = snapshot.status === "completed" && !snapshot.failed
        ? "AI không phát hiện vùng cần kiểm tra thêm."
        : "Không có kết quả đánh dấu để hiển thị.";
      host.appendChild(empty);
      return;
    }
    const heading = document.createElement("strong");
    heading.className = "chapter-qc-results-title";
    heading.textContent = `${results.length} kết quả cần xem`;
    host.appendChild(heading);
    results.forEach((result) => {
      const issues = result?.issues?.length ? result.issues : [null];
      issues.forEach((issue) => host.appendChild(makeResultButton(workspace, result, issue)));
    });
  }

  function renderPanel(workspace) {
    if (!workspace?.isConnected) return;
    syncChapterState();
    const panel = workspace.querySelector(".chapter-qc-panel");
    if (!panel) return;
    const snapshot = state.snapshot;
    const running = isRunning(snapshot);
    setLocked(workspace, running);
    updateProgress(panel, snapshot);
    updateButtons(workspace, panel, snapshot, running);
    const resultsHost = panel.querySelector(".chapter-qc-results");
    if (resultsHost) renderResultList(workspace, resultsHost, snapshot);
  }

  function renderAll() {
    syncChapterState();
    document.querySelectorAll(".review-workspace-shell[data-chapter-qc-bound='1']").forEach(renderPanel);
  }

  function schedulePoll(jobId, generation) {
    window.clearTimeout(state.pollTimer);
    if (!jobId || generation !== state.generation) return;
    state.pollTimer = window.setTimeout(() => pollJob(jobId, generation), POLL_INTERVAL_MS);
  }

  async function pollJob(jobId, generation) {
    try {
      const snapshot = await requestJson(`/api/visual_qc/chapter/${encodeURIComponent(jobId)}`);
      if (generation !== state.generation || snapshot?.chapter_id !== window.currentChapterId) return;
      state.snapshot = snapshot;
      renderAll();
      if (isRunning()) schedulePoll(jobId, generation);
    } catch (err) {
      if (generation === state.generation) showError("Không thể cập nhật tiến độ kiểm tra toàn chương: ", err);
    }
  }

  function showError(prefix, err) {
    if (typeof window.showToast === "function") window.showToast(prefix + err.message, "error");
  }

  function chapterRequest(workspace, chapterId) {
    const panel = workspace.querySelector(".chapter-qc-panel");
    const provider = panel?.querySelector(".chapter-qc-provider")?.value || "gemini";
    const budgetRaw = Number(panel?.querySelector(".chapter-qc-budget-input")?.value);
    const budget = Number.isFinite(budgetRaw) ? Math.max(0.005, Math.min(0.15, budgetRaw)) : 0.08;
    return { chapter_id: chapterId, concurrency: 2, provider, budget_usd: budget };
  }

  async function startChapterQC(workspace) {
    syncChapterState();
    const chapterId = window.currentChapterId;
    if (!chapterId || isRunning()) return;
    if (workspace?.classList.contains("review-busy")) {
      if (typeof window.showToast === "function") window.showToast("Hãy hoàn tất kiểm tra trang hiện tại trước khi kiểm tra toàn chương.", "info");
      return;
    }
    const request = chapterRequest(workspace, chapterId);
    if (request.provider === "deepseek" && window.deepseekVisualQCConfigured === false) {
      if (typeof window.showToast === "function") window.showToast("Hãy cấu hình DeepSeek API key trong Cài đặt trước.", "error");
      return;
    }
    const generation = ++state.generation;
    try {
      const snapshot = await requestJson("/api/visual_qc/chapter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;
      state.snapshot = snapshot;
      renderAll();
      schedulePoll(snapshot.job_id, generation);
    } catch (err) {
      if (generation === state.generation) showError("Không thể bắt đầu kiểm tra toàn chương: ", err);
      renderPanel(workspace);
    }
  }

  async function cancelChapterQC() {
    const current = state.snapshot;
    if (!current?.job_id || !isRunning(current) || current.cancel_requested) return;
    const chapterId = window.currentChapterId;
    try {
      const snapshot = await requestJson(`/api/visual_qc/chapter/${encodeURIComponent(current.job_id)}/cancel`, { method: "POST" });
      if (chapterId !== window.currentChapterId) return;
      state.snapshot = snapshot;
      const generation = ++state.generation;
      window.clearTimeout(state.pollTimer);
      renderAll();
      if (isRunning(snapshot)) schedulePoll(snapshot.job_id, generation);
    } catch (err) {
      showError("Không thể hủy kiểm tra: ", err);
    }
  }

  async function retryChapterQC() {
    const current = state.snapshot;
    if (!current?.job_id || isRunning(current)) return;
    const chapterId = window.currentChapterId;
    const generation = ++state.generation;
    try {
      const snapshot = await requestJson(`/api/visual_qc/chapter/${encodeURIComponent(current.job_id)}/retry`, { method: "POST" });
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;
      state.snapshot = snapshot;
      renderAll();
      schedulePoll(snapshot.job_id, generation);
    } catch (err) {
      showError("Không thể thử lại kiểm tra: ", err);
    }
  }

  function createPanel() {
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
      <div class="chapter-qc-options">
        <label>Provider
          <select class="chapter-qc-provider">
            <option value="gemini">Gemini</option>
            <option value="deepseek">DeepSeek Vision Exp</option>
          </select>
        </label>
        <label class="chapter-qc-budget" hidden>Giới hạn chi phí
          <span>$<input class="chapter-qc-budget-input" type="number" min="0.005" max="0.15" step="0.005" value="0.08" inputmode="decimal"></span>
        </label>
      </div>
      <p class="chapter-qc-summary">Chưa chạy kiểm tra toàn chương.</p>
      <p class="chapter-qc-usage" hidden></p>
      <progress class="chapter-qc-progress" max="1" value="0" hidden></progress>
      <div class="chapter-qc-results"></div>`;
    panel.querySelector(".chapter-qc-cancel")?.addEventListener("click", cancelChapterQC);
    panel.querySelector(".chapter-qc-retry")?.addEventListener("click", retryChapterQC);
    panel.querySelector(".chapter-qc-provider")?.addEventListener("change", () => {
      syncProviderControls(panel, state.snapshot, isRunning());
    });
    return panel;
  }

  function bindWorkspace(workspace) {
    syncChapterState();
    if (!workspace || workspace.dataset.chapterQcBound === "1") return;
    const actions = workspace.querySelector(".review-actions-group");
    const toolbar = workspace.querySelector(".review-sticky-toolbar");
    const nav = workspace.querySelector(".review-page-nav");
    if (!actions || !toolbar || !nav) return;
    workspace.dataset.chapterQcBound = "1";

    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.className = "ui-btn ui-btn-ghost chapter-qc-run";
    runBtn.textContent = "Kiểm tra toàn chương bằng AI";
    runBtn.addEventListener("click", () => startChapterQC(workspace));
    actions.prepend(runBtn);

    const panel = createPanel();
    toolbar.after(panel);
    const observer = new MutationObserver(() => {
      if (isRunning()) setLocked(workspace, true);
    });
    observer.observe(workspace, { childList: true, subtree: true });
    renderPanel(workspace);
  }

  function scan() {
    syncChapterState();
    document.querySelectorAll(".review-workspace-shell").forEach(bindWorkspace);
  }

  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", scan);
  scan();
})();
