(() => {
  const POLL_INTERVAL_MS = 700;
  const state = { snapshot: null, timer: null, generation: 0 };

  function apiHelpers() {
    return {
      parse: typeof window.parseApiResponse === "function"
        ? window.parseApiResponse
        : async (response) => response.json().catch(() => ({})),
      error: typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, data) => data?.detail || `Máy chủ trả về ${status}`,
    };
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const helpers = apiHelpers();
    const data = await helpers.parse(response);
    if (!response.ok) throw new Error(helpers.error(response.status, data));
    return data;
  }

  function isRunning(snapshot = state.snapshot) {
    return snapshot?.status === "pending" || snapshot?.status === "running";
  }

  function currentLang() {
    return document.getElementById("lang-select")?.value || "ja";
  }

  function syncChapter() {
    const chapterId = window.currentChapterId || null;
    if (!state.snapshot || state.snapshot.chapter_id === chapterId) return;
    state.generation += 1;
    window.clearTimeout(state.timer);
    state.timer = null;
    state.snapshot = null;
  }

  function statusLabel(status) {
    if (status === "cancelled") return "Đã hủy";
    if (status === "completed") return "Hoàn tất";
    return "Đang nhận dạng";
  }

  function summaryText(snapshot) {
    if (!snapshot) return "Chưa chạy OCR toàn chương.";
    const total = Number(snapshot.total) || 0;
    const done = Number(snapshot.completed) + Number(snapshot.stale) + Number(snapshot.failed);
    const status = statusLabel(snapshot.status);
    return `${status}: ${done}/${total} · có chữ ${snapshot.recognized || 0} · rỗng ${snapshot.empty || 0} · cache ${snapshot.cached || 0} · stale ${snapshot.stale || 0} · lỗi ${snapshot.failed || 0}`;
  }

  function renderPanel(workspace) {
    const panel = workspace?.querySelector(".chapter-ocr-panel");
    if (!panel) return;
    const snapshot = state.snapshot;
    const running = isRunning(snapshot);
    const run = workspace.querySelector(".chapter-ocr-run");
    if (run) {
      run.disabled = running;
      run.textContent = running ? "OCR toàn chương đang chạy…" : "OCR toàn chương";
    }
    const summary = panel.querySelector(".chapter-ocr-summary");
    if (summary) summary.textContent = summaryText(snapshot);
    const progress = panel.querySelector(".chapter-ocr-progress");
    if (progress) {
      const total = Math.max(1, Number(snapshot?.total) || 1);
      const done = Number(snapshot?.completed || 0) + Number(snapshot?.stale || 0) + Number(snapshot?.failed || 0);
      progress.max = total;
      progress.value = Math.min(total, done);
      progress.hidden = !snapshot;
    }
    const cancel = panel.querySelector(".chapter-ocr-cancel");
    if (cancel) cancel.hidden = !running;
    const retry = panel.querySelector(".chapter-ocr-retry");
    if (retry) {
      retry.hidden = running || !snapshot || (Number(snapshot.failed || 0) + Number(snapshot.stale || 0) === 0);
    }
  }

  function renderAll() {
    syncChapter();
    document.querySelectorAll(".review-workspace-shell").forEach(renderPanel);
  }

  function schedulePoll(jobId, generation) {
    window.clearTimeout(state.timer);
    state.timer = window.setTimeout(() => poll(jobId, generation), POLL_INTERVAL_MS);
  }

  async function poll(jobId, generation) {
    try {
      const snapshot = await requestJson(`/api/ocr/chapter/${encodeURIComponent(jobId)}`);
      if (generation !== state.generation || snapshot.chapter_id !== window.currentChapterId) return;
      state.snapshot = snapshot;
      renderAll();
      if (isRunning(snapshot)) {
        schedulePoll(jobId, generation);
      } else if (snapshot.status === "completed") {
        showToast("OCR toàn chương đã hoàn tất.", "info");
      }
    } catch (err) {
      if (generation === state.generation) {
        showToast("Không cập nhật được tiến độ OCR: " + err.message, "error");
      }
    }
  }

  async function startChapterOCR() {
    if (!window.currentChapterId || isRunning()) return;
    const chapterId = window.currentChapterId;
    const generation = ++state.generation;
    try {
      const snapshot = await requestJson("/api/ocr/chapter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          lang: currentLang(),
          concurrency: 1,
          force: false,
        }),
      });
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;
      state.snapshot = snapshot;
      renderAll();
      if (isRunning(snapshot)) schedulePoll(snapshot.job_id, generation);
    } catch (err) {
      showToast("Không thể chạy OCR toàn chương: " + err.message, "error");
    }
  }

  async function cancelChapterOCR() {
    const snapshot = state.snapshot;
    if (!snapshot?.job_id || !isRunning(snapshot)) return;
    const generation = state.generation;
    try {
      state.snapshot = await requestJson(
        `/api/ocr/chapter/${encodeURIComponent(snapshot.job_id)}/cancel`,
        { method: "POST" },
      );
      renderAll();
      if (isRunning(state.snapshot)) schedulePoll(snapshot.job_id, generation);
    } catch (err) {
      showToast("Không thể hủy OCR: " + err.message, "error");
    }
  }

  async function retryChapterOCR() {
    const snapshot = state.snapshot;
    if (!snapshot?.job_id || isRunning(snapshot)) return;
    const chapterId = window.currentChapterId;
    const generation = ++state.generation;
    try {
      const next = await requestJson(
        `/api/ocr/chapter/${encodeURIComponent(snapshot.job_id)}/retry`,
        { method: "POST" },
      );
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;
      state.snapshot = next;
      renderAll();
      if (isRunning(next)) schedulePoll(next.job_id, generation);
    } catch (err) {
      showToast("Không thể thử lại OCR: " + err.message, "error");
    }
  }

  function createPanel() {
    const panel = document.createElement("section");
    panel.className = "chapter-ocr-panel";
    panel.setAttribute("aria-live", "polite");
    panel.innerHTML = `
      <div class="chapter-ocr-head">
        <div><span class="ui-eyebrow">OCR</span><strong>Nhận dạng toàn chương</strong></div>
        <div class="chapter-ocr-actions">
          <button type="button" class="ui-btn ui-btn-ghost chapter-ocr-retry" hidden>Thử lại phần lỗi</button>
          <button type="button" class="ui-btn ui-btn-ghost chapter-ocr-cancel" hidden>Hủy OCR</button>
        </div>
      </div>
      <p class="chapter-ocr-summary">Chưa chạy OCR toàn chương.</p>
      <progress class="chapter-ocr-progress" max="1" value="0" hidden></progress>`;
    panel.querySelector(".chapter-ocr-cancel")?.addEventListener("click", cancelChapterOCR);
    panel.querySelector(".chapter-ocr-retry")?.addEventListener("click", retryChapterOCR);
    return panel;
  }

  function bindWorkspace(workspace) {
    syncChapter();
    if (!workspace || workspace.dataset.chapterOcrBound === "1") return;
    const actions = workspace.querySelector(".review-actions-group");
    const toolbar = workspace.querySelector(".review-sticky-toolbar");
    if (!actions || !toolbar) return;
    workspace.dataset.chapterOcrBound = "1";

    const run = document.createElement("button");
    run.type = "button";
    run.className = "ui-btn ui-btn-ghost chapter-ocr-run";
    run.textContent = "OCR toàn chương";
    run.addEventListener("click", startChapterOCR);
    actions.prepend(run);

    const panel = createPanel();
    const qcPanel = workspace.querySelector(".chapter-qc-panel");
    (qcPanel || toolbar).after(panel);
    renderPanel(workspace);
  }

  function scan() {
    syncChapter();
    document.querySelectorAll(".review-workspace-shell").forEach(bindWorkspace);
  }

  async function safeFetchOcr(pageIndex, boxIndex, originalEl) {
    const chapterId = window.currentChapterId;
    if (!chapterId) return;
    const lang = currentLang();
    try {
      const data = await requestJson("/api/ocr_box", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          page_index: pageIndex,
          box_index: boxIndex,
          lang,
        }),
      });
      if (chapterId !== window.currentChapterId) return;
      const page = window.currentManifest?.pages?.[pageIndex];
      const box = (page?.boxes || []).find((item) => String(item?.id) === String(data.box_id));
      if (box) {
        box.ocr_text = data.text || "";
        box.ocr_lang = lang;
        box.ocr_engine = data.engine || "";
      }
      if (originalEl) originalEl.textContent = data.text || "(không nhận dạng được)";
    } catch (err) {
      if (originalEl) originalEl.textContent = "(Lỗi OCR: " + err.message + ")";
      showToast("Không thể hoàn tất OCR: " + err.message, "error");
    }
  }

  window.fetchOcr = safeFetchOcr;

  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", scan);
  scan();
})();
