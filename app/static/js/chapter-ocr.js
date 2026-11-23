(() => {
  const POLL_INTERVAL_MS = 700;
  const LIVE_SYNC_MIN_MS = 2200;
  const state = {
    snapshot: null,
    timer: null,
    generation: 0,
    detecting: false,
    lastLiveRecognized: 0,
    lastLiveSyncAt: 0,
    liveSyncInFlight: false,
  };
  const autoAttemptedChapters = new Set();

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

  function langLabel(code) {
    const map = window.SOURCE_LANG_LABELS || {
      ja: "Tiếng Nhật",
      ch: "Tiếng Trung",
      korean: "Tiếng Hàn",
      en: "Tiếng Anh",
    };
    return map[code] || code || "";
  }

  function syncChapter() {
    const chapterId = window.currentChapterId || null;
    if (!state.snapshot || state.snapshot.chapter_id === chapterId) return;
    state.generation += 1;
    window.clearTimeout(state.timer);
    state.timer = null;
    state.snapshot = null;
    state.detecting = false;
    state.lastLiveRecognized = 0;
    state.lastLiveSyncAt = 0;
  }

  function statusLabel(status) {
    if (status === "cancelled") return "Đã hủy";
    if (status === "completed") return "Hoàn tất";
    return "Đang nhận dạng";
  }

  function summaryText(snapshot) {
    if (state.detecting) return "Đang tự động nhận diện ngôn ngữ truyện…";
    if (!snapshot) return "Chưa chạy OCR toàn chương.";
    const total = Number(snapshot.total) || 0;
    const done = Number(snapshot.completed) + Number(snapshot.stale) + Number(snapshot.failed);
    const status = statusLabel(snapshot.status);
    const langTag = snapshot.lang ? ` (${langLabel(snapshot.lang)})` : "";
    return `${status}${langTag}: ${done}/${total} · có chữ ${snapshot.recognized || 0} · trống ${snapshot.empty || 0} · bộ nhớ đệm ${snapshot.cached || 0} · cần cập nhật ${snapshot.stale || 0} · lỗi ${snapshot.failed || 0}`;
  }

  function renderPanel(workspace) {
    const snapshot = state.snapshot;
    const running = isRunning(snapshot);
    const busy = state.detecting || running;
    const total = Number(snapshot?.total) || 0;
    const done = Number(snapshot?.completed || 0)
      + Number(snapshot?.stale || 0)
      + Number(snapshot?.failed || 0);

    const run = workspace?.querySelector(".chapter-ocr-run");
    if (run) {
      run.disabled = busy;
      if (state.detecting) {
        run.textContent = "Đang nhận diện ngôn ngữ…";
      } else if (running && total > 0) {
        run.textContent = `OCR ${done}/${total}`;
      } else {
        run.textContent = "OCR toàn chương";
      }
      run.title = snapshot || state.detecting ? summaryText(snapshot) : "Tự nhận diện ngôn ngữ và OCR toàn chương";
    }

    const pill = workspace?.querySelector(".review-ocr-status-pill");
    if (pill) {
      pill.hidden = !busy;
      const textEl = pill.querySelector(".review-ocr-pill-text");
      const stopEl = pill.querySelector(".review-ocr-pill-cancel");
      if (textEl) {
        if (state.detecting) {
          textEl.textContent = "Đang nhận diện ngôn ngữ…";
        } else {
          const langTag = snapshot?.lang ? ` ${langLabel(snapshot.lang)}` : "";
          textEl.textContent = total > 0
            ? `OCR${langTag} · ${done}/${total}`
            : `Đang OCR${langTag}…`;
        }
      }
      if (stopEl) stopEl.hidden = !running;
      pill.title = summaryText(snapshot);
    }

    const panel = workspace?.querySelector(".chapter-ocr-panel");
    if (!panel) return;
    panel.hidden = !snapshot && !state.detecting;
    if (running) panel.open = true;
    const summary = panel.querySelector(".chapter-ocr-summary");
    if (summary) summary.textContent = summaryText(snapshot);
    const progress = panel.querySelector(".chapter-ocr-progress");
    if (progress) {
      progress.max = Math.max(1, total);
      progress.value = Math.min(Math.max(1, total), done);
      progress.hidden = !snapshot;
    }
    const cancel = panel.querySelector(".chapter-ocr-cancel");
    if (cancel) cancel.hidden = !running;
    const retry = panel.querySelector(".chapter-ocr-retry");
    if (retry) {
      retry.hidden = running || !snapshot || (Number(snapshot.failed || 0) + Number(snapshot.stale || 0) === 0);
    }
  }

  function workspaceRoot() {
    return document.getElementById("page-view") || document;
  }

  function renderAll() {
    syncChapter();
    workspaceRoot().querySelectorAll(".review-workspace-shell").forEach(renderPanel);
  }

  function schedulePoll(jobId, generation) {
    window.clearTimeout(state.timer);
    state.timer = window.setTimeout(() => poll(jobId, generation), POLL_INTERVAL_MS);
  }

  function syncDetectedLanguage(chapterId, lang) {
    if (!lang || chapterId !== window.currentChapterId || !window.currentManifest) return;
    if (window.currentManifest.source_lang !== lang) {
      window.currentManifest.source_lang = lang;
      window.currentManifest.source_lang_origin = window.currentManifest.source_lang_origin || "auto";
      document.dispatchEvent(new CustomEvent("source-lang-changed", {
        detail: {
          chapter_id: chapterId,
          source_lang: lang,
          source_lang_origin: window.currentManifest.source_lang_origin,
        },
      }));
    }
  }

  async function refreshManifestSilently(chapterId, generation) {
    if (!chapterId || generation !== state.generation || chapterId !== window.currentChapterId) return false;
    if (typeof window.refreshChapterManifest === "function") {
      const manifest = await window.refreshChapterManifest(chapterId);
      if (!manifest || generation !== state.generation || chapterId !== window.currentChapterId) return false;
    } else {
      const manifest = await requestJson(`/api/chapter/${encodeURIComponent(chapterId)}`);
      if (generation !== state.generation || chapterId !== window.currentChapterId) return false;
      window.currentManifest = manifest;
    }
    return true;
  }

  async function syncLiveCanvas(chapterId, generation) {
    if (state.liveSyncInFlight) return;
    state.liveSyncInFlight = true;
    try {
      const ok = await refreshManifestSilently(chapterId, generation);
      if (!ok) return;
      window.refreshReviewOverlays?.();
      const shell = document.querySelector("#page-view.review-mode .review-document-shell");
      shell?._rebuildProofPanel?.();
    } catch (_) {
      // Non-fatal mid-job live refresh error; final refresh runs on completion.
    } finally {
      state.liveSyncInFlight = false;
    }
  }

  async function refreshCanonicalChapter(chapterId, generation) {
    const ok = await refreshManifestSilently(chapterId, generation);
    if (!ok) return false;
    if (typeof window.refreshReviewOverlays === "function") {
      window.refreshReviewOverlays();
      const shell = document.querySelector("#page-view.review-mode .review-document-shell");
      shell?._rebuildProofPanel?.();
    } else if (typeof window.renderReview === "function") {
      window.renderReview();
    }
    return true;
  }

  function hasPendingOcrWork(manifest = window.currentManifest) {
    const pages = manifest?.pages;
    if (!Array.isArray(pages) || !pages.length) return false;
    for (const page of pages) {
      if (!page || page.skipped || page.process_required) continue;
      const boxes = Array.isArray(page.boxes) ? page.boxes : [];
      if (!boxes.length) continue;
      for (const box of boxes) {
        if (!box || box.removed || box.ocr_eligible === false) continue;
        if (!box.id) continue;
        const hasCachedText = Boolean(String(box.ocr_text || "").trim());
        const hasStaleStamp = !box.ocr_engine || !box.ocr_lang;
        if (!hasCachedText && hasStaleStamp) return true;
      }
    }
    return false;
  }

  async function poll(jobId, generation) {
    try {
      const snapshot = await requestJson(`/api/ocr/chapter/${encodeURIComponent(jobId)}`);
      if (generation !== state.generation || snapshot.chapter_id !== window.currentChapterId) return;
      state.snapshot = snapshot;
      syncDetectedLanguage(snapshot.chapter_id, snapshot.lang);
      renderAll();

      if (isRunning(snapshot)) {
        const recognized = Number(snapshot.recognized) || 0;
        const now = Date.now();
        if (
          recognized > state.lastLiveRecognized
          && (recognized - state.lastLiveRecognized >= 3 || now - state.lastLiveSyncAt >= LIVE_SYNC_MIN_MS)
        ) {
          state.lastLiveRecognized = recognized;
          state.lastLiveSyncAt = now;
          void syncLiveCanvas(snapshot.chapter_id, generation);
        }
        schedulePoll(jobId, generation);
        return;
      }

      try {
        const refreshed = await refreshCanonicalChapter(snapshot.chapter_id, generation);
        if (!refreshed) return;
      } catch (refreshErr) {
        if (generation !== state.generation || snapshot.chapter_id !== window.currentChapterId) return;
        console.error("Could not refresh chapter after OCR job:", refreshErr);
        showToast("OCR đã dừng nhưng không thể làm mới dữ liệu chương. Hãy tải lại chương.", "error");
      }

      if (snapshot.status === "completed") {
        const failed = Number(snapshot.failed) || 0;
        const total = Number(snapshot.total) || 0;
        const recognized = Number(snapshot.recognized) || 0;
        const langTag = snapshot.lang ? ` (${langLabel(snapshot.lang)})` : "";
        const reason = snapshot.fatal_error || snapshot.errors?.[0]?.detail || snapshot.errors?.[0]?.error_type || "";
        if (failed > 0 && recognized === 0) {
          showToast(
            reason
              ? `OCR toàn chương thất bại (${failed}/${total} vùng lỗi): ${reason}`
              : `OCR toàn chương thất bại (${failed}/${total} vùng lỗi).`,
            "error",
          );
        } else if (failed > 0) {
          showToast(
            `OCR${langTag} hoàn tất: nhận dạng ${recognized}/${total} vùng, nhưng có ${failed} vùng lỗi${reason ? ` (${reason})` : ""}.`,
            "error",
          );
        } else if (total === 0) {
          showToast("Không có vùng chữ nào cần OCR trên chương này.", "info");
        } else {
          showToast(`OCR toàn chương${langTag} hoàn tất: nhận dạng được ${recognized}/${total} vùng chữ.`, "success");
        }
      }
    } catch (err) {
      if (generation === state.generation) {
        showToast("Không cập nhật được tiến độ OCR: " + err.message, "error");
      }
    }
  }

  async function startChapterOCR({ auto = false, force = false } = {}) {
    if (!window.currentChapterId || isRunning() || state.detecting) return;
    const chapterId = window.currentChapterId;
    const generation = ++state.generation;
    state.lastLiveRecognized = 0;
    state.lastLiveSyncAt = 0;

    try {
      let lang = window.currentSourceLang?.() || null;
      if (!lang) {
        state.detecting = true;
        renderAll();
        try {
          lang = await window.detectSourceLang?.();
        } catch (err) {
          console.warn("Source language detection failed", err);
          lang = null;
        } finally {
          if (generation === state.generation) {
            state.detecting = false;
            renderAll();
          }
        }
      }
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;

      if (!lang) {
        if (!auto) {
          document.dispatchEvent(new CustomEvent("source-lang-needed"));
          showToast("Chưa tự nhận diện được ngôn ngữ gốc. Vui lòng chọn tại mục Ngôn ngữ gốc (⋯).", "error");
        } else {
          document.dispatchEvent(new CustomEvent("source-lang-needed"));
          showToast("Chưa nhận diện được ngôn ngữ truyện để tự OCR — hãy chọn Ngôn ngữ gốc trong trình đơn (⋯).", "info");
        }
        return;
      }

      const snapshot = await requestJson("/api/ocr/chapter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          lang,
          concurrency: 1,
          force: Boolean(force),
        }),
      });
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;
      state.snapshot = snapshot;
      syncDetectedLanguage(chapterId, snapshot.lang);
      renderAll();
      if (isRunning(snapshot)) {
        if (auto && Number(snapshot.total) > 0) {
          showToast(`Tự động nhận diện ${langLabel(snapshot.lang)} · đang OCR ${snapshot.total} vùng chữ…`, "info");
        }
        schedulePoll(snapshot.job_id, generation);
      } else if (snapshot.status === "completed") {
        await refreshCanonicalChapter(chapterId, generation);
      }
    } catch (err) {
      if (generation === state.generation && chapterId === window.currentChapterId) {
        showToast("Không thể chạy OCR toàn chương: " + err.message, "error");
      }
    }
  }

  async function maybeAutoChapterOCR() {
    const chapterId = window.currentChapterId;
    if (!chapterId) return;
    syncChapter();
    if (isRunning() || state.detecting) return;

    // 1. Reconnect if server is already running OCR for this chapter
    try {
      const active = await requestJson(`/api/ocr/chapter/active/${encodeURIComponent(chapterId)}`);
      if (chapterId !== window.currentChapterId) return;
      if (active?.active && active.job?.job_id) {
        const generation = ++state.generation;
        state.snapshot = active.job;
        state.lastLiveRecognized = Number(active.job.recognized) || 0;
        syncDetectedLanguage(chapterId, active.job.lang);
        renderAll();
        schedulePoll(active.job.job_id, generation);
        return;
      }
    } catch (_) {
      // Ignore reconnect probe failure and proceed normally.
    }

    // 2. Auto-run chapter OCR once per chapter if there are un-OCR'd detected regions
    if (autoAttemptedChapters.has(chapterId)) return;
    if (!hasPendingOcrWork(window.currentManifest)) return;
    autoAttemptedChapters.add(chapterId);
    void startChapterOCR({ auto: true });
  }

  async function cancelChapterOCR() {
    const snapshot = state.snapshot;
    if (!snapshot?.job_id || !isRunning(snapshot)) return;
    const generation = state.generation;
    const chapterId = snapshot.chapter_id;
    try {
      const next = await requestJson(
        `/api/ocr/chapter/${encodeURIComponent(snapshot.job_id)}/cancel`,
        { method: "POST" },
      );
      if (generation !== state.generation || chapterId !== window.currentChapterId) return;
      state.snapshot = next;
      renderAll();
      if (isRunning(next)) schedulePoll(snapshot.job_id, generation);
      else await refreshCanonicalChapter(next.chapter_id, generation);
    } catch (err) {
      if (generation === state.generation && chapterId === window.currentChapterId) {
        showToast("Không thể hủy OCR: " + err.message, "error");
      }
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
      else await refreshCanonicalChapter(chapterId, generation);
    } catch (err) {
      if (generation === state.generation && chapterId === window.currentChapterId) {
        showToast("Không thể thử lại OCR: " + err.message, "error");
      }
    }
  }

  function createPanel() {
    const panel = document.createElement("details");
    panel.className = "ui-disclosure inspector-section chapter-ocr-panel";
    panel.setAttribute("aria-live", "polite");
    panel.innerHTML = `<summary>Nhận dạng toàn chương</summary>
      <div class="chapter-ocr-content">
      <div class="chapter-ocr-head">
        <div><strong>Nhận dạng toàn chương</strong></div>
        <div class="chapter-ocr-actions">
          <button type="button" class="ui-btn ui-btn-ghost chapter-ocr-retry" hidden>Thử lại phần lỗi</button>
          <button type="button" class="ui-btn ui-btn-ghost chapter-ocr-cancel" hidden>Hủy OCR</button>
        </div>
      </div>
      <p class="chapter-ocr-summary">Chưa chạy OCR toàn chương.</p>
          <progress class="ui-progress chapter-ocr-progress" max="1" value="0" hidden></progress>
      </div>`;
    panel.querySelector(".chapter-ocr-cancel")?.addEventListener("click", cancelChapterOCR);
    panel.querySelector(".chapter-ocr-retry")?.addEventListener("click", retryChapterOCR);
    return panel;
  }

  function createStatusPill() {
    const pill = document.createElement("div");
    pill.className = "review-ocr-status-pill";
    pill.setAttribute("role", "status");
    pill.setAttribute("aria-live", "polite");
    pill.hidden = true;

    const dot = document.createElement("span");
    dot.className = "review-ocr-pill-dot";
    const text = document.createElement("span");
    text.className = "review-ocr-pill-text";
    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "review-ocr-pill-cancel";
    cancelBtn.textContent = "×";
    cancelBtn.title = "Hủy OCR toàn chương";
    cancelBtn.setAttribute("aria-label", "Hủy OCR toàn chương");
    cancelBtn.addEventListener("click", cancelChapterOCR);

    pill.append(dot, text, cancelBtn);
    return pill;
  }

  function bindWorkspace(workspace) {
    syncChapter();
    if (!workspace || workspace.dataset.chapterOcrBound === "1") return;
    const actions = workspace.querySelector(".review-more-actions")
      || workspace.querySelector(".review-docbar-actions")
      || workspace.querySelector(".review-actions-group");
    if (!actions) return;
    workspace.dataset.chapterOcrBound = "1";

    const run = document.createElement("button");
    run.type = "button";
    run.className = "ui-btn ui-btn-ghost ui-btn-compact chapter-ocr-run";
    run.textContent = "OCR toàn chương";
    run.title = "Tự nhận diện ngôn ngữ và nhận dạng chữ toàn chương";
    run.addEventListener("click", () => startChapterOCR({ auto: false, force: false }));
    actions.prepend(run);

    const docbarActions = workspace.querySelector(".review-docbar-actions");
    if (docbarActions && !docbarActions.querySelector(".review-ocr-status-pill")) {
      docbarActions.prepend(createStatusPill());
    }

    const inspector = workspace.querySelector(".review-inspector")
      || workspace.querySelector(".review-more-panel");
    if (inspector) {
      const panel = createPanel();
      inspector.appendChild(panel);
    }
    renderPanel(workspace);
  }

  function scan() {
    syncChapter();
    workspaceRoot().querySelectorAll(".review-workspace-shell").forEach(bindWorkspace);
  }

  window.mountChapterOCR = scan;
  window.startChapterOCR = startChapterOCR;
  window.maybeAutoChapterOCR = maybeAutoChapterOCR;
})();
