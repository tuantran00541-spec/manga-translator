const PROCESS_JOB_POLL_MS = 700;

function processingButtons() {
  return Array.from(document.querySelectorAll(".preview-primary-action, #start-action"));
}

function setProcessingUi(snapshot) {
  const total = Number(snapshot?.total) || 0;
  const completed = Number(snapshot?.completed) || 0;
  const active = snapshot?.status === "pending" || snapshot?.status === "running";
  processingButtons().forEach((btn) => {
    if (!btn || !btn.isConnected) return;
    if (active) {
      btn.setAttribute("aria-busy", "true");
      btn.textContent = `Đang xử lý ${completed}/${total}…`;
      return;
    }
    btn.removeAttribute("aria-busy");
    if (document.body?.dataset?.appStage === "preview") {
      btn.textContent = completed > 0 ? "Tiếp tục xử lý" : "Bắt đầu xử lý";
    }
  });
}

async function processingJson(url, options = undefined) {
  const resp = await fetch(url, options);
  const data = await parseApiResponse(resp);
  if (!resp.ok) {
    throw new Error(getErrorMessage(resp.status, data));
  }
  return data;
}

function sleepProcessingPoll() {
  return new Promise((resolve) => setTimeout(resolve, PROCESS_JOB_POLL_MS));
}

async function getCurrentProcessingJob(chapterId) {
  if (!chapterId) return null;
  return processingJson(`/api/process/chapter/${encodeURIComponent(chapterId)}`);
}
window.getCurrentProcessingJob = getCurrentProcessingJob;

async function finishSuccessfulProcessing(chapterId) {
  if (chapterId !== currentChapterId) return;
  await refreshChapterManifest(chapterId);
  if (chapterId !== currentChapterId) return;
  if (document.body?.dataset?.appStage === "preview") {
    renderReview();
  }
}

async function finishFailedProcessing(chapterId, snapshot, { reconnect = false } = {}) {
  if (chapterId !== currentChapterId) return;
  try {
    await refreshChapterManifest(chapterId);
  } catch (err) {
    console.error("Could not resync chapter after processing failure:", err);
  }
  if (chapterId !== currentChapterId) return;
  setProcessingUi(snapshot);
  if (!reconnect) {
    const first = Array.isArray(snapshot?.errors) ? snapshot.errors[0] : null;
    showToast(first?.message || "Xử lý trang thất bại.", "error");
  }
  if (document.body?.dataset?.appStage === "preview") {
    renderPreview();
  }
}

async function monitorProcessingJob(initialSnapshot, chapterId, options = {}) {
  let snapshot = initialSnapshot;
  const reconnect = options.reconnect === true;

  while (
    chapterId === currentChapterId
    && (snapshot?.status === "pending" || snapshot?.status === "running")
  ) {
    setProcessingUi(snapshot);
    await sleepProcessingPoll();
    if (chapterId !== currentChapterId) return snapshot;
    snapshot = await processingJson(
      `/api/process/jobs/${encodeURIComponent(snapshot.job_id)}`,
    );
  }

  if (chapterId !== currentChapterId) return snapshot;
  setProcessingUi(snapshot);
  if (snapshot?.status === "completed") {
    await finishSuccessfulProcessing(chapterId);
  } else if (snapshot?.status === "failed") {
    await finishFailedProcessing(chapterId, snapshot, { reconnect });
  }
  return snapshot;
}

// api.js owns duplicate-click protection. The inner run now transfers the
// complete page plan to one backend job instead of submitting browser-owned
// 16-page batches. Refreshing/closing the tab therefore cannot truncate the
// remaining inpaint queue.
window._processSelectedPagesOnce = async function serverOwnedProcessSelectedPagesOnce() {
  const pages = currentManifest?.pages || [];
  const indices = pages
    .map((page, index) => (page?.skipped ? null : index))
    .filter((index) => index !== null);

  if (!currentChapterId) {
    showToast("Chưa có chương để xử lý.", "error");
    return;
  }
  if (indices.length === 0) {
    showToast("Không có trang nào để xử lý.", "error");
    return;
  }

  const chapterId = currentChapterId;
  processingButtons().forEach((btn) => {
    if (!btn || !btn.isConnected) return;
    btn.setAttribute("aria-busy", "true");
    btn.textContent = "Đang lưu vùng giữ nguyên…";
  });

  try {
    if (typeof window.flushPreserveRegionSaves === "function") {
      await window.flushPreserveRegionSaves(chapterId);
    }
    if (chapterId !== currentChapterId) return;

    const snapshot = await processingJson("/api/process/chapter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: chapterId,
        page_indices: indices,
        workers: getWorkersSetting(),
      }),
    });
    return await monitorProcessingJob(snapshot, chapterId);
  } catch (err) {
    if (chapterId === currentChapterId) {
      showToast("Xử lý trang thất bại: " + err.message, "error");
    }
  } finally {
    if (chapterId === currentChapterId && document.body?.dataset?.appStage === "preview") {
      try {
        const latest = await getCurrentProcessingJob(chapterId);
        if (!latest?.active) setProcessingUi(latest);
      } catch (_) {
        setProcessingUi({ status: "idle", completed: 0, total: indices.length });
      }
    }
  }
};

async function reconnectPageProcessingJob(chapterId) {
  if (!chapterId || chapterId !== currentChapterId) return null;
  let snapshot;
  try {
    snapshot = await getCurrentProcessingJob(chapterId);
  } catch (err) {
    console.error("Could not query processing job after resume:", err);
    return null;
  }
  if (!snapshot?.active) {
    setProcessingUi(snapshot || { status: "idle", completed: 0, total: 0 });
    return snapshot;
  }
  return monitorProcessingJob(snapshot, chapterId, { reconnect: true });
}
window.reconnectPageProcessingJob = reconnectPageProcessingJob;

const resumeChapterWithoutProcessingReconnect = window.resumeChapter;
if (typeof resumeChapterWithoutProcessingReconnect === "function") {
  window.resumeChapter = async function resumeChapterWithProcessingReconnect(chapterId) {
    await resumeChapterWithoutProcessingReconnect(chapterId);
    if (
      chapterId === currentChapterId
      && document.body?.dataset?.appStage === "preview"
    ) {
      await reconnectPageProcessingJob(chapterId);
    }
  };
}

// The old Editor screen is now a compatibility route only. Lettering tools live
// inside the stitched Review document, so old checkpoints and stale callers are
// redirected to that unified workspace instead of mounting a second editor UI.
const legacyRenderEditor = window.renderEditor;
window.renderEditor = function renderUnifiedReviewFromLegacyEditor() {
  const pages = window.currentManifest?.pages || [];
  const rawIndex = Number(window.editorState?.activePageIndex ?? window.currentManifest?.workflow?.page_index ?? 0);
  const pageIndex = Math.max(0, Math.min(Number.isFinite(rawIndex) ? rawIndex : 0, Math.max(0, pages.length - 1)));
  window.initialReviewCanonicalPageIndex = pageIndex;
  window.setWorkflowCheckpoint?.("review", pageIndex);
  return window.renderReview?.();
};
window.legacyRenderEditor = legacyRenderEditor;

const renderUnifiedReview = window.renderReview;
if (typeof renderUnifiedReview === "function") {
  window.renderReview = function renderReviewWithoutEditorHandoff(...args) {
    const result = renderUnifiedReview(...args);
    queueMicrotask(() => {
      document.querySelectorAll(".review-primary-action").forEach((button) => button.remove());
    });
    return result;
  };
}

document.addEventListener("DOMContentLoaded", () => {
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn && typeof loadChapter === "function") {
    loadBtn.addEventListener("click", loadChapter);
  }

  const workersEl = document.getElementById("workers-select");
  if (workersEl) {
    const saved = localStorage.getItem("mt_workers");
    if (saved && [...workersEl.options].some((o) => o.value === saved)) {
      workersEl.value = saved;
    }
    workersEl.addEventListener("change", () => {
      localStorage.setItem("mt_workers", workersEl.value);
    });
  }

  if (typeof initUpload === "function") initUpload();
  if (typeof loadRecentChapters === "function") loadRecentChapters();
  if (typeof loadFonts === "function") loadFonts();

  const urlHash = (window.location.hash || "").replace(/^#/, "").trim();
  // An explicit deep link resumes immediately. resumeChapter is wrapped above,
  // so F5 also reconnects to an active backend processing job.
  if (urlHash && typeof resumeChapter === "function") {
    resumeChapter(urlHash);
  }
});
