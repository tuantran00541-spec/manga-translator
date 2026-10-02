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

async function finishSuccessfulProcessing(chapterId) {
  if (chapterId !== currentChapterId) return;
  await refreshChapterManifest(chapterId);
  if (chapterId !== currentChapterId) return;
  if (document.body?.dataset?.appStage === "preview") {
    renderReview();
    // New users found empty source text after processing; reading it starts right away.
    window.startChapterOCR?.({ auto: true });
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

(() => {
  "use strict";

  const baseRenderTranslations = window.renderTranslations;
  if (typeof baseRenderTranslations === "function") {
    window.renderTranslations = async function renderTranslationsWithReviewUrl(pageIndex) {
      const result = await baseRenderTranslations(pageIndex);
      const page = window.currentManifest?.pages?.[Number(pageIndex)];
      if (page?.rendered && window.currentChapterId) {
        const url = `/api/download/${encodeURIComponent(window.currentChapterId)}/${Number(pageIndex)}`;
        page.rendered = url;
        page._reviewRenderedUrl = url;
      }
      return result;
    };
  }

  function restoreReviewWorkspaceState(workspace) {
    const state = workspace?._reviewRestoreState;
    if (!state) return;
    if (state.chapterId !== (window.currentChapterId || "")) {
      delete workspace._reviewRestoreState;
      return;
    }
    const shell = workspace.querySelector(".review-document-shell");
    const viewport = workspace.querySelector(".review-document-viewport");
    const ready = Boolean(shell?._descriptors?.length && viewport?.isConnected);
    if (!ready) {
      state._adapterAttempts = Number(state._adapterAttempts || 0) + 1;
      if (state._adapterAttempts < 20) window.setTimeout(scheduleRefresh, 50);
      else delete workspace._reviewRestoreState;
      return;
    }
    if (!state._scrollRestored) {
      viewport.scrollLeft = Math.max(0, Number(state.scrollLeft || 0));
      viewport.scrollTop = Math.max(0, Number(state.scrollTop || 0));
      state._scrollRestored = true;
    }
    const draft = state.draft;
    if (!draft) {
      delete workspace._reviewRestoreState;
      return;
    }
    const overlay = [...workspace.querySelectorAll(".review-text-object-overlay")].find((node) =>
      Number(node.dataset.pageIndex) === Number(draft.pageIndex)
      && String(node.dataset.objectId || "") === String(draft.objectId || "")
    );
    let editor = null;
    if (draft.surface === "inline") {
      editor = overlay?.querySelector(".review-inline-translation") || null;
    } else {
      const className = draft.field === "ocr_text" ? "ocr-textarea" : "translation-textarea";
      editor = [...workspace.querySelectorAll(`.translation-panel-host textarea.${className}`)].find((node) =>
        String(node.dataset.textObjectId || "") === String(draft.objectId || "")
        && Number(node.dataset.pageIndex ?? draft.pageIndex) === Number(draft.pageIndex)
      ) || null;
    }
    if (!editor) {
      state._adapterAttempts = Number(state._adapterAttempts || 0) + 1;
      if (state._adapterAttempts < 20) window.setTimeout(scheduleRefresh, 50);
      else delete workspace._reviewRestoreState;
      return;
    }
    try { editor.focus({ preventScroll: true }); } catch (_) { editor.focus(); }
    const length = editor.value.length;
    const start = Math.max(0, Math.min(length, Number(draft.selectionStart ?? length)));
    const end = Math.max(start, Math.min(length, Number(draft.selectionEnd ?? start)));
    try { editor.setSelectionRange(start, end); } catch (_) {}
    delete workspace._reviewRestoreState;
  }

  function refreshReviewAdapters() {
    const workspace = document.querySelector("#page-view.review-mode .review-workspace-shell");
    if (!workspace) return;
    restoreReviewWorkspaceState(workspace);
  }

  let refreshQueued = false;
  function scheduleRefresh() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      refreshReviewAdapters();
    });
  }

  let spaceDown = false;
  let pan = null;
  const editableTarget = (target) => {
    const tag = target?.tagName?.toLowerCase();
    return Boolean(target?.isContentEditable || tag === "input" || tag === "textarea" || tag === "select");
  };
  const currentReviewWorkspace = () => document.querySelector("#page-view.review-mode .review-workspace-shell");

  document.addEventListener("keydown", (event) => {
    const workspace = currentReviewWorkspace();
    if (!workspace) return;
    if (event.code === "Space" && !editableTarget(event.target)) {
      spaceDown = true;
      event.preventDefault();
      event.stopImmediatePropagation();
      workspace.querySelector(".review-document-viewport")?.classList.add("review-space-pan");
      return;
    }
  }, true);

  document.addEventListener("keyup", (event) => {
    if (event.code !== "Space") return;
    spaceDown = false;
    if (pan?.viewport) pan.viewport.classList.remove("is-panning");
    pan = null;
    currentReviewWorkspace()?.querySelector(".review-document-viewport")?.classList.remove("review-space-pan");
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);

  document.addEventListener("pointerdown", (event) => {
    const workspace = currentReviewWorkspace();
    if (!workspace) return;
    const viewport = event.target?.closest?.(".review-document-viewport");
    if (spaceDown && viewport && event.button === 0) {
      pan = { viewport, x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop, pointerId: event.pointerId };
      try { viewport.setPointerCapture?.(event.pointerId); } catch (_) {}
      viewport.classList.add("is-panning", "review-space-pan");
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
  }, true);

  document.addEventListener("pointermove", (event) => {
    if (!pan?.viewport) return;
    pan.viewport.scrollLeft = pan.left - (event.clientX - pan.x);
    pan.viewport.scrollTop = pan.top - (event.clientY - pan.y);
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);

  const finishPan = (event) => {
    if (!pan?.viewport) return;
    pan.viewport.classList.remove("is-panning");
    try { pan.viewport.releasePointerCapture?.(pan.pointerId); } catch (_) {}
    pan = null;
    event?.preventDefault?.();
    event?.stopImmediatePropagation?.();
  };
  document.addEventListener("pointerup", finishPan, true);
  document.addEventListener("pointercancel", finishPan, true);

  const pageView = document.getElementById("page-view");
  if (pageView) {
    const observer = new MutationObserver(scheduleRefresh);
    observer.observe(pageView, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  }
  window.addEventListener("resize", scheduleRefresh);
  queueMicrotask(scheduleRefresh);
})();

document.addEventListener("DOMContentLoaded", () => {
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn && typeof loadChapter === "function") {
    loadBtn.addEventListener("click", loadChapter);
  }

  if (typeof initUpload === "function") initUpload();
  if (typeof loadRecentChapters === "function") loadRecentChapters();
  if (typeof loadFonts === "function") loadFonts();

  const urlHash = (window.location.hash || "").replace(/^#/, "").trim();
  if (urlHash && typeof resumeChapter === "function") {
    resumeChapter(urlHash);
  }
});
