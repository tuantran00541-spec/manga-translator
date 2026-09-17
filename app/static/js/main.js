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

(() => {
  "use strict";

  const sourcePageOf = (page, fallback) => Number.isInteger(page?.source_page) ? page.source_page : fallback;

  const baseRenderTranslations = window.renderTranslations;
  if (typeof baseRenderTranslations === "function" && !window._unifiedReviewRenderWrapped) {
    window._unifiedReviewRenderWrapped = true;
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

  function ensureAdapterStyles() {
    if (document.getElementById("unified-review-adapter-styles")) return;
    const style = document.createElement("style");
    style.id = "unified-review-adapter-styles";
    style.textContent = `
      .review-inline-translation{max-height:240px!important;overflow:auto!important}
      .review-qc-compat-card{position:absolute!important;z-index:35!important;display:block!important;margin:0!important;padding:0!important;border:0!important;background:transparent!important;box-shadow:none!important;pointer-events:none!important;overflow:visible!important}
      .review-qc-compat-card>.review-image-wrap{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;max-width:none!important;margin:0!important;overflow:visible!important;pointer-events:none!important}
      .review-qc-compat-card>.review-image-wrap>img{display:block!important;width:100%!important;height:100%!important;max-width:none!important;opacity:0!important;pointer-events:none!important}
      .review-workspace-shell.review-chapter-qc-running .review-render-text-btn,
      .review-workspace-shell.review-chapter-qc-running .chapter-translate-controls{opacity:.45;pointer-events:none!important}
      .review-workspace-shell.review-chapter-qc-running .review-text-object-overlay{opacity:.5!important;pointer-events:none!important}
      .review-document-viewport.review-space-pan{cursor:grab!important}
      .review-document-viewport.review-space-pan.is-panning{cursor:grabbing!important}
    `;
    document.head.appendChild(style);
  }

  function sourceCoverage(desc) {
    return {
      y1: Number(desc.sourceY1) - Number(desc.localY1),
      y2: Number(desc.sourceY1) + (Number(desc.img?.naturalHeight || 0) - Number(desc.localY1)),
    };
  }

  function ensureQcCompatibility(workspace) {
    const shell = workspace?.querySelector(".review-document-shell");
    const image = shell?.querySelector(".review-stitched-image");
    const descriptors = Array.isArray(shell?._descriptors) ? shell._descriptors : [];
    if (!image || !descriptors.length) return;
    const signature = descriptors.map((desc) => [
      Number(desc.item?.canonicalIndex),
      Number(desc.sourceY1), Number(desc.sourceY2),
      Number(desc.localY1), Number(desc.localY2),
      Number(desc.img?.naturalWidth), Number(desc.img?.naturalHeight),
    ].join(":" )).join("|");
    const existing = image.querySelectorAll(":scope > .review-qc-compat-card");
    if (image.dataset.qcCompatSignature === signature && existing.length === descriptors.length) return;
    existing.forEach((node) => node.remove());
    image.dataset.qcCompatSignature = signature;

    descriptors.forEach((desc) => {
      const pageIndex = Number(desc.item?.canonicalIndex);
      const page = window.currentManifest?.pages?.[pageIndex];
      if (!page || !Number.isInteger(pageIndex)) return;
      const coverage = sourceCoverage(desc);
      const card = document.createElement("div");
      card.className = "review-card review-qc-compat-card";
      card.dataset.pageIndex = String(pageIndex);
      Object.assign(card.style, {
        left: "0px",
        top: `${coverage.y1}px`,
        width: `${Number(desc.img?.naturalWidth || page.width || 1)}px`,
        height: `${Number(desc.img?.naturalHeight || page.height || 1)}px`,
      });
      const wrap = document.createElement("div");
      wrap.className = "review-image-wrap";
      const img = document.createElement("img");
      img.alt = "";
      img.src = desc.img?.currentSrc || desc.img?.src || page.clean || page.original || "";
      wrap.appendChild(img);
      card.appendChild(wrap);
      image.appendChild(card);
    });
  }

  function rewriteQcLabels(workspace) {
    workspace?.querySelectorAll?.(".chapter-qc-result > span:first-child").forEach((label) => {
      if (!label.dataset.qcCanonicalPage) {
        const match = String(label.textContent || "").match(/Trang\s+(\d+)/i);
        if (!match) return;
        label.dataset.qcCanonicalPage = String(Number(match[1]) - 1);
      }
      const canonical = Number(label.dataset.qcCanonicalPage);
      const page = window.currentManifest?.pages?.[canonical];
      if (!page) return;
      const next = `Trang ${sourcePageOf(page, canonical) + 1}`;
      if (label.textContent !== next) label.textContent = next;
    });
  }

  function setQcDisabled(control, locked) {
    if (!(control instanceof HTMLButtonElement || control instanceof HTMLInputElement || control instanceof HTMLSelectElement || control instanceof HTMLTextAreaElement)) return;
    if (locked) {
      if (control.dataset.qcWasDisabled === undefined) control.dataset.qcWasDisabled = control.disabled ? "1" : "0";
      control.disabled = true;
      return;
    }
    if (control.dataset.qcWasDisabled === undefined) return;
    control.disabled = control.dataset.qcWasDisabled === "1";
    delete control.dataset.qcWasDisabled;
  }

  function syncQcLock(workspace) {
    if (!workspace) return;
    const locked = workspace.classList.contains("review-chapter-qc-running");
    document.querySelectorAll(".review-rail-tool").forEach((control) => setQcDisabled(control, locked));
    workspace.querySelectorAll(".review-render-text-btn,.chapter-translate-controls button,.review-inline-translation").forEach((control) => setQcDisabled(control, locked));
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
    const editor = overlay?.querySelector(".review-inline-translation");
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
    ensureQcCompatibility(workspace);
    rewriteQcLabels(workspace);
    syncQcLock(workspace);
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
    if (!workspace.classList.contains("review-chapter-qc-running") || editableTarget(event.target)) return;
    const key = String(event.key || "").toLowerCase();
    const mutatingShortcut = ["v", "r", "o", "b", "e", "[", "]", "delete", "backspace"].includes(key);
    if (mutatingShortcut) {
      event.preventDefault();
      event.stopImmediatePropagation();
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
    if (workspace.classList.contains("review-chapter-qc-running") && event.target?.closest?.(".review-stitched-image")) {
      event.preventDefault();
      event.stopImmediatePropagation();
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

  ensureAdapterStyles();
  const pageView = document.getElementById("page-view");
  if (pageView) {
    const observer = new MutationObserver(scheduleRefresh);
    observer.observe(pageView, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  }
  window.addEventListener("resize", scheduleRefresh);
  queueMicrotask(scheduleRefresh);
})();

document.addEventListener("DOMContentLoaded", () => {
  // Lettering is part of the stitched Review workspace now. Keep the legacy
  // renderer only for old checkpoints, but do not expose a second Editor stage.
  document.querySelector('.sidebar-link[data-stage="editor"]')?.remove();
  const reviewLabel = document.querySelector('.sidebar-link[data-stage="review"] span');
  if (reviewLabel) reviewLabel.textContent = "Xử lý & Biên tập";

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
