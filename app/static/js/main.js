const RESPONSIVE_PROCESS_BATCH_SIZE = 2;

function yieldProcessUi() {
  return new Promise((resolve) => {
    if (
      typeof requestAnimationFrame === "function"
      && document.visibilityState !== "hidden"
    ) {
      requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 0);
    }
  });
}

// api.js owns the public processing guard. Replace only the inner run so the
// existing duplicate-click protection stays authoritative while batches stay
// short enough to keep progress and local UI responsive.
window._processSelectedPagesOnce = async function responsiveProcessSelectedPagesOnce() {
  const pages = currentManifest?.pages || [];
  const indices = pages
    .map((page, index) => (page?.skipped ? null : index))
    .filter((index) => index !== null);

  if (indices.length === 0) {
    showToast("Không có trang nào để xử lý.", "error");
    return;
  }

  const chapterId = currentChapterId;
  const total = indices.length;
  const btn = document.querySelector("#preview-toolbar .preview-primary-action")
    || document.querySelector("#preview-toolbar button");
  let completed = 0;
  let excludedRegionsSaved = false;

  if (btn) {
    // Keep the control focusable/clickable. The outer guard rejects duplicate
    // starts, while the rest of the application remains interactive.
    btn.disabled = false;
    btn.setAttribute("aria-busy", "true");
  }

  try {
    if (btn) btn.textContent = "Đang lưu vùng loại trừ…";
    if (typeof window.flushExcludedRegionSaves === "function") {
      await window.flushExcludedRegionSaves(chapterId);
    }
    excludedRegionsSaved = true;
    if (chapterId !== currentChapterId) return;

    for (let start = 0; start < total; start += RESPONSIVE_PROCESS_BATCH_SIZE) {
      const batch = indices.slice(
        start,
        start + RESPONSIVE_PROCESS_BATCH_SIZE,
      );
      if (btn) btn.textContent = `Đang xử lý ${completed}/${total}…`;

      // Give Chromium/WebView a paint turn before each CPU-heavy request.
      await yieldProcessUi();

      const resp = await fetch("/api/process_pages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: chapterId,
          page_indices: batch,
          workers: getWorkersSetting(),
        }),
      });
      const data = await parseApiResponse(resp);
      if (!resp.ok) {
        throw new Error(getErrorMessage(resp.status, data));
      }
      if (chapterId !== currentChapterId) return;

      currentManifest = data;
      completed += batch.length;
      if (btn) btn.textContent = `Đã xử lý ${completed}/${total}…`;
    }

    if (
      chapterId === currentChapterId
      && document.body?.dataset?.appStage === "preview"
    ) {
      renderReview();
    }
  } catch (err) {
    if (!excludedRegionsSaved) {
      showToast(
        "Không thể bắt đầu xử lý vì vùng loại trừ chưa được lưu: " + err.message,
        "error",
      );
      return;
    }

    let resynced = false;
    if (chapterId === currentChapterId) {
      try {
        await refreshChapterManifest(chapterId);
        resynced = true;
      } catch (syncErr) {
        console.error(
          "Could not resync chapter after partial process failure:",
          syncErr,
        );
      }
    }

    const prefix = completed > 0
      ? `Đã xử lý ít nhất ${completed}/${total} trang. Phần tiếp theo thất bại: `
      : "Xử lý trang thất bại: ";
    showToast(prefix + err.message, "error");

    if (
      resynced
      && document.body?.dataset?.appStage === "preview"
    ) {
      window.previewActivePageIndex = Math.min(
        Number(window.previewActivePageIndex) || 0,
        Math.max(0, (currentManifest?.pages?.length || 1) - 1),
      );
      renderPreview();
    }
  } finally {
    if (btn && btn.isConnected) {
      btn.disabled = false;
      btn.removeAttribute("aria-busy");
      if (
        chapterId === currentChapterId
        && document.body?.dataset?.appStage === "preview"
      ) {
        btn.textContent = completed > 0
          ? "Tiếp tục xử lý"
          : "Bắt đầu xử lý";
      }
    }
  }
};

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
  const savedActive = urlHash || sessionStorage.getItem("mt_active_chapter");
  if (savedActive && typeof resumeChapter === "function") {
    resumeChapter(savedActive);
  }
});
