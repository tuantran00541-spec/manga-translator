const PROCESS_PROGRESS_POLL_MS = 1000;
const PROCESS_PROGRESS_FETCH_TIMEOUT_MS = 4500;
const PROCESS_PROGRESS_MAX_WATCH_MS = 2 * 60 * 60 * 1000;
let _processingProgressWatchSeq = 0;

function processRevisionOf(page) {
  const value = Number(page?.process_revision || 0);
  return Number.isFinite(value) ? value : 0;
}

async function watchProcessingProgress(button, chapterId, indices, baseline, seq) {
  const startedAt = Date.now();
  while (
    seq === _processingProgressWatchSeq
    && chapterId === currentChapterId
    && button.isConnected
    && Date.now() - startedAt < PROCESS_PROGRESS_MAX_WATCH_MS
  ) {
    await new Promise((resolve) => setTimeout(resolve, PROCESS_PROGRESS_POLL_MS));
    if (
      seq !== _processingProgressWatchSeq
      || chapterId !== currentChapterId
      || !button.isConnected
    ) {
      return;
    }

    const buttonText = String(button.textContent || "");
    const processingVisible = /^(Đang lưu vùng loại trừ|Đang xử lý|Đã xử lý)/.test(buttonText);
    if (!button.disabled && !processingVisible) return;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), PROCESS_PROGRESS_FETCH_TIMEOUT_MS);
    try {
      const response = await fetch(`/api/chapter/${encodeURIComponent(chapterId)}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) continue;
      const manifest = await response.json();
      const pages = Array.isArray(manifest?.pages) ? manifest.pages : [];
      let completed = 0;
      indices.forEach((pageIndex) => {
        if (processRevisionOf(pages[pageIndex]) > (baseline.get(pageIndex) || 0)) {
          completed += 1;
        }
      });

      if (
        button.disabled
        && /^(Đang xử lý|Đã xử lý)/.test(String(button.textContent || ""))
      ) {
        button.textContent = `Đang xử lý ${completed}/${indices.length}…`;
      }
    } catch (err) {
      if (err?.name !== "AbortError") {
        console.debug("Processing progress poll skipped:", err);
      }
    } finally {
      clearTimeout(timeout);
    }
  }
}

function installProcessingProgressWatch() {
  document.addEventListener("click", (event) => {
    const target = event.target;
    const button = target?.closest?.("#preview-toolbar .preview-primary-action");
    if (!button || !currentChapterId || !currentManifest) return;

    const pages = Array.isArray(currentManifest.pages) ? currentManifest.pages : [];
    const indices = pages
      .map((page, index) => (page?.skipped ? null : index))
      .filter((index) => index !== null);
    if (!indices.length) return;

    const baseline = new Map(
      indices.map((pageIndex) => [pageIndex, processRevisionOf(pages[pageIndex])]),
    );
    const seq = ++_processingProgressWatchSeq;
    void watchProcessingProgress(button, currentChapterId, indices, baseline, seq);
  }, true);
}

document.addEventListener("DOMContentLoaded", () => {
  installProcessingProgressWatch();

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