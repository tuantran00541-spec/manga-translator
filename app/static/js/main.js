document.addEventListener("DOMContentLoaded", () => {
  if (!document.querySelector('script[data-review-stitch-inspector="1"]')) {
    const reviewInspector = document.createElement("script");
    reviewInspector.src = "/static/js/review-stitch-inspector.js";
    reviewInspector.dataset.reviewStitchInspector = "1";
    reviewInspector.defer = true;
    document.head.appendChild(reviewInspector);
  }

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
