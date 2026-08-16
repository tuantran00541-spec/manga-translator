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
});
