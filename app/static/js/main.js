// main.js - Khởi tạo trạng thái toàn cục & Gắn sự kiện chính

let currentChapterId = null;
let currentManifest = null;
let _saveDraftTimer = null;
let availableFonts = [];

document.addEventListener("DOMContentLoaded", () => {
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) {
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

  initUpload();
  loadRecentChapters();
  loadFonts();
});
