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
  initUpload();
  loadRecentChapters();
  loadFonts();
});
