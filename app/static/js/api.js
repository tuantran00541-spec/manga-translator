// api.js - Quản lý tất cả các kết nối API backend

async function loadRecentChapters() {
  const container = document.getElementById("recent-chapters");
  if (!container) return;
  try {
    const resp = await fetch("/api/chapters");
    const chapters = await resp.json();
    if (!chapters || chapters.length === 0) {
      container.innerHTML = "";
      return;
    }
    container.innerHTML = '<div class="recent-title">📂 Các Chapter đang dịch dở</div>';
    const list = document.createElement("div");
    list.className = "recent-list";
    chapters.forEach((ch) => {
      const card = document.createElement("div");
      card.className = "recent-card";
      const info = document.createElement("div");
      info.className = "recent-info";
      info.innerHTML = `<strong>${ch.chapter_id}</strong><br><span class="recent-url">${ch.source_url || "(không có URL)"}</span><br><span class="recent-meta">${ch.total_pages} trang</span>`;
      card.appendChild(info);
      const btn = document.createElement("button");
      btn.className = "recent-resume-btn";
      btn.textContent = "Tiếp tục dịch";
      btn.addEventListener("click", () => resumeChapter(ch.chapter_id));
      card.appendChild(btn);
      list.appendChild(card);
    });
    container.appendChild(list);
  } catch (e) {
    container.innerHTML = "";
  }
}

async function resumeChapter(chapterId) {
  currentChapterId = chapterId;
  try {
    const resp = await fetch(`/api/chapter/${chapterId}`);
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `lỗi ${resp.status}`);
    }
    currentManifest = await resp.json();
    const recentEl = document.getElementById("recent-chapters");
    if (recentEl) recentEl.innerHTML = "";
    renderEditor();
  } catch (err) {
    showToast("Không tiếp tục được chapter: " + err.message, "error");
  }
}

async function loadChapter() {
  const urlEl = document.getElementById("chapter-url");
  if (!urlEl) return;
  const url = urlEl.value.trim();
  if (!url) return;

  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) {
    loadBtn.disabled = true;
    loadBtn.textContent = "Đang tải...";
  }

  try {
    const resp = await fetch("/api/chapter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, workers: getWorkersSetting() }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `Server trả về lỗi ${resp.status}`);
    }
    currentManifest = await resp.json();
    currentChapterId = currentManifest.chapter_id;
    renderPreview();
  } catch (err) {
    showToast("Không tải được chapter: " + err.message, "error");
  } finally {
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.textContent = "Tải chapter";
    }
  }
}

async function toggleSkip(pageIndex, card, btn) {
  const page = currentManifest.pages[pageIndex];
  const newSkipped = !page.skipped;

  try {
    const resp = await fetch("/api/skip_pages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: currentChapterId,
        page_indices: [pageIndex],
        skipped: newSkipped,
      }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `lỗi ${resp.status}`);
    }
    page.skipped = newSkipped;
    card.classList.toggle("skipped", newSkipped);
    btn.textContent = newSkipped ? "Đã bỏ qua (bấm để hủy)" : "Bỏ qua trang này";
  } catch (err) {
    showToast("Không đổi được trạng thái bỏ qua: " + err.message, "error");
  }
}

function getWorkersSetting() {
  const el = document.getElementById("workers-select");
  const n = el ? parseInt(el.value, 10) : 2;
  if (!Number.isFinite(n) || n < 1) return 2;
  return Math.min(8, n);
}

async function processSelectedPages() {
  const indices = currentManifest.pages
    .map((p, i) => (p.skipped ? null : i))
    .filter((i) => i !== null);

  if (indices.length === 0) {
    showToast("Không có trang nào để xử lý.", "error");
    return;
  }

  const btn = document.querySelector("#preview-toolbar button");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Đang xử lý, vui lòng đợi...";
  }

  try {
    const resp = await fetch("/api/process_pages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: currentChapterId,
        page_indices: indices,
        workers: getWorkersSetting(),
      }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `Server trả về lỗi ${resp.status}`);
    }
    currentManifest = await resp.json();
    renderReview();
  } catch (err) {
    showToast("Xử lý trang thất bại: " + err.message, "error");
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Xử lý các trang đã chọn (bỏ qua trang đã đánh dấu)";
    }
  }
}

async function fetchOcr(pageIndex, boxIndex, originalEl) {
  const page = currentManifest.pages[pageIndex];
  if (page && page.boxes && page.boxes[boxIndex] && page.boxes[boxIndex].ocr_text) {
    originalEl.textContent = page.boxes[boxIndex].ocr_text;
    return;
  }
  const langEl = document.getElementById("lang-select");
  const lang = langEl ? langEl.value : "ja";
  try {
    const resp = await fetch("/api/ocr_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: currentChapterId,
        page_index: pageIndex,
        box_index: boxIndex,
        lang,
      }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `lỗi ${resp.status}`);
    }
    const data = await resp.json();
    originalEl.textContent = data.text || "(không đọc được)";
  } catch (err) {
    originalEl.textContent = "(OCR lỗi: " + err.message + ")";
  }
}

function scheduleSaveDraft() {
  if (_saveDraftTimer) clearTimeout(_saveDraftTimer);
  _saveDraftTimer = setTimeout(saveDraftNow, 800);
}

async function saveDraftNow() {
  if (!currentChapterId) return;
  const textareas = document.querySelectorAll("textarea[data-page-index]");
  const drafts = {};
  textareas.forEach((ta) => {
    const key = `${ta.dataset.pageIndex}_${ta.dataset.boxIndex}`;
    drafts[key] = {
      text: ta.value,
      color: ta.dataset.color || "auto",
      font: ta.dataset.font || "default",
      fontSize: ta.dataset.fontSize || "auto",
      bold: ta.dataset.bold === "true",
      strokeWidth: ta.dataset.strokeWidth || "auto",
      strokeColor: ta.dataset.strokeColor || "auto",
      bgColor: ta.dataset.bgColor || "transparent",
      cornerRadius: ta.dataset.cornerRadius || "0",
    };
  });
  try {
    await fetch("/api/save_draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, drafts }),
    });
  } catch (e) { /* silent fail */ }
}

async function renderTranslations(pageIndex) {
  const textareas = document.querySelectorAll(
    `textarea[data-page-index="${pageIndex}"]`
  );
  const translations = {};
  const colors = {};
  const fonts = {};
  const font_sizes = {};
  const bolds = {};
  const stroke_widths = {};
  const stroke_colors = {};
  const bg_colors = {};
  const corner_radii = {};

  textareas.forEach((ta) => {
    if (ta.value.trim()) {
      const idx = ta.dataset.boxIndex;
      translations[idx] = ta.value.trim();
      colors[idx] = ta.dataset.color || "auto";
      fonts[idx] = ta.dataset.font || "default";
      font_sizes[idx] = ta.dataset.fontSize || "auto";
      bolds[idx] = ta.dataset.bold === "true";
      stroke_widths[idx] = ta.dataset.strokeWidth || "auto";
      stroke_colors[idx] = ta.dataset.strokeColor || "auto";
      bg_colors[idx] = ta.dataset.bgColor || "transparent";
      corner_radii[idx] = parseInt(ta.dataset.cornerRadius || "0");
    }
  });

  const btn = document.querySelector(
    `.page-block[data-page-index="${pageIndex}"] .render-btn`
  );
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Đang chèn chữ...";
  }

  const resp = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chapter_id: currentChapterId,
      page_index: pageIndex,
      translations,
      colors,
      fonts,
      font_sizes,
      bolds,
      stroke_widths,
      stroke_colors,
      bg_colors,
      corner_radii,
    }),
  });

  if (btn) {
    btn.disabled = false;
    btn.textContent = "Chèn chữ vào ảnh";
  }

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    showToast("Chèn chữ thất bại: " + (data.detail || `lỗi ${resp.status}`), "error");
    return;
  }

  const data = await resp.json();
  showRenderResult(pageIndex, data.output);
}

async function loadFonts() {
  try {
    const resp = await fetch("/api/fonts");
    availableFonts = await resp.json();
  } catch (e) {
    availableFonts = [{ id: "default", name: "Mặc định (Comic)" }];
  }
}

async function submitManualBox(pageIndex, x1, y1, x2, y2) {
  try {
    const resp = await fetch("/api/add_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, page_index: pageIndex, x1, y1, x2, y2 }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `lỗi ${resp.status}`);
    }
    const manifest = await resp.json();
    const newPage = manifest.pages[pageIndex];
    currentManifest.pages[pageIndex] = newPage;
    refreshPageAfterAddBox(pageIndex, newPage);
  } catch (err) {
    showToast("Thêm vùng thoại thất bại: " + err.message, "error");
  }
}

async function removeBoxAndRepaint(pageIndex, boxIndex, item) {
  item.remove();
  const overlay = document.querySelector(
    `.box-overlay[data-page-index="${pageIndex}"][data-box-index="${boxIndex}"]`
  );
  if (overlay) overlay.remove();

  try {
    const resp = await fetch("/api/remove_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, page_index: pageIndex, box_index: boxIndex }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `lỗi ${resp.status}`);
    }
    const manifest = await resp.json();
    currentManifest.pages[pageIndex] = manifest.pages[pageIndex];

    const block = document.querySelector(`.page-block[data-page-index="${pageIndex}"]`);
    if (!block) return;
    const img = block.querySelector(".page-image-wrap img");
    img.src = manifest.pages[pageIndex].clean + "?t=" + Date.now();
  } catch (err) {
    showToast("Xóa vùng thoại thất bại: " + err.message + " — vui lòng tải lại trang để đồng bộ.", "error");
  }
}

async function saveExcludedRegions(pageIndex, excludedRegions) {
  try {
    const resp = await fetch("/api/save_excluded_regions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: currentChapterId,
        page_index: pageIndex,
        excluded_regions: excludedRegions,
      }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `lỗi ${resp.status}`);
    }
    const manifest = await resp.json();
    currentManifest.pages[pageIndex] = manifest.pages[pageIndex];
    return manifest;
  } catch (err) {
    showToast("Không lưu được vùng cấm dịch: " + err.message, "error");
    throw err;
  }
}
