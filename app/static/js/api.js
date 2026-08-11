async function parseApiResponse(resp) {
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    try {
      data = { detail: await resp.text() };
    } catch (_) {
      data = {};
    }
  }
  return data || {};
}
window.parseApiResponse = parseApiResponse;

function getErrorMessage(status, data) {
  if (data && data.detail) {
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((e) => (typeof e === "string" ? e : e.msg || JSON.stringify(e)))
        .join("; ");
    }
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail.trim();
    }
  }
  const statusMessages = {
    400: "Yêu cầu không hợp lệ (400)",
    404: "Không tìm thấy dữ liệu (404)",
    422: "Dữ liệu gửi lên không đúng định dạng (422)",
    500: "Lỗi nội bộ máy chủ (500)",
    502: "Máy chủ cổng không phản hồi (502)",
    504: "Hết thời gian chờ phản hồi từ máy chủ (504)",
  };
  return statusMessages[status] || `Máy chủ trả về lỗi HTTP ${status}`;
}
window.getErrorMessage = getErrorMessage;

async function loadRecentChapters() {
  const container = document.getElementById("recent-chapters");
  if (!container) return;
  try {
    const resp = await fetch("/api/chapters");
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      showToast("Không tải được danh sách chapter: " + getErrorMessage(resp.status, data), "error");
      container.innerHTML = "";
      return;
    }
    const chapters = Array.isArray(data) ? data : [];
    if (chapters.length === 0) {
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
    showToast("Không tải được danh sách chapter: " + e.message, "error");
    container.innerHTML = "";
  }
}

async function resumeChapter(chapterId) {
  currentChapterId = chapterId;
  try {
    const resp = await fetch(`/api/chapter/${chapterId}`);
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    currentManifest = data;
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
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    currentManifest = data;
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
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
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
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    currentManifest = data;
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
  const langEl = document.getElementById("lang-select");
  const lang = langEl ? langEl.value : "ja";
  const box = page && page.boxes ? page.boxes[boxIndex] : null;

  if (box && box.ocr_text && box.ocr_lang === lang) {
    originalEl.textContent = box.ocr_text;
    return;
  }
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
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (box) {
      box.ocr_text = data.text || "";
      box.ocr_lang = lang;
    }
    originalEl.textContent = data.text || "(không đọc được)";
  } catch (err) {
    originalEl.textContent = "(OCR lỗi: " + err.message + ")";
    showToast("Lỗi OCR: " + err.message, "error");
  }
}

function scheduleSaveDraft() {
  if (_saveDraftTimer) clearTimeout(_saveDraftTimer);
  _saveDraftTimer = setTimeout(saveDraftNow, 800);
}

async function saveDraftNow() {
  if (!currentChapterId) return;
  const textareas = document.querySelectorAll("textarea[data-page-index]");
  const drafts = currentManifest ? (currentManifest.drafts || (currentManifest.drafts = {})) : {};
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
    const resp = await fetch("/api/save_draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, drafts }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      console.error("Save draft failed:", getErrorMessage(resp.status, data));
    }
  } catch (e) {
    console.error("Save draft network error:", e);
  }
}
window.saveDraftNow = saveDraftNow;
window.scheduleSaveDraft = scheduleSaveDraft;

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

  try {
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

    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      showToast("Chèn chữ thất bại: " + getErrorMessage(resp.status, data), "error");
      return;
    }

    showRenderResult(pageIndex, data.output);
    if (data.warning) {
      showToast(data.warning, "info");
    }
  } catch (err) {
    showToast("Chèn chữ thất bại: " + err.message, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Chèn chữ vào ảnh";
    }
  }
}

async function loadFonts() {
  try {
    const resp = await fetch("/api/fonts");
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    availableFonts = Array.isArray(data) ? data : [{ id: "default", name: "Mặc định (Comic)" }];
  } catch (e) {
    availableFonts = [{ id: "default", name: "Mặc định (Comic)" }];
    showToast("Không thể tải danh sách phông chữ, dùng phông mặc định.", "info");
  }
}

async function submitManualBox(pageIndex, x1, y1, x2, y2) {
  try {
    if (typeof window.cancelPendingPersist === "function") window.cancelPendingPersist();
    const resp = await fetch("/api/add_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, page_index: pageIndex, x1, y1, x2, y2 }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    const newPage = data.pages[pageIndex];
    currentManifest.pages[pageIndex] = newPage;
    refreshPageAfterAddBox(pageIndex, newPage);
  } catch (err) {
    showToast("Thêm vùng thoại thất bại: " + err.message, "error");
  }
}

async function removeBoxAndRepaint(pageIndex, boxIndex, item) {
  try {
    if (typeof window.cancelPendingPersist === "function") window.cancelPendingPersist();
    const resp = await fetch("/api/remove_box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, page_index: pageIndex, box_index: boxIndex }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    const newPage = data.pages[pageIndex];
    currentManifest.pages[pageIndex] = newPage;
    refreshPageAfterRemoveBox(pageIndex, newPage);
  } catch (err) {
    showToast("Xóa vùng thoại thất bại: " + err.message + " — vui lòng tải lại trang để đồng bộ.", "error");
  }
}
window.removeBoxAndRepaint = removeBoxAndRepaint;

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
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    currentManifest.pages[pageIndex] = data.pages[pageIndex];
    return data;
  } catch (err) {
    showToast("Không lưu được vùng cấm dịch: " + err.message, "error");
    throw err;
  }
}

async function resetManualMask(pageIndex, img, canvas, ctx, resetBtn) {
  if (resetBtn) {
    resetBtn.disabled = true;
    resetBtn.textContent = "Đang xóa...";
  }
  try {
    const resp = await fetch("/api/reset_manual_mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, page_index: pageIndex }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    currentManifest.pages[pageIndex] = data.pages[pageIndex];
    if (img) img.src = data.pages[pageIndex].clean + "?t=" + Date.now();
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
  } catch (err) {
    showToast("Không xóa được vùng tô tay: " + err.message, "error");
  } finally {
    if (resetBtn) {
      resetBtn.disabled = false;
      resetBtn.textContent = "Xóa vùng tô tay";
    }
  }
}
