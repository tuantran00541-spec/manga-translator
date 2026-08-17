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

async function setWorkflowCheckpoint(stage, pageIndex) {
  if (!currentChapterId) return;
  const canonicalIndex = Math.max(0, parseInt(pageIndex, 10) || 0);
  try {
    const resp = await fetch("/api/workflow_checkpoint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: currentChapterId,
        stage,
        page_index: canonicalIndex,
      }),
    });
    if (resp.ok && currentManifest) {
      currentManifest.workflow = { stage, page_index: canonicalIndex };
    }
  } catch (err) {
    console.error("Workflow checkpoint save failed:", err);
  }
}
window.setWorkflowCheckpoint = setWorkflowCheckpoint;

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
    const stageLabels = {
      preview: "Xem trước",
      review: "Sửa lỗi ảnh",
      editor: "Biên tập dịch",
    };
    chapters.forEach((ch) => {
      const card = document.createElement("div");
      card.className = "recent-card";
      const info = document.createElement("div");
      info.className = "recent-info";
      const stageText = stageLabels[ch.workflow?.stage] || (ch.workflow?.stage || "Đang dịch");
      info.innerHTML = `<strong>${ch.chapter_id}</strong><br><span class="recent-url">${ch.source_url || "(không có URL)"}</span><br><span class="recent-meta">${ch.total_pages} trang &middot; <span class="recent-stage-badge">${stageText}</span></span>`;
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
  if (!chapterId) return;
  currentChapterId = chapterId;
  try {
    const resp = await fetch(`/api/chapter/${chapterId}`);
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    currentManifest = data;
    currentChapterId = currentManifest.chapter_id;
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      if (window.location.hash !== `#${currentChapterId}`) {
        window.history.replaceState(null, "", `#${currentChapterId}`);
      }
    } catch (_) {}

    const recentEl = document.getElementById("recent-chapters");
    if (recentEl) recentEl.innerHTML = "";

    const pages = currentManifest.pages || [];
    let workflow = currentManifest.workflow;
    if (!workflow || !workflow.stage) {
      if (pages.some((p) => p.rendered)) {
        workflow = { stage: "editor", page_index: 0 };
      } else if (pages.some((p) => p.clean)) {
        workflow = { stage: "review", page_index: 0 };
      } else {
        workflow = { stage: "preview", page_index: 0 };
      }
    }

    const stage = workflow.stage;
    const rawIndex = parseInt(workflow.page_index, 10) || 0;
    const pageIndex = Math.max(0, Math.min(rawIndex, Math.max(0, pages.length - 1)));

    if (stage === "preview") {
      window.previewActivePageIndex = pageIndex;
      renderPreview();
    } else if (stage === "review") {
      window.initialReviewCanonicalPageIndex = pageIndex;
      renderReview();
    } else {
      if (window.editorState) {
        window.editorState.activePageIndex = pageIndex;
      }
      renderEditor();
    }
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
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      window.history.replaceState(null, "", `#${currentChapterId}`);
    } catch (_) {}
    window.previewActivePageIndex = 0;
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
  try {
    if (typeof window.flushAllPendingPersists === "function") {
      await window.flushAllPendingPersists(pageIndex);
    } else if (typeof window.flushTextObjectPersist === "function") {
      await window.flushTextObjectPersist(pageIndex);
    }
  } catch (err) {
    showToast("Không thể chèn chữ vì lưu dữ liệu không thành công.", "error");
    return;
  }
  const page = currentManifest && currentManifest.pages ? currentManifest.pages[pageIndex] : null;
  if (!page) return;

  const translations = {};
  const colors = {};
  const fonts = {};
  const font_sizes = {};
  const bolds = {};
  const stroke_widths = {};
  const stroke_colors = {};
  const bg_colors = {};
  const corner_radii = {};
  const horizontal_aligns = {};
  const vertical_aligns = {};
  (page.text_objects || []).forEach((obj) => {
    if (!obj || !obj.translation || !obj.translation.trim()) return;
    translations[obj.id] = obj.translation.trim();
    const s = obj.style || {};
    colors[obj.id] = s.color || "auto";
    fonts[obj.id] = s.font || "default";
    font_sizes[obj.id] = s.fontSize || "auto";
    bolds[obj.id] = s.bold === true;
    stroke_widths[obj.id] = s.strokeWidth || "auto";
    stroke_colors[obj.id] = s.strokeColor || "auto";
    bg_colors[obj.id] = s.bgColor || "transparent";
    corner_radii[obj.id] = parseInt(s.cornerRadius || "0", 10);
    horizontal_aligns[obj.id] = ["left", "center", "right"].includes(s.horizontalAlign)
      ? s.horizontalAlign
      : "center";
    vertical_aligns[obj.id] = ["top", "middle", "bottom"].includes(s.verticalAlign)
      ? s.verticalAlign
      : "middle";
  });

  const btn = document.querySelector(".editor-render-btn");
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
        horizontal_aligns,
        vertical_aligns,
      }),
    });

    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      showToast("Chèn chữ thất bại: " + getErrorMessage(resp.status, data), "error");
      return;
    }

    if (currentManifest && currentManifest.pages && currentManifest.pages[pageIndex]) {
      currentManifest.pages[pageIndex].rendered = true;
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
